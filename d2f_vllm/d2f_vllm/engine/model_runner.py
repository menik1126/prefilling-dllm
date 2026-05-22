import time

import pickle
import torch
import torch.distributed as dist

from typing import List
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from d2f_vllm.config import Config
from d2f_vllm.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM, SequenceBase
from d2f_vllm.models.auto_model import AutoModelLM
from d2f_vllm.layers.sampler import AutoSampler
from d2f_vllm.utils.checker import CHECK_SLOT_MAPPING
from d2f_vllm.utils.context import (
    set_context_causal_lm, 
    get_context_causal_lm, 
    reset_context_causal_lm,
    set_context_diffusion_lm,
    get_context_diffusion_lm,
    reset_context_diffusion_lm
)

class ModelRunnerBase(ABC):
    """Base class for model runners supporting different model types."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        self.config = config
        self.model_type = config.model_type
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize model, sampler, and kv cache
        # Check if process group is already initialized (e.g., by accelerate)
        self._process_group_initialized_by_us = False
        # print(f"[DEBUG][Rank {rank}] Checking distributed initialization...")
        if not dist.is_initialized():
            init_method = f"tcp://{config.master_addr}:{config.master_port}"
            # print(f"[DEBUG][Rank {rank}] Initializing process group with {init_method}, world_size={self.world_size}")
            dist.init_process_group("nccl", init_method, world_size=self.world_size, rank=rank)
            self._process_group_initialized_by_us = True
            # print(f"[DEBUG][Rank {rank}] Process group initialized successfully")
        else:
            # print(f"[DEBUG][Rank {rank}] Process group already initialized (by accelerate)")
            pass
        
        # Use the actual distributed rank if available, otherwise use the passed rank
        actual_rank = dist.get_rank() if dist.is_initialized() else rank
        self.rank = actual_rank  # Update rank to the actual distributed rank
        
        # When using accelerate, each process should use its local rank as device_id
        # Check if we're in an accelerate environment
        if dist.is_initialized() and not self._process_group_initialized_by_us:
            # We're using accelerate or similar, use local rank for device assignment
            device_id = actual_rank
        else:
            # Traditional tensor parallel setup
            device_id = (getattr(config, "device_start", 0) or 0) + actual_rank
        assert 0 <= device_id <= torch.cuda.device_count(), f"Invalid device_id {device_id}."
        # print(f"[DEBUG][Rank {self.rank}] Setting device to cuda:{device_id}")
        torch.cuda.set_device(device_id)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(f"cuda:{device_id}")
        # print(f"[DEBUG][Rank {self.rank}] Initializing model on device cuda:{device_id}")
        self.model = AutoModelLM.from_config(config)
        # print(f"[DEBUG][Rank {self.rank}] Model initialized successfully")
        self.sampler = AutoSampler.from_config(config)
        # print(f"[DEBUG][Rank {self.rank}] Sampler initialized successfully")
        self.warmup_model()
        self.allocate_kv_cache()  # NOCHANGE
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Allocate shared memory for inter-process communication
        # NOCHANGE
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        # Only set up shared memory communication if we have multiple processes in tensor parallel mode
        if self.world_size > 1:
            # Use the actual distributed rank for shared memory coordination
            actual_rank = dist.get_rank() if dist.is_initialized() else rank
            
            # In distributed environments (like accelerate), we don't use shared memory communication
            # Each rank handles its own model runner independently
            if dist.is_initialized():
                # Skip shared memory setup in distributed environments
                # In distributed mode, each rank will handle model computation directly
                # without inter-process communication via shared memory
                pass
            elif actual_rank == 0:
                # Clean up any existing shared memory with the same name
                try:
                    shm = SharedMemory(name=config.shm_name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    # print(f"[DEBUG][Rank {actual_rank}] Warning: Failed to clean up existing shared memory: {e}")
                    pass
                
                # Create new shared memory
                shm_size = 2**25 if self.model_type == "diffusion_lm" else 2**20
                try:
                    self.shm = SharedMemory(name=config.shm_name, create=True, size=shm_size)
                    # print(f"[DEBUG][Rank {actual_rank}] Successfully created shared memory: {config.shm_name}")
                except Exception as e:
                    # print(f"[DEBUG][Rank {actual_rank}] Failed to create shared memory: {e}")
                    # Try with a unique name if the original fails
                    import time, os
                    unique_name = f"{config.shm_name}_{os.getpid()}_{int(time.time())}"
                    # print(f"[DEBUG][Rank {actual_rank}] Trying with unique name: {unique_name}")
                    self.shm = SharedMemory(name=unique_name, create=True, size=shm_size)
                    # Store the actual name in a temporary file for other ranks to find
                    import tempfile
                    temp_file = f"/tmp/d2f_vllm_shm_name_{os.getpid()}"
                    with open(temp_file, 'w') as f:
                        f.write(unique_name)
                    # print(f"[DEBUG][Rank {actual_rank}] Stored shared memory name in: {temp_file}")
                
                # Ensure all processes wait for rank 0 to finish setup
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                
                # Connect to existing shared memory
                max_retries = 10
                retry_count = 0
                shm_name = config.shm_name
                
                while retry_count < max_retries:
                    try:
                        self.shm = SharedMemory(name=shm_name)
                        # print(f"[DEBUG][Rank {actual_rank}] Successfully connected to shared memory: {shm_name}")
                        break
                    except FileNotFoundError:
                        # Check if rank 0 created a shared memory with a different name
                        import os
                        temp_file = f"/tmp/d2f_vllm_shm_name_{os.getppid()}"  # Use parent process ID
                        if os.path.exists(temp_file):
                            try:
                                with open(temp_file, 'r') as f:
                                    shm_name = f.read().strip()
                                # print(f"[DEBUG][Rank {actual_rank}] Found alternative shared memory name: {shm_name}")
                                continue  # Try again with the new name
                            except Exception as e:
                                # print(f"[DEBUG][Rank {actual_rank}] Failed to read temp file: {e}")
                                pass
                        
                        retry_count += 1
                        if retry_count < max_retries:
                            # print(f"[DEBUG][Rank {actual_rank}] Shared memory not found, retrying ({retry_count}/{max_retries})...")
                            pass
                            import time
                            time.sleep(0.1)
                        else:
                            raise RuntimeError(f"Failed to connect to shared memory after {max_retries} retries")
                
                # Only enter loop mode in traditional tensor parallel setup
                self.loop()

    def exit(self):
        if self.world_size > 1 and hasattr(self, 'shm'):
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and hasattr(self, 'graphs'):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        # Only destroy process group if we initialized it
        if hasattr(self, '_process_group_initialized_by_us') and self._process_group_initialized_by_us:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        # For non-rank-0 processes, self.event should be a single event object
        # But we need to handle the case where it might be passed as a list due to rank confusion
        if isinstance(self.event, list):
            # If we received a list, we need to find the correct event for this rank
            # Use the actual distributed rank to index into the event list
            actual_rank = dist.get_rank() if dist.is_initialized() else self.rank
            if actual_rank > 0 and len(self.event) >= actual_rank:
                event_obj = self.event[actual_rank - 1]
            else:
                # Fallback: use the first event if indexing fails
                event_obj = self.event[0] if self.event else None
                if event_obj is None:
                    raise RuntimeError(f"No valid event object found for rank {actual_rank}")
        else:
            event_obj = self.event
        
        event_obj.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        event_obj.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        if n + 4 > len(self.shm.buf):
            raise ValueError(f"Serialized data size ({n} bytes) exceeds shared memory buffer size ({len(self.shm.buf)} bytes). "
                           f"Consider increasing shared memory size or reducing batch size.")
        
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        
        # For rank 0, self.event should be a list of events for all other ranks
        if isinstance(self.event, list):
            for event in self.event:
                event.set()
        else:
            # Fallback: if it's a single event, just set it
            self.event.set()

    def call(self, method_name, *args):
        # Only use shared memory communication in traditional tensor parallel setup
        # In distributed environments (accelerate), each rank handles its own calls
        if self.world_size > 1 and self.rank == 0 and hasattr(self, 'shm'):
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @abstractmethod
    def warmup_model(self):
        """Model-specific warmup logic."""
        pass

    @abstractmethod
    def allocate_kv_cache(self):
        pass

    def prepare_block_tables(self, seqs: List[SequenceBase]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    @abstractmethod
    def prepare_prefill(self, seqs: List[SequenceBase]):
        """Model-specific prefill preparation."""
        pass

    @abstractmethod
    def prepare_decode(self, seqs: List[SequenceBase]):
        """Model-specific decode preparation."""
        pass

    def prepare_sample(self, seqs: List[SequenceBase]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @abstractmethod
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Model-specific forward pass."""
        pass

    @abstractmethod
    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        """Main inference pipeline."""
        pass

    @abstractmethod
    @torch.inference_mode()
    def capture_cudagraph(self):
        """Model-specific CUDA graph capture."""
        pass


class ModelRunnerForCausalLM(ModelRunnerBase):
    """Model runner for Causal Language Models."""
    def warmup_model(self):
        # return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForCausalLM(test_input_ids) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * 
                       head_dim * hf_config.torch_dtype.itemsize)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - 
                                        used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context_causal_lm(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            seqlen_k = len(seq)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_causal_lm(False, cu_seqlens_k=cu_seqlens_k, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context_causal_lm()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run_verbose(self, seqs: List[SequenceForCausalLM], is_prefill: bool) -> List[int]:
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        s = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) #if self.rank == 0 else None
        print(f"Prepared input in {time.time() - s:.2f} seconds")
        s = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - s:.2f} seconds")
        s = time.time()
        token_ids = self.sampler(logits, temperatures).tolist() #if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - s:.2f} seconds")
        reset_context_causal_lm()
        return token_ids

    def run(self, seqs: List[SequenceForCausalLM], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) #if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context_causal_lm()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context_causal_lm(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context_causal_lm()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )


class ModelRunnerForDiffusionLM(ModelRunnerBase):
    """Model runner for Diffusion Language Models. TODO: Implement DLM-specific logic."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        super().__init__(config, rank, event)
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
            
    def warmup_model(self):
        # return
        print("Warming up model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForDiffusionLM(test_input_ids, config=self.config) for _ in range(num_seqs)]
        self.run(seqs, True)
        for seq in seqs:
            seq.post_process()
        torch.cuda.empty_cache()
        
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)
        get_num_kvcache_blocks = lambda gpu_memory_utilization: int(total * gpu_memory_utilization - 
                                                                    used - peak + current) // block_bytes
        try:
            num_kvcache_blocks = get_num_kvcache_blocks(config.gpu_memory_utilization)
            assert num_kvcache_blocks > 0
        except:
            gpu_memory_utilization = config.gpu_memory_utilization
            while num_kvcache_blocks <= 200: 
                print(f"Warning: GPU memory utilization {gpu_memory_utilization} is too low to allocate kv cache. "
                    f"Automatically adding 0.05, which is {gpu_memory_utilization + 0.05:.2f} now.")
                gpu_memory_utilization += 0.05
                num_kvcache_blocks = get_num_kvcache_blocks(gpu_memory_utilization)
            print(f"Set gpu_memory_utilization to {gpu_memory_utilization:.2f} to allocate kv cache.")
            config.gpu_memory_utilization = gpu_memory_utilization
            
        config.num_kvcache_blocks = num_kvcache_blocks
        print(f"Allocated {config.num_kvcache_blocks} blocks of size {self.block_size} for kv cache on rank {self.rank}.")

        if config.kv_cache_layout == "distinct":
            max_needed = (config.max_model_len + self.block_size - 1) // self.block_size
            max_blocks_cap = max(max_needed * config.max_num_seqs * 8, 240)
            if config.num_kvcache_blocks > max_blocks_cap:
                config.num_kvcache_blocks = max_blocks_cap
                print(f"  Capped to {max_blocks_cap} blocks for distinct layout (max_needed_per_seq={max_needed}).")
            # k_cache: [layer_id, block_id, head, head_dim // x, block_size(segmented seq_len), x]
            # v_cache: [layer_id, block_id, head, head_dim, block_size(segmented seq_len)]
            x = config.k_cache_hdim_split_factor_x
            
            self.k_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim // x, self.block_size, x
            )
            self.v_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim, self.block_size
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.k_cache[layer_id]
                    module.v_cache = self.v_cache[layer_id]
                    layer_id += 1
        elif config.kv_cache_layout == "unified":
            # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
            self.kv_cache = torch.zeros(
                2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                self.block_size, num_kv_heads, head_dim)
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        else:
            raise ValueError(f"Unsupported kv_cache_layout: {config.kv_cache_layout}. "
                             f"Supported values are 'distinct' and 'unified'.")

    def prepare_prefill(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = []
        seq_lens = []

        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)

            total_seqlen = len(seq)
            # tokens and positions to run in this prefill step
            input_ids.extend(seq[seq.cached_num_tokens:])
            if seq.prompt_positions is not None:
                custom_pos = list(seq.prompt_positions[seq.cached_num_tokens:])
                gen_start = max(seq.prompt_positions) + 1
                gen_count = total_seqlen - len(seq.prompt_positions)
                custom_pos.extend(list(range(gen_start, gen_start + gen_count)))
                positions.extend(custom_pos)
            else:
                positions.extend(list(range(seq.cached_num_tokens, total_seqlen)))
            seq_lens.append(total_seqlen)
            context_lens.append(0)
            assert len(input_ids) == len(positions), (
                f"prepare_prefill(diffusion): len(input_ids) {len(input_ids)} != len(positions) {len(positions)}"
            )
            
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            # build slot mapping for prefix cache prompt blocks
            for i in range(0, seq.num_prompt_blocks):
                if seq.block_cache_missed[i]:
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_prompt_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_prompt_num_tokens
                    slot_mapping.extend(list(range(start, end)))
                else:
                    slot_mapping.extend([-1] * self.block_size)
            # pad to a full diffusion block
            slot_mapping.extend([-1] * seq.diffusion_block_size)

        # For diffusion prefill we always need block tables for prefix cache bookkeeping
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # More checks to avoid downstream rotary errors
        assert cu_seqlens_q[-1].item() == input_ids.numel(), (
            f"prepare_prefill(diffusion): cu_seqlens_q[-1]={cu_seqlens_q[-1].item()} != num_tokens={input_ids.numel()}"
        )
        assert cu_seqlens_k[-1].item() == sum(seq_lens), (
            f"prepare_prefill(diffusion): cu_seqlens_k[-1]={cu_seqlens_k[-1].item()} != sum(seq_lens)={sum(seq_lens)}"
        )

        set_context_diffusion_lm(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            seqs=seqs,
            kv_cache_layout=self.config.kv_cache_layout,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        seq_lens = []
        seq_id_to_queue_id = {}
        need_kv_cache_store = False
        # if sum((sum(seq.active_blocks) + sum(seq.to_cache_blocks)) * seq.diffusion_block_size for seq in seqs) == 1536:
        #     pass
        for seq_idx_in_queue, seq in enumerate(seqs): 
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            seq.next_diffusion_step()
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()
            
            seq_lens.append(len(cur_input_ids))
            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)
            
            total_seqlen = len(seq)
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            context_len = context_lens[seq_id_to_queue_id[seq_id]]
            for mem_block_idx in range(0, seq.num_blocks):
                start_idx = mem_block_idx * seq.block_size
                end_idx = start_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                is_last_block = False
                meet_active_block = False
                while start_idx < end_idx and not is_last_block and not meet_active_block:
                    local_start_idx = lambda: start_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_start_idx()]]
                    if diffusion_block.block_id == 0 and diffusion_block.cursor != start_idx:
                        diffusion_block.cursor = start_idx
                    if cur_map[local_start_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True
                    get_step = lambda diff_blk, start_idx: (
                        diff_blk.remaining_length(start_idx)
                        if diff_blk.remaining_length(start_idx) + local_start_idx() <= seq.block_size
                        else seq.block_size - local_start_idx()
                    )
                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        start_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        cur_diffusion_block_start = 0
                        cur_diffusion_block_end = step
                        start_idx += step
                        mem_block_start = seq.block_table[mem_block_idx] * self.block_size + context_len % seq.block_size
                        context_len += step
                        slot_mapping.extend(list(range(mem_block_start + cur_diffusion_block_start,
                                                       mem_block_start + cur_diffusion_block_end)))
                        need_kv_cache_store = True
                    elif diffusion_block.is_active:
                        meet_active_block = True
                        
                if meet_active_block:
                    # Covering all the after-active blocks
                    active = seq.active_blocks
                    first_active_idx = next((i for i, v in enumerate(active) if v), None)
                    if first_active_idx is not None:
                        num_blocks_to_pad = len(active) - first_active_idx
                        padding_slots = [-1] * (num_blocks_to_pad * seq.diffusion_block_size)
                        slot_mapping.extend(padding_slots)
                    break
            assert len(input_ids) == len(positions), f"Input IDs length {len(input_ids)} does not match positions length {len(positions)}"
            assert len(input_ids) == len(slot_mapping), f"Input IDs length {len(input_ids)} does not match slot mapping length {len(slot_mapping)}"

        # CHECK_SLOT_MAPPING(seqs, slot_mapping)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_diffusion_lm(False, slot_mapping=slot_mapping, context_lens=context_lens, 
                                 cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                 block_tables=block_tables, seqs=seqs, 
                                 seq_lens=seq_lens, seq_lens_ts=seq_lens_ts, 
                                 kv_cache_layout=self.config.kv_cache_layout, need_kv_cache_store=need_kv_cache_store)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # print(f"[DEBUG][Rank {self.rank}] run_model called with input_ids.shape={input_ids.shape}, positions.shape={positions.shape}, is_prefill={is_prefill}")
        # print(f"[DEBUG][Rank {self.rank}] world_size={self.world_size}, device={torch.cuda.current_device()}")
        
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # print(f"[DEBUG][Rank {self.rank}] Using eager mode")
            try:
                model_output = self.model(input_ids, positions)
                # print(f"[DEBUG][Rank {self.rank}] EAGER_MODEL_OUTPUT.shape={model_output.shape if model_output is not None else 'None'}")
                logits = self.model.compute_logits(model_output)
                # print(f"[DEBUG][Rank {self.rank}] EAGER_LOGITS.shape={logits.shape if logits is not None else 'None'}")
                return logits
            except Exception as e:
                # print(f"[DEBUG][Rank {self.rank}] Error in eager mode: {e}")
                raise
        else:
            # print(f"[DEBUG][Rank {self.rank}] Using CUDA graph mode")
            try:
                bs = input_ids.size(0)
                context = get_context_diffusion_lm()
                graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
                graph_vars = self.graph_vars
                for k, v in graph_vars.items():
                    if k != "outputs":
                        v.zero_()
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
                graph.replay()
                logits = self.model.compute_logits(graph_vars["outputs"][:bs])
                # print(f"[DEBUG][Rank {self.rank}] CUDAGRAPH_LOGITS.shape={logits.shape if logits is not None else 'None'}")
                return logits
            except Exception as e:
                # print(f"[DEBUG][Rank {self.rank}] Error in CUDA graph mode: {e}")
                raise

    def run_verbose(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        s = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) #if self.rank == 0 else None
        print(f"Prepared input in {time.time() - s:.2f} seconds")
        s = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - s:.2f} seconds")
        s = time.time()
        sample_output = self.sampler(logits, temperatures) #if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - s:.2f} seconds")
        reset_context_diffusion_lm()
        return sample_output

    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        # print(f"[DEBUG][Rank {self.rank}] run() called with {len(seqs)} sequences, is_prefill={is_prefill}")
        
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # print(f"[DEBUG][Rank {self.rank}] prepared input_ids.shape={input_ids.shape}, positions.shape={positions.shape}")
        
        temperatures = self.prepare_sample(seqs) #if self.rank == 0 else None
        # print(f"[DEBUG][Rank {self.rank}] temperatures.shape={temperatures.shape if temperatures is not None else 'None'}")
        
        logits = self.run_model(input_ids, positions, is_prefill)
        # print(f"[DEBUG][Rank {self.rank}] FINAL_LOGITS after run_model: {'None' if logits is None else f'tensor with shape {logits.shape}'}")
        
        if logits is None:
            # print(f"[DEBUG][Rank {self.rank}] WARNING: FINAL_LOGITS is None, this will cause sampler to fail!")
            pass
        
        sample_output = self.sampler(logits, temperatures) #if self.rank == 0 else None
        # print(f"[DEBUG][Rank {self.rank}] sampling completed, sample_output type: {type(sample_output)}")
        
        reset_context_diffusion_lm()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        '''
            TODO: Varlen decoding does not support CUDA graph capture yet.
            Can be implemented, but requires drastically high overhead.
        '''
        raise NotImplementedError("CUDA graph capture for DiffusionLM is not implemented yet.")
    

class AutoModelRunner:
    @classmethod
    def from_config(cls, config: Config, rank: int, event: Event | List[Event]):
        """Factory method to create a model runner based on the model type."""
        if config.model_type == "causal_lm":
            return ModelRunnerForCausalLM(config, rank, event)
        elif config.model_type == "diffusion_lm":
            return ModelRunnerForDiffusionLM(config, rank, event)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")