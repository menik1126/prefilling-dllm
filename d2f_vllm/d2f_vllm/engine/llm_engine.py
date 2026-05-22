import atexit

import torch.multiprocessing as mp

from dataclasses import fields, replace
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from typing import List

from d2f_vllm.config import Config
from d2f_vllm.sampling_params import SamplingParams
from d2f_vllm.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM
from d2f_vllm.engine.scheduler import AutoScheduler, SchedulerBase
from d2f_vllm.engine.model_runner import AutoModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = config = Config(model, **config_kwargs)
        self.engine_type = config.model_type
        self.ps = []
        self.events = []
        
        # Check if we're in a distributed environment (e.g., using accelerate)
        import torch.distributed as dist
        if dist.is_initialized():
            # We're in a distributed environment
            current_rank = dist.get_rank()
            world_size = dist.get_world_size()
            # print(f"[DEBUG] Detected distributed environment, rank={current_rank}, world_size={world_size}, tensor_parallel_size={config.tensor_parallel_size}")
            
            # Validate that tensor parallel size matches the distributed setup
            if config.tensor_parallel_size != world_size:
                raise ValueError(f"tensor_parallel_size ({config.tensor_parallel_size}) must match distributed world_size ({world_size}) when using accelerate")
            
            # Create model runner for this rank without spawning processes
            # In distributed mode, each rank is a tensor parallel rank
            self.model_runner = AutoModelRunner.from_config(config, current_rank, [])
        else:
            # Traditional setup: spawn processes for tensor parallelism
            ctx = mp.get_context("spawn")
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=AutoModelRunner.from_config, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
            self.model_runner = AutoModelRunner.from_config(config, 0, self.events)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler: SchedulerBase = AutoScheduler.from_config(config)
        self._exited = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_exited", False):
            return
        self._exited = True
        if hasattr(self, "model_runner") and self.model_runner is not None:
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            try:
                del self.model_runner
            except Exception:
                pass
        for p in getattr(self, "ps", []):
            try:
                p.join()
            except Exception:
                pass

    def _prepare_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        stops = sampling_params.stop
        if stops is None:
            return sampling_params
        if isinstance(stops, str):
            stops = [stops]
        return replace(sampling_params, stop=list(stops), stop_token_ids=None)

    @staticmethod
    def _truncate_stop_text(text: str, stops: str | List[str] | None) -> str:
        if stops is None:
            return text
        if isinstance(stops, str):
            stops = [stops]
        cut_positions = [text.find(stop) for stop in stops if stop and text.find(stop) >= 0]
        return text[:min(cut_positions)] if cut_positions else text

    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams, prompt_positions: List[int] = None):
        sampling_params = self._prepare_sampling_params(sampling_params)
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        if self.engine_type == "causal_lm":
            seq = SequenceForCausalLM(prompt, sampling_params)
        elif self.engine_type == "diffusion_lm":
            seq = SequenceForDiffusionLM(prompt, sampling_params, config=self.config, prompt_positions=prompt_positions)
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")

        self.scheduler.add(seq)
        # Return seq_id so caller can build a stable mapping
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        sample_output = self.model_runner.call("run", seqs, is_prefill)
        n_diff_steps = self.scheduler.postprocess(seqs, sample_output)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if self.engine_type == "causal_lm":
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else len(seqs)
        else:
            num_tokens = sum(seq.input_num_tokens + seq.new_tokens for seq in seqs) if is_prefill else sum(seq.new_tokens for seq in seqs)
        return outputs, num_tokens, is_prefill, n_diff_steps

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
        use_tqdm: bool = True,
        prompt_positions: List[List[int]] = None,
    ) -> List[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        sampling_params = [self._prepare_sampling_params(sp) for sp in sampling_params]
        # Map internal seq_id -> input index to keep output order stable
        seqid_to_idx = {}
        for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            pos = prompt_positions[idx] if prompt_positions is not None else None
            sid = self.add_request(prompt, sp, prompt_positions=pos)
            seqid_to_idx[sid] = idx
        outputs = [None] * len(prompts)
        prefill_throughput = decode_throughput = 0.
        n_steps = 0
        n_diff_steps = [-1] * len(prompts)
        while not self.is_finished():
            t = perf_counter()
            n_steps += 1
            output, num_tokens, is_prefill, cur_n_diff_steps = self.step()
            if use_tqdm:
                if is_prefill:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            if cur_n_diff_steps:
                for seq_id, n_step in cur_n_diff_steps.items():
                    if seq_id in seqid_to_idx and n_step >= 0:
                        n_diff_steps[seqid_to_idx[seq_id]] = n_step
            for seq_id, token_ids in output:
                if seq_id in seqid_to_idx:
                    outputs[seqid_to_idx[seq_id]] = token_ids
                if use_tqdm:
                    pbar.update(1)
        print(f"Finished in {n_steps} steps, prefill throughput: {prefill_throughput:.2f} tok/s, decode throughput: {decode_throughput:.2f} tok/s")
        # Ensure all outputs are present
        assert all(toks is not None for toks in outputs), "Some sequences did not produce outputs"
        formatted_outputs = []
        for token_ids, n_diff_step, sp in zip(outputs, n_diff_steps, sampling_params):
            if self.config.eos in token_ids:
                token_ids = token_ids[:token_ids.index(self.config.eos)]
            text = self.tokenizer.decode(token_ids)
            text = self._truncate_stop_text(text, sp.stop)
            formatted_outputs.append({
                "text": text,
                "token_ids": token_ids,
                "n_diff_steps": n_diff_step,
            })
        outputs = formatted_outputs
        if use_tqdm:
            pbar.close()
        return outputs
