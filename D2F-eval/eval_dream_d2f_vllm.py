import logging
import gc
import time
import json
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
from peft import PeftConfig, PeftModel
import numpy as np

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")
import random
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def shift_logits(logits):
    shifted_logits = torch.zeros_like(logits)
    shifted_logits[:, 1:, :] = logits[:, :-1, :]
    shifted_logits[:, 0, :] = 1.0
    return shifted_logits

def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    """
    Creates a complete attention mask for the entire sequence with block-based causal attention.
    
    Args:
        prompt_length: Length of the prompt (first irregular block)
        max_length: Maximum total sequence length
        block_size: Size of each regular block
        device: Device to create tensor on
        dtype: Data type for the attention mask
        
    Returns:
        attention_mask: Tensor of shape [1, 1, max_length, max_length]
    """
    # Use the provided dtype or default to bfloat16
    if dtype is None:
        dtype = torch.bfloat16
    
    # Initialize mask with -inf (no attention)
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    
    # Block 0: Prompt (can see itself)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    
    # Calculate the number of regular blocks after prompt
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    
    # Process each regular block
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        
        # Current block can see the prompt
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        
        # Current block can see all previous regular blocks
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        
        # Current block can see itself (full attention within block)
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    
    return attention_mask

def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    """
    Extract the relevant portion of attention mask for current forward pass.
    
    Args:
        full_mask: Complete attention mask [1, 1, max_length, max_length]
        start_pos: Starting position in the full sequence
        input_length: Length of current input sequence
        cache_length: Length of cached sequence
        
    Returns:
        attention_mask: Extracted mask [1, 1, input_length, cache_length + input_length]
    """
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    
    # Extract the relevant rows (current input positions)
    # and columns (cache + current input positions)
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, 
                               device=full_mask.device, dtype=full_mask.dtype)
    
    # Copy cache columns (0 to cache_length in the extracted mask corresponds to 0 to cache_length in full mask)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    
    # Copy current input columns
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    
    return extracted_mask

def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None, dtype=None):
    B, seq_len = input_ids.shape
    # Use the provided dtype or default to float32
    if dtype is None:
        dtype = torch.float32
    # Initialize to all -inf
    attn_mask = torch.full((B, 1, seq_len, seq_len), float('-inf'), dtype=dtype, device=device)
    # 1. Prompt part: each token can attend to the entire prompt
    for i in range(B):
        attn_mask[i, :, :, :prompt_length[i]] = 0.0  # Allow all tokens to see the prompt

        # 2. Block division: divide into blocks starting from prompt_length
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            block_end = min(block_start + block_size, seq_len)

            # Full attention within the block
            attn_mask[i, :, block_start:block_end, block_start:block_end] = 0.0

            # Causal attention between blocks (can only see previous blocks)
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # Current block can see previous blocks
                attn_mask[i, :, block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    
    # Save initial confidence
    confidence = initial_confidence.clone()
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0, initial_confidence

@register_model("dream_lora")
class DreamLoRA(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        lora_path: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        max_length: Optional[int] = 2048,  # Updated to match example code
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.2,  # Updated default
        top_p: Optional[float] = None,  # Updated default
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        block_size: Optional[int] = 4,  # Updated to match example code
        mask_token_id: Optional[int] = 151666,  # Added mask_token_id parameter
        block_add_threshold: Optional[float] = 0.5,  # Added block_add_threshold parameter
        decoded_token_threshold: Optional[int] = 0.9,  # Added decoded_token_threshold parameter
        skip_threshold: Optional[float] = 1.0,  # Added skip_threshold parameter
        sampling_strategy: Optional[str] = "default",  # Added sampling_strategy parameter
        save_dir: Optional[str] = None,
        tensor_parallel_size: Optional[int] = 1,  # Added tensor_parallel_size parameter
        master_addr: Optional[str] = "localhost",  # Added master_addr parameter
        master_port: Optional[int] = 2333,  # Added master_port parameter
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        # Save LoRA path and block_size
        self.lora_path = lora_path
        self.block_size = block_size
        self.block_add_threshold = block_add_threshold  # New block_add_threshold attribute
        self.skip_threshold = skip_threshold  # New skip_threshold attribute
        self.sampling_strategy = sampling_strategy  # Save sampling strategy parameter
        self.decoded_token_threshold = decoded_token_threshold  # New decoded_token_threshold attribute
        self.save_dir = save_dir
        self.tensor_parallel_size = tensor_parallel_size  # Save tensor_parallel_size parameter
        self.master_addr = master_addr  # Save master_addr parameter
        self.master_port = master_port  # Save master_port parameter
        
        # Add metric tracking
        self.total_forward_passes = 0
        self.total_generated_tokens = 0
        self.total_prompts = 0
        # Add time and token statistics
        self.total_generation_time = 0.0
        self.total_block_tokens = 0  # Number of blocks * block_size
        self.total_actual_tokens = 0  # Actual generated tokens (excluding EOS)
        self.total_non_eos_tokens = 0  # Total non-EOS tokens in the entire sequence
        self.all_generation_times = []
        self.all_block_tokens = []
        self.all_actual_tokens = []
        self.all_non_eos_tokens = []
        
        # Save target_dtype for later use
        self.target_dtype = get_dtype(dtype)

        # if isinstance(pretrained, str):
        #     if gpus >= 1 or str(self.device) == "mps":
        #         # TODO: can remove this whole snippet except in the mps case, perhaps?
        #         if not (parallelize or autogptq or hasattr(self, "accelerator")):
        #             # place model onto device requested manually,
        #             # if not using HF Accelerate or device_map
        #             # or any other option that preloads model onto device
        #             try:
        #                 self.model.to(self.device)
        #             except ValueError:
        #                 eval_logger.debug(
        #                     "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
        #                 )
        #     # multigpu data-parallel support when launched with accelerate
        #     if gpus > 1:
        #         if accelerator.num_processes > 1:
        #             if parallelize:
        #                 eval_logger.warning(
        #                     "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
        #                 )
        #             elif gpus > accelerator.num_processes:
        #                 eval_logger.warning(
        #                     "WARNING: The number of total system GPUs does not match the number of spawned processes. "
        #                     "If you would like to use data parallelism, please launch the script "
        #                     "with 'accelerate launch *script*'. "
        #                     f"Current run will proceed with {accelerator.num_processes} devices."
        #                 )
        #                 if self.accelerator.is_local_main_process:
        #                     eval_logger.info(
        #                         f"Using {gpus} devices with data parallelism"
        #                     )

        #             self._device = torch.device(f"{accelerator.device}")
        #             self.accelerator = accelerator

        #             self._rank = self.accelerator.local_process_index
        #             self._world_size = self.accelerator.num_processes
        #         else:
        #             # if we aren't launching via accelerate, ditch
        #             self._rank = 0
        #             self._world_size = 1
        # else:
        #     # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
        #     eval_logger.warning(
        #         "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
        #     )
        #     self._rank = 0
        #     self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until
        self.block_size = block_size
        self.mask_token_id = mask_token_id

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps
        
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        from d2f_vllm import LLM, SamplingParams
        
        self.LLM = LLM(
            pretrained, 
            lora_path=self.lora_path,
            use_lora=True,
            model_name="dream", 
            model_type="diffusion_lm",
            enforce_eager=True, 
            tensor_parallel_size=self.tensor_parallel_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            gpu_memory_utilization=0.60,
            max_num_batched_tokens=2048,
            max_num_seqs=20,
            max_model_len=1024,
            accept_threshold=self.skip_threshold,
            complete_threshold=self.decoded_token_threshold,
            add_new_block_threshold=1-self.block_add_threshold,
            kv_cache_layout="unified"
        )
        self.tokenizer = self.LLM.tokenizer
        self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
        

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        
        # Initialize statistics counters
        if not hasattr(self, 'total_generated_tokens'):
            self.total_generated_tokens = 0
        num_tokens = 0
        num_nfe = 0  # Number of Forward Evaluations
        
        prompts, gen_args = [], []
        print("Preparing prompts...")
        for req in tqdm(requests):
            prompts.append(self.tokenizer.bos_token + req.arguments[0])
            gen_args.append(req.arguments[1])

        start_time = time.time()
        
        outputs = self.LLM.generate(prompts, self.sampling_params)

        end_time = time.time()
        total_time = end_time - start_time
        
        # Accumulate statistics
        res = [output['text'] for output in outputs]
        num_tokens = sum(len(output['token_ids']) for output in outputs)
        num_nfe = sum(output['n_diff_steps'] for output in outputs)

        # Save final statistics
        final_stats = {
            'processed_samples': len(requests),
            'total_samples': len(requests),
            'total_tokens': num_tokens,
            'total_nfe': num_nfe,
            'total_time': total_time,
            'tokens_per_second': num_tokens / total_time if total_time > 0 else 0,
            'nfe_per_token': num_nfe / num_tokens if num_tokens > 0 else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save statistics to file
        if self.save_dir is not None:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            
            # Save response results
            save_path = os.path.join(self.save_dir, f'rank_{self.rank}_responses.jsonl')
            with open(save_path, 'w', encoding='utf-8') as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            
            # Save statistics results
            stats_path = os.path.join(self.save_dir, f'rank_{self.rank}_final_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        
        # Print final statistics
        print("\n" + "="*60)
        print("=== Final Statistics ===")
        print("="*60)
        print(f"Processed Samples: {final_stats['processed_samples']}")
        print(f"Total Samples: {final_stats['total_samples']}")
        print(f"Total Tokens: {final_stats['total_tokens']}")
        print(f"Total NFE: {final_stats['total_nfe']}")
        print(f"Total Time: {final_stats['total_time']:.4f}s")
        print(f"Tokens/Second: {final_stats['tokens_per_second']:.2f}")
        print(f"NFE/Token: {final_stats['nfe_per_token']:.4f}")
        print(f"Completion Time: {final_stats['timestamp']}")
        print("="*60)

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_token_id
            batch = torch.cat([batch, un_batch])

        input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input).logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            # eval_logger.info("before noising")
            perturbed_seq_, p_mask = self._forward_process(seq)
            # eval_logger.info("end noising")
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # by default truncate on the left
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len-cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning(f"All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        print(ds[0])
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                # likelihood calculations are modified from https://github.com/ML-GSAI/SMDM/blob/main/evaluate_diff.py
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()