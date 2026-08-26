import logging
import gc
import time
import json
import math
import importlib
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
import sys
from unittest.mock import MagicMock
import torch

# Stub out the problematic module before it's imported by lm_eval
mock_vlms = MagicMock()
sys.modules["lm_eval.models.hf_vlms"] = mock_vlms
sys.modules["lm_eval.models.vllm_causallms"] = MagicMock()
sys.modules["lm_eval.models.vllm_vlms"] = MagicMock()

# Stub out torch.distributed.tensor to fix PEFT/Lora loading issue on old torch versions
dist_tensor_spec = importlib.util.find_spec("torch.distributed.tensor")
if not hasattr(torch.distributed, "tensor") and dist_tensor_spec is None:
    class DTensorStub: pass
    mock_dist_tensor = MagicMock()
    mock_dist_tensor.DTensor = DTensorStub
    sys.modules["torch.distributed.tensor"] = mock_dist_tensor
    torch.distributed.tensor = mock_dist_tensor

# Ensure transformers has the attribute too if anyone checks
if not hasattr(transformers, "AutoModelForVision2Seq"):
    transformers.AutoModelForVision2Seq = type("AutoModelForVision2Seq", (), {})
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

def build_sparse_block_attention_mask(
    query_block_ids: torch.Tensor,
    cached_length: int,
    query_prompt_window_mask: Optional[torch.Tensor] = None,
    update_kvcache: int = 0,
    device=None,
    dtype=None,
):
    """
    Build the Prefilling-dLLM block-structured attention mask from block ids in physical order.

    RoPE positions are handled separately; this mask only controls which cached/current
    tokens are visible according to Prefilling-dLLM block scheduling.
    """
    if dtype is None:
        dtype = torch.bfloat16
    if device is None:
        device = query_block_ids.device

    query_block_ids = query_block_ids.to(device=device, dtype=torch.long)

    q_len = query_block_ids.shape[0]
    if q_len == 0:
        return torch.empty((1, 1, 0, cached_length), device=device, dtype=dtype)

    # Raw NTK/full-context baseline prefill is a single full-visible prompt block.
    # Passing no mask lets SDPA use its efficient full-attention path instead of
    # materializing an all-zero q_len x q_len mask.
    if cached_length == 0 and torch.all(query_block_ids == query_block_ids[0]):
        return None

    k_len = cached_length + q_len
    attention_mask = torch.full((1, 1, q_len, k_len), -torch.inf, device=device, dtype=dtype)

    if cached_length > 0:
        attention_mask[:, :, :, :cached_length] = 0

    if query_prompt_window_mask is None:
        query_prompt_window_mask = torch.zeros(q_len, device=device, dtype=torch.bool)
    else:
        query_prompt_window_mask = query_prompt_window_mask.to(device=device, dtype=torch.bool)

    for row_idx, q_block_id in enumerate(query_block_ids.tolist()):
        if row_idx < update_kvcache and query_prompt_window_mask[row_idx]:
            visible_current = query_block_ids == q_block_id
        else:
            visible_current = query_block_ids <= q_block_id
        attention_mask[0, 0, row_idx, cached_length:][visible_current] = 0

    return attention_mask

def build_per_layer_sparse_block_attention_mask(
    query_block_ids: torch.Tensor,
    cached_positions_per_layer: List[torch.Tensor],
    query_prompt_window_mask: Optional[torch.Tensor] = None,
    update_kvcache: int = 0,
    device=None,
    dtype=None,
) -> List[torch.Tensor]:
    return [
        build_sparse_block_attention_mask(
            query_block_ids=query_block_ids,
            cached_length=layer_cached_positions.shape[0],
            query_prompt_window_mask=query_prompt_window_mask,
            update_kvcache=update_kvcache,
            device=device,
            dtype=dtype,
        )
        for layer_cached_positions in cached_positions_per_layer
    ]

def build_unified_sparse_block_attention_mask(
    query_block_ids: torch.Tensor,
    cached_positions_per_layer: List[torch.Tensor],
    query_prompt_window_mask: Optional[torch.Tensor] = None,
    update_kvcache: int = 0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    max_cached_length = 0
    if cached_positions_per_layer:
        max_cached_length = max(
            int(layer_cached_positions.shape[0]) for layer_cached_positions in cached_positions_per_layer
        )
    return build_sparse_block_attention_mask(
        query_block_ids=query_block_ids,
        cached_length=max_cached_length,
        query_prompt_window_mask=query_prompt_window_mask,
        update_kvcache=update_kvcache,
        device=device,
        dtype=dtype,
    )

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
        parallelcomp_mode: Optional[bool] = False,
        parallelcomp_pre_runtime_mode: Optional[bool] = False,
        parallelcomp_chunk_size: Optional[int] = 256,
        parallelcomp_query_tokens: Optional[int] = 0,
        parallelcomp_topk_chunks: Optional[int] = 4,
        parallelcomp_min_prompt_tokens: Optional[int] = 1024,
        parallelcomp_keep_first_chunk: Optional[bool] = False,
        parallelcomp_split_from_tail: Optional[bool] = False,
        parallelcomp_recent_token_window: Optional[int] = 0,
        parallelcomp_chunk_score_query_window: Optional[int] = 0,
        parallelcomp_chunk_score_attention_mask: Optional[str] = "query_to_chunk",
        parallelcomp_hidden_topk: Optional[int] = 32,
        parallelcomp_token_capacity: Optional[int] = 128,
        parallelcomp_token_keep_min: Optional[int] = 32,
        parallelcomp_high_score_threshold: Optional[float] = None,
        parallelcomp_cache_compress_mode: Optional[bool] = False,
        parallelcomp_structural_bias: Optional[bool] = False,
        parallelcomp_structural_bias_strength: Optional[float] = 0.2,
        parallelcomp_select_low_score_chunks: Optional[bool] = False,
        parallelcomp_fixed_query_text: Optional[str] = "Please complete the preceding code.",
        parallelcomp_pooling: Optional[str] = "maxpool",
        parallelcomp_pooling_kernel_size: Optional[int] = 7,
        parallelcomp_tail_replay_full_mask: Optional[bool] = True,
        parallelcomp_query_free_cache_rebuild: Optional[bool] = False,
        parallelcomp_score_mode: Optional[str] = "self_information",
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
        self.parallelcomp_mode = parallelcomp_mode
        self.parallelcomp_pre_runtime_mode = parallelcomp_pre_runtime_mode
        self.parallelcomp_chunk_size = parallelcomp_chunk_size
        self.parallelcomp_query_tokens = parallelcomp_query_tokens
        self.parallelcomp_topk_chunks = parallelcomp_topk_chunks
        self.parallelcomp_min_prompt_tokens = parallelcomp_min_prompt_tokens
        self.parallelcomp_keep_first_chunk = parallelcomp_keep_first_chunk
        self.parallelcomp_split_from_tail = parallelcomp_split_from_tail
        self.parallelcomp_recent_token_window = parallelcomp_recent_token_window
        self.parallelcomp_chunk_score_query_window = parallelcomp_chunk_score_query_window
        self.parallelcomp_chunk_score_attention_mask = str(parallelcomp_chunk_score_attention_mask or "query_to_chunk").lower()
        if self.parallelcomp_chunk_score_attention_mask not in {"causal", "full", "full_visible", "query_to_chunk", "prefix_full"}:
            raise ValueError(
                f"Unsupported parallelcomp_chunk_score_attention_mask: {parallelcomp_chunk_score_attention_mask}"
            )
        self.parallelcomp_hidden_topk = parallelcomp_hidden_topk
        self.parallelcomp_token_capacity = parallelcomp_token_capacity
        self.parallelcomp_token_keep_min = parallelcomp_token_keep_min
        self.parallelcomp_high_score_threshold = parallelcomp_high_score_threshold
        self.parallelcomp_cache_compress_mode = parallelcomp_cache_compress_mode
        self.parallelcomp_structural_bias = parallelcomp_structural_bias
        self.parallelcomp_structural_bias_strength = parallelcomp_structural_bias_strength
        self.parallelcomp_select_low_score_chunks = parallelcomp_select_low_score_chunks
        self.parallelcomp_fixed_query_text = parallelcomp_fixed_query_text
        self.parallelcomp_pooling = parallelcomp_pooling
        self.parallelcomp_pooling_kernel_size = parallelcomp_pooling_kernel_size
        self.parallelcomp_tail_replay_full_mask = parallelcomp_tail_replay_full_mask
        self.parallelcomp_query_free_cache_rebuild = bool(parallelcomp_query_free_cache_rebuild)
        self.parallelcomp_score_mode = str(parallelcomp_score_mode or "self_information").lower()
        chunk_bos_env = os.environ.get("PARALLELCOMP_CHUNK_BOS_ABLATION")
        self.parallelcomp_chunk_bos_ablation = (
            True
            if chunk_bos_env is None or chunk_bos_env == ""
            else str(chunk_bos_env).lower() not in {"0", "false", "no", "off"}
        )
        self.parallelcomp_generation_block_bos_ablation = str(
            os.environ.get("PARALLELCOMP_GENERATION_BLOCK_BOS_ABLATION", "")
        ).lower() in {"1", "true", "yes", "on"}
        self._parallelcomp_post_compression_position_offset = None
        self._parallelcomp_active_scoring_query_ids = None
        self._parallelcomp_active_scoring_query_source = None
        self._parallelcomp_active_query_start = None
        self._parallelcomp_active_query_end = None
        
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
        
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

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
        # Get correct data type
        from model_cache.dream.model_dream import DreamModel
        from model_cache.dream.configuration_dream import DreamConfig
        target_dtype = get_dtype(dtype)
        
        # Load base model, using DreamModel and DreamConfig
        model_config = DreamConfig.from_pretrained(pretrained)
        import inspect
        signature = inspect.signature(DreamModel.from_pretrained)
        load_kwargs = {"config": model_config, "torch_dtype": target_dtype, "trust_remote_code": False}
        if "weights_only" in signature.parameters:
            load_kwargs["weights_only"] = True
            
        self.model = DreamModel.from_pretrained(
            pretrained, 
            **load_kwargs
        ).eval()
        
        # Load LoRA config and model
        config = PeftConfig.from_pretrained(self.lora_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_path)
        
        # Only convert data type if target_dtype is not None and not "auto"
        if target_dtype is not None and target_dtype != "auto":
            self.model = self.model.to(target_dtype)
        
        # Move to specified device
        self.model = self.model.to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def _build_full_visible_attention_mask(self, seq_len: int) -> torch.Tensor:
        dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        return torch.zeros((1, 1, seq_len, seq_len), device=self.device, dtype=dtype)

    def _build_causal_attention_mask(self, seq_len: int) -> torch.Tensor:
        dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        mask = torch.full((1, 1, seq_len, seq_len), -torch.inf, device=self.device, dtype=dtype)
        tri = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        mask[0, 0, tri] = 0
        return mask

    def _build_cached_prefix_causal_attention_mask(
        self,
        cached_length: int,
        query_length: int,
    ) -> torch.Tensor:
        dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        mask = torch.full(
            (1, 1, query_length, cached_length + query_length),
            -torch.inf,
            device=self.device,
            dtype=dtype,
        )
        if cached_length > 0:
            mask[:, :, :, :cached_length] = 0
        tri = torch.tril(torch.ones((query_length, query_length), device=self.device, dtype=torch.bool))
        mask[0, 0, :, cached_length:][tri] = 0
        return mask

    def _build_cached_prefix_full_visible_attention_mask(
        self,
        cached_length: int,
        query_length: int,
    ) -> torch.Tensor:
        dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        return torch.zeros(
            (1, 1, query_length, cached_length + query_length),
            device=self.device,
            dtype=dtype,
        )

    def _build_parallelcomp_chunk_query_attention_mask(
        self,
        chunk_len: int,
        query_len: int,
    ) -> torch.Tensor:
        seq_len = chunk_len + query_len
        dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        mask = torch.zeros((1, 1, seq_len, seq_len), device=self.device, dtype=dtype)
        if chunk_len > 0 and query_len > 0:
            mask[:, :, :chunk_len, chunk_len:] = -torch.inf
        return mask

    def _build_parallelcomp_scoring_attention_mask(
        self,
        seq_len: int,
        chunk_len: Optional[int] = None,
        query_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.parallelcomp_chunk_score_attention_mask in {"full", "full_visible"}:
            return self._build_full_visible_attention_mask(seq_len)
        if self.parallelcomp_chunk_score_attention_mask in {"query_to_chunk", "prefix_full"}:
            if chunk_len is None or query_len is None:
                raise ValueError("query_to_chunk scoring mask requires chunk_len and query_len")
            return self._build_parallelcomp_chunk_query_attention_mask(chunk_len, query_len)
        return self._build_causal_attention_mask(seq_len)

    def _pool_parallelcomp_token_scores(
        self,
        token_scores: torch.Tensor,
    ) -> torch.Tensor:
        if token_scores.numel() == 0:
            return token_scores

        pooling = str(self.parallelcomp_pooling or "none").lower()
        if pooling in {"none", "off"}:
            return token_scores

        kernel_size = max(1, int(self.parallelcomp_pooling_kernel_size or 1))
        if kernel_size <= 1:
            return token_scores

        if token_scores.dim() == 1:
            pooled_input = token_scores.view(1, 1, -1)
        elif token_scores.dim() == 2:
            pooled_input = token_scores.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported token score rank for pooling: {token_scores.dim()}")

        if pooling in {"max", "maxpool"}:
            pooled = F.max_pool1d(
                pooled_input,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        elif pooling in {"avg", "avgpool"}:
            pooled = F.avg_pool1d(
                pooled_input,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError(f"Unsupported ParallelComp pooling mode: {self.parallelcomp_pooling}")

        if token_scores.dim() == 1:
            return pooled.view(-1)
        return pooled.squeeze(0)

    def _get_parallelcomp_query_window_size(
        self,
        query_len: int,
    ) -> int:
        if query_len <= 0:
            return 0
        recent_window = int(self.parallelcomp_recent_token_window or 0)
        if recent_window <= 0:
            return query_len
        return min(query_len, recent_window)

    def _get_parallelcomp_chunk_score_query_window_size(
        self,
        query_len: int,
    ) -> int:
        if query_len <= 0:
            return 0
        score_window = int(self.parallelcomp_chunk_score_query_window or 0)
        if score_window <= 0:
            return query_len
        return min(query_len, score_window)

    def _analyze_chunk_with_attentions(
        self, chunk_ids: List[int], query_ids: List[int]
    ) -> Tuple[float, torch.Tensor]:
        joint_ids = torch.tensor([chunk_ids + query_ids], device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_ids.shape[1],
            chunk_len=len(chunk_ids),
            query_len=len(query_ids),
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        attn_layers = outputs.attentions
        if attn_layers is None or len(attn_layers) == 0:
            return float("-inf"), torch.empty(0, device=self.device)

        selected_layers = attn_layers[-4:] if len(attn_layers) >= 4 else attn_layers
        stacked_attn = torch.stack(selected_layers, dim=0).mean(dim=0)[0]

        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        if chunk_len == 0 or query_len == 0:
            return float("-inf"), torch.empty(0, device=self.device)
        query_window = self._get_parallelcomp_query_window_size(query_len)
        if query_window <= 0:
            return float("-inf"), torch.empty(0, device=self.device)

        query_start = chunk_len + query_len - query_window
        cross_attn = stacked_attn[:, query_start:chunk_len + query_len, :chunk_len]
        if cross_attn.numel() == 0:
            return float("-inf"), torch.empty(0, device=self.device)

        token_scores = cross_attn.mean(dim=0).sum(dim=0)
        token_scores = self._pool_parallelcomp_token_scores(token_scores)
        topk = min(max(1, self.parallelcomp_hidden_topk), token_scores.shape[0])
        score = token_scores.topk(topk).values.mean().item()
        return float(score), token_scores

    def _score_chunk_with_self_information(
        self, chunk_ids: List[int], query_ids: List[int]
    ) -> float:
        if len(chunk_ids) == 0 or len(query_ids) == 0:
            return float("inf")

        joint_ids = torch.tensor([chunk_ids + query_ids], device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_ids.shape[1],
            chunk_len=len(chunk_ids),
            query_len=len(query_ids),
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        query_window = self._get_parallelcomp_chunk_score_query_window_size(query_len)
        if query_window <= 0:
            return float("inf")
        if logits.shape[1] < chunk_len + query_len - 1:
            return float("inf")

        query_logits = logits[:, chunk_len + query_len - query_window - 1:chunk_len + query_len - 1, :]
        query_labels = joint_ids[:, chunk_len + query_len - query_window:chunk_len + query_len]
        log_probs = F.log_softmax(query_logits, dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
        return float(token_nll.mean().item())

    def _score_chunk_with_next_block_logits(
        self, chunk_ids: List[int], query_ids: List[int], next_block_size: Optional[int] = None
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Score chunk by predicting next block content from query tail logits.

        Forward pass on [chunk + query], then use the last `next_block_size`
        positions of query logits to predict pseudo next-block tokens. The score
        is the negative NLL of those argmax pseudo-labels, so higher is better.

        Returns:
            (score, predicted_tokens, predicted_confidences)
            - score: negative mean NLL of predicted next-block tokens
            - predicted_tokens: [next_block_size] tensor of argmax token ids
            - predicted_confidences: [next_block_size] tensor of top-1 probabilities
        """
        if next_block_size is None:
            next_block_size = self.block_size
        if len(chunk_ids) == 0 or len(query_ids) == 0:
            return float("-inf"), torch.tensor([], device=self.device), torch.tensor([], device=self.device)

        chunk_len = len(chunk_ids)
        query_len = len(query_ids)

        joint_ids = torch.tensor([chunk_ids + query_ids], device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_ids.shape[1],
            chunk_len=chunk_len,
            query_len=query_len,
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits
        total_len = chunk_len + query_len

        # Take logits from the last next_block_size positions of the sequence.
        # logits[:, t, :] predicts token at position t+1.
        # So logits[:, -next_block_size:, :] predicts the next block's tokens.
        predict_len = min(next_block_size, total_len)
        next_block_logits = logits[:, total_len - predict_len:total_len, :]

        log_probs = F.log_softmax(next_block_logits.squeeze(0), dim=-1)
        predicted_tokens = log_probs.argmax(dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=predicted_tokens.unsqueeze(-1)).squeeze(-1)
        predicted_confidences = log_probs.max(dim=-1).values.exp()

        score = -float(token_nll.mean().item())
        return score, predicted_tokens, predicted_confidences

    def _score_prompt_blocks_with_hidden_resonance(
        self,
        x_t: torch.Tensor,
        block_states,
        candidate_block_ids: List[int],
        query_block_id: int,
        stable_prefix_len: int,
    ) -> dict:
        if not candidate_block_ids or query_block_id not in block_states or stable_prefix_len <= 0:
            return {}

        prompt_ids = x_t[:, :stable_prefix_len]
        attention_mask = self._build_causal_attention_mask(prompt_ids.shape[1])

        with torch.inference_mode():
            outputs = self.model(
                prompt_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            return {}

        selected_hidden_layers = hidden_states[-min(4, len(hidden_states) - 1):]
        hidden = torch.stack([layer_hidden[0] for layer_hidden in selected_hidden_layers], dim=0).mean(dim=0)

        query_state = block_states[query_block_id]
        query_start = min(query_state["start_pos"], stable_prefix_len)
        query_end = min(query_state["end_pos"], stable_prefix_len)
        if query_end <= query_start:
            return {}

        query_hidden = hidden[query_start:query_end]
        if query_hidden.numel() == 0:
            return {}

        query_hidden = F.normalize(query_hidden, dim=-1)
        scores = {}
        for block_id in candidate_block_ids:
            state = block_states[block_id]
            block_start = min(state["start_pos"], stable_prefix_len)
            block_end = min(state["end_pos"], stable_prefix_len)
            if block_end <= block_start:
                continue
            block_hidden = hidden[block_start:block_end]
            if block_hidden.numel() == 0:
                continue
            block_hidden = F.normalize(block_hidden, dim=-1)
            similarity = torch.matmul(query_hidden, block_hidden.transpose(0, 1))
            token_resonance = similarity.max(dim=0).values
            topk = min(max(1, int(self.parallelcomp_hidden_topk)), token_resonance.shape[0])
            score = float(token_resonance.topk(topk).values.mean().item())
            scores[block_id] = score

        return scores

    def _score_prompt_blocks_with_attention_resonance(
        self,
        x_t: torch.Tensor,
        block_states,
        candidate_block_ids: List[int],
        query_block_id: int,
        stable_prefix_len: int,
    ) -> dict:
        if not candidate_block_ids or query_block_id not in block_states or stable_prefix_len <= 0:
            return {}

        prompt_ids = x_t[:, :stable_prefix_len]
        attention_mask = self._build_causal_attention_mask(prompt_ids.shape[1])

        with torch.inference_mode():
            outputs = self.model(
                prompt_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        attn_layers = outputs.attentions
        if attn_layers is None or len(attn_layers) == 0:
            return {}

        selected_layers = attn_layers[-min(4, len(attn_layers)):]
        query_state = block_states[query_block_id]
        query_start = min(query_state["start_pos"], stable_prefix_len)
        query_end = min(query_state["end_pos"], stable_prefix_len)
        if query_end <= query_start:
            return {}

        scores = {}
        for block_id in candidate_block_ids:
            state = block_states[block_id]
            block_start = min(state["start_pos"], stable_prefix_len)
            block_end = min(state["end_pos"], stable_prefix_len)
            if block_end <= block_start:
                continue

            layer_scores = []
            for layer_attn in selected_layers:
                layer_attn = layer_attn[0]
                cross_attn = layer_attn[:, query_start:query_end, block_start:block_end]
                if cross_attn.numel() == 0:
                    continue
                token_scores = cross_attn.mean(dim=0).sum(dim=0)
                topk = min(max(1, int(self.parallelcomp_hidden_topk)), token_scores.shape[0])
                layer_scores.append(token_scores.topk(topk).values.mean())

            if layer_scores:
                scores[block_id] = float(torch.stack(layer_scores).mean().item())

        return scores

    def _analyze_chunk_with_attentions_per_layer(
        self, chunk_ids: List[int], query_ids: List[int]
    ) -> List[torch.Tensor]:
        joint_ids = torch.tensor([chunk_ids + query_ids], device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_ids.shape[1],
            chunk_len=len(chunk_ids),
            query_len=len(query_ids),
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        attn_layers = outputs.attentions
        if attn_layers is None or len(attn_layers) == 0:
            return []

        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        if chunk_len == 0 or query_len == 0:
            return []
        query_window = self._get_parallelcomp_query_window_size(query_len)
        if query_window <= 0:
            return []

        token_scores_per_layer = []
        for layer_attn in attn_layers:
            layer_attn = layer_attn[0]
            query_start = chunk_len + query_len - query_window
            cross_attn = layer_attn[:, query_start:chunk_len + query_len, :chunk_len]
            if cross_attn.numel() == 0:
                token_scores_per_layer.append(torch.empty(0, device=self.device))
                continue
            token_scores = cross_attn.mean(dim=0).sum(dim=0)
            token_scores = self._pool_parallelcomp_token_scores(token_scores)
            token_scores_per_layer.append(token_scores)
        return token_scores_per_layer

    def _analyze_chunk_with_attentions_per_layer_per_head(
        self,
        chunk_ids: List[int],
        query_ids: List[int],
        num_cache_heads_per_layer: List[int],
    ) -> List[torch.Tensor]:
        joint_ids = torch.tensor([chunk_ids + query_ids], device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_ids.shape[1],
            chunk_len=len(chunk_ids),
            query_len=len(query_ids),
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        attn_layers = outputs.attentions
        if attn_layers is None or len(attn_layers) == 0:
            return []

        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        if chunk_len == 0 or query_len == 0:
            return []
        query_window = self._get_parallelcomp_query_window_size(query_len)
        if query_window <= 0:
            return []

        token_scores_per_layer_per_head = []
        for layer_idx, layer_attn in enumerate(attn_layers):
            layer_attn = layer_attn[0]
            query_start = chunk_len + query_len - query_window
            cross_attn = layer_attn[:, query_start:chunk_len + query_len, :chunk_len]
            if cross_attn.numel() == 0:
                num_cache_heads = num_cache_heads_per_layer[layer_idx] if layer_idx < len(num_cache_heads_per_layer) else 1
                token_scores_per_layer_per_head.append(
                    torch.empty((num_cache_heads, 0), device=self.device)
                )
                continue

            query_head_scores = cross_attn.sum(dim=1)
            num_cache_heads = num_cache_heads_per_layer[layer_idx] if layer_idx < len(num_cache_heads_per_layer) else query_head_scores.shape[0]
            query_head_scores = self._pool_parallelcomp_token_scores(query_head_scores)
            grouped_scores = []
            for head_group in torch.tensor_split(query_head_scores, num_cache_heads, dim=0):
                if head_group.shape[0] == 0:
                    grouped_scores.append(torch.zeros(chunk_len, device=self.device, dtype=query_head_scores.dtype))
                else:
                    grouped_scores.append(head_group.mean(dim=0))
            token_scores_per_layer_per_head.append(torch.stack(grouped_scores, dim=0))
        return token_scores_per_layer_per_head

    def _score_parallelcomp_query_from_logits(
        self,
        logits: torch.Tensor,
        joint_ids: torch.Tensor,
        chunk_len: int,
        query_len: int,
    ) -> float:
        query_window = self._get_parallelcomp_chunk_score_query_window_size(query_len)
        if query_window <= 0 or logits.shape[1] < chunk_len + query_len - 1:
            return float("-inf")
        query_logits = logits[:, chunk_len + query_len - query_window - 1:chunk_len + query_len - 1, :]
        query_labels = joint_ids[:, chunk_len + query_len - query_window:chunk_len + query_len]
        if query_logits.shape[1] != query_labels.shape[1]:
            return float("-inf")
        log_probs = F.log_softmax(query_logits, dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
        return float(-token_nll.mean().item())

    def _token_scores_from_parallelcomp_attentions(
        self,
        attn_layers,
        chunk_len: int,
        query_len: int,
        num_cache_heads_per_layer: List[int],
    ) -> List[torch.Tensor]:
        if attn_layers is None or len(attn_layers) == 0 or chunk_len <= 0 or query_len <= 0:
            return []
        query_window = self._get_parallelcomp_query_window_size(query_len)
        if query_window <= 0:
            return []

        query_start = chunk_len + query_len - query_window
        query_end = chunk_len + query_len
        token_scores_per_layer_per_head = []
        for layer_idx, layer_attn in enumerate(attn_layers):
            layer_attn = layer_attn[0]
            cross_attn = layer_attn[:, query_start:query_end, :chunk_len]
            num_cache_heads = (
                num_cache_heads_per_layer[layer_idx]
                if layer_idx < len(num_cache_heads_per_layer)
                else layer_attn.shape[0]
            )
            if cross_attn.numel() == 0:
                token_scores_per_layer_per_head.append(
                    torch.empty((num_cache_heads, 0), device=self.device)
                )
                continue

            query_head_scores = cross_attn.sum(dim=1)
            query_head_scores = self._pool_parallelcomp_token_scores(query_head_scores)
            grouped_scores = []
            for head_group in torch.tensor_split(query_head_scores, num_cache_heads, dim=0):
                if head_group.shape[0] == 0:
                    grouped_scores.append(torch.zeros(chunk_len, device=self.device, dtype=query_head_scores.dtype))
                else:
                    grouped_scores.append(head_group.mean(dim=0))
            token_scores_per_layer_per_head.append(torch.stack(grouped_scores, dim=0))
        return token_scores_per_layer_per_head

    def _select_cache_block_token_indices_from_scores_per_layer_per_head(
        self,
        token_scores_per_layer_per_head: List[torch.Tensor],
        block_len: int,
        num_cache_heads_per_layer: List[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        if block_len <= 0:
            return [], []
        if not token_scores_per_layer_per_head:
            default_indices = []
            for num_heads in num_cache_heads_per_layer:
                base = torch.arange(block_len, device=self.device, dtype=torch.long)
                default_indices.append(base.unsqueeze(0).expand(num_heads, -1).clone())
            return default_indices, [0 for _ in num_cache_heads_per_layer]

        keep_indices_per_layer_per_head = []
        evicted_high_per_layer = []
        total_layers = max(1, len(token_scores_per_layer_per_head))
        for layer_idx, head_scores in enumerate(token_scores_per_layer_per_head):
            num_heads = (
                num_cache_heads_per_layer[layer_idx]
                if layer_idx < len(num_cache_heads_per_layer)
                else head_scores.shape[0]
            )
            if head_scores.numel() == 0:
                base = torch.arange(block_len, device=self.device, dtype=torch.long)
                keep_indices_per_layer_per_head.append(base.unsqueeze(0).expand(num_heads, -1).clone())
                evicted_high_per_layer.append(0)
                continue

            keep_min = min(block_len, max(1, int(self.parallelcomp_token_keep_min)))
            token_capacity = max(1, int(self.parallelcomp_token_capacity))
            keep_count = min(block_len, max(keep_min, token_capacity))

            head_keep_indices = []
            biased_head_scores = []
            evicted_high = 0
            for head_idx in range(head_scores.shape[0]):
                biased_scores = self._apply_layer_structural_bias(
                    token_scores=head_scores[head_idx],
                    layer_idx=layer_idx,
                    num_layers=total_layers,
                )
                biased_head_scores.append(biased_scores)
                selected_indices, head_evicted_high = self._select_token_indices_from_scores(
                    token_scores=biased_scores,
                    keep_count=keep_count,
                    keep_min=keep_min,
                )
                head_keep_indices.append(torch.tensor(selected_indices, device=self.device, dtype=torch.long))
                evicted_high += head_evicted_high

            padded_head_indices = []
            for head_idx, index_tensor in enumerate(head_keep_indices):
                if index_tensor.shape[0] < keep_count:
                    token_scores = biased_head_scores[head_idx]
                    remaining_mask = torch.ones(token_scores.shape[0], dtype=torch.bool, device=token_scores.device)
                    remaining_mask[index_tensor] = False
                    remaining_indices = torch.arange(token_scores.shape[0], device=token_scores.device)[remaining_mask]
                    if remaining_indices.numel() > 0:
                        supplement_scores = token_scores[remaining_indices]
                        supplement_local = supplement_scores.topk(
                            min(keep_count - index_tensor.shape[0], remaining_indices.shape[0]),
                            largest=False,
                        ).indices
                        supplement_indices = remaining_indices[supplement_local]
                        index_tensor = torch.cat([index_tensor, supplement_indices], dim=0)
                if index_tensor.shape[0] < keep_count:
                    raise RuntimeError(
                        "Failed to build a fixed-width per-head selection set for ParallelComp token eviction"
                    )
                padded_head_indices.append(torch.sort(index_tensor[:keep_count]).values)

            keep_indices_per_layer_per_head.append(torch.stack(padded_head_indices, dim=0))
            evicted_high_per_layer.append(evicted_high)

        return keep_indices_per_layer_per_head, evicted_high_per_layer

    def _run_parallelcomp_local_block_forward(
        self,
        block_input_ids: torch.Tensor,
        query_ids: List[int],
        reused_window_size: int,
    ) -> Optional[Dict[str, Any]]:
        block_len = int(block_input_ids.shape[1])
        query_len = len(query_ids)
        if block_len <= 0 or query_len <= 0:
            return None

        # One local ParallelComp pass: score the chunk from query NLL, score
        # tokens from query->chunk attention, and keep the chunk KV slice.
        query_input_ids = torch.tensor([query_ids], device=self.device, dtype=torch.long)
        joint_ids = torch.cat([block_input_ids, query_input_ids], dim=1)
        joint_len = block_len + query_len
        block_positions = torch.arange(block_len, device=self.device, dtype=torch.long)
        query_positions = torch.arange(
            reused_window_size,
            reused_window_size + query_len,
            device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.cat([block_positions, query_positions], dim=0).unsqueeze(0)
        cache_position = torch.arange(joint_len, device=self.device, dtype=torch.long)
        attention_mask = self._build_parallelcomp_scoring_attention_mask(
            joint_len,
            chunk_len=block_len,
            query_len=query_len,
        )

        with torch.inference_mode():
            outputs = self.model(
                joint_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                output_attentions=True,
                return_dict=True,
                use_cache=True,
                update_kvcache=joint_len,
            )

        block_cache = outputs.past_key_values
        num_cache_heads_per_layer = [layer_k.shape[1] for layer_k in block_cache.key_cache]
        token_scores_per_layer_per_head = self._token_scores_from_parallelcomp_attentions(
            outputs.attentions,
            chunk_len=block_len,
            query_len=query_len,
            num_cache_heads_per_layer=num_cache_heads_per_layer,
        )
        per_layer_kept_indices, evicted_high_layers = self._select_cache_block_token_indices_from_scores_per_layer_per_head(
            token_scores_per_layer_per_head=token_scores_per_layer_per_head,
            block_len=block_len,
            num_cache_heads_per_layer=num_cache_heads_per_layer,
        )
        score = self._score_parallelcomp_query_from_logits(
            logits=outputs.logits,
            joint_ids=joint_ids,
            chunk_len=block_len,
            query_len=query_len,
        )

        return {
            "score": score,
            "key_cache": [layer_k[:, :, :block_len, :] for layer_k in block_cache.key_cache],
            "value_cache": [layer_v[:, :, :block_len, :] for layer_v in block_cache.value_cache],
            "kept_indices": per_layer_kept_indices,
            "evicted_high_layers": evicted_high_layers,
            "num_cache_heads_per_layer": num_cache_heads_per_layer,
            "block_len": block_len,
        }

    def _build_layer_structural_prior(
        self,
        seq_len: int,
        layer_idx: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if seq_len <= 1 or num_layers <= 0:
            return torch.zeros(seq_len, device=device, dtype=dtype)

        position = torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=dtype)
        start_prior = 1.0 - position
        end_prior = position
        middle_prior = 1.0 - torch.abs(position - 0.5) * 2.0

        early_end = max(1, int(math.ceil(num_layers * 0.25)))
        middle_end = max(early_end + 1, int(math.ceil(num_layers * 0.5)))
        deep_start = max(middle_end, int(math.floor(num_layers * 0.75)))

        if layer_idx < early_end:
            prior = start_prior
        elif layer_idx < middle_end:
            prior = end_prior
        elif layer_idx >= deep_start:
            prior = middle_prior
        else:
            prior = torch.zeros_like(position)

        return prior - prior.mean()

    def _apply_layer_structural_bias(
        self,
        token_scores: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        if (
            not self.parallelcomp_structural_bias
            or token_scores.numel() == 0
            or num_layers <= 0
        ):
            return token_scores

        prior = self._build_layer_structural_prior(
            seq_len=token_scores.shape[-1],
            layer_idx=layer_idx,
            num_layers=num_layers,
            device=token_scores.device,
            dtype=token_scores.dtype,
        )
        score_scale = token_scores.mean().abs().clamp_min(1e-6)
        return token_scores + prior * score_scale * float(self.parallelcomp_structural_bias_strength)

    def _select_token_indices_from_scores(
        self,
        token_scores: torch.Tensor,
        keep_count: int,
        keep_min: int,
    ) -> Tuple[List[int], int]:
        candidate_indices = torch.arange(token_scores.shape[0], device=token_scores.device)
        high_threshold = self.parallelcomp_high_score_threshold
        evicted_high = 0
        evicted_pool = torch.empty(0, device=token_scores.device, dtype=torch.long)
        if high_threshold is not None and token_scores.shape[0] > keep_min:
            candidate_mask = token_scores <= float(high_threshold)
            if candidate_mask.sum().item() >= keep_min:
                evicted_high = int((~candidate_mask).sum().item())
                evicted_pool = candidate_indices[~candidate_mask]
                candidate_indices = candidate_indices[candidate_mask]

        candidate_scores = token_scores[candidate_indices]
        if candidate_scores.numel() >= keep_count:
            selected_local = candidate_scores.topk(keep_count).indices
            selected_indices = candidate_indices[selected_local]
        else:
            selected_indices = candidate_indices
            need = keep_count - selected_indices.shape[0]
            if need > 0:
                if evicted_pool.numel() > 0:
                    supplement_scores = token_scores[evicted_pool]
                    supplement_local = supplement_scores.topk(min(need, evicted_pool.shape[0]), largest=False).indices
                    supplement_indices = evicted_pool[supplement_local]
                else:
                    remaining_mask = torch.ones(token_scores.shape[0], dtype=torch.bool, device=token_scores.device)
                    if selected_indices.numel() > 0:
                        remaining_mask[selected_indices] = False
                    remaining_indices = torch.arange(token_scores.shape[0], device=token_scores.device)[remaining_mask]
                    remaining_scores = token_scores[remaining_indices]
                    supplement_local = remaining_scores.topk(min(need, remaining_indices.shape[0]), largest=False).indices
                    supplement_indices = remaining_indices[supplement_local]
                selected_indices = torch.cat([selected_indices, supplement_indices], dim=0)

        selected_indices = torch.sort(torch.unique(selected_indices)).values.tolist()
        if len(selected_indices) > keep_count:
            selected_indices = selected_indices[:keep_count]
        if len(selected_indices) == 0 and token_scores.shape[0] > 0:
            selected_indices = [int(torch.argmax(token_scores).item())]
        return selected_indices, evicted_high

    def _prune_chunk_tokens_parallelcomp(
        self, chunk_ids: List[int], token_scores: torch.Tensor
    ) -> Tuple[List[int], int]:
        if len(chunk_ids) == 0 or token_scores.numel() == 0:
            return chunk_ids, 0

        keep_min = min(len(chunk_ids), max(1, int(self.parallelcomp_token_keep_min)))
        token_capacity = max(1, int(self.parallelcomp_token_capacity))
        keep_count = min(len(chunk_ids), max(keep_min, token_capacity))
        keep_count = min(len(chunk_ids), keep_count)

        candidate_indices = torch.arange(token_scores.shape[0], device=token_scores.device)
        high_threshold = self.parallelcomp_high_score_threshold
        evicted_high = 0
        if high_threshold is not None and token_scores.shape[0] > keep_min:
            candidate_mask = token_scores <= float(high_threshold)
            if candidate_mask.sum().item() >= keep_min:
                evicted_high = int((~candidate_mask).sum().item())
                candidate_indices = candidate_indices[candidate_mask]

        candidate_scores = token_scores[candidate_indices]
        if candidate_scores.numel() <= keep_count:
            selected_indices = candidate_indices
        else:
            selected_local = candidate_scores.topk(keep_count).indices
            selected_indices = candidate_indices[selected_local]

        selected_indices = torch.sort(selected_indices).values.tolist()
        pruned_chunk_ids = [chunk_ids[idx] for idx in selected_indices]
        return pruned_chunk_ids, evicted_high

    def _select_cache_block_token_indices(
        self, block_ids: List[int], query_ids: List[int]
    ) -> Tuple[List[int], int]:
        if len(block_ids) == 0:
            return [], 0
        score, token_scores = self._analyze_chunk_with_attentions(block_ids, query_ids)
        del score
        pruned_block_ids, evicted_high = self._prune_chunk_tokens_parallelcomp(block_ids, token_scores)
        kept_token_set = set(pruned_block_ids)
        kept_indices = [idx for idx, tok in enumerate(block_ids) if tok in kept_token_set]
        if len(kept_indices) != len(pruned_block_ids):
            # Fall back to positional top-k when repeated token ids make set matching ambiguous.
            keep_count = len(pruned_block_ids)
            candidate_indices = torch.arange(token_scores.shape[0], device=token_scores.device)
            high_threshold = self.parallelcomp_high_score_threshold
            if high_threshold is not None and token_scores.shape[0] > keep_count:
                candidate_mask = token_scores <= float(high_threshold)
                if candidate_mask.sum().item() >= keep_count:
                    candidate_indices = candidate_indices[candidate_mask]
            candidate_scores = token_scores[candidate_indices]
            if candidate_scores.numel() <= keep_count:
                selected_indices = candidate_indices
            else:
                selected_local = candidate_scores.topk(keep_count).indices
                selected_indices = candidate_indices[selected_local]
            kept_indices = torch.sort(selected_indices).values.tolist()
        return kept_indices, evicted_high

    def _select_cache_block_token_indices_per_layer(
        self, block_ids: List[int], query_ids: List[int], num_layers: int
    ) -> Tuple[List[List[int]], List[int]]:
        token_scores_per_layer = self._analyze_chunk_with_attentions_per_layer(block_ids, query_ids)
        if not token_scores_per_layer:
            default_indices = [list(range(len(block_ids))) for _ in range(num_layers)]
            return default_indices, [0 for _ in range(num_layers)]

        keep_indices_per_layer = []
        evicted_high_per_layer = []
        total_layers = max(1, num_layers)
        for layer_idx in range(num_layers):
            if layer_idx < len(token_scores_per_layer):
                token_scores = token_scores_per_layer[layer_idx]
            else:
                token_scores = token_scores_per_layer[-1]

            if token_scores.numel() == 0:
                keep_indices_per_layer.append(list(range(len(block_ids))))
                evicted_high_per_layer.append(0)
                continue

            token_scores = self._apply_layer_structural_bias(
                token_scores=token_scores,
                layer_idx=layer_idx,
                num_layers=total_layers,
            )
            keep_min = min(len(block_ids), max(1, int(self.parallelcomp_token_keep_min)))
            token_capacity = max(1, int(self.parallelcomp_token_capacity))
            keep_count = min(len(block_ids), max(keep_min, token_capacity))
            selected_indices, evicted_high = self._select_token_indices_from_scores(
                token_scores=token_scores,
                keep_count=keep_count,
                keep_min=keep_min,
            )
            keep_indices_per_layer.append(selected_indices)
            evicted_high_per_layer.append(evicted_high)

        return keep_indices_per_layer, evicted_high_per_layer

    def _select_cache_block_token_indices_per_layer_per_head(
        self,
        block_ids: List[int],
        query_ids: List[int],
        num_cache_heads_per_layer: List[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        token_scores_per_layer_per_head = self._analyze_chunk_with_attentions_per_layer_per_head(
            block_ids,
            query_ids,
            num_cache_heads_per_layer,
        )
        if not token_scores_per_layer_per_head:
            default_indices = []
            for num_heads in num_cache_heads_per_layer:
                base = torch.arange(len(block_ids), device=self.device, dtype=torch.long)
                default_indices.append(base.unsqueeze(0).expand(num_heads, -1).clone())
            return default_indices, [0 for _ in num_cache_heads_per_layer]

        keep_indices_per_layer_per_head = []
        evicted_high_per_layer = []
        total_layers = max(1, len(token_scores_per_layer_per_head))
        for layer_idx, head_scores in enumerate(token_scores_per_layer_per_head):
            num_heads = num_cache_heads_per_layer[layer_idx] if layer_idx < len(num_cache_heads_per_layer) else head_scores.shape[0]
            if head_scores.numel() == 0:
                base = torch.arange(len(block_ids), device=self.device, dtype=torch.long)
                keep_indices_per_layer_per_head.append(base.unsqueeze(0).expand(num_heads, -1).clone())
                evicted_high_per_layer.append(0)
                continue

            keep_min = min(len(block_ids), max(1, int(self.parallelcomp_token_keep_min)))
            token_capacity = max(1, int(self.parallelcomp_token_capacity))
            keep_count = min(len(block_ids), max(keep_min, token_capacity))

            head_keep_indices = []
            biased_head_scores = []
            evicted_high = 0
            for head_idx in range(head_scores.shape[0]):
                biased_scores = self._apply_layer_structural_bias(
                    token_scores=head_scores[head_idx],
                    layer_idx=layer_idx,
                    num_layers=total_layers,
                )
                biased_head_scores.append(biased_scores)
                selected_indices, head_evicted_high = self._select_token_indices_from_scores(
                    token_scores=biased_scores,
                    keep_count=keep_count,
                    keep_min=keep_min,
                )
                head_keep_indices.append(torch.tensor(selected_indices, device=self.device, dtype=torch.long))
                evicted_high += head_evicted_high

            padded_head_indices = []
            for head_idx, index_tensor in enumerate(head_keep_indices):
                if index_tensor.shape[0] < keep_count:
                    token_scores = biased_head_scores[head_idx]
                    remaining_mask = torch.ones(token_scores.shape[0], dtype=torch.bool, device=token_scores.device)
                    remaining_mask[index_tensor] = False
                    remaining_indices = torch.arange(token_scores.shape[0], device=token_scores.device)[remaining_mask]
                    if remaining_indices.numel() > 0:
                        supplement_scores = token_scores[remaining_indices]
                        supplement_local = supplement_scores.topk(
                            min(keep_count - index_tensor.shape[0], remaining_indices.shape[0]),
                            largest=False,
                        ).indices
                        supplement_indices = remaining_indices[supplement_local]
                        index_tensor = torch.cat([index_tensor, supplement_indices], dim=0)
                if index_tensor.shape[0] < keep_count:
                    raise RuntimeError(
                        "Failed to build a fixed-width per-head selection set for ParallelComp token eviction"
                    )
                padded_head_indices.append(torch.sort(index_tensor[:keep_count]).values)

            keep_indices_per_layer_per_head.append(torch.stack(padded_head_indices, dim=0))
            evicted_high_per_layer.append(evicted_high)

        return keep_indices_per_layer_per_head, evicted_high_per_layer

    def _replay_parallelcomp_query_span(
        self,
        x_t: torch.Tensor,
        replay_start: int,
        replay_end: int,
        replay_range_key,
        past_key_values,
        cached_positions_per_layer: List[torch.Tensor],
        cached_block_ranges_per_layer: List[dict],
    ) -> Tuple[object, List[torch.Tensor], List[dict], int, torch.Tensor]:
        replay_input_ids = x_t[:, replay_start:replay_end]
        replay_len = replay_input_ids.shape[1]
        if replay_len <= 0:
            raise ValueError("ParallelComp query replay requested with an empty span")

        cached_length = (
            cached_positions_per_layer[0].shape[0]
            if cached_positions_per_layer
            else 0
        )
        reused_window_size = max(1, int(self.parallelcomp_chunk_size or 1))
        replay_positions = torch.arange(
            reused_window_size,
            reused_window_size + replay_len,
            device=self.device,
            dtype=torch.long,
        )
        replay_cache_positions = self._build_cache_slot_positions(
            cache_start=cached_length,
            length=replay_len,
        )
        if self.parallelcomp_tail_replay_full_mask:
            attention_mask = self._build_cached_prefix_full_visible_attention_mask(
                cached_length=cached_length,
                query_length=replay_len,
            )
        else:
            attention_mask = self._build_cached_prefix_causal_attention_mask(
                cached_length=cached_length,
                query_length=replay_len,
            )

        outputs = self.model(
            replay_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=replay_positions.unsqueeze(0),
            cache_position=replay_cache_positions,
            use_cache=True,
            update_kvcache=replay_len,
        )

        past_key_values = outputs.past_key_values
        if not cached_positions_per_layer:
            num_layers = len(past_key_values.key_cache)
            cached_positions_per_layer = [
                torch.empty(0, device=self.device, dtype=torch.long) for _ in range(num_layers)
            ]
            cached_block_ranges_per_layer = [{} for _ in range(num_layers)]

        new_cached_positions_per_layer = []
        new_cached_block_ranges_per_layer = []
        for layer_idx, layer_cached_positions in enumerate(cached_positions_per_layer):
            layer_start = layer_cached_positions.shape[0]
            layer_end = layer_start + replay_len
            new_cached_positions_per_layer.append(
                torch.cat([layer_cached_positions, replay_cache_positions], dim=0)
            )
            layer_ranges = dict(cached_block_ranges_per_layer[layer_idx])
            layer_ranges[replay_range_key] = (layer_start, layer_end)
            new_cached_block_ranges_per_layer.append(layer_ranges)

        last_logits = outputs.logits[:, replay_len - 1, :].unsqueeze(1)
        total_cached_length = cached_length + replay_len
        return (
            past_key_values,
            new_cached_positions_per_layer,
            new_cached_block_ranges_per_layer,
            total_cached_length,
            last_logits,
        )

    def _rebuild_parallelcomp_context_cache_with_query(
        self,
        x_t: torch.Tensor,
        past_key_values,
        block_states,
        stable_prompt_block_ids: List[int],
        cached_positions_per_layer: List[torch.Tensor],
        cached_block_ranges_per_layer: List[dict],
    ) -> Tuple[object, List[torch.Tensor], List[dict]]:
        if past_key_values is None or len(stable_prompt_block_ids) <= 1:
            return past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer

        tail_prompt_block_id = stable_prompt_block_ids[-1]
        context_prompt_block_ids = stable_prompt_block_ids[:-1]
        if not context_prompt_block_ids:
            return past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer

        tail_state = block_states[tail_prompt_block_id]
        tail_input_ids = x_t[:, tail_state["start_pos"]:tail_state["end_pos"]]
        tail_len = int(tail_input_ids.shape[1])
        if tail_len <= 0:
            return past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer

        reused_window_size = max(1, self._get_reused_window_size(block_states))
        rebuilt_key_parts = None
        rebuilt_value_parts = None
        rebuilt_cache = None
        rebuilt_ranges = {}
        cursor = 0

        for block_id in context_prompt_block_ids:
            state = block_states[block_id]
            block_input_ids = x_t[:, state["start_pos"]:state["end_pos"]]
            block_len = int(block_input_ids.shape[1])
            if block_len <= 0:
                continue

            block_positions = torch.arange(block_len, device=self.device, dtype=torch.long)
            if self.parallelcomp_query_free_cache_rebuild:
                rebuild_input_ids = block_input_ids
                rebuild_len = block_len
                position_ids = block_positions.unsqueeze(0)
                attention_mask = self._build_full_visible_attention_mask(rebuild_len)
            else:
                rebuild_input_ids = torch.cat([block_input_ids, tail_input_ids], dim=1)
                rebuild_len = int(rebuild_input_ids.shape[1])
                tail_positions = torch.arange(
                    reused_window_size,
                    reused_window_size + tail_len,
                    device=self.device,
                    dtype=torch.long,
                )
                position_ids = torch.cat([block_positions, tail_positions], dim=0).unsqueeze(0)
                attention_mask = self._build_full_visible_attention_mask(rebuild_len)
            cache_position = torch.arange(rebuild_len, device=self.device, dtype=torch.long)

            outputs = self.model(
                rebuild_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                update_kvcache=rebuild_len,
            )
            block_cache = outputs.past_key_values
            if rebuilt_cache is None:
                rebuilt_cache = block_cache
                rebuilt_key_parts = [[] for _ in block_cache.key_cache]
                rebuilt_value_parts = [[] for _ in block_cache.value_cache]

            for layer_idx, (layer_k, layer_v) in enumerate(
                zip(block_cache.key_cache, block_cache.value_cache)
            ):
                rebuilt_key_parts[layer_idx].append(layer_k[:, :, :block_len, :])
                rebuilt_value_parts[layer_idx].append(layer_v[:, :, :block_len, :])

            rebuilt_ranges[block_id] = (cursor, cursor + block_len)
            cursor += block_len

        if rebuilt_cache is None:
            return past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer

        rebuilt_cache.key_cache = [
            torch.cat(parts, dim=2) if parts else layer_k[:, :, :0, :]
            for parts, layer_k in zip(rebuilt_key_parts, rebuilt_cache.key_cache)
        ]
        rebuilt_cache.value_cache = [
            torch.cat(parts, dim=2) if parts else layer_v[:, :, :0, :]
            for parts, layer_v in zip(rebuilt_value_parts, rebuilt_cache.value_cache)
        ]

        rebuilt_cached_positions_per_layer = [
            torch.arange(cursor, device=self.device, dtype=torch.long)
            for _ in rebuilt_cache.key_cache
        ]
        rebuilt_cached_block_ranges_per_layer = [
            dict(rebuilt_ranges) for _ in rebuilt_cache.key_cache
        ]

        self._emit_parallelcomp_runtime_summary(
            "[ParallelComp] query_conditioned_context_rebuild "
            f"context_blocks={len(context_prompt_block_ids)} "
            f"tail_prompt_block={tail_prompt_block_id} "
            f"tail_prompt_len={tail_len} "
            f"rebuilt_context_tokens={cursor} "
            f"reused_window_size={reused_window_size} "
            f"query_free_cache={self.parallelcomp_query_free_cache_rebuild}"
        )
        return (
            rebuilt_cache,
            rebuilt_cached_positions_per_layer,
            rebuilt_cached_block_ranges_per_layer,
        )

    def _resolve_prompt_scoring_query(
        self,
        prompt: Optional[Dict[str, Any]],
    ) -> Tuple[List[int], str]:
        if isinstance(prompt, dict):
            scoring_query_text = prompt.get("scoring_query")
            if scoring_query_text is None:
                scoring_query_text = prompt.get("query", "")
                if scoring_query_text:
                    query_ids = self._encode_text_fragment(scoring_query_text)
                    if query_ids:
                        return query_ids, "prompt.query"
            else:
                query_ids = self._encode_text_fragment(scoring_query_text)
                if query_ids:
                    return query_ids, "prompt.scoring_query"

        query_ids = self._build_parallelcomp_scoring_query_ids()
        return query_ids, "fixed_fallback"

    def _build_parallelcomp_scoring_query_ids(self) -> List[int]:
        if (
            int(self.parallelcomp_query_tokens or 0) > 0
            and not getattr(self, "_parallelcomp_warned_query_tokens_ignored", False)
        ):
            self._emit_parallelcomp_runtime_summary(
                "[ParallelComp] prompt-tail query blocks are disabled; ignoring parallelcomp_query_tokens"
            )
            self._parallelcomp_warned_query_tokens_ignored = True

        if self._parallelcomp_active_scoring_query_ids:
            return list(self._parallelcomp_active_scoring_query_ids)

        query_text = (self.parallelcomp_fixed_query_text or "").strip()
        if not query_text:
            query_text = "Please complete the next line of code."
        return self.tokenizer.encode(query_text, add_special_tokens=False)

    def _emit_parallelcomp_runtime_summary(self, message: str) -> None:
        # Keep this mirrored to stdout so Slurm logs capture compression summaries
        # even when the logger stays at WARNING.
        eval_logger.info(message)
        print(message, flush=True)

    def _get_reused_window_size(self, block_states) -> int:
        if not block_states:
            return 0
        return max(
            max(1, state.get("rope_span", state["end_pos"] - state["start_pos"]))
            for state in block_states.values()
        )

    def _remap_parallelcomp_runtime_positions(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        position_offset = getattr(self, "_parallelcomp_post_compression_position_offset", None)
        if position_offset is None or positions.numel() == 0:
            return positions
        return positions - int(position_offset)

    def _build_cache_slot_positions(
        self,
        cache_start: int,
        length: int,
    ) -> torch.Tensor:
        return torch.arange(
            cache_start,
            cache_start + length,
            device=self.device,
            dtype=torch.long,
        )

    def _build_forward_cache_positions(
        self,
        position_ids: torch.Tensor,
        cached_length: int,
        update_kvcache: int,
    ) -> torch.Tensor:
        cache_positions = position_ids.clone()
        if update_kvcache > 0:
            cache_positions[:update_kvcache] = self._build_cache_slot_positions(
                cache_start=cached_length,
                length=update_kvcache,
            )
        return cache_positions

    def _build_block_local_positions(self, block_len: int) -> torch.Tensor:
        return torch.arange(block_len, device=self.device, dtype=torch.long)

    def _build_block_runtime_positions(self, state, take: int) -> torch.Tensor:
        if state.get("is_prompt_window", False):
            block_len = state["end_pos"] - state["start_pos"]
            block_rope_span = max(1, state.get("rope_span", block_len))
            return self._build_block_local_positions(block_rope_span)[:take]
        positions = torch.arange(
            state["start_pos"],
            state["start_pos"] + take,
            device=self.device,
            dtype=torch.long,
        )
        return self._remap_parallelcomp_runtime_positions(positions)

    def _should_split_prompt_windows(self, prompt_length: int) -> bool:
        if not self.parallelcomp_cache_compress_mode:
            return False
        if self.parallelcomp_chunk_size is None or self.parallelcomp_chunk_size <= 0:
            return False
        min_prompt_tokens = max(1, int(self.parallelcomp_min_prompt_tokens))
        return prompt_length >= min_prompt_tokens and prompt_length > self.parallelcomp_chunk_size

    def _init_prompt_block_states(self, prompt_length: int):
        if not self._should_split_prompt_windows(prompt_length):
            return {
                0: {
                    'start_pos': 0,
                    'end_pos': prompt_length,
                    'mask_count': 0,
                    'total_masks': prompt_length,
                    'rope_span': prompt_length,
                    'state': 'to_cache',
                    'is_complete': True,
                    'is_prompt_window': True,
                },
            }

        block_states = {}
        chunk_size = int(self.parallelcomp_chunk_size)
        cursor = 0
        block_id = 0

        # Ablation option: assign the remainder to the first prompt chunk so the
        # tail prompt chunk can stay at (or very close to) chunk_size tokens.
        if self.parallelcomp_split_from_tail and prompt_length > chunk_size:
            leading_remainder = prompt_length % chunk_size
            if leading_remainder > 0:
                block_states[block_id] = {
                    'start_pos': 0,
                    'end_pos': leading_remainder,
                    'mask_count': 0,
                    'total_masks': leading_remainder,
                    'rope_span': leading_remainder,
                    'state': 'to_cache',
                    'is_complete': True,
                    'is_prompt_window': True,
                }
                cursor = leading_remainder
                block_id += 1

        while cursor < prompt_length:
            block_end = min(cursor + chunk_size, prompt_length)
            block_len = block_end - cursor
            block_states[block_id] = {
                'start_pos': cursor,
                'end_pos': block_end,
                'mask_count': 0,
                'total_masks': block_len,
                'rope_span': block_len,
                'state': 'to_cache',
                'is_complete': True,
                'is_prompt_window': True,
            }
            cursor = block_end
            block_id += 1

        if block_states:
            block_states[max(block_states.keys())]['is_tail_prompt_window'] = True
        return block_states

    def _init_canonical_block_tokens(
        self,
        x_t: torch.Tensor,
        block_states,
    ) -> dict:
        canonical_block_tokens = {}
        for block_id, state in block_states.items():
            canonical_block_tokens[block_id] = x_t[0, state["start_pos"]:state["end_pos"]].tolist()
        return canonical_block_tokens

    def _snapshot_canonical_block_tokens(
        self,
        x_t: torch.Tensor,
        block_states,
        block_ids: List[int],
        canonical_block_tokens: dict,
    ) -> None:
        for block_id in block_ids:
            if block_id in canonical_block_tokens:
                continue
            state = block_states.get(block_id)
            if state is None:
                continue
            canonical_block_tokens[block_id] = x_t[0, state["start_pos"]:state["end_pos"]].tolist()

    def _build_input_block_metadata(
        self,
        block_states,
        process_start_pos: int,
        input_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_end_pos = process_start_pos + input_length
        block_ids = torch.full((input_length,), -1, device=self.device, dtype=torch.long)
        prompt_window_mask = torch.zeros((input_length,), device=self.device, dtype=torch.bool)

        for block_id in sorted(block_states.keys()):
            state = block_states[block_id]
            overlap_start = max(process_start_pos, state["start_pos"])
            overlap_end = min(input_end_pos, state["end_pos"])
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - process_start_pos
            local_end = overlap_end - process_start_pos
            block_ids[local_start:local_end] = block_id
            if state.get("is_prompt_window", False):
                prompt_window_mask[local_start:local_end] = True

        if (block_ids < 0).any():
            raise ValueError("Failed to assign block ids to all input tokens")

        return block_ids, prompt_window_mask

    def _build_input_position_ids(
        self,
        block_states,
        process_start_pos: int,
        input_length: int,
        update_kvcache: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_ids, prompt_window_mask = self._build_input_block_metadata(
            block_states=block_states,
            process_start_pos=process_start_pos,
            input_length=input_length,
        )

        rope_positions = torch.empty(input_length, device=self.device, dtype=torch.long)

        active_tail_start = update_kvcache
        if active_tail_start < input_length:
            rope_positions[active_tail_start:] = self._remap_parallelcomp_runtime_positions(
                torch.arange(
                    process_start_pos + active_tail_start,
                    process_start_pos + input_length,
                    device=self.device,
                    dtype=torch.long,
                )
            )

        cursor = 0
        if update_kvcache > 0:
            for block_id in sorted(block_states.keys()):
                state = block_states[block_id]
                if state["state"] != "to_cache":
                    continue
                block_len = state["end_pos"] - state["start_pos"]
                take = min(block_len, update_kvcache - cursor)
                if take <= 0:
                    break
                rope_positions[cursor:cursor + take] = self._build_block_runtime_positions(state, take)
                cursor += take
                if cursor >= update_kvcache:
                    break

        if cursor < min(update_kvcache, input_length):
            rope_positions[cursor:update_kvcache] = self._remap_parallelcomp_runtime_positions(
                torch.arange(
                    process_start_pos + cursor,
                    process_start_pos + update_kvcache,
                    device=self.device,
                    dtype=torch.long,
                )
            )

        return block_ids, prompt_window_mask, rope_positions

    def _build_cache_write_positions(
        self,
        cache_start: int,
        update_kvcache: int,
    ) -> torch.Tensor:
        if update_kvcache <= 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return self._build_cache_slot_positions(cache_start=cache_start, length=update_kvcache)

    def _rerun_active_blocks_after_cache_compression(
        self,
        x_t: torch.Tensor,
        block_states,
        past_key_values,
        cached_positions_per_layer: List[torch.Tensor],
        attn_dtype: torch.dtype,
    ) -> Tuple[Optional[object], Optional[int]]:
        active_block_ids = [
            block_id for block_id, state in sorted(block_states.items())
            if state["state"] == "active"
        ]
        if not active_block_ids:
            return None, None

        process_start_pos = min(block_states[block_id]["start_pos"] for block_id in active_block_ids)
        active_input_seq = x_t[:, process_start_pos:]
        if active_input_seq.shape[1] == 0:
            return None, None

        input_block_ids, input_prompt_window_mask, input_rope_positions = self._build_input_position_ids(
            block_states=block_states,
            process_start_pos=process_start_pos,
            input_length=active_input_seq.shape[1],
            update_kvcache=0,
        )
        if cached_positions_per_layer:
            attention_mask = build_unified_sparse_block_attention_mask(
                query_block_ids=input_block_ids,
                cached_positions_per_layer=cached_positions_per_layer,
                query_prompt_window_mask=input_prompt_window_mask,
                update_kvcache=0,
                device=self.device,
                dtype=attn_dtype,
            )
        else:
            attention_mask = build_sparse_block_attention_mask(
                query_block_ids=input_block_ids,
                cached_length=0,
                query_prompt_window_mask=input_prompt_window_mask,
                update_kvcache=0,
                device=self.device,
                dtype=attn_dtype,
            )

        outputs = self.model(
            active_input_seq,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=input_rope_positions.unsqueeze(0),
            cache_position=self._build_forward_cache_positions(
                position_ids=input_rope_positions,
                cached_length=cached_positions_per_layer[0].shape[0] if cached_positions_per_layer else 0,
                update_kvcache=0,
            ),
            use_cache=True,
            update_kvcache=0,
        )
        return outputs, process_start_pos

    def _count_generated_blocks(self, block_states) -> int:
        return sum(1 for state in block_states.values() if not state.get("is_prompt_window", False))

    def _select_global_cache_blocks(
        self,
        block_states,
        stable_block_ids: List[int],
        stable_prefix_len: int,
        query_ids: List[int],
        canonical_block_tokens: dict,
        x_t: Optional[torch.Tensor] = None,
        query_block_id: Optional[int] = None,
    ) -> List[int]:
        if len(stable_block_ids) <= 1:
            return stable_block_ids

        scored_blocks = []
        self_information_scores = {}
        attention_scores = {}
        hidden_scores = {}

        for block_id in stable_block_ids:
            block_ids = canonical_block_tokens.get(block_id)
            if block_ids is None:
                block_ids = []
            if self.parallelcomp_score_mode == "next_block_logits":
                score, _, _ = self._score_chunk_with_next_block_logits(block_ids, query_ids)
            else:
                score = self._score_chunk_with_self_information(block_ids, query_ids)
                if math.isfinite(score):
                    score = -score
            if math.isfinite(score):
                self_information_scores[block_id] = score

        if len(self_information_scores) != len(stable_block_ids) and x_t is not None and query_block_id is not None:
            attention_scores = self._score_prompt_blocks_with_attention_resonance(
                x_t=x_t,
                block_states=block_states,
                candidate_block_ids=[
                    block_id for block_id in stable_block_ids if block_id not in self_information_scores
                ],
                query_block_id=query_block_id,
                stable_prefix_len=stable_prefix_len,
            )
            if not attention_scores:
                hidden_scores = self._score_prompt_blocks_with_hidden_resonance(
                    x_t=x_t,
                    block_states=block_states,
                    candidate_block_ids=[
                        block_id for block_id in stable_block_ids if block_id not in self_information_scores
                    ],
                    query_block_id=query_block_id,
                    stable_prefix_len=stable_prefix_len,
                )

        if self_information_scores:
            for block_id in stable_block_ids:
                score = self_information_scores.get(block_id, float("-inf"))
                scored_blocks.append((block_id, score))
        elif attention_scores:
            for block_id in stable_block_ids:
                score = attention_scores.get(block_id, float("-inf"))
                scored_blocks.append((block_id, score))
        elif hidden_scores:
            for block_id in stable_block_ids:
                score = hidden_scores.get(block_id, float("-inf"))
                scored_blocks.append((block_id, score))
        else:
            for block_id in stable_block_ids:
                scored_blocks.append((block_id, float("-inf")))

        keep_ids = set()
        if self.parallelcomp_keep_first_chunk and 0 in stable_block_ids:
            keep_ids.add(0)

        candidates = [item for item in scored_blocks if item[0] not in keep_ids]
        topk = len(candidates) if self.parallelcomp_topk_chunks is None else min(max(1, self.parallelcomp_topk_chunks), len(candidates))
        prefer_low_scores = bool(self.parallelcomp_select_low_score_chunks)
        selected = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=not prefer_low_scores,
        )[:topk]
        keep_ids.update(block_id for block_id, _ in selected)

        kept_block_ids = [block_id for block_id in stable_block_ids if block_id in keep_ids]
        eval_logger.info(
            "Prefix-aware global block selection: stable_blocks=%d, kept_blocks=%d, stable_prefix_tokens=%d, scoring=%s, prefer_low_scores=%s",
            len(stable_block_ids),
            len(kept_block_ids),
            stable_prefix_len,
            "self_information" if self_information_scores else ("attention" if attention_scores else ("hidden" if hidden_scores else "none")),
            prefer_low_scores,
        )
        return kept_block_ids

    def _compress_cached_prefix_blocks(
        self,
        x_t: torch.Tensor,
        past_key_values,
        block_states,
        mask_id: int,
        cached_positions_per_layer: List[torch.Tensor],
        cached_block_ranges_per_layer: List[dict],
        shared_cached_length: int,
        canonical_block_tokens: dict,
    ) -> Tuple[torch.Tensor, Optional[object], List[torch.Tensor], List[dict], int, bool, Optional[torch.Tensor]]:
        if not self.parallelcomp_cache_compress_mode or past_key_values is None:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None
        if getattr(self, "_parallelcomp_prompt_cache_compression_done", False):
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        stable_prompt_block_ids = [
            block_id for block_id, state in sorted(block_states.items())
            if state["state"] == "in_cache" and state.get("is_prompt_window", False)
        ]
        if not stable_prompt_block_ids:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None
        if len(stable_prompt_block_ids) <= 1:
            # If the prompt never split into multiple ParallelComp windows, skip
            # prompt-side compression entirely.
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        stable_prefix_len = shared_cached_length
        if stable_prefix_len <= 0:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        query_span = self._get_active_prompt_query_span(prompt_length=x_t.shape[1])
        if query_span is None:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None
        query_start, query_end = query_span
        if query_end > stable_prefix_len:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        context_prompt_units = []
        query_prompt_block_id = None
        for block_id in stable_prompt_block_ids:
            state = block_states[block_id]
            if state["start_pos"] <= query_start < state["end_pos"]:
                query_prompt_block_id = block_id
            context_start = state["start_pos"]
            context_end = min(state["end_pos"], query_start)
            if context_end <= context_start:
                continue
            context_prompt_units.append(
                {
                    "block_id": block_id,
                    "start_pos": context_start,
                    "end_pos": context_end,
                    "original_start_pos": state["start_pos"],
                    "original_end_pos": state["end_pos"],
                    "is_partial": context_end < state["end_pos"],
                }
            )
        context_prompt_block_ids = [unit["block_id"] for unit in context_prompt_units]
        context_prompt_unit_by_id = {unit["block_id"]: unit for unit in context_prompt_units}
        if not context_prompt_units:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        scoring_query_ids = self._build_parallelcomp_scoring_query_ids()
        if not scoring_query_ids:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None
        prev_context_block_id = context_prompt_block_ids[-1] if context_prompt_block_ids else None
        reused_window_size = max(1, self._get_reused_window_size(block_states))

        local_block_results = {}
        local_block_scores = {}
        for unit in context_prompt_units:
            block_id = unit["block_id"]
            block_input_ids = x_t[:, unit["start_pos"]:unit["end_pos"]]
            local_result = self._run_parallelcomp_local_block_forward(
                block_input_ids=block_input_ids,
                query_ids=scoring_query_ids,
                reused_window_size=reused_window_size,
            )
            if local_result is None:
                continue
            local_block_results[block_id] = local_result
            local_block_scores[block_id] = local_result["score"]

        self._parallelcomp_prompt_cache_compression_done = True
        if not local_block_results:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        keep_ids = set()
        if self.parallelcomp_keep_first_chunk and 0 in local_block_results:
            keep_ids.add(0)
        candidates = [(block_id, local_block_scores.get(block_id, float("-inf"))) for block_id in context_prompt_block_ids if block_id in local_block_results and block_id not in keep_ids]
        if self.parallelcomp_topk_chunks is None:
            topk = len(candidates)
        else:
            topk = min(max(1, int(self.parallelcomp_topk_chunks)), len(candidates)) if candidates else 0
        prefer_low_scores = bool(self.parallelcomp_select_low_score_chunks)
        selected = sorted(candidates, key=lambda item: item[1], reverse=not prefer_low_scores)[:topk]
        keep_ids.update(block_id for block_id, _ in selected)
        kept_block_ids = [block_id for block_id in context_prompt_block_ids if block_id in keep_ids]
        if not kept_block_ids:
            return x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, False, None

        self._emit_parallelcomp_runtime_summary(
            "[ParallelComp] local_forward_once "
            f"context_blocks={len(context_prompt_block_ids)} "
            f"kept_blocks={len(kept_block_ids)} "
            f"query_prompt_block={query_prompt_block_id} "
            f"query_span=[{query_start},{query_end}) "
            f"query_len={query_end - query_start} "
            f"scoring_query_len={len(scoring_query_ids)} "
            f"score_query_window={self._get_parallelcomp_chunk_score_query_window_size(len(scoring_query_ids))} "
            f"token_query_window={self._get_parallelcomp_query_window_size(len(scoring_query_ids))} "
            f"local_mask={self.parallelcomp_chunk_score_attention_mask} "
            f"reused_window_size={reused_window_size}"
        )

        compressed_block_meta = []
        block_keep_widths = {}
        total_evicted_high = 0
        total_removed = 0
        for block_id in kept_block_ids:
            unit = context_prompt_unit_by_id[block_id]
            local_result = local_block_results[block_id]
            block_len = int(local_result["block_len"])
            per_layer_kept_indices = local_result["kept_indices"]
            evicted_high_layers = local_result["evicted_high_layers"]
            total_evicted_high += sum(evicted_high_layers)
            layer_keep_width = max(
                (kept_indices.shape[1] if isinstance(kept_indices, torch.Tensor) else len(kept_indices))
                for kept_indices in per_layer_kept_indices
            ) if per_layer_kept_indices else block_len
            total_removed += max(0, block_len - layer_keep_width)
            block_keep_widths[block_id] = layer_keep_width
            compressed_block_meta.append(
                (block_id, unit["start_pos"], unit["end_pos"], per_layer_kept_indices)
            )

        dropped_block_ids = [unit["block_id"] for unit in context_prompt_units if unit["block_id"] not in kept_block_ids]
        total_removed += sum(
            unit["end_pos"] - unit["start_pos"]
            for unit in context_prompt_units
            if unit["block_id"] in dropped_block_ids
        )

        new_key_cache = []
        new_value_cache = []
        new_cached_positions_per_layer = []
        new_cached_block_ranges_per_layer = []
        for layer_idx, (layer_k, layer_v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
            compressed_k_parts = []
            compressed_v_parts = []
            compressed_pos_parts = []
            layer_cached_positions = cached_positions_per_layer[layer_idx]
            layer_block_ranges = cached_block_ranges_per_layer[layer_idx] if layer_idx < len(cached_block_ranges_per_layer) else {}
            new_layer_block_ranges = {}
            layer_cursor = 0
            for block_id, unit_start, unit_end, per_layer_kept_indices in compressed_block_meta:
                local_result = local_block_results.get(block_id)
                if local_result is None or layer_idx >= len(local_result["key_cache"]):
                    continue
                per_head_kept_indices = per_layer_kept_indices[layer_idx] if layer_idx < len(per_layer_kept_indices) else per_layer_kept_indices[0]
                per_head_kept_indices = per_head_kept_indices.to(device=layer_k.device, dtype=torch.long)
                keep_count = per_head_kept_indices.shape[1]
                block_slice_k = local_result["key_cache"][layer_idx].to(device=layer_k.device)
                block_slice_v = local_result["value_cache"][layer_idx].to(device=layer_v.device)
                expanded_index = per_head_kept_indices.unsqueeze(0).unsqueeze(-1).expand(
                    block_slice_k.shape[0],
                    per_head_kept_indices.shape[0],
                    keep_count,
                    block_slice_k.shape[-1],
                )
                block_k = block_slice_k.gather(2, expanded_index)
                block_v = block_slice_v.gather(2, expanded_index)
                block_pos = torch.arange(
                    layer_cursor,
                    layer_cursor + block_k.shape[2],
                    device=layer_cached_positions.device,
                    dtype=torch.long,
                )
                compressed_k_parts.append(block_k)
                compressed_v_parts.append(block_v)
                compressed_pos_parts.append(block_pos)
                new_layer_block_ranges[block_id] = (layer_cursor, layer_cursor + block_k.shape[2])
                layer_cursor += block_k.shape[2]

            compressed_k = torch.cat(compressed_k_parts, dim=2) if compressed_k_parts else layer_k[:, :, :0, :]
            compressed_v = torch.cat(compressed_v_parts, dim=2) if compressed_v_parts else layer_v[:, :, :0, :]
            new_key_cache.append(compressed_k)
            new_value_cache.append(compressed_v)
            new_cached_positions_per_layer.append(
                torch.cat(compressed_pos_parts, dim=0) if compressed_pos_parts else layer_cached_positions[:0]
            )
            new_cached_block_ranges_per_layer.append(new_layer_block_ranges)

        past_key_values.key_cache = new_key_cache
        past_key_values.value_cache = new_value_cache

        old_stable_prefix_len = stable_prefix_len
        new_stable_prefix_len = (
            new_cached_positions_per_layer[0].shape[0]
            if new_cached_positions_per_layer
            else 0
        )
        reused_window_size = max(1, self._get_reused_window_size(block_states))
        self._parallelcomp_post_compression_position_offset = (
            query_start - reused_window_size
        )

        layer_selection_divergence = 0
        if compressed_block_meta:
            for _, _, _, per_layer_kept_indices in compressed_block_meta:
                reference_set = None
                for kept_indices in per_layer_kept_indices:
                    if isinstance(kept_indices, torch.Tensor):
                        for head_indices in kept_indices.tolist():
                            head_set = set(head_indices)
                            if reference_set is None:
                                reference_set = head_set
                            else:
                                layer_selection_divergence += len(reference_set.symmetric_difference(head_set))
                    elif reference_set is not None:
                        layer_selection_divergence += len(reference_set.symmetric_difference(set(kept_indices)))

        eval_logger.info(
            "ParallelComp-style per-layer prompt cache compression: logical_prefix_tokens=%d, context_blocks=%d, kept_context_blocks=%d, query_prompt_block=%s, query_len=%d, removed_tokens=%d, high_score_evicted=%d, layer_selection_divergence=%d, layer_cache_lengths=%s",
            old_stable_prefix_len,
            len(context_prompt_block_ids),
            len(kept_block_ids),
            query_prompt_block_id,
            query_end - query_start,
            total_removed,
            total_evicted_high,
            layer_selection_divergence,
            [layer_positions.shape[0] for layer_positions in new_cached_positions_per_layer],
        )
        prev_block_summary = "prev_block=None"
        if prev_context_block_id is not None and prev_context_block_id in context_prompt_unit_by_id:
            prev_unit = context_prompt_unit_by_id[prev_context_block_id]
            prev_original_len = prev_unit["end_pos"] - prev_unit["start_pos"]
            prev_kept = prev_context_block_id in block_keep_widths
            prev_kept_tokens = block_keep_widths.get(prev_context_block_id, 0)
            prev_block_summary = (
                f"prev_block={prev_context_block_id} "
                f"span=[{prev_unit['start_pos']},{prev_unit['end_pos']}) "
                f"len={prev_original_len} kept={prev_kept} kept_tokens={prev_kept_tokens}"
            )

        recent_context_blocks = context_prompt_block_ids[-min(3, len(context_prompt_block_ids)):]
        recent_kept_blocks = [block_id for block_id in recent_context_blocks if block_id in block_keep_widths]
        self._emit_parallelcomp_runtime_summary(
            "[ParallelComp] prompt_cache_compress "
            f"scoring_query_source={getattr(self, '_parallelcomp_active_scoring_query_source', 'fixed_fallback')} "
            f"scoring_query_len={len(scoring_query_ids)} "
            f"context_blocks={len(context_prompt_block_ids)} "
            f"kept_blocks={len(kept_block_ids)} "
            f"query_prompt_block={query_prompt_block_id} "
            f"query_span=[{query_start},{query_end}) "
            f"query_len={query_end - query_start} "
            f"recent_context_blocks={recent_context_blocks} "
            f"recent_kept_blocks={recent_kept_blocks} "
            f"{prev_block_summary} "
            f"removed_tokens={total_removed} "
            f"high_score_evicted={total_evicted_high} "
            f"local_forward_once=True "
            f"score_query_window={self._get_parallelcomp_chunk_score_query_window_size(len(scoring_query_ids))} "
            f"token_query_window={self._get_parallelcomp_query_window_size(len(scoring_query_ids))} "
            f"local_mask={self.parallelcomp_chunk_score_attention_mask}"
        )
        (
            replayed_past_key_values,
            replayed_cached_positions_per_layer,
            replayed_cached_block_ranges_per_layer,
            replayed_cached_length,
            replay_last_logits,
        ) = self._replay_parallelcomp_query_span(
            x_t=x_t,
            replay_start=query_start,
            replay_end=query_end,
            replay_range_key=f"query:{query_start}:{query_end}",
            past_key_values=past_key_values,
            cached_positions_per_layer=new_cached_positions_per_layer,
            cached_block_ranges_per_layer=new_cached_block_ranges_per_layer,
        )
        self._emit_parallelcomp_runtime_summary(
            "[ParallelComp] query_replay "
            f"query_prompt_block={query_prompt_block_id} "
            f"query_span=[{query_start},{query_end}) "
            f"query_len={query_end - query_start} "
            f"stitched_context_tokens={new_stable_prefix_len} "
            f"reused_window_size={reused_window_size} "
            f"position_offset={self._parallelcomp_post_compression_position_offset} "
            f"cached_after_replay={replayed_cached_length}"
        )
        return (
            x_t,
            replayed_past_key_values,
            replayed_cached_positions_per_layer,
            replayed_cached_block_ranges_per_layer,
            replayed_cached_length,
            True,
            replay_last_logits,
        )
    
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

    def _count_non_eos_tokens_before_truncation(self, generated_sequence, prompt_length):
        """
        Unified token counting function: counts non-EOS tokens in the generated sequence (before truncation).
        """
        # Get the generated part (excluding the prompt)
        generated_tokens = generated_sequence[prompt_length:]
        # Count non-EOS tokens
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            # If it's a tensor, convert to list for counting
            if hasattr(generated_tokens, 'tolist'):
                generated_tokens_list = generated_tokens.tolist()
            else:
                generated_tokens_list = generated_tokens
            non_eos_count = sum(1 for token in generated_tokens_list if token != eos_token_id)
        else:
            non_eos_count = len(generated_tokens)
        return non_eos_count

    def _get_prompt_token_budget(self) -> int:
        return max(1, int(self.max_length - self.max_new_tokens))

    def _get_bos_token_ids(self) -> List[int]:
        if not self.add_bos_token:
            return []
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return [int(bos_token_id)]
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if bos_token:
            return self.tokenizer.encode(bos_token, add_special_tokens=False)
        return []

    def _encode_text_fragment(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _get_active_prompt_query_span(self, prompt_length: int) -> Optional[Tuple[int, int]]:
        if self._parallelcomp_active_query_start is None or self._parallelcomp_active_query_end is None:
            return None
        query_start = max(0, int(self._parallelcomp_active_query_start))
        query_end = min(prompt_length, int(self._parallelcomp_active_query_end))
        if query_end <= query_start:
            return None
        return query_start, query_end

    def _split_parallelcomp_token_chunks(self, token_ids: List[int]) -> List[List[int]]:
        if not token_ids:
            return []
        chunk_size = int(self.parallelcomp_chunk_size or 0)
        if chunk_size <= 0 or len(token_ids) <= chunk_size:
            return [token_ids]

        chunks = []
        cursor = 0
        if self.parallelcomp_split_from_tail and len(token_ids) > chunk_size:
            leading_remainder = len(token_ids) % chunk_size
            if leading_remainder > 0:
                chunks.append(token_ids[:leading_remainder])
                cursor = leading_remainder

        while cursor < len(token_ids):
            chunks.append(token_ids[cursor:cursor + chunk_size])
            cursor += chunk_size
        return chunks

    def _maybe_prepend_bos_to_parallelcomp_chunk(self, chunk_ids: List[int]) -> List[int]:
        if not self.parallelcomp_chunk_bos_ablation:
            return chunk_ids
        bos_ids = self._get_bos_token_ids()
        if not bos_ids:
            return chunk_ids
        if chunk_ids[:len(bos_ids)] == bos_ids:
            return chunk_ids
        chunk_size = int(self.parallelcomp_chunk_size or 0)
        with_bos = list(bos_ids) + list(chunk_ids)
        if chunk_size > 0 and len(with_bos) > chunk_size:
            with_bos = with_bos[:chunk_size]
        return with_bos

    def _build_pre_runtime_candidate_chunks(
        self,
        prompt_spec: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates = []
        context_segments = prompt_spec.get("context_segments")
        context_text = prompt_spec.get("context", "")

        if context_segments is not None:
            for segment_idx, segment_text in enumerate(context_segments):
                segment_ids = self._encode_text_fragment(segment_text)
                for chunk_idx, chunk_ids in enumerate(self._split_parallelcomp_token_chunks(segment_ids)):
                    chunk_ids = self._maybe_prepend_bos_to_parallelcomp_chunk(chunk_ids)
                    candidates.append(
                        {
                            "token_ids": chunk_ids,
                            "segment_idx": segment_idx,
                            "chunk_idx": chunk_idx,
                        }
                    )
        elif context_text:
            context_ids = self._encode_text_fragment(context_text)
            for chunk_idx, chunk_ids in enumerate(self._split_parallelcomp_token_chunks(context_ids)):
                chunk_ids = self._maybe_prepend_bos_to_parallelcomp_chunk(chunk_ids)
                candidates.append(
                    {
                        "token_ids": chunk_ids,
                        "segment_idx": 0,
                        "chunk_idx": chunk_idx,
                    }
                )

        return candidates

    def _score_pre_runtime_candidate_chunks(
        self,
        candidate_chunks: List[Dict[str, Any]],
        scoring_query_ids: List[int],
    ) -> Dict[int, float]:
        scores = {}
        for idx, chunk in enumerate(candidate_chunks):
            chunk_ids = chunk["token_ids"]
            if self.parallelcomp_score_mode == "next_block_logits":
                score, _, _ = self._score_chunk_with_next_block_logits(chunk_ids, scoring_query_ids)
            else:
                score = -self._score_chunk_with_self_information(chunk_ids, scoring_query_ids)
            if not math.isfinite(score):
                score, _ = self._analyze_chunk_with_attentions(chunk_ids, scoring_query_ids)
            scores[idx] = float(score) if math.isfinite(score) else float("-inf")
        return scores

    def _select_pre_runtime_chunk_indices(
        self,
        candidate_chunks: List[Dict[str, Any]],
        scoring_query_ids: List[int],
    ) -> Tuple[List[int], Dict[int, float]]:
        if not candidate_chunks:
            return [], {}

        if not scoring_query_ids:
            return list(range(len(candidate_chunks))), {}

        scores = self._score_pre_runtime_candidate_chunks(candidate_chunks, scoring_query_ids)
        forced = []
        remaining = list(range(len(candidate_chunks)))
        if self.parallelcomp_keep_first_chunk and remaining:
            forced = [0]
            remaining = remaining[1:]

        prefer_low_scores = bool(self.parallelcomp_select_low_score_chunks)
        ranked = sorted(
            remaining,
            key=lambda idx: scores.get(idx, float("-inf")),
            reverse=not prefer_low_scores,
        )

        topk = len(ranked)
        if self.parallelcomp_topk_chunks is not None:
            topk = min(max(1, int(self.parallelcomp_topk_chunks)), len(ranked))

        selected = forced + ranked[:topk]
        return sorted(set(selected)), scores

    def _pack_pre_runtime_chunks_to_budget(
        self,
        candidate_chunks: List[Dict[str, Any]],
        selected_indices: List[int],
        separator_ids: List[int],
        available_context_tokens: int,
    ) -> Tuple[List[List[int]], List[int], int]:
        if available_context_tokens <= 0:
            return [], [], 0

        packed_chunks = []
        packed_indices = []
        used_tokens = 0

        for idx in selected_indices:
            chunk_ids = candidate_chunks[idx]["token_ids"]
            separator_cost = len(separator_ids) if packed_chunks else 0
            remaining_after_separator = available_context_tokens - used_tokens - separator_cost
            if remaining_after_separator <= 0:
                continue

            if len(chunk_ids) <= remaining_after_separator:
                if separator_cost > 0:
                    used_tokens += separator_cost
                packed_chunks.append(chunk_ids)
                packed_indices.append(idx)
                used_tokens += len(chunk_ids)
                continue

            if packed_chunks:
                continue

            truncated_ids = chunk_ids[-remaining_after_separator:]
            if not truncated_ids:
                continue
            packed_chunks.append(truncated_ids)
            packed_indices.append(idx)
            used_tokens += len(truncated_ids)

        return packed_chunks, packed_indices, used_tokens

    def _assemble_prompt_ids_from_chunks(
        self,
        prefix_ids: List[int],
        chunk_token_ids: List[List[int]],
        separator_ids: List[int],
        query_ids: List[int],
    ) -> List[int]:
        prompt_ids = list(prefix_ids)
        for idx, chunk_ids in enumerate(chunk_token_ids):
            if idx > 0 and separator_ids:
                prompt_ids.extend(separator_ids)
            prompt_ids.extend(chunk_ids)
        prompt_ids.extend(query_ids)
        return prompt_ids

    def _prepare_prompt_ids(self, prompt: Union[str, Dict[str, Any]]) -> Tuple[List[int], Dict[str, Any]]:
        bos_ids = self._get_bos_token_ids()
        prompt_budget = self._get_prompt_token_budget()

        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_ids = bos_ids + prompt_ids
            return prompt_ids, {"mode": "raw_string", "raw_prompt_tokens": len(prompt_ids)}

        if not isinstance(prompt, dict):
            raise TypeError(f"Unsupported prompt type for generation: {type(prompt)!r}")

        if "prompt" in prompt and not any(
            key in prompt for key in ("prefix", "context", "context_segments", "query")
        ):
            prompt_ids = self.tokenizer.encode(prompt["prompt"])
            prompt_ids = bos_ids + prompt_ids
            return prompt_ids, {
                "mode": "raw_prompt_dict",
                "raw_prompt_tokens": len(prompt_ids),
                "label": prompt.get("metadata_label"),
            }

        prefix_ids = bos_ids + self._encode_text_fragment(prompt.get("prefix", ""))
        query_ids = self._encode_text_fragment(prompt.get("query", ""))
        separator_ids = self._encode_text_fragment(prompt.get("segment_separator", "\n\n"))
        candidate_chunks = self._build_pre_runtime_candidate_chunks(prompt)
        scoring_query_ids, scoring_query_source = self._resolve_prompt_scoring_query(prompt)
        ntk_truncation_strategy = prompt.get("ntk_truncation_strategy")

        raw_context_tokens = 0
        if candidate_chunks:
            raw_context_tokens = sum(len(chunk["token_ids"]) for chunk in candidate_chunks)
            raw_context_tokens += len(separator_ids) * max(0, len(candidate_chunks) - 1)
        raw_prompt_tokens = len(prefix_ids) + raw_context_tokens + len(query_ids)

        if not candidate_chunks:
            prompt_ids = prefix_ids + query_ids
            query_start = len(prefix_ids)
            query_end = query_start + len(query_ids)
            return prompt_ids, {
                "mode": "structured_no_context",
                "raw_prompt_tokens": len(prompt_ids),
                "label": prompt.get("metadata_label"),
                "scoring_query_source": scoring_query_source,
                "scoring_query_ids": scoring_query_ids,
                "ntk_truncation_strategy": ntk_truncation_strategy,
                "query_start": query_start,
                "query_end": query_end,
            }

        should_compress = bool(self.parallelcomp_pre_runtime_mode)
        should_compress = should_compress and raw_prompt_tokens > int(
            self.parallelcomp_min_prompt_tokens or 0
        )

        if not should_compress:
            prompt_ids = self._assemble_prompt_ids_from_chunks(
                prefix_ids=prefix_ids,
                chunk_token_ids=[chunk["token_ids"] for chunk in candidate_chunks],
                separator_ids=separator_ids,
                query_ids=query_ids,
            )
            query_start = len(prompt_ids) - len(query_ids)
            query_end = len(prompt_ids)
            return prompt_ids, {
                "mode": "structured_passthrough",
                "raw_prompt_tokens": raw_prompt_tokens,
                "candidate_chunks": len(candidate_chunks),
                "label": prompt.get("metadata_label"),
                "scoring_query_source": scoring_query_source,
                "scoring_query_ids": scoring_query_ids,
                "ntk_truncation_strategy": ntk_truncation_strategy,
                "query_start": query_start,
                "query_end": query_end,
            }

        selected_indices, scores = self._select_pre_runtime_chunk_indices(
            candidate_chunks=candidate_chunks,
            scoring_query_ids=scoring_query_ids,
        )

        available_context_tokens = prompt_budget - len(prefix_ids) - len(query_ids)
        packed_chunks, packed_indices, packed_context_tokens = self._pack_pre_runtime_chunks_to_budget(
            candidate_chunks=candidate_chunks,
            selected_indices=selected_indices,
            separator_ids=separator_ids,
            available_context_tokens=available_context_tokens,
        )
        prompt_ids = self._assemble_prompt_ids_from_chunks(
            prefix_ids=prefix_ids,
            chunk_token_ids=packed_chunks,
            separator_ids=separator_ids,
            query_ids=query_ids,
        )
        query_start = len(prompt_ids) - len(query_ids)
        query_end = len(prompt_ids)

        selected_scores = {
            idx: scores[idx]
            for idx in packed_indices
            if idx in scores and math.isfinite(scores[idx])
        }
        return prompt_ids, {
            "mode": "structured_pre_runtime_compressed",
            "label": prompt.get("metadata_label"),
            "raw_prompt_tokens": raw_prompt_tokens,
            "prompt_budget": prompt_budget,
            "candidate_chunks": len(candidate_chunks),
            "scoring_query_tokens": len(scoring_query_ids),
            "scoring_query_source": scoring_query_source,
            "scoring_query_ids": scoring_query_ids,
            "ntk_truncation_strategy": ntk_truncation_strategy,
            "selected_chunks": len(packed_indices),
            "selected_chunk_indices": packed_indices,
            "selected_chunk_scores": selected_scores,
            "context_tokens_after_pack": packed_context_tokens,
            "query_start": query_start,
            "query_end": query_end,
        }

    def _generate_batch(self, prompts: List[Union[str, Dict[str, Any]]]) -> List[str]:
        responses = []

        # Generate for each prompt individually (block generation usually processes one by one)
        for i, prompt in enumerate(prompts):
            prompt_ids, prompt_meta = self._prepare_prompt_ids(prompt)
            self._parallelcomp_active_scoring_query_ids = prompt_meta.get("scoring_query_ids")
            self._parallelcomp_active_scoring_query_source = prompt_meta.get(
                "scoring_query_source"
            )
            label = prompt_meta.get("label") or f"prompt_{i}"
            if prompt_meta.get("mode") == "structured_pre_runtime_compressed":
                score_values = list(prompt_meta.get("selected_chunk_scores", {}).values())
                score_summary = ""
                if score_values:
                    score_summary = (
                        f" score_min={min(score_values):.4f}"
                        f" score_max={max(score_values):.4f}"
                    )
                self._emit_parallelcomp_runtime_summary(
                    "[ParallelComp][pre_runtime] "
                    f"label={label} "
                    f"raw_prompt_tokens={prompt_meta.get('raw_prompt_tokens', len(prompt_ids))} "
                    f"budget={prompt_meta.get('prompt_budget', self._get_prompt_token_budget())} "
                    f"candidate_chunks={prompt_meta.get('candidate_chunks', 0)} "
                    f"scoring_query_source={prompt_meta.get('scoring_query_source', 'fixed_fallback')} "
                    f"kept_chunks={prompt_meta.get('selected_chunks', 0)} "
                    f"kept_chunk_indices={prompt_meta.get('selected_chunk_indices', [])} "
                    f"context_tokens_after_pack={prompt_meta.get('context_tokens_after_pack', 0)}"
                    f"{score_summary}"
                )

            query_start = prompt_meta.get("query_start")
            query_end = prompt_meta.get("query_end")
            ntk_truncation_strategy = prompt_meta.get("ntk_truncation_strategy")
            prompt_offset = 0
            prompt_budget = self.max_length - self.max_new_tokens

            if len(prompt_ids) > prompt_budget:
                can_preserve_query = (
                    query_start is not None
                    and query_end is not None
                    and 0 <= int(query_start) < int(query_end) <= len(prompt_ids)
                )
                if can_preserve_query:
                    original_prompt_len = len(prompt_ids)
                    query_start_int = int(query_start)
                    query_end_int = int(query_end)
                    query_ids = prompt_ids[query_start_int:query_end_int]
                    if ntk_truncation_strategy == "head_tail":
                        tail_keep = max(len(query_ids), prompt_budget // 2)
                        tail_keep = min(tail_keep, prompt_budget)
                        head_keep = prompt_budget - tail_keep
                        tail_start = original_prompt_len - tail_keep
                        if tail_start > query_start_int:
                            tail_keep = original_prompt_len - query_start_int
                            tail_keep = min(tail_keep, prompt_budget)
                            head_keep = prompt_budget - tail_keep
                            tail_start = original_prompt_len - tail_keep
                        eval_logger.warning(
                            f"Prompt length {original_prompt_len} is larger than {prompt_budget}, "
                            f"cutoff prompt middle; "
                            f"kept_head_tokens={head_keep}, "
                            f"kept_tail_tokens={tail_keep}, "
                            f"query_tokens={len(query_ids)}, "
                            f"dropped_middle_tokens={tail_start - head_keep}"
                        )
                        prompt_ids = prompt_ids[:head_keep] + prompt_ids[tail_start:]
                        query_start = head_keep + max(0, query_start_int - tail_start)
                        query_end = head_keep + max(0, query_end_int - tail_start)
                    else:
                        prefix_context_budget = prompt_budget - len(query_ids)
                        if prefix_context_budget > 0:
                            kept_prefix_context_len = min(query_start_int, prefix_context_budget)
                            dropped_context_tail = query_start_int - kept_prefix_context_len
                            eval_logger.warning(
                                f"Prompt length {original_prompt_len} is larger than {prompt_budget}, "
                                f"cutoff context tail before query; "
                                f"kept_prefix_context_tokens={kept_prefix_context_len}, "
                                f"query_tokens={len(query_ids)}, "
                                f"dropped_context_tail_tokens={dropped_context_tail}"
                            )
                            prompt_ids = prompt_ids[:kept_prefix_context_len] + query_ids
                            query_start = kept_prefix_context_len
                            query_end = len(prompt_ids)
                        else:
                            eval_logger.warning(
                                f"Prompt length {original_prompt_len} is larger than {prompt_budget}, "
                                f"query length {len(query_ids)} exceeds the prompt budget; cutoff on the left side"
                            )
                            prompt_offset = original_prompt_len - prompt_budget
                            prompt_ids = prompt_ids[-prompt_budget:]
                else:
                    eval_logger.warning(
                        f"Prompt length {len(prompt_ids)} is larger than {prompt_budget}, cutoff on the left side"
                    )
                    prompt_offset = len(prompt_ids) - prompt_budget
                    prompt_ids = prompt_ids[-prompt_budget:]

            prompt_tensor = torch.tensor([prompt_ids], device=self.device, dtype=torch.long)
            if query_start is not None and query_end is not None:
                adjusted_query_start = max(0, int(query_start) - prompt_offset)
                adjusted_query_end = min(prompt_tensor.shape[1], int(query_end) - prompt_offset)
                if adjusted_query_end > adjusted_query_start:
                    self._parallelcomp_active_query_start = adjusted_query_start
                    self._parallelcomp_active_query_end = adjusted_query_end
                else:
                    self._parallelcomp_active_query_start = None
                    self._parallelcomp_active_query_end = None
            else:
                self._parallelcomp_active_query_start = None
                self._parallelcomp_active_query_end = None

            # Use generate_block_single method to generate, returns EOS-truncated response text
            try:
                response = self._generate_block_single(prompt_tensor)
                responses.append(response)
            finally:
                self._parallelcomp_active_scoring_query_ids = None
                self._parallelcomp_active_scoring_query_source = None
                self._parallelcomp_active_query_start = None
                self._parallelcomp_active_query_end = None

        return responses
    
    def _generate_block_single(self, prompt):
        """
        Generates a response for a single prompt using parallel block generation, based on KV cache,
        and using pre-generated attention masks.
        Returns: EOS-truncated response text.
        """
        self.model.eval()
        
        mask_id = self.mask_token_id
        block_size = self.block_size
        block_add_threshold = self.block_add_threshold
        skip_threshold = self.skip_threshold
        decoded_token_threshold = self.decoded_token_threshold
        
        attn_dtype = self.target_dtype if self.target_dtype is not None and self.target_dtype != "auto" else torch.bfloat16
        
        with torch.inference_mode():
            # Initialization
            x_t = prompt.to(self.device)
            self._parallelcomp_prompt_cache_compression_done = False
            self._parallelcomp_post_compression_position_offset = None
            cached_positions_per_layer = []
            cached_block_ranges_per_layer = []
            shared_cached_length = 0
            
            # Track block states - state can be: 'active', 'to_cache', 'in_cache'
            # Added 'is_complete' field to indicate whether it's a complete state (True) or incomplete (False)
            block_states = self._init_prompt_block_states(prompt.shape[1])
            canonical_block_tokens = self._init_canonical_block_tokens(x_t, block_states)
            
            # Initialize cache
            past_key_values = None
            last_logits = None
            
            current_blocks = 0  # Number of active blocks
            step = 0
            eos_detected = False  # EOS detection flag
            
            while current_blocks >= 0:
                step += 1
                
                # Check if a new block needs to be added
                generated_block_count = self._count_generated_blocks(block_states)
                max_generation_blocks = 0
                if self.max_new_tokens > 0:
                    max_generation_blocks = max(
                        1, (self.max_new_tokens + block_size - 1) // block_size
                    )
                pending_cache_blocks = any(
                    state["state"] == "to_cache" for state in block_states.values()
                )
                defer_generation_for_raw_prefill = (
                    pending_cache_blocks and not self.parallelcomp_cache_compress_mode
                )
                if (
                    generated_block_count < max_generation_blocks
                    and not eos_detected
                    and not defer_generation_for_raw_prefill
                ):
                    last_block_id = max(block_states.keys())
                    current_progress = (block_states[last_block_id]['total_masks'] - 
                                      block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']
                    if current_progress >= block_add_threshold:
                        # Add new block - defaults to incomplete state
                        new_block_id = max(block_states.keys()) + 1
                        new_start_pos = x_t.shape[1]
                        new_block_tokens = [mask_id] * block_size
                        if self.parallelcomp_generation_block_bos_ablation:
                            bos_ids = self._get_bos_token_ids()
                            if bos_ids:
                                bos_prefix = bos_ids[:block_size]
                                new_block_tokens[:len(bos_prefix)] = bos_prefix
                        new_mask_count = sum(1 for token_id in new_block_tokens if token_id == mask_id)
                        x_t = torch.cat([x_t, torch.tensor([new_block_tokens]).to(self.device)], dim=1)
                        
                        block_states[new_block_id] = {
                            'start_pos': new_start_pos,
                            'end_pos': new_start_pos + block_size,
                            'mask_count': new_mask_count,
                            'total_masks': max(1, new_mask_count),
                            'rope_span': block_size,
                            'state': 'active',
                            'is_complete': False,  # New block defaults to incomplete state
                            'is_prompt_window': False,
                        }
                        current_blocks += 1
                
                # At the beginning of each loop, update block completion states
                self._update_block_completion_states(block_states, decoded_token_threshold)
                # Determine which blocks need to be added to cache
                blocks_to_cache = [bid for bid, state in block_states.items() 
                                if state['state'] == 'to_cache']
                # Check if there are still mask tokens
                mask_index = (x_t == mask_id)
                if mask_index.sum() == 0 and current_blocks == 0 and not blocks_to_cache:
                    break
                    
                # Determine the part to process
                cache_length = shared_cached_length if past_key_values is not None else 0
                
                # Determine content to add to cache
                update_kvcache = 0
                if blocks_to_cache:
                    # Find the earliest block that needs to be cached
                    earliest_block_id = min(blocks_to_cache)
                    earliest_pos = block_states[earliest_block_id]['start_pos']
                    
                    # Find the latest block that needs to be cached
                    latest_block_id = max(blocks_to_cache)
                    latest_pos = block_states[latest_block_id]['end_pos']
                    
                    # Update cache for all blocks within this range
                    update_kvcache = latest_pos - earliest_pos
                
                # Create input sequence for forward pass
                process_start_pos = cache_length
                
                if update_kvcache > 0:
                    # Need to update cache - use completed blocks
                    earliest_block_to_cache = min(blocks_to_cache)
                    input_seq = x_t[:, block_states[earliest_block_to_cache]['start_pos']:]
                    process_start_pos = block_states[earliest_block_to_cache]['start_pos']
                else:
                    # Only process active blocks
                    active_blocks = [bid for bid in block_states.keys() if block_states[bid]['state'] == 'active']
                    if active_blocks:
                        # Get all active blocks after the cache
                        earliest_active_after_cache = float('inf')
                        for bid in active_blocks:
                            if block_states[bid]['start_pos'] >= cache_length:
                                earliest_active_after_cache = min(earliest_active_after_cache, block_states[bid]['start_pos'])
                        
                        if earliest_active_after_cache < float('inf'):
                            input_seq = x_t[:, earliest_active_after_cache:]
                            process_start_pos = earliest_active_after_cache
                        else:
                            # No active blocks after cache, this shouldn't happen
                            input_seq = x_t[:, cache_length:]
                            # If cache length is already equal to or exceeds sequence length, exit
                            if cache_length >= x_t.shape[1]:
                                print(f"Cache length ({cache_length}) >= sequence length ({x_t.shape[1]}) at step {step}. Exiting generation loop.")
                                raise Exception("Cache length >= sequence length")
                    else:
                        # No active blocks, but might have blocks to cache in next iteration
                        break
                
                # Check if input_seq is empty
                if input_seq.shape[1] == 0:
                    print(f"Warning: input_seq is empty at step {step}. Breaking generation loop.")
                    raise Exception("input_seq is empty")
                
                input_length = input_seq.shape[1]
                input_block_ids, input_prompt_window_mask, input_rope_positions = self._build_input_position_ids(
                    block_states=block_states,
                    process_start_pos=process_start_pos,
                    input_length=input_length,
                    update_kvcache=update_kvcache,
                )
                input_cache_positions = self._build_forward_cache_positions(
                    position_ids=input_rope_positions,
                    cached_length=cache_length,
                    update_kvcache=update_kvcache,
                )
                if cached_positions_per_layer:
                    attention_mask = build_unified_sparse_block_attention_mask(
                        query_block_ids=input_block_ids,
                        cached_positions_per_layer=cached_positions_per_layer,
                        query_prompt_window_mask=input_prompt_window_mask,
                        update_kvcache=update_kvcache,
                        device=self.device,
                        dtype=attn_dtype,
                    )
                else:
                    attention_mask = build_sparse_block_attention_mask(
                        query_block_ids=input_block_ids,
                        cached_length=0,
                        query_prompt_window_mask=input_prompt_window_mask,
                        update_kvcache=update_kvcache,
                        device=self.device,
                        dtype=attn_dtype,
                    )
                
                # Forward pass
                has_active_blocks_in_forward = any(
                    state["state"] == "active" and state["end_pos"] > process_start_pos
                    for state in block_states.values()
                )
                logits_to_keep = 1 if update_kvcache > 0 and not has_active_blocks_in_forward else 0
                outputs = self.model(
                    input_seq,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=input_rope_positions.unsqueeze(0),
                    cache_position=input_cache_positions,
                    use_cache=True,
                    update_kvcache=update_kvcache,
                    num_logits_to_keep=logits_to_keep,
                )
                
                # If needed, update cache
                if update_kvcache > 0:
                    # Store logits of the last cached position for next token prediction.
                    cache_end_idx = outputs.logits.shape[1] - 1 if logits_to_keep == 1 else update_kvcache - 1
                    last_logits = outputs.logits[:, cache_end_idx, :].unsqueeze(1)
                    
                    # Update cache
                    past_key_values = outputs.past_key_values
                    if not cached_positions_per_layer:
                        num_layers = len(past_key_values.key_cache)
                        cached_positions_per_layer = [
                            torch.empty(0, device=self.device, dtype=torch.long) for _ in range(num_layers)
                        ]
                        cached_block_ranges_per_layer = [{} for _ in range(num_layers)]
                    old_layer_lengths = [
                        layer_cached_positions.shape[0] for layer_cached_positions in cached_positions_per_layer
                    ]
                    new_cached_slice = self._build_cache_write_positions(
                        cache_start=old_layer_lengths[0] if old_layer_lengths else 0,
                        update_kvcache=update_kvcache,
                    )
                    cached_positions_per_layer = [
                        torch.cat([layer_cached_positions, new_cached_slice], dim=0)
                        for layer_cached_positions in cached_positions_per_layer
                    ]
                    sorted_blocks_to_cache = sorted(blocks_to_cache)
                    self._snapshot_canonical_block_tokens(
                        x_t=x_t,
                        block_states=block_states,
                        block_ids=sorted_blocks_to_cache,
                        canonical_block_tokens=canonical_block_tokens,
                    )
                    block_cursor = 0
                    for layer_idx, layer_block_ranges in enumerate(cached_block_ranges_per_layer):
                        layer_cursor = old_layer_lengths[layer_idx]
                        for block_id in sorted_blocks_to_cache:
                            block_len = block_states[block_id]['end_pos'] - block_states[block_id]['start_pos']
                            layer_block_ranges[block_id] = (layer_cursor + block_cursor, layer_cursor + block_cursor + block_len)
                            block_cursor += block_len
                        block_cursor = 0
                    
                    # Mark blocks as cached
                    for block_id in blocks_to_cache:
                        block_states[block_id]['state'] = 'in_cache'

                    shared_cached_length += update_kvcache

                    x_t, past_key_values, cached_positions_per_layer, cached_block_ranges_per_layer, shared_cached_length, cache_compressed, replay_last_logits = self._compress_cached_prefix_blocks(
                        x_t=x_t,
                        past_key_values=past_key_values,
                        block_states=block_states,
                        mask_id=mask_id,
                        cached_positions_per_layer=cached_positions_per_layer,
                        cached_block_ranges_per_layer=cached_block_ranges_per_layer,
                        shared_cached_length=shared_cached_length,
                        canonical_block_tokens=canonical_block_tokens,
                    )
                    if cache_compressed:
                        last_logits = replay_last_logits
                        mask_index = (x_t == mask_id)
                        rerun_outputs, rerun_process_start_pos = self._rerun_active_blocks_after_cache_compression(
                            x_t=x_t,
                            block_states=block_states,
                            past_key_values=past_key_values,
                            cached_positions_per_layer=cached_positions_per_layer,
                            attn_dtype=attn_dtype,
                        )
                        if rerun_outputs is not None:
                            outputs = rerun_outputs
                            process_start_pos = rerun_process_start_pos

                if not any(state["state"] == "active" for state in block_states.values()):
                    continue
                
                # Get correctly shifted logits for prediction
                logits = self._shift_logits(outputs.logits, last_logit=last_logits)
                
                # Process mask tokens for each active block
                blocks_to_deactivate = []
                
                for block_id in sorted(block_states.keys()):
                    if block_states[block_id]['state'] != 'active':
                        continue
                    
                    # Get mask positions for this block
                    block_start = block_states[block_id]['start_pos']
                    block_end = block_states[block_id]['end_pos']
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False

                    # If the current block has no masks, skip it
                    if block_mask_index.sum() == 0:
                        blocks_to_deactivate.append(block_id)
                        continue
                    
                    # Calculate relative position for logits
                    logit_offset = block_start - process_start_pos
                    block_rel_positions = torch.where(block_mask_index[0, block_start:block_end])[0]
                    
                    if block_rel_positions.size(0) > 0:
                        # Get logits for masked positions
                        block_mask_logits = logits[:, logit_offset + block_rel_positions, :]
                    
                        # Sample tokens
                        confidence, x0, initial_confidence = sample_tokens(
                            block_mask_logits.squeeze(0), 
                            self.temperature, 
                            top_p=self.top_p, 
                            top_k=self.top_k, 
                            neg_entropy=(self.sampling_strategy == "neg_entropy"),
                            margin_confidence=(self.sampling_strategy == "margin_confidence")
                        )
                        
                        # Apply different sampling strategies based on the block's complete/incomplete state
                        is_complete = block_states[block_id]['is_complete']
                        
                        if is_complete:
                            # Complete state: apply confidence threshold, if no high confidence, select highest
                            high_conf_indices = torch.where(initial_confidence > skip_threshold)[0]
                            
                            if len(high_conf_indices) == 0:
                                number_transfer_tokens = 1
                                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                            else:
                                transfer_index = torch.tensor([], device=self.device, dtype=torch.long)
                            
                            # Merge indices
                            all_indices = torch.unique(torch.cat([transfer_index, high_conf_indices]))
                        else:
                            # Incomplete state: only apply confidence threshold, if none exceed, select no tokens
                            high_conf_indices = torch.where(initial_confidence > skip_threshold)[0]
                            all_indices = high_conf_indices
                        
                        # Update tokens
                        if len(all_indices) > 0:
                            x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_id
                            x0_[all_indices] = x0[all_indices].clone()
                                
                            # Map indices back to original positions
                            for i, idx in enumerate(all_indices):
                                abs_pos = block_start + block_rel_positions[idx]
                                x_t[0, abs_pos] = x0_[idx]
                            
                            # Update block state
                            block_states[block_id]['mask_count'] -= len(all_indices)
                            
                            # Check EOS token
                            eos_token_id = self.tokenizer.eos_token_id
                            if eos_token_id is not None:
                                for idx in all_indices:
                                    if x0[idx].item() == eos_token_id:
                                        eos_detected = True
                                        break

                    # If no masks remain in this block, deactivate it
                    mask_index = (x_t == mask_id)
                    block_mask_index = mask_index.clone()
                    block_mask_index[:, :block_start] = False
                    block_mask_index[:, block_end:] = False
                    if block_mask_index.sum() == 0:
                        blocks_to_deactivate.append(block_id)
                        continue
                
                # Deactivate completed blocks and mark them for caching in the next iteration
                for block_id in blocks_to_deactivate:
                    if block_states[block_id]['state'] == 'active':
                        # Check if all preceding blocks are already non-active
                        can_deactivate = True
                        for prev_block_id in range(block_id):
                            if prev_block_id in block_states and block_states[prev_block_id]['state'] == 'active':
                                can_deactivate = False
                                break
                        
                        # Only mark the current block as 'to_cache' if all preceding blocks are non-active
                        if can_deactivate:
                            block_states[block_id]['state'] = 'to_cache'
                            current_blocks -= 1
                        # If there are active blocks before, keep current block as active (do nothing)

                # Safety check
                if step > 10000:
                    print(f"WARNING: Hit safety check at step {step}. Exiting generation loop.")
                    break
        
        # First, calculate non-EOS tokens for the full generated sequence
        generated_sequence = x_t[0, prompt.shape[1]:].tolist()
        non_eos_tokens = self._count_non_eos_tokens_before_truncation(
            x_t[0].tolist(), prompt.shape[1]
        )
        
        # Accumulate to total tokens
        if not hasattr(self, 'total_generated_tokens'):
            self.total_generated_tokens = 0
        self.total_generated_tokens += non_eos_tokens
        
        # Generate EOS-truncated response text (consistent with other file logic)
        response = self.tokenizer.decode(generated_sequence).split(self.tokenizer.eos_token)[0]
        
        return response

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        """
        Updates the complete/incomplete state of blocks.
        Iterates through blocks from front to back. If a block's decoded token count
        is greater than the threshold, the next block to its right (if it exists)
        is set to a complete state.
        """
        for block_id in sorted(block_states.keys()):
            # if block_id == 0:  # Skip prompt block
            #     continue
            
            # Calculate decoded tokens for the current block
            decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
            decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
            # If the current block's decoded token count is greater than the threshold,
            # then the next block (if it exists) is set to a complete state.
            # print("decode_ratio",decode_ratio)
            # print("decoded_token_threshold",decoded_token_threshold)
            if decode_ratio >= decoded_token_threshold:
                next_block_id = block_id + 1
                if next_block_id in block_states:
                    block_states[next_block_id]['is_complete'] = True

    def _shift_logits(self, logits, last_logit=None, block_size=None):
        """Shifts logits to the right by one position, for autoregressive generation"""
        # Check if logits are empty
        if logits.shape[1] == 0:
            print("Warning: logits sequence length is 0, returning empty logits")
            raise Exception("logits sequence length is 0")
            
        shifted_logits = torch.zeros_like(logits)
        shifted_logits[:, 1:, :] = logits[:, :-1, :]
        if last_logit is not None:
            shifted_logits[:, 0, :] = last_logit
            return shifted_logits
        shifted_logits[:, 0, :] = 1.0
        return shifted_logits

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        
        # Initialize statistics counters
        if not hasattr(self, 'total_generated_tokens'):
            self.total_generated_tokens = 0
        num_tokens = 0
        num_nfe = 0  # Number of Forward Evaluations

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        
        start_time = time.time()

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]['until']:
                        r = r.split(s)[0]
                    responses[i] = r

            res.extend(responses)
            pbar.update(len(contexts))

        end_time = time.time()
        total_time = end_time - start_time
        
        # Accumulate statistics
        num_tokens = self.total_generated_tokens
        num_nfe = self.diffusion_steps * len(requests)  # Estimate NFE
        
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
