#!/usr/bin/env python3
"""Fast-dLLM v1 + ParallelComp runtime.

This module implements the full KV path used by our ParallelComp experiments:

1. split long context into chunks and prepend BOS to every chunk by default;
2. select context chunks with the real benchmark query;
3. run each selected chunk together with the query/draft query for local scoring;
4. optionally evict tokens inside each chunk;
5. concatenate the retained KV cache with controlled/reused RoPE positions;
6. run Fast-dLLM blockwise diffusion generation over the compressed cache;
7. write completed generated blocks back into KV so later blocks can attend them.

The implementation is intentionally benchmark-agnostic.  Evaluation scripts
should tokenize prefix/context/query and call ``FastDLLMParallelComp.generate``.
"""

from __future__ import annotations

import math
import os
import sys
import time
import types
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributions as dists
import torch.nn.functional as F
import transformers

from fastdllm_v1_model import _resolve_dtype, default_fastdllm_dream_dir


def default_fastdllm_llada_dir() -> str:
    return os.environ.get("FASTDLLM_LLADA_DIR", "/home/ma-user/work/Fast-dLLM/v1/llada")


Cache = Optional[List[Tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class FastDLLMParallelCompConfig:
    fastdllm_dream_dir: str
    pretrained: str
    model_backend: str = "dream"
    fastdllm_llada_dir: Optional[str] = None
    device: str = "cuda"
    dtype: str = "auto"
    trust_remote_code: bool = True
    llada_score_batch_size: int = 8

    max_new_tokens: int = 32
    max_length: int = 4096
    block_length: int = 32
    diffusion_steps: Optional[int] = None
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    alg: str = "confidence_threshold"
    alg_temp: Optional[float] = 0.0
    threshold: float = 0.9
    add_bos_token: bool = True
    rope_scale_factor: float = 1.0
    rope_scaling_type: str = "yarn"

    chunk_size: int = 1024
    topk_chunks: int = 4
    keep_first_chunk: bool = False
    split_from_tail: bool = False
    chunk_bos: bool = True
    force_keep_chunk_bos: bool = True
    cache_build_mode: str = "chunk_query"

    score_mode: str = "draft_self_information"
    score_query_window: int = 0
    score_draft_tokens: int = 16
    score_draft_steps: Optional[int] = None
    score_draft_fixed_steps: bool = False
    score_draft_partial_steps: Optional[int] = None
    score_draft_partial_rounds: Optional[int] = None
    score_draft_score_all_slots: bool = False
    score_llada_shift_logits: bool = False
    score_attention_mask: str = "causal"
    score_context_mode: str = "single_chunk"
    score_batch_size: int = 8
    attention_score_layers: int = 4
    attention_query_window: int = 0

    token_capacity: int = 0
    token_score_query_window: int = 8
    token_score_layers: int = 0
    token_score_layer_mode: str = "all"
    token_score_reduce: str = "sum"
    token_score_pooling: str = "maxpool"
    token_score_pool_kernel: int = 7
    token_score_head_reduce: str = "sum"
    token_score_layer_reduce: str = "mean"
    token_score_direction: str = "query_to_chunk"
    token_score_keep: str = "high"
    token_score_include_prefix: bool = True
    token_attention_mask: str = "causal"
    token_score_use_generated: bool = False
    token_eviction_granularity: str = "global"

    chunk_position_mode: str = "reuse"
    chunk_query_position_mode: str = "after_reused_window"
    query_position_mode: str = "after_reused_window"
    generation_position_mode: str = "after_query"


@dataclass
class FastDLLMChunkMeta:
    chunk_index: int
    original_tokens: int
    kept_tokens: int
    removed_tokens: int
    cache_start: int
    cache_end: int
    rope_start: int
    rope_end: int
    score: Optional[float] = None
    kept_positions: List[int] = field(default_factory=list)


@dataclass
class FastDLLMParallelCompResult:
    text: str
    sequences: List[int]
    selected_chunk_indices: List[int]
    chunk_scores: Dict[int, float]
    cache_build_mode: str
    prefix_tokens: int
    query_tokens: int
    raw_context_tokens: int
    candidate_chunks: int
    cache_tokens: int
    removed_tokens: int
    chunk_meta: List[FastDLLMChunkMeta]
    generation_blocks: int
    generation_block_length: int


def dataclass_to_jsonable(obj):
    if is_dataclass(obj):
        return {
            field.name: dataclass_to_jsonable(getattr(obj, field.name))
            for field in fields(obj)
        }
    if isinstance(obj, dict):
        return {key: dataclass_to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(value) for value in obj]
    return obj


def split_token_chunks(
    token_ids: Sequence[int],
    chunk_size: int,
    split_from_tail: bool = False,
) -> List[List[int]]:
    ids = list(token_ids)
    if not ids:
        return []
    if chunk_size <= 0 or len(ids) <= chunk_size:
        return [ids]

    chunks: List[List[int]] = []
    cursor = 0
    if split_from_tail:
        leading_remainder = len(ids) % chunk_size
        if leading_remainder > 0:
            chunks.append(ids[:leading_remainder])
            cursor = leading_remainder

    while cursor < len(ids):
        chunks.append(ids[cursor:cursor + chunk_size])
        cursor += chunk_size
    return chunks


def _top_p_logits(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None:
        return logits
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature > 0:
        logits = logits / temperature
    logits = _top_p_logits(logits, top_p)
    logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
    if neg_entropy:
        log_probs = torch.log(probs + 1e-10)
        confidence = torch.sum(probs * log_probs, dim=-1)
    return confidence, x0


def _patch_fastdllm_base_model_position_ids(model) -> None:
    """Make Fast-dLLM v1 respect explicit full-key position_ids.

    The upstream v1 DreamBaseModel regenerates position_ids from cache length.
    That is fine for normal Fast-dLLM, but it breaks ParallelComp position reuse
    because compressed cache slots no longer have continuous positions.  This
    local monkey patch keeps upstream behavior when position_ids is None and
    only changes the explicit-position path.
    """

    base = getattr(model, "model", None)
    if base is None or getattr(base, "_parallelcomp_position_ids_patch", False):
        return

    original_forward = base.forward
    forward_globals = original_forward.__func__.__globals__
    BaseModelOutputWithPast = forward_globals["BaseModelOutputWithPast"]
    logger = forward_globals["logger"]

    def patched_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        dual_cache: Optional[bool] = False,
        replace_position: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = past_key_values[0][0].shape[1] if past_key_values is not None else 0
        if position_ids is None:
            if not dual_cache:
                position_ids = torch.arange(
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                ).unsqueeze(0)
            elif past_key_values is not None:
                position_ids = torch.arange(past_seen_tokens, device=inputs_embeds.device).unsqueeze(0)
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        else:
            position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)

        hidden_states = inputs_embeds
        attn_key_values: Cache = [] if use_cache else None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past_key_value = past_key_values[layer_idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    layer_past_key_value,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    dual_cache,
                    replace_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    dual_cache=dual_cache,
                    replace_position=replace_position,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if use_cache:
                present_index = 2 if output_attentions else 1
                attn_key_values.append(layer_outputs[present_index])

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            values = [hidden_states, all_hidden_states, all_self_attns, attn_key_values]
            return tuple(value for value in values if value is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            past_key_values=attn_key_values,
        )

    base.forward = types.MethodType(patched_forward, base)
    base._parallelcomp_position_ids_patch = True


class FastDLLMParallelComp:
    def __init__(self, config: FastDLLMParallelCompConfig):
        self.config = config
        self.model_backend = (config.model_backend or "dream").lower()
        if self.model_backend not in {"dream", "llada"}:
            raise ValueError(f"Unsupported model_backend: {config.model_backend}")
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(config.max_new_tokens)
        self.block_length = int(config.block_length)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        self.diffusion_steps = (
            int(config.diffusion_steps)
            if config.diffusion_steps is not None
            else max(1, math.ceil(self.max_new_tokens / self.block_length))
        )
        if config.token_score_direction not in {"query_to_chunk", "chunk_to_query", "bidirectional"}:
            raise ValueError(f"Unsupported token_score_direction: {config.token_score_direction}")
        if config.token_score_keep not in {"high", "low"}:
            raise ValueError(f"Unsupported token_score_keep: {config.token_score_keep}")
        if config.token_eviction_granularity not in {"global", "per_head"}:
            raise ValueError(f"Unsupported token_eviction_granularity: {config.token_eviction_granularity}")

        self._llada_attention_capture_patched = False
        self._llada_collect_attentions = False
        self._llada_apply_attention_bias = False
        self._llada_capture_layer_indices: Optional[set[int]] = None
        self._llada_captured_attentions: List[Optional[torch.Tensor]] = []

        target_dtype = _resolve_dtype(config.dtype)
        self._load_backend_model(target_dtype)
        if target_dtype not in (None, "auto"):
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.pretrained,
            trust_remote_code=config.trust_remote_code,
        )
        self.generated_token_num = 0
        self.total_generation_time = 0.0

    def _prepare_fastdllm_module_path(self, source_dir: str) -> None:
        source_dir = os.path.abspath(source_dir)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Fast-dLLM source dir not found: {source_dir}")
        for module_name in list(sys.modules):
            if module_name == "model" or module_name.startswith("model."):
                del sys.modules[module_name]
        if source_dir in sys.path:
            sys.path.remove(source_dir)
        sys.path.insert(0, source_dir)

    def _load_backend_model(self, target_dtype):
        if self.model_backend == "dream":
            self._prepare_fastdllm_module_path(self.config.fastdllm_dream_dir)
            from model.configuration_dream import DreamConfig
            from model.modeling_dream import DreamModel, DreamRotaryEmbedding

            model_config = DreamConfig.from_pretrained(self.config.pretrained)
            self.model = DreamModel.from_pretrained(
                self.config.pretrained,
                config=model_config,
                torch_dtype=target_dtype,
                trust_remote_code=False,
            ).eval()
            self._apply_rope_scaling(DreamRotaryEmbedding, float(self.config.rope_scale_factor or 1.0))
            _patch_fastdllm_base_model_position_ids(self.model)
            return

        llada_dir = self.config.fastdllm_llada_dir or default_fastdllm_llada_dir()
        self._prepare_fastdllm_module_path(llada_dir)
        from model.configuration_llada import LLaDAConfig
        from model.modeling_llada import LLaDAModelLM, RotaryEmbedding

        model_config = LLaDAConfig.from_pretrained(self.config.pretrained)
        if hasattr(model_config, "flash_attention"):
            model_config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(
            self.config.pretrained,
            config=model_config,
            torch_dtype=target_dtype,
            trust_remote_code=self.config.trust_remote_code,
        ).eval()
        self._apply_llada_rope_scaling(
            RotaryEmbedding,
            float(self.config.rope_scale_factor or 1.0),
            self.config.rope_scaling_type,
        )
        self._patch_llada_attention_capture()

    def _patch_llada_attention_capture(self) -> None:
        if self.model_backend != "llada" or self._llada_attention_capture_patched:
            return

        layer_idx = 0
        for module in self.model.modules():
            scaled_attn = getattr(module, "_scaled_dot_product_attention", None)
            attention = getattr(module, "attention", None)
            if not callable(scaled_attn) or not callable(attention):
                continue

            module._parallelcomp_layer_idx = layer_idx
            module._parallelcomp_current_attention_bias = None
            original_scaled_attn = scaled_attn
            original_attention = attention
            parent = self

            def wrapped_attention(
                q,
                k,
                v,
                mask=None,
                attention_bias=None,
                layer_past=None,
                use_cache=False,
                replace_position=None,
                _module=module,
                _original_attention=original_attention,
            ):
                _module._parallelcomp_current_attention_bias = attention_bias if attention_bias is not None else mask
                try:
                    return _original_attention(
                        q,
                        k,
                        v,
                        mask=mask,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        replace_position=replace_position,
                    )
                finally:
                    _module._parallelcomp_current_attention_bias = None

            def wrapped_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                _module=module,
                _original_scaled_attn=original_scaled_attn,
            ):
                if not parent._llada_collect_attentions:
                    return _original_scaled_attn(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                    )
                return parent._llada_attention_with_capture(
                    _module,
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

            module.attention = wrapped_attention
            module._scaled_dot_product_attention = wrapped_scaled_dot_product_attention
            layer_idx += 1

        self._llada_attention_capture_patched = True

    def _apply_rope_scaling(self, rotary_cls, factor: float) -> None:
        if factor <= 1.0:
            return
        original_max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if original_max_pos is None:
            raise ValueError("Cannot apply RoPE scaling: config has no max_position_embeddings")
        self.model.config.rope_scaling = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_pos,
        }
        for module in self.model.modules():
            if isinstance(module, rotary_cls):
                module.__init__(config=self.model.config)
                module.to(self.device)

    def _apply_llada_rope_scaling(self, rotary_cls, factor: float, scaling_type: str = "yarn") -> None:
        if factor <= 1.0:
            return

        scaling_type = (scaling_type or "yarn").lower()
        if scaling_type not in {"linear", "yarn"}:
            raise ValueError(f"Unsupported LLaDA RoPE scaling type: {scaling_type}")

        model_config = getattr(getattr(self.model, "model", None), "config", None)
        if model_config is None:
            raise ValueError("Cannot apply LLaDA RoPE scaling: model config not found")
        original_max_pos = getattr(model_config, "max_sequence_length", None)
        if original_max_pos is not None:
            setattr(model_config, "parallelcomp_original_max_sequence_length", original_max_pos)
            setattr(model_config, "max_sequence_length", int(math.ceil(original_max_pos * factor)))
        setattr(
            model_config,
            "parallelcomp_rope_scaling",
            {
                "rope_type": scaling_type,
                "factor": factor,
                "original_max_sequence_length": original_max_pos,
            },
        )

        def yarn_inv_freq(
            module,
            dim: int,
            device: torch.device,
            max_position_embeddings: int,
            rope_factor: float,
        ) -> Tuple[torch.Tensor, float]:
            attention_factor = 0.1 * math.log(rope_factor) + 1.0
            beta_fast = 32
            beta_slow = 1

            def find_correction_dim(num_rotations: int) -> float:
                return (
                    dim
                    * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
                    / (2 * math.log(module.rope_theta))
                )

            def find_correction_range() -> Tuple[int, int]:
                low = math.floor(find_correction_dim(beta_fast))
                high = math.ceil(find_correction_dim(beta_slow))
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(low: int, high: int) -> torch.Tensor:
                if low == high:
                    high += 1
                ramp = (torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
                return torch.clamp(ramp, 0, 1)

            pos_freqs = module.rope_theta ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
            )
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (rope_factor * pos_freqs)
            low, high = find_correction_range()
            extrapolation_weight = 1 - linear_ramp_factor(low, high)
            inv_freq = (
                inv_freq_interpolation * (1 - extrapolation_weight)
                + inv_freq_extrapolation * extrapolation_weight
            )
            return inv_freq, attention_factor

        for module in self.model.modules():
            if not isinstance(module, rotary_cls):
                continue

            module._parallelcomp_rope_scale_factor = float(factor)
            module._parallelcomp_rope_scaling_type = scaling_type

            def scaled_get_rotary_embedding(
                seq_len: int,
                device: torch.device,
                _module=module,
                _factor=float(factor),
                _scaling_type=scaling_type,
                _original_max_pos=int(original_max_pos or 4096),
            ):
                cache = getattr(_module, "_RotaryEmbedding__cache")
                key_sin = f"parallelcomp_rope_pos_sin_{_scaling_type}_{_factor:g}"
                key_cos = f"parallelcomp_rope_pos_cos_{_scaling_type}_{_factor:g}"
                pos_sin = cache.get(key_sin)
                pos_cos = cache.get(key_cos)
                if (
                    pos_sin is not None
                    and pos_cos is not None
                    and pos_sin.shape[-2] >= seq_len
                    and pos_cos.shape[-2] >= seq_len
                ):
                    if pos_sin.device != device:
                        pos_sin = pos_sin.to(device)
                        cache[key_sin] = pos_sin
                    if pos_cos.device != device:
                        pos_cos = pos_cos.to(device)
                        cache[key_cos] = pos_cos
                    return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

                with torch.autocast(device.type, enabled=False):
                    dim = _module.config.d_model // _module.config.n_heads
                    attention_factor = 1.0
                    if _scaling_type == "yarn":
                        inv_freq, attention_factor = yarn_inv_freq(
                            _module,
                            dim,
                            device,
                            _original_max_pos,
                            _factor,
                        )
                    else:
                        inv_freq = 1.0 / (
                            _module.rope_theta
                            ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
                        )
                    seq = torch.arange(seq_len, device=device, dtype=torch.float)
                    if _scaling_type == "linear":
                        seq = seq / _factor
                    freqs = torch.outer(seq, inv_freq)
                    positions = torch.cat((freqs, freqs), dim=-1)
                    pos_sin = positions.sin()[None, None, :, :] * attention_factor
                    pos_cos = positions.cos()[None, None, :, :] * attention_factor
                cache[key_sin] = pos_sin
                cache[key_cos] = pos_cos
                return pos_sin, pos_cos

            module.get_rotary_embedding = scaled_get_rotary_embedding

    def _mask_token_id(self) -> int:
        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.model.config, "mask_token_id", None)
        if mask_token_id is None:
            generation_config = getattr(self.model, "generation_config", None)
            mask_token_id = getattr(generation_config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(f"Fast-dLLM {self.model_backend} mask_token_id is not available")
        return int(mask_token_id)

    def _ids_tensor(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(ids)], device=self.device, dtype=torch.long)

    def _ids_batch_tensor(self, rows: Sequence[Sequence[int]]) -> torch.Tensor:
        return torch.tensor([list(row) for row in rows], device=self.device, dtype=torch.long)

    def _model_forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask="full",
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Cache = None,
        use_cache: bool = False,
        return_dict: bool = True,
        output_attentions: bool = False,
        dual_cache: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
    ):
        if self.model_backend == "dream":
            kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
            if output_attentions:
                kwargs["output_attentions"] = True
            if dual_cache is not None:
                kwargs["dual_cache"] = dual_cache
            if replace_position is not None:
                kwargs["replace_position"] = replace_position
            return self.model(input_ids, **kwargs)

        if output_attentions:
            raise NotImplementedError("Fast-dLLM LLaDA model does not expose output_attentions")
        kwargs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "return_dict": return_dict,
        }
        if not isinstance(attention_mask, str) or attention_mask != "full":
            kwargs["attention_bias"] = attention_mask
        if replace_position is not None:
            kwargs["replace_position"] = replace_position
        return self.model(**kwargs)

    def _llada_additive_attention_bias(
        self,
        raw_bias: Optional[torch.Tensor],
        *,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if raw_bias is None:
            return None
        bias = raw_bias.to(device=device)
        if bias.dtype in (torch.int8, torch.bool):
            additive = bias.to(dtype=torch.float32)
            additive = additive.masked_fill(additive == 0.0, torch.finfo(torch.float32).min)
            additive = additive.masked_fill(additive == 1.0, 0.0)
        else:
            additive = bias.to(dtype=torch.float32)
        additive = additive[:, :, key_len - query_len:key_len, :key_len]
        return additive.to(dtype=dtype)

    def _llada_attention_with_capture(
        self,
        module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if q.size(1) != k.size(1):
            if q.size(1) % k.size(1) != 0:
                raise ValueError(f"Cannot expand LLaDA KV heads {k.size(1)} to query heads {q.size(1)}")
            repeat = q.size(1) // k.size(1)
            k_attn = k.repeat_interleave(repeat, dim=1, output_size=q.size(1))
            v_attn = v.repeat_interleave(repeat, dim=1, output_size=q.size(1))
        else:
            k_attn = k
            v_attn = v

        query_len = q.shape[-2]
        key_len = k_attn.shape[-2]
        scores = torch.matmul(q.float(), k_attn.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])

        if self._llada_apply_attention_bias:
            raw_bias = getattr(module, "_parallelcomp_current_attention_bias", None)
            additive_bias = self._llada_additive_attention_bias(
                raw_bias,
                query_len=query_len,
                key_len=key_len,
                device=scores.device,
                dtype=scores.dtype,
            )
            if additive_bias is not None:
                scores = scores + additive_bias
            elif attn_mask is not None:
                scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)

        if is_causal:
            causal_mask = torch.full(
                (query_len, key_len),
                torch.finfo(scores.dtype).min,
                device=scores.device,
                dtype=scores.dtype,
            )
            offset = key_len - query_len
            for row in range(query_len):
                causal_mask[row, :offset + row + 1] = 0
            scores = scores + causal_mask.view(1, 1, query_len, key_len)

        probs = F.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0:
            probs = F.dropout(probs, p=dropout_p, training=self.model.training)

        layer_idx = getattr(module, "_parallelcomp_layer_idx", None)
        should_capture = (
            layer_idx is not None
            and (
                self._llada_capture_layer_indices is None
                or layer_idx in self._llada_capture_layer_indices
            )
        )
        if should_capture:
            store_dtype = torch.float16 if probs.device.type == "cuda" else torch.float32
            self._llada_captured_attentions[layer_idx] = probs.detach().to(store_dtype)

        return torch.matmul(probs.to(v_attn.dtype), v_attn)

    def _model_forward_with_attentions(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask="full",
        use_cache: bool = False,
        return_dict: bool = True,
        capture_layer_indices: Optional[Sequence[int]] = None,
        apply_attention_bias: bool = True,
    ):
        if self.model_backend == "dream":
            outputs = self._model_forward(
                input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=True,
                return_dict=return_dict,
            )
            return outputs, getattr(outputs, "attentions", None)

        self._patch_llada_attention_capture()
        total_layers = self._num_hidden_layers()
        self._llada_captured_attentions = [None for _ in range(total_layers)]
        self._llada_capture_layer_indices = (
            set(int(idx) for idx in capture_layer_indices)
            if capture_layer_indices is not None
            else None
        )
        self._llada_apply_attention_bias = bool(apply_attention_bias)
        self._llada_collect_attentions = True
        try:
            outputs = self._model_forward(
                input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=False,
                return_dict=return_dict,
            )
            attentions = tuple(self._llada_captured_attentions)
        finally:
            self._llada_collect_attentions = False
            self._llada_apply_attention_bias = False
            self._llada_capture_layer_indices = None
            self._llada_captured_attentions = []
        return outputs, attentions

    @property
    def _mask_dtype(self) -> torch.dtype:
        dtype = next(self.model.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return dtype
        return torch.float32

    def _get_bos_token_ids(self) -> List[int]:
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            return [int(bos_token_id)]
        bos_token = getattr(self.tokenizer, "bos_token", None)
        if bos_token:
            return self.tokenizer.encode(bos_token, add_special_tokens=False)
        return []

    def _maybe_prepend_bos_to_chunk(self, chunk_ids: Sequence[int]) -> List[int]:
        ids = list(chunk_ids)
        if not self.config.chunk_bos:
            return ids
        bos_ids = self._get_bos_token_ids()
        if not bos_ids or ids[:len(bos_ids)] == bos_ids:
            return ids
        with_bos = bos_ids + ids
        chunk_size = int(self.config.chunk_size or 0)
        if chunk_size > 0 and len(with_bos) > chunk_size:
            with_bos = with_bos[:chunk_size]
        return with_bos

    def _window_query(self, query_ids: Sequence[int], window: int) -> List[int]:
        ids = list(query_ids)
        if window and window > 0:
            return ids[-window:]
        return ids

    def _position_ids_from_list(self, positions: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(positions)], device=self.device, dtype=torch.long)

    def _range_positions(self, start: int, length: int) -> List[int]:
        return list(range(int(start), int(start) + int(length)))

    def _chunk_rope_start(self, prefix_len: int, chunk_order: int, chunk_index: int) -> int:
        mode = self.config.chunk_position_mode
        if mode == "reuse":
            return prefix_len
        if mode == "continuous":
            return prefix_len + chunk_order * max(1, self.config.chunk_size)
        if mode == "absolute":
            return prefix_len + chunk_index * max(1, self.config.chunk_size)
        raise ValueError(f"Unsupported chunk_position_mode: {mode}")

    def _chunk_query_rope_start(self, prefix_len: int, chunk_start: int, chunk_len: int) -> int:
        mode = self.config.chunk_query_position_mode
        if mode == "after_reused_window":
            return prefix_len + max(1, self.config.chunk_size)
        if mode == "after_chunk":
            return chunk_start + chunk_len
        raise ValueError(f"Unsupported chunk_query_position_mode: {mode}")

    def _final_query_rope_start(self, prefix_len: int, cache_positions: Sequence[int], selected_count: int) -> int:
        mode = self.config.query_position_mode
        if mode == "after_reused_window":
            return prefix_len + max(1, self.config.chunk_size)
        if mode == "after_selected_chunks":
            return prefix_len + selected_count * max(1, self.config.chunk_size)
        if mode == "after_cache":
            return (max(cache_positions) + 1) if cache_positions else prefix_len
        raise ValueError(f"Unsupported query_position_mode: {mode}")

    def _causal_attention_mask(self, q_len: int, key_len: Optional[int] = None) -> torch.Tensor:
        key_len = q_len if key_len is None else key_len
        mask = torch.full(
            (1, 1, q_len, key_len),
            -torch.inf,
            device=self.device,
            dtype=self._mask_dtype,
        )
        offset = key_len - q_len
        for row in range(q_len):
            mask[:, :, row, :offset + row + 1] = 0
        return mask

    def _query_to_chunk_mask(
        self,
        prefix_len: int,
        chunk_len: int,
        query_len: int,
        *,
        current_only: bool,
    ) -> torch.Tensor:
        key_len = prefix_len + chunk_len + query_len
        q_len = chunk_len + query_len if current_only else key_len
        row_offset = prefix_len if current_only else 0
        mask = torch.zeros((1, 1, q_len, key_len), device=self.device, dtype=self._mask_dtype)
        neg = torch.finfo(mask.dtype).min
        for local_row in range(q_len):
            row = row_offset + local_row
            if prefix_len <= row < prefix_len + chunk_len and query_len > 0:
                mask[:, :, local_row, prefix_len + chunk_len:] = neg
        return mask

    def _attention_mask(
        self,
        mode: str,
        *,
        q_len: int,
        key_len: Optional[int] = None,
        prefix_len: int = 0,
        chunk_len: int = 0,
        query_len: int = 0,
        current_only: bool = False,
    ):
        mode = (mode or "full").lower()
        if mode in {"full", "none"}:
            return "full"
        if mode == "causal":
            return self._causal_attention_mask(q_len, key_len)
        if mode == "query_to_chunk":
            return self._query_to_chunk_mask(
                prefix_len,
                chunk_len,
                query_len,
                current_only=current_only,
            )
        raise ValueError(f"Unsupported attention mask mode: {mode}")

    def _shift_logits(self, logits: torch.Tensor, last_logit: Optional[torch.Tensor] = None) -> torch.Tensor:
        shifted = torch.empty_like(logits)
        if last_logit is None:
            shifted[:, :1, :] = logits[:, :1, :]
        else:
            shifted[:, 0, :] = last_logit
        if logits.shape[1] > 1:
            shifted[:, 1:, :] = logits[:, :-1, :]
        return shifted

    def _align_generation_logits(
        self,
        logits: torch.Tensor,
        last_logit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_backend == "dream":
            return self._shift_logits(logits, last_logit)
        return logits

    def _num_key_value_heads(self) -> int:
        model_config = getattr(self.model, "config", None)
        num_heads = getattr(model_config, "effective_n_kv_heads", None)
        if num_heads is None:
            num_heads = getattr(model_config, "n_kv_heads", None)
        if num_heads is None:
            num_heads = getattr(model_config, "num_key_value_heads", None)
        if num_heads is None:
            num_heads = getattr(model_config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(model_config, "n_heads", None)
        return max(1, int(num_heads or 1))

    def _num_hidden_layers(self) -> int:
        model_config = getattr(self.model, "config", None)
        num_layers = getattr(model_config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(model_config, "n_layers", None)
        return max(1, int(num_layers or 1))

    def _cache_seq_dim(self, tensor: torch.Tensor) -> int:
        return 2 if tensor.dim() == 4 else 1

    def _cache_len(self, cache: Cache) -> int:
        if not cache:
            return 0
        key = cache[0][0]
        return int(key.shape[self._cache_seq_dim(key)])

    def _slice_cache(self, cache: Cache, start: int, end: int) -> Cache:
        if cache is None:
            return None
        sliced = []
        for key, value in cache:
            if self._cache_seq_dim(key) == 2:
                sliced.append((key[:, :, start:end, :].contiguous(), value[:, :, start:end, :].contiguous()))
            else:
                sliced.append((key[:, start:end, :].contiguous(), value[:, start:end, :].contiguous()))
        return sliced

    def _gather_cache(self, cache: Cache, keep_positions: Sequence[int]) -> Cache:
        if cache is None:
            return None
        keep = torch.tensor(list(keep_positions), device=self.device, dtype=torch.long)
        gathered = []
        for key, value in cache:
            dim = self._cache_seq_dim(key)
            gathered.append((key.index_select(dim, keep).contiguous(), value.index_select(dim, keep).contiguous()))
        return gathered

    def _gather_cache_per_layer_per_head(
        self,
        cache: Cache,
        keep_indices_per_layer_per_head: Sequence[torch.Tensor],
    ) -> Cache:
        if cache is None:
            return None
        gathered: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (key, value) in enumerate(cache):
            keep = keep_indices_per_layer_per_head[layer_idx]
            keep = keep.to(device=key.device, dtype=torch.long)
            num_heads, keep_count = keep.shape
            if self._cache_seq_dim(key) == 2:
                key_index = keep.unsqueeze(0).unsqueeze(-1).expand(key.shape[0], num_heads, keep_count, key.shape[-1])
                value_index = keep.unsqueeze(0).unsqueeze(-1).expand(
                    value.shape[0],
                    num_heads,
                    keep_count,
                    value.shape[-1],
                )
                gathered.append(
                    (
                        key.gather(2, key_index).contiguous(),
                        value.gather(2, value_index).contiguous(),
                    )
                )
                continue
            if key.shape[-1] % num_heads != 0 or value.shape[-1] % num_heads != 0:
                raise ValueError(
                    f"Cannot gather per-head cache: hidden dims {key.shape[-1]}/{value.shape[-1]} "
                    f"are not divisible by {num_heads} KV heads"
                )
            key_dim = key.shape[-1] // num_heads
            value_dim = value.shape[-1] // num_heads
            key_heads = key.view(key.shape[0], key.shape[1], num_heads, key_dim).transpose(1, 2)
            value_heads = value.view(value.shape[0], value.shape[1], num_heads, value_dim).transpose(1, 2)
            key_index = keep.unsqueeze(0).unsqueeze(-1).expand(key.shape[0], num_heads, keep_count, key_dim)
            value_index = keep.unsqueeze(0).unsqueeze(-1).expand(value.shape[0], num_heads, keep_count, value_dim)
            gathered_key = key_heads.gather(2, key_index).transpose(1, 2).contiguous().view(
                key.shape[0],
                keep_count,
                key.shape[-1],
            )
            gathered_value = value_heads.gather(2, value_index).transpose(1, 2).contiguous().view(
                value.shape[0],
                keep_count,
                value.shape[-1],
            )
            gathered.append((gathered_key, gathered_value))
        return gathered

    def _concat_caches(self, caches: Sequence[Cache]) -> Cache:
        parts = [cache for cache in caches if cache is not None and self._cache_len(cache) > 0]
        if not parts:
            return None
        num_layers = len(parts[0])
        combined: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            cat_dim = self._cache_seq_dim(parts[0][layer_idx][0])
            combined.append(
                (
                    torch.cat([cache[layer_idx][0] for cache in parts], dim=cat_dim).contiguous(),
                    torch.cat([cache[layer_idx][1] for cache in parts], dim=cat_dim).contiguous(),
                )
            )
        return combined

    def _prefill_plain(
        self,
        token_ids: Sequence[int],
        positions: Sequence[int],
        *,
        past_key_values: Cache = None,
        past_positions: Optional[Sequence[int]] = None,
        attention_mask="full",
    ):
        if not token_ids:
            return past_key_values, None
        full_positions = list(past_positions or []) + list(positions)
        outputs = self._model_forward(
            self._ids_tensor(token_ids),
            attention_mask=attention_mask,
            position_ids=self._position_ids_from_list(full_positions),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        last_logits = outputs.logits[:, -1, :].detach()
        return outputs.past_key_values, last_logits

    def _llada_leave_one_out_score(
        self,
        joint_ids: Sequence[int],
        label_positions: Sequence[int],
        label_ids: Sequence[int],
        attention_mask,
    ) -> float:
        if not label_positions:
            return float("-inf")
        mask_token_id = self._mask_token_id()
        batch_size = max(1, int(self.config.llada_score_batch_size or 1))
        nll_parts: List[torch.Tensor] = []
        positions = [int(pos) for pos in label_positions]
        labels = [int(token_id) for token_id in label_ids]
        for offset in range(0, len(positions), batch_size):
            batch_positions = positions[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            rows = [list(joint_ids) for _ in batch_positions]
            for row, pos in zip(rows, batch_positions):
                row[pos] = mask_token_id
            with torch.inference_mode():
                outputs = self._model_forward(
                    self._ids_batch_tensor(rows),
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            pos_tensor = torch.tensor(batch_positions, device=self.device, dtype=torch.long)
            row_tensor = torch.arange(len(batch_positions), device=self.device, dtype=torch.long)
            logits = outputs.logits[row_tensor, pos_tensor, :]
            label_tensor = torch.tensor(batch_labels, device=self.device, dtype=torch.long)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            nll_parts.append(-log_probs.gather(dim=-1, index=label_tensor.unsqueeze(-1)).squeeze(-1))
        if not nll_parts:
            return float("-inf")
        return float(-torch.cat(nll_parts, dim=0).mean().item())

    def _self_information_score_from_targets(
        self,
        *,
        joint_ids: Sequence[int],
        label_positions: Sequence[int],
        label_ids: Sequence[int],
        attention_mask,
    ) -> float:
        if not label_positions:
            return float("-inf")
        if self.model_backend == "llada":
            if self.config.score_llada_shift_logits:
                positions = torch.tensor(label_positions, device=self.device, dtype=torch.long)
                if int(positions.min().item()) <= 0:
                    return float("-inf")
                with torch.inference_mode():
                    outputs = self._model_forward(
                        self._ids_tensor(joint_ids),
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                logits = outputs.logits
                if int(positions.max().item()) - 1 >= logits.shape[1]:
                    return float("-inf")
                query_logits = logits.index_select(1, positions - 1)
                query_labels = self._ids_tensor(label_ids)
                if query_logits.shape[1] != query_labels.shape[1]:
                    return float("-inf")
                log_probs = F.log_softmax(query_logits.float(), dim=-1)
                token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
                return float(-token_nll.mean().item())
            return self._llada_leave_one_out_score(joint_ids, label_positions, label_ids, attention_mask)

        positions = torch.tensor(label_positions, device=self.device, dtype=torch.long)
        if int(positions.min().item()) <= 0:
            return float("-inf")
        with torch.inference_mode():
            outputs = self._model_forward(
                self._ids_tensor(joint_ids),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits
        if int(positions.max().item()) - 1 >= logits.shape[1]:
            return float("-inf")
        query_logits = logits.index_select(1, positions - 1)
        query_labels = self._ids_tensor(label_ids)
        if query_logits.shape[1] != query_labels.shape[1]:
            return float("-inf")
        log_probs = F.log_softmax(query_logits.float(), dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
        return float(-token_nll.mean().item())

    def score_chunk_self_information(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
        score_token_count: Optional[int] = None,
        score_token_mask: Optional[Sequence[bool]] = None,
    ) -> float:
        if score_token_count is None and score_token_mask is None:
            query_ids = self._window_query(query_ids, self.config.score_query_window)
        else:
            query_ids = list(query_ids)
        if not chunk_ids or not query_ids:
            return float("-inf")
        joint_ids = list(prefix_ids) + list(chunk_ids) + list(query_ids)
        prefix_len = len(prefix_ids)
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        attention_mask = self._attention_mask(
            self.config.score_attention_mask,
            q_len=len(joint_ids),
            key_len=len(joint_ids),
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            query_len=query_len,
            current_only=False,
        )
        if score_token_mask is not None:
            if len(score_token_mask) != query_len:
                return float("-inf")
            local_indices = [idx for idx, keep in enumerate(score_token_mask) if keep]
            if not local_indices:
                return float("-inf")
            label_positions = [prefix_len + chunk_len + idx for idx in local_indices]
            label_ids = [query_ids[idx] for idx in local_indices]
        elif score_token_count is None:
            window = min(query_len, self.config.score_query_window or query_len)
            start = prefix_len + chunk_len + query_len - window
            end = prefix_len + chunk_len + query_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        else:
            target_len = min(query_len, max(0, int(score_token_count)))
            if target_len <= 0:
                return float("-inf")
            start = prefix_len + chunk_len
            end = start + target_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        return self._self_information_score_from_targets(
            joint_ids=joint_ids,
            label_positions=label_positions,
            label_ids=label_ids,
            attention_mask=attention_mask,
        )

    def score_chunks_self_information_batched(
        self,
        prefix_ids: Sequence[int],
        candidate_chunks: Sequence[Sequence[int]],
        query_ids: Sequence[int],
        score_token_count: Optional[int] = None,
        score_token_mask: Optional[Sequence[bool]] = None,
    ) -> Dict[int, float]:
        if self.model_backend == "llada" and not self.config.score_llada_shift_logits:
            return {
                idx: self.score_chunk_self_information(
                    prefix_ids,
                    chunk_ids,
                    query_ids,
                    score_token_count=score_token_count,
                    score_token_mask=score_token_mask,
                )
                for idx, chunk_ids in enumerate(candidate_chunks)
            }

        if score_token_count is None and score_token_mask is None:
            query_ids = self._window_query(query_ids, self.config.score_query_window)
        else:
            query_ids = list(query_ids)
        prefix_ids = list(prefix_ids)
        if not query_ids:
            return {idx: float("-inf") for idx in range(len(candidate_chunks))}

        prefix_len = len(prefix_ids)
        query_len = len(query_ids)
        scores: Dict[int, float] = {}
        groups: Dict[int, List[Tuple[int, List[int]]]] = {}
        for idx, chunk_ids in enumerate(candidate_chunks):
            chunk_ids = list(chunk_ids)
            if not chunk_ids:
                scores[idx] = float("-inf")
                continue
            groups.setdefault(len(chunk_ids), []).append((idx, chunk_ids))

        batch_size = max(1, int(self.config.score_batch_size or 1))
        for chunk_len, group in groups.items():
            joint_len = prefix_len + chunk_len + query_len
            attention_mask = self._attention_mask(
                self.config.score_attention_mask,
                q_len=joint_len,
                key_len=joint_len,
                prefix_len=prefix_len,
                chunk_len=chunk_len,
                query_len=query_len,
                current_only=False,
            )
            if score_token_mask is not None:
                if len(score_token_mask) != query_len:
                    for idx, _ in group:
                        scores[idx] = float("-inf")
                    continue
                local_indices = [idx for idx, keep in enumerate(score_token_mask) if keep]
                if not local_indices:
                    for idx, _ in group:
                        scores[idx] = float("-inf")
                    continue
                label_positions = [prefix_len + chunk_len + idx for idx in local_indices]
                label_ids = [query_ids[idx] for idx in local_indices]
            elif score_token_count is None:
                window = min(query_len, self.config.score_query_window or query_len)
                start = prefix_len + chunk_len + query_len - window
                end = prefix_len + chunk_len + query_len
                label_positions = list(range(start, end))
                label_ids = query_ids[-window:]
            else:
                target_len = min(query_len, max(0, int(score_token_count)))
                if target_len <= 0:
                    for idx, _ in group:
                        scores[idx] = float("-inf")
                    continue
                start = prefix_len + chunk_len
                end = start + target_len
                label_positions = list(range(start, end))
                label_ids = query_ids[:target_len]

            positions = torch.tensor(label_positions, device=self.device, dtype=torch.long)
            if int(positions.min().item()) <= 0:
                for idx, _ in group:
                    scores[idx] = float("-inf")
                continue
            labels = torch.tensor(label_ids, device=self.device, dtype=torch.long)
            for offset in range(0, len(group), batch_size):
                batch = group[offset:offset + batch_size]
                rows = [prefix_ids + chunk_ids + list(query_ids) for _, chunk_ids in batch]
                with torch.inference_mode():
                    outputs = self._model_forward(
                        self._ids_batch_tensor(rows),
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                logits = outputs.logits
                if int(positions.max().item()) - 1 >= logits.shape[1]:
                    for idx, _ in batch:
                        scores[idx] = float("-inf")
                    continue
                query_logits = logits.index_select(1, positions - 1)
                log_probs = F.log_softmax(query_logits.float(), dim=-1)
                label_tensor = labels.view(1, -1, 1).expand(log_probs.shape[0], -1, 1)
                token_nll = -log_probs.gather(dim=-1, index=label_tensor).squeeze(-1)
                batch_scores = -token_nll.mean(dim=-1)
                for (idx, _), score in zip(batch, batch_scores):
                    value = float(score.item())
                    scores[idx] = value if math.isfinite(value) else float("-inf")
        return scores

    def score_chunk_self_information_joint_chunks(
        self,
        prefix_ids: Sequence[int],
        candidate_chunks: Sequence[Sequence[int]],
        target_index: int,
        query_ids: Sequence[int],
        score_token_count: Optional[int] = None,
        score_token_mask: Optional[Sequence[bool]] = None,
    ) -> float:
        """Score a target chunk while all chunks are present in the scoring pass.

        The target chunk is moved immediately before the query so the shifted
        logits used for the first query token are still anchored on the target
        chunk, matching the single-chunk score's boundary semantics.  Other
        chunks are included before it, allowing chunk representations to see
        each other when ``score_attention_mask=full``.
        """

        if score_token_count is None and score_token_mask is None:
            query_ids = self._window_query(query_ids, self.config.score_query_window)
        else:
            query_ids = list(query_ids)
        if not candidate_chunks or not query_ids:
            return float("-inf")
        if target_index < 0 or target_index >= len(candidate_chunks):
            return float("-inf")

        target_chunk = list(candidate_chunks[target_index])
        if not target_chunk:
            return float("-inf")

        context_ids: List[int] = []
        for idx, chunk_ids in enumerate(candidate_chunks):
            if idx != target_index:
                context_ids.extend(list(chunk_ids))
        context_ids.extend(target_chunk)
        if not context_ids:
            return float("-inf")

        joint_ids = list(prefix_ids) + context_ids + list(query_ids)
        prefix_len = len(prefix_ids)
        context_len = len(context_ids)
        query_len = len(query_ids)
        attention_mask = self._attention_mask(
            self.config.score_attention_mask,
            q_len=len(joint_ids),
            key_len=len(joint_ids),
            prefix_len=prefix_len,
            chunk_len=context_len,
            query_len=query_len,
            current_only=False,
        )
        if score_token_mask is not None:
            if len(score_token_mask) != query_len:
                return float("-inf")
            local_indices = [idx for idx, keep in enumerate(score_token_mask) if keep]
            if not local_indices:
                return float("-inf")
            label_positions = [prefix_len + context_len + idx for idx in local_indices]
            label_ids = [query_ids[idx] for idx in local_indices]
        elif score_token_count is None:
            window = min(query_len, self.config.score_query_window or query_len)
            start = prefix_len + context_len + query_len - window
            end = prefix_len + context_len + query_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        else:
            target_len = min(query_len, max(0, int(score_token_count)))
            if target_len <= 0:
                return float("-inf")
            start = prefix_len + context_len
            end = start + target_len
            label_positions = list(range(start, end))
            label_ids = joint_ids[start:end]
        return self._self_information_score_from_targets(
            joint_ids=joint_ids,
            label_positions=label_positions,
            label_ids=label_ids,
            attention_mask=attention_mask,
        )

    def score_chunk_next_block_logits(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
    ) -> float:
        draft_len = int(self.config.score_draft_tokens or self.config.block_length or 1)
        draft_len = max(1, draft_len)
        if not chunk_ids or not query_ids:
            return float("-inf")
        mask_token_id = self._mask_token_id()
        joint_ids = list(prefix_ids) + list(chunk_ids) + list(query_ids) + [mask_token_id] * draft_len
        with torch.inference_mode():
            outputs = self._model_forward(
                self._ids_tensor(joint_ids),
                attention_mask="full",
                use_cache=False,
                return_dict=True,
            )
        logits = self._align_generation_logits(outputs.logits)
        draft_logits = logits[:, -draft_len:, :]
        probs = torch.softmax(draft_logits.float(), dim=-1)
        confidence = probs.max(dim=-1).values
        return float(confidence.mean().item())

    def query_attention_token_scores(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
        *,
        query_window: int,
        layer_window: int,
        layer_mode: str,
        reduce_mode: str,
        pooling: Optional[str] = None,
        pool_kernel: Optional[int] = None,
        head_reduce: Optional[str] = None,
        layer_reduce: Optional[str] = None,
    ) -> torch.Tensor:
        query_ids = self._window_query(query_ids, query_window)
        if not chunk_ids or not query_ids:
            return torch.empty(0, device=self.device)

        score_prefix_ids = list(prefix_ids) if self.config.token_score_include_prefix else []
        prefix_len = len(score_prefix_ids)
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        joint_ids = score_prefix_ids + list(chunk_ids) + list(query_ids)
        attention_mask = self._attention_mask(
            self.config.token_attention_mask,
            q_len=len(joint_ids),
            key_len=len(joint_ids),
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            query_len=query_len,
            current_only=False,
        )
        if attention_mask == "full":
            attention_mask = None
        selected_indices = self._select_attention_layer_indices(
            self._num_hidden_layers(),
            layer_window,
            layer_mode,
        )
        if not selected_indices:
            return torch.empty(0, device=self.device)
        with torch.inference_mode():
            _, attentions = self._model_forward_with_attentions(
                self._ids_tensor(joint_ids),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                capture_layer_indices=selected_indices,
            )

        if not attentions:
            return torch.empty(0, device=self.device)

        selected_layers = [attentions[idx] for idx in selected_indices]
        layer_scores: List[torch.Tensor] = []
        query_start = prefix_len + chunk_len
        chunk_start = prefix_len
        pooling = (pooling or self.config.token_score_pooling or "none").lower()
        pool_kernel = int(pool_kernel or self.config.token_score_pool_kernel or 1)
        head_reduce = (head_reduce or self.config.token_score_head_reduce or "sum").lower()
        layer_reduce = (layer_reduce or self.config.token_score_layer_reduce or "mean").lower()
        for attn in selected_layers:
            if attn is None:
                continue
            head_scores = self._chunk_attention_head_scores(
                attn,
                query_start=query_start,
                query_len=query_len,
                chunk_start=chunk_start,
                chunk_len=chunk_len,
                reduce_mode=reduce_mode,
            )
            if head_scores.numel() == 0:
                continue

            if pooling != "none" and head_scores.shape[-1] > 0:
                kernel = max(1, min(pool_kernel, head_scores.shape[-1]))
                padding = kernel // 2
                pooled = head_scores.unsqueeze(1)
                if pooling == "avgpool":
                    pooled = F.avg_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                elif pooling == "maxpool":
                    pooled = F.max_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                else:
                    raise ValueError(f"Unsupported token_score_pooling: {pooling}")
                head_scores = pooled.squeeze(1)[..., :chunk_len]

            if head_reduce == "mean":
                layer_scores.append(head_scores.mean(dim=0))
            elif head_reduce == "sum":
                layer_scores.append(head_scores.sum(dim=0))
            elif head_reduce == "max":
                layer_scores.append(head_scores.max(dim=0).values)
            else:
                raise ValueError(f"Unsupported token_score_head_reduce: {head_reduce}")

        if not layer_scores:
            return torch.empty(0, device=self.device)
        stacked = torch.stack(layer_scores, dim=0)
        if layer_reduce == "mean":
            return stacked.mean(dim=0)
        if layer_reduce == "sum":
            return stacked.sum(dim=0)
        if layer_reduce == "max":
            return stacked.max(dim=0).values
        raise ValueError(f"Unsupported token_score_layer_reduce: {layer_reduce}")

    def _chunk_attention_head_scores(
        self,
        attn: torch.Tensor,
        *,
        query_start: int,
        query_len: int,
        chunk_start: int,
        chunk_len: int,
        reduce_mode: str,
    ) -> torch.Tensor:
        direction = self.config.token_score_direction
        parts: List[torch.Tensor] = []
        if direction in {"query_to_chunk", "bidirectional"}:
            query_to_chunk = attn[0, :, query_start:query_start + query_len, chunk_start:chunk_start + chunk_len].float()
            if query_to_chunk.numel() > 0:
                if reduce_mode == "mean":
                    parts.append(query_to_chunk.mean(dim=1))
                else:
                    parts.append(query_to_chunk.sum(dim=1))
        if direction in {"chunk_to_query", "bidirectional"}:
            chunk_to_query = attn[0, :, chunk_start:chunk_start + chunk_len, query_start:query_start + query_len].float()
            if chunk_to_query.numel() > 0:
                if reduce_mode == "mean":
                    parts.append(chunk_to_query.mean(dim=2))
                else:
                    parts.append(chunk_to_query.sum(dim=2))
        if not parts:
            return torch.empty(0, device=self.device)
        return torch.stack(parts, dim=0).sum(dim=0)

    def query_attention_token_scores_per_layer_per_head(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
        *,
        query_window: int,
        layer_window: int,
        layer_mode: str,
        reduce_mode: str,
        pooling: Optional[str] = None,
        pool_kernel: Optional[int] = None,
    ) -> List[torch.Tensor]:
        query_ids = self._window_query(query_ids, query_window)
        if not chunk_ids or not query_ids:
            return []

        score_prefix_ids = list(prefix_ids) if self.config.token_score_include_prefix else []
        prefix_len = len(score_prefix_ids)
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        joint_ids = score_prefix_ids + list(chunk_ids) + list(query_ids)
        attention_mask = self._attention_mask(
            self.config.token_attention_mask,
            q_len=len(joint_ids),
            key_len=len(joint_ids),
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            query_len=query_len,
            current_only=False,
        )
        if attention_mask == "full":
            attention_mask = None
        selected_indices = self._select_attention_layer_indices(
            self._num_hidden_layers(),
            layer_window,
            layer_mode,
        )
        if not selected_indices:
            return []
        with torch.inference_mode():
            _, attentions = self._model_forward_with_attentions(
                self._ids_tensor(joint_ids),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                capture_layer_indices=selected_indices,
            )

        if not attentions:
            return []

        num_kv_heads = self._num_key_value_heads()
        pooling = (pooling or self.config.token_score_pooling or "none").lower()
        pool_kernel = int(pool_kernel or self.config.token_score_pool_kernel or 1)
        query_start = prefix_len + chunk_len
        chunk_start = prefix_len
        scores_by_layer: List[Optional[torch.Tensor]] = [None for _ in attentions]
        for layer_idx in selected_indices:
            attn = attentions[layer_idx]
            if attn is None:
                continue
            query_head_scores = self._chunk_attention_head_scores(
                attn,
                query_start=query_start,
                query_len=query_len,
                chunk_start=chunk_start,
                chunk_len=chunk_len,
                reduce_mode=reduce_mode,
            )
            if query_head_scores.numel() == 0:
                continue

            if pooling != "none" and query_head_scores.shape[-1] > 0:
                kernel = max(1, min(pool_kernel, query_head_scores.shape[-1]))
                padding = kernel // 2
                pooled = query_head_scores.unsqueeze(1)
                if pooling == "avgpool":
                    pooled = F.avg_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                elif pooling == "maxpool":
                    pooled = F.max_pool1d(pooled, kernel_size=kernel, padding=padding, stride=1)
                else:
                    raise ValueError(f"Unsupported token_score_pooling: {pooling}")
                query_head_scores = pooled.squeeze(1)[..., :chunk_len]

            grouped_scores = []
            for head_group in torch.tensor_split(query_head_scores, num_kv_heads, dim=0):
                if head_group.shape[0] == 0:
                    grouped_scores.append(torch.zeros(chunk_len, device=self.device, dtype=query_head_scores.dtype))
                else:
                    grouped_scores.append(head_group.mean(dim=0))
            scores_by_layer[layer_idx] = torch.stack(grouped_scores, dim=0)

        fallback = next((score for score in scores_by_layer if score is not None), None)
        if fallback is None:
            return []
        return [
            score if score is not None else fallback.clone()
            for score in scores_by_layer
        ]

    def _select_attention_layer_indices(self, total_layers: int, layer_window: int, layer_mode: str) -> List[int]:
        if total_layers <= 0:
            return []
        if layer_mode == "all" or layer_window <= 0:
            return list(range(total_layers))
        window = min(max(1, int(layer_window)), total_layers)
        if layer_mode == "first":
            return list(range(window))
        if layer_mode == "last":
            return list(range(total_layers - window, total_layers))
        raise ValueError(f"Unsupported attention layer mode: {layer_mode}")

    def _select_attention_layers(self, attentions, layer_window: int, layer_mode: str):
        return [attentions[idx] for idx in self._select_attention_layer_indices(len(attentions), layer_window, layer_mode)]

    def _selection_query_ids(
        self,
        prefix_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
    ) -> List[int]:
        selection_query_ids, _, _ = self._selection_query_ids_with_score_target(prefix_ids, scoring_query_ids)
        return selection_query_ids

    def _selection_query_ids_with_score_target(
        self,
        prefix_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
    ) -> Tuple[List[int], Optional[int], Optional[List[bool]]]:
        selection_query_ids = list(scoring_query_ids)
        score_token_count: Optional[int] = None
        score_token_mask: Optional[List[bool]] = None
        if self.config.score_mode == "draft_self_information" and self.config.score_draft_tokens > 0:
            if self.config.score_draft_partial_rounds is not None:
                draft_ids, draft_mask = self.generate_partial_draft_rounds_from_cache(
                    past_key_values=None,
                    cache_positions=[],
                    query_ids=list(prefix_ids) + list(scoring_query_ids),
                    query_rope_start=0,
                    max_new_tokens=int(self.config.score_draft_tokens),
                    partial_rounds=int(self.config.score_draft_partial_rounds),
                )
                if self.config.score_draft_score_all_slots:
                    score_token_mask = [True] * len(scoring_query_ids) + [True] * len(draft_ids)
                else:
                    score_token_mask = [True] * len(scoring_query_ids) + list(draft_mask)
            elif self.config.score_draft_partial_steps is not None:
                draft_ids, filled_count = self.generate_partial_draft_from_cache(
                    past_key_values=None,
                    cache_positions=[],
                    query_ids=list(prefix_ids) + list(scoring_query_ids),
                    query_rope_start=0,
                    max_new_tokens=int(self.config.score_draft_tokens),
                    partial_steps=int(self.config.score_draft_partial_steps),
                )
                mask_token_id = self._mask_token_id()
                score_token_mask = [True] * len(scoring_query_ids) + [
                    token_id != mask_token_id for token_id in draft_ids
                ]
            else:
                draft_ids = self.generate_from_cache(
                    past_key_values=None,
                    cache_positions=[],
                    query_ids=list(prefix_ids) + list(scoring_query_ids),
                    query_rope_start=0,
                    max_new_tokens=int(self.config.score_draft_tokens),
                    diffusion_steps=int(self.config.score_draft_steps or 1),
                    force_diffusion=bool(self.config.score_draft_fixed_steps),
                )
            selection_query_ids = list(scoring_query_ids) + draft_ids
        return selection_query_ids, score_token_count, score_token_mask

    def _token_eviction_query_ids(
        self,
        scoring_query_ids: Sequence[int],
        selection_query_ids: Sequence[int],
        score_token_mask: Optional[Sequence[bool]],
    ) -> List[int]:
        if not self.config.token_score_use_generated:
            return list(scoring_query_ids)
        if score_token_mask is not None and len(score_token_mask) == len(selection_query_ids):
            return [
                int(token_id)
                for token_id, keep in zip(selection_query_ids, score_token_mask)
                if keep
            ]
        return list(selection_query_ids)

    def select_chunks(
        self,
        prefix_ids: Sequence[int],
        candidate_chunks: Sequence[Sequence[int]],
        scoring_query_ids: Sequence[int],
    ) -> Tuple[List[int], Dict[int, float], List[int], Optional[List[bool]]]:
        if not candidate_chunks:
            return [], {}, list(scoring_query_ids), None

        selection_query_ids, score_token_count, score_token_mask = self._selection_query_ids_with_score_target(
            prefix_ids,
            scoring_query_ids,
        )
        if self.config.score_mode == "none" or not selection_query_ids:
            selected = list(range(len(candidate_chunks)))
            if self.config.topk_chunks > 0:
                selected = selected[:self.config.topk_chunks]
            return selected, {}, selection_query_ids, score_token_mask

        scores: Dict[int, float] = {}
        if (
            self.config.score_mode in {"self_information", "draft_self_information"}
            and self.config.score_context_mode == "single_chunk"
            and int(self.config.score_batch_size or 1) > 1
        ):
            scores.update(
                self.score_chunks_self_information_batched(
                    prefix_ids,
                    candidate_chunks,
                    selection_query_ids,
                    score_token_count=score_token_count,
                    score_token_mask=score_token_mask,
                )
            )
        for idx, chunk_ids in enumerate(candidate_chunks):
            if idx in scores:
                continue
            if self.config.score_mode in {"self_information", "draft_self_information"}:
                if self.config.score_context_mode == "single_chunk":
                    score = self.score_chunk_self_information(
                        prefix_ids,
                        chunk_ids,
                        selection_query_ids,
                        score_token_count=score_token_count,
                        score_token_mask=score_token_mask,
                    )
                elif self.config.score_context_mode == "joint_chunks_target_last":
                    score = self.score_chunk_self_information_joint_chunks(
                        prefix_ids,
                        candidate_chunks,
                        idx,
                        selection_query_ids,
                        score_token_count=score_token_count,
                        score_token_mask=score_token_mask,
                    )
                else:
                    raise ValueError(f"Unsupported score_context_mode: {self.config.score_context_mode}")
            elif self.config.score_mode == "per_chunk_draft_self_information":
                per_chunk_query_ids = list(scoring_query_ids)
                if self.config.score_draft_tokens > 0:
                    draft_ids = self.generate_from_cache(
                        past_key_values=None,
                        cache_positions=[],
                        query_ids=list(prefix_ids) + list(chunk_ids) + list(scoring_query_ids),
                        query_rope_start=0,
                        max_new_tokens=int(self.config.score_draft_tokens),
                        diffusion_steps=int(self.config.score_draft_steps or 1),
                        force_diffusion=bool(self.config.score_draft_fixed_steps),
                    )
                    per_chunk_query_ids = list(scoring_query_ids) + draft_ids
                score = self.score_chunk_self_information(prefix_ids, chunk_ids, per_chunk_query_ids)
            elif self.config.score_mode == "next_block_logits":
                score = self.score_chunk_next_block_logits(prefix_ids, chunk_ids, scoring_query_ids)
            elif self.config.score_mode == "attention":
                token_scores = self.query_attention_token_scores(
                    prefix_ids,
                    chunk_ids,
                    selection_query_ids,
                    query_window=self.config.attention_query_window,
                    layer_window=self.config.attention_score_layers,
                    layer_mode="last",
                    reduce_mode=self.config.token_score_reduce,
                    pooling=self.config.token_score_pooling,
                    pool_kernel=self.config.token_score_pool_kernel,
                    head_reduce=self.config.token_score_head_reduce,
                    layer_reduce=self.config.token_score_layer_reduce,
                )
                score = float(token_scores.mean().item()) if token_scores.numel() else float("-inf")
            else:
                raise ValueError(f"Unsupported score_mode: {self.config.score_mode}")
            scores[idx] = score if math.isfinite(score) else float("-inf")

        forced = [0] if self.config.keep_first_chunk and candidate_chunks else []
        remaining = [idx for idx in range(len(candidate_chunks)) if idx not in forced]
        ranked = sorted(remaining, key=lambda idx: scores.get(idx, float("-inf")), reverse=True)
        topk = len(ranked) if self.config.topk_chunks <= 0 else min(self.config.topk_chunks, len(ranked))
        return sorted(set(forced + ranked[:topk])), scores, selection_query_ids, score_token_mask

    def _keep_positions_for_chunk(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
    ) -> List[int]:
        capacity = int(self.config.token_capacity or 0)
        chunk_len = len(chunk_ids)
        if capacity <= 0 or chunk_len <= capacity:
            return list(range(chunk_len))

        token_scores = self.query_attention_token_scores(
            prefix_ids,
            chunk_ids,
            query_ids,
            query_window=self.config.token_score_query_window,
            layer_window=self.config.token_score_layers,
            layer_mode=self.config.token_score_layer_mode,
            reduce_mode=self.config.token_score_reduce,
            pooling=self.config.token_score_pooling,
            pool_kernel=self.config.token_score_pool_kernel,
            head_reduce=self.config.token_score_head_reduce,
            layer_reduce=self.config.token_score_layer_reduce,
        )
        if token_scores.numel() != chunk_len:
            head = capacity // 2
            tail = capacity - head
            keep = sorted(set(list(range(head)) + list(range(chunk_len - tail, chunk_len))))
        else:
            keep = self._select_positions_from_token_scores(
                token_scores,
                capacity=capacity,
                chunk_len=chunk_len,
            ).tolist()

        if self.config.force_keep_chunk_bos and self.config.chunk_bos and chunk_len > 0:
            keep = sorted(set([0] + keep))
            if len(keep) > capacity:
                keep = [0] + [idx for idx in keep if idx != 0][-max(0, capacity - 1):]
                keep = sorted(keep)
        return keep

    def _select_positions_from_token_scores(
        self,
        token_scores: torch.Tensor,
        *,
        capacity: int,
        chunk_len: int,
    ) -> torch.Tensor:
        keep_count = min(max(1, int(capacity)), chunk_len)
        if token_scores.numel() != chunk_len:
            head = keep_count // 2
            tail = keep_count - head
            keep = sorted(set(list(range(head)) + list(range(chunk_len - tail, chunk_len))))
            return torch.tensor(keep[:keep_count], device=self.device, dtype=torch.long)

        largest = self.config.token_score_keep == "high"
        if self.config.force_keep_chunk_bos and self.config.chunk_bos and chunk_len > 0:
            if keep_count == 1:
                return torch.zeros(1, device=self.device, dtype=torch.long)
            candidate_indices = torch.arange(1, chunk_len, device=self.device, dtype=torch.long)
            if candidate_indices.numel() <= keep_count - 1:
                selected = candidate_indices
            else:
                candidate_scores = token_scores.index_select(0, candidate_indices)
                selected = candidate_indices[torch.topk(candidate_scores, k=keep_count - 1, largest=largest).indices]
            return torch.sort(torch.cat([torch.zeros(1, device=self.device, dtype=torch.long), selected], dim=0)).values

        return torch.topk(token_scores, k=keep_count, largest=largest).indices.sort().values

    def _keep_positions_per_layer_per_head_for_chunk(
        self,
        prefix_ids: Sequence[int],
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
    ) -> List[torch.Tensor]:
        capacity = int(self.config.token_capacity or 0)
        chunk_len = len(chunk_ids)
        num_layers = self._num_hidden_layers()
        num_heads = self._num_key_value_heads()
        if chunk_len <= 0:
            return []
        if capacity <= 0 or chunk_len <= capacity:
            base = torch.arange(chunk_len, device=self.device, dtype=torch.long)
            return [base.unsqueeze(0).expand(num_heads, -1).clone() for _ in range(num_layers)]

        per_layer_scores = self.query_attention_token_scores_per_layer_per_head(
            prefix_ids,
            chunk_ids,
            query_ids,
            query_window=self.config.token_score_query_window,
            layer_window=self.config.token_score_layers,
            layer_mode=self.config.token_score_layer_mode,
            reduce_mode=self.config.token_score_reduce,
            pooling=self.config.token_score_pooling,
            pool_kernel=self.config.token_score_pool_kernel,
        )
        keep_count = min(capacity, chunk_len)
        if not per_layer_scores:
            fallback = self._select_positions_from_token_scores(
                torch.empty(0, device=self.device),
                capacity=keep_count,
                chunk_len=chunk_len,
            )
            return [fallback.unsqueeze(0).expand(num_heads, -1).clone() for _ in range(num_layers)]

        per_layer_keep: List[torch.Tensor] = []
        for layer_scores in per_layer_scores:
            if layer_scores.numel() == 0 or layer_scores.shape[-1] != chunk_len:
                fallback = self._select_positions_from_token_scores(
                    torch.empty(0, device=self.device),
                    capacity=keep_count,
                    chunk_len=chunk_len,
                )
                per_layer_keep.append(fallback.unsqueeze(0).expand(num_heads, -1).clone())
                continue
            head_keeps = [
                self._select_positions_from_token_scores(
                    layer_scores[head_idx],
                    capacity=keep_count,
                    chunk_len=chunk_len,
                )
                for head_idx in range(layer_scores.shape[0])
            ]
            per_layer_keep.append(torch.stack(head_keeps, dim=0))

        if len(per_layer_keep) < num_layers:
            fallback = per_layer_keep[-1]
            per_layer_keep.extend(fallback.clone() for _ in range(num_layers - len(per_layer_keep)))
        return per_layer_keep[:num_layers]

    def _union_keep_positions_per_layer_per_head(self, per_layer_keep: Sequence[torch.Tensor]) -> List[int]:
        if not per_layer_keep:
            return []
        flat_parts = [keep.reshape(-1) for keep in per_layer_keep if keep.numel() > 0]
        if not flat_parts:
            return []
        return torch.sort(torch.unique(torch.cat(flat_parts, dim=0))).values.tolist()

    def _full_prompt_keep_indices_per_layer_per_head(
        self,
        *,
        seq_len: int,
        chunk_spans: Sequence[Tuple[int, int, List[torch.Tensor]]],
    ) -> List[torch.Tensor]:
        num_layers = self._num_hidden_layers()
        if not chunk_spans:
            base = torch.arange(seq_len, device=self.device, dtype=torch.long)
            return [
                base.unsqueeze(0).expand(self._num_key_value_heads(), -1).clone()
                for _ in range(num_layers)
            ]

        per_layer_global: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            layer_template = chunk_spans[0][2][min(layer_idx, len(chunk_spans[0][2]) - 1)]
            num_heads = int(layer_template.shape[0])
            head_global_indices = []
            for head_idx in range(num_heads):
                pieces = []
                cursor = 0
                for start, end, per_layer_keep in chunk_spans:
                    if start > cursor:
                        pieces.append(torch.arange(cursor, start, device=self.device, dtype=torch.long))
                    layer_keep = per_layer_keep[min(layer_idx, len(per_layer_keep) - 1)]
                    local_keep = layer_keep[min(head_idx, layer_keep.shape[0] - 1)]
                    pieces.append(local_keep.to(device=self.device, dtype=torch.long) + int(start))
                    cursor = int(end)
                if cursor < seq_len:
                    pieces.append(torch.arange(cursor, seq_len, device=self.device, dtype=torch.long))
                head_global_indices.append(torch.cat(pieces, dim=0))
            per_layer_global.append(torch.stack(head_global_indices, dim=0))
        return per_layer_global

    def _build_chunk_cache(
        self,
        *,
        prefix_cache: Cache,
        prefix_ids: Sequence[int],
        prefix_positions: Sequence[int],
        chunk_ids: Sequence[int],
        chunk_positions: Sequence[int],
        query_ids: Sequence[int],
        query_positions: Sequence[int],
    ) -> Cache:
        if self.config.cache_build_mode == "chunk_only":
            if not chunk_ids:
                return None
            outputs = self._model_forward(
                self._ids_tensor(chunk_ids),
                attention_mask="full",
                position_ids=self._position_ids_from_list(chunk_positions),
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            return outputs.past_key_values

        if self.config.cache_build_mode != "chunk_query":
            raise ValueError(f"_build_chunk_cache does not support cache_build_mode={self.config.cache_build_mode}")

        input_ids = list(chunk_ids) + list(query_ids)
        if not input_ids:
            return None
        full_positions = list(prefix_positions) + list(chunk_positions) + list(query_positions)
        attention_mask = self._attention_mask(
            self.config.token_attention_mask,
            q_len=len(input_ids),
            key_len=len(full_positions),
            prefix_len=len(prefix_ids),
            chunk_len=len(chunk_ids),
            query_len=len(query_ids),
            current_only=True,
        )
        outputs = self._model_forward(
            self._ids_tensor(input_ids),
            attention_mask=attention_mask,
            position_ids=self._position_ids_from_list(full_positions),
            past_key_values=prefix_cache,
            use_cache=True,
            return_dict=True,
        )
        chunk_start = len(prefix_ids)
        chunk_end = chunk_start + len(chunk_ids)
        return self._slice_cache(outputs.past_key_values, chunk_start, chunk_end)

    def _prepare_candidate_chunks(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
    ):
        candidate_chunks = split_token_chunks(
            context_ids,
            self.config.chunk_size,
            split_from_tail=self.config.split_from_tail,
        )
        candidate_chunks = [self._maybe_prepend_bos_to_chunk(chunk_ids) for chunk_ids in candidate_chunks]
        selected_indices, chunk_scores, selection_query_ids, score_token_mask = self.select_chunks(
            prefix_ids,
            candidate_chunks,
            scoring_query_ids,
        )
        return candidate_chunks, selected_indices, chunk_scores, selection_query_ids, score_token_mask

    def build_prefill_cache(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        scoring_query_ids: Optional[Sequence[int]] = None,
    ):
        prefix_ids = list(prefix_ids)
        context_ids = list(context_ids)
        query_ids = list(query_ids)
        scoring_query_ids = list(scoring_query_ids if scoring_query_ids is not None else query_ids)

        if self.config.cache_build_mode in {"full_prompt_mask", "full_prompt_query"}:
            raise ValueError(f"{self.config.cache_build_mode} uses generate_full_prompt_mask instead of build_prefill_cache")

        candidate_chunks, selected_indices, chunk_scores, selection_query_ids, score_token_mask = self._prepare_candidate_chunks(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            scoring_query_ids=scoring_query_ids,
        )
        eviction_query_ids = self._token_eviction_query_ids(
            scoring_query_ids=scoring_query_ids,
            selection_query_ids=selection_query_ids,
            score_token_mask=score_token_mask,
        )

        prefix_positions = self._range_positions(0, len(prefix_ids))
        prefix_cache: Cache = None
        if prefix_ids:
            prefix_cache, _ = self._prefill_plain(prefix_ids, prefix_positions, attention_mask="full")

        cache_parts: List[Cache] = []
        cache_positions: List[int] = []
        cache_len = 0
        if prefix_cache is not None:
            cache_parts.append(prefix_cache)
            cache_positions.extend(prefix_positions)
            cache_len += len(prefix_ids)

        chunk_meta: List[FastDLLMChunkMeta] = []
        total_removed = 0
        for chunk_order, chunk_index in enumerate(selected_indices):
            chunk_ids = list(candidate_chunks[chunk_index])
            chunk_start = self._chunk_rope_start(len(prefix_ids), chunk_order, chunk_index)
            chunk_positions = self._range_positions(chunk_start, len(chunk_ids))
            tmp_query_start = self._chunk_query_rope_start(len(prefix_ids), chunk_start, len(chunk_ids))
            tmp_query_ids = self._window_query(selection_query_ids, self.config.token_score_query_window)
            tmp_query_positions = self._range_positions(tmp_query_start, len(tmp_query_ids))

            per_layer_per_head_keep: Optional[List[torch.Tensor]] = None
            if self.config.token_eviction_granularity == "per_head":
                per_layer_per_head_keep = self._keep_positions_per_layer_per_head_for_chunk(
                    prefix_ids,
                    chunk_ids,
                    eviction_query_ids,
                )
                keep_positions = self._union_keep_positions_per_layer_per_head(per_layer_per_head_keep)
                kept_count = (
                    int(per_layer_per_head_keep[0].shape[1])
                    if per_layer_per_head_keep
                    else len(chunk_ids)
                )
            else:
                keep_positions = self._keep_positions_for_chunk(prefix_ids, chunk_ids, eviction_query_ids)
                kept_count = len(keep_positions)
            chunk_cache = self._build_chunk_cache(
                prefix_cache=prefix_cache,
                prefix_ids=prefix_ids,
                prefix_positions=prefix_positions,
                chunk_ids=chunk_ids,
                chunk_positions=chunk_positions,
                query_ids=tmp_query_ids,
                query_positions=tmp_query_positions,
            )
            if (
                self.config.token_eviction_granularity == "per_head"
                and per_layer_per_head_keep
                and kept_count < len(chunk_ids)
            ):
                chunk_cache = self._gather_cache_per_layer_per_head(chunk_cache, per_layer_per_head_keep)
            elif keep_positions and len(keep_positions) < len(chunk_ids):
                chunk_cache = self._gather_cache(chunk_cache, keep_positions)
            kept_positions_abs = (
                chunk_positions[:kept_count]
                if self.config.token_eviction_granularity == "per_head"
                else [chunk_positions[idx] for idx in keep_positions]
            )
            removed = max(0, len(chunk_ids) - kept_count)
            cache_start = cache_len
            if chunk_cache is not None and kept_count > 0:
                cache_parts.append(chunk_cache)
                cache_positions.extend(kept_positions_abs)
                cache_len += kept_count
            total_removed += removed
            chunk_meta.append(
                FastDLLMChunkMeta(
                    chunk_index=chunk_index,
                    original_tokens=len(chunk_ids),
                    kept_tokens=kept_count,
                    removed_tokens=removed,
                    cache_start=cache_start,
                    cache_end=cache_len,
                    rope_start=chunk_start,
                    rope_end=chunk_start + len(chunk_ids),
                    score=chunk_scores.get(chunk_index),
                    kept_positions=list(keep_positions),
                )
            )

        past_key_values = self._concat_caches(cache_parts)
        query_rope_start = self._final_query_rope_start(
            len(prefix_ids),
            cache_positions,
            selected_count=len(selected_indices),
        )
        return past_key_values, cache_positions, {
            "selected_indices": selected_indices,
            "chunk_scores": chunk_scores,
            "cache_build_mode": self.config.cache_build_mode,
            "prefix_tokens": len(prefix_ids),
            "query_rope_start": query_rope_start,
            "raw_context_tokens": len(context_ids),
            "candidate_chunks": len(candidate_chunks),
            "cache_tokens": cache_len,
            "removed_tokens": total_removed,
            "chunk_meta": chunk_meta,
        }

    def _forward_generation_block(
        self,
        block_ids: torch.Tensor,
        block_positions: Sequence[int],
        *,
        past_key_values: Cache,
        cache_positions: Sequence[int],
    ) -> torch.Tensor:
        outputs = self._model_forward(
            block_ids,
            attention_mask="full",
            position_ids=self._position_ids_from_list(list(cache_positions) + list(block_positions)),
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits

    def _forward_generation_block_replace(
        self,
        block_ids: torch.Tensor,
        *,
        full_cache: Cache,
        full_positions: Sequence[int],
        replace_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, Cache]:
        outputs = self._model_forward(
            block_ids,
            attention_mask="full",
            position_ids=self._position_ids_from_list(full_positions),
            past_key_values=full_cache,
            use_cache=True,
            return_dict=True,
            dual_cache=True,
            replace_position=replace_position,
        )
        return outputs.logits, outputs.past_key_values

    def _initialize_fastdllm_suffix_cache(
        self,
        *,
        active_cache: Cache,
        active_positions: Sequence[int],
        suffix_len: int,
        suffix_pos_start: int,
        last_context_logit: Optional[torch.Tensor],
    ) -> Tuple[Cache, List[int], torch.Tensor, torch.Tensor]:
        mask_token_id = self._mask_token_id()
        suffix_ids = [mask_token_id] * int(suffix_len)
        suffix_positions = self._range_positions(suffix_pos_start, suffix_len)
        full_positions = list(active_positions) + suffix_positions
        outputs = self._model_forward(
            self._ids_tensor(suffix_ids),
            attention_mask="full",
            position_ids=self._position_ids_from_list(full_positions),
            past_key_values=active_cache,
            use_cache=True,
            return_dict=True,
        )
        shifted_logits = self._align_generation_logits(outputs.logits, last_context_logit)
        first_logits = shifted_logits[:, :1, :].reshape(-1, shifted_logits.shape[-1])
        _, first_token = sample_tokens(
            first_logits,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        return outputs.past_key_values, full_positions, shifted_logits, first_token

    def _denoise_block_confidence_threshold(
        self,
        block_ids: torch.Tensor,
        *,
        full_cache: Cache,
        full_positions: Sequence[int],
        replace_position: torch.Tensor,
        last_context_logit: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Cache]:
        mask_token_id = self._mask_token_id()
        guard = 0
        while (block_ids == mask_token_id).any():
            guard += 1
            if guard > block_ids.shape[1] + 8:
                break
            mask_index = block_ids == mask_token_id
            logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_positions,
                replace_position=replace_position,
            )
            shifted_logits = self._align_generation_logits(logits, last_context_logit)
            mask_logits = shifted_logits[mask_index]
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            candidate = torch.full_like(block_ids, mask_token_id, device=self.device)
            candidate[mask_index] = x0.clone()
            full_confidence = torch.full_like(block_ids, -torch.inf, device=self.device, dtype=shifted_logits.dtype)
            full_confidence[mask_index] = confidence
            transfer_count = int(mask_index.sum().item())
            selected_confidence, select_index = torch.topk(full_confidence, transfer_count)
            transfer_index = torch.zeros_like(block_ids, dtype=torch.bool, device=self.device)
            transfer_index[0, select_index[0, 0]] = True
            for idx in range(1, transfer_count):
                if selected_confidence[0, idx] >= self.config.threshold:
                    transfer_index[0, select_index[0, idx]] = True
            block_ids[transfer_index] = candidate[transfer_index]
        return block_ids, full_cache

    def _denoise_block_diffusion(
        self,
        block_ids: torch.Tensor,
        *,
        full_cache: Cache,
        full_positions: Sequence[int],
        replace_position: torch.Tensor,
        last_context_logit: Optional[torch.Tensor],
        steps_per_block: int,
        confidence_alg: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Cache]:
        mask_token_id = self._mask_token_id()
        timesteps = torch.linspace(1, 1e-3, steps_per_block + 1, device=self.device)
        alg = confidence_alg or self.config.alg
        for step in range(steps_per_block):
            mask_index = block_ids == mask_token_id
            if not mask_index.any():
                break
            logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_positions,
                replace_position=replace_position,
            )
            shifted_logits = self._align_generation_logits(logits, last_context_logit)
            mask_logits = shifted_logits[mask_index]
            if alg in {"origin", "confidence_threshold"}:
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                )
            elif alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    margin_confidence=True,
                )
            else:
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    neg_entropy=True,
                )

            t = timesteps[step]
            s = timesteps[step + 1]
            mask_count = int(mask_index.sum().item())
            transfer_count = int(mask_count * (1 - s / t)) if step < steps_per_block - 1 else mask_count
            if transfer_count <= 0:
                continue
            candidate = torch.full_like(block_ids, mask_token_id, device=self.device)
            candidate[mask_index] = x0.clone()
            full_confidence = torch.full_like(block_ids, -torch.inf, device=self.device, dtype=shifted_logits.dtype)
            full_confidence[mask_index] = confidence
            if self.config.alg_temp is None or self.config.alg_temp == 0:
                _, transfer_index_ids = torch.topk(full_confidence, transfer_count)
            else:
                scaled = F.softmax(full_confidence / self.config.alg_temp, dim=-1)
                transfer_index_ids = torch.multinomial(scaled, num_samples=transfer_count)
            row_indices = torch.arange(block_ids.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index_ids)
            block_ids[row_indices, transfer_index_ids] = candidate[row_indices, transfer_index_ids]
        return block_ids, full_cache

    def generate_partial_draft_from_cache(
        self,
        *,
        past_key_values: Cache,
        cache_positions: Sequence[int],
        query_ids: Sequence[int],
        query_rope_start: int,
        max_new_tokens: int,
        partial_steps: int,
    ) -> Tuple[List[int], int]:
        mask_token_id = self._mask_token_id()
        query_ids = list(query_ids)
        cache_positions = list(cache_positions)
        draft_len = int(max_new_tokens)
        if draft_len <= 0:
            return [], 0

        query_positions = self._range_positions(query_rope_start, len(query_ids))
        active_cache = past_key_values
        active_positions = list(cache_positions)
        last_context_logit: Optional[torch.Tensor] = None
        if query_ids:
            active_cache, last_context_logit = self._prefill_plain(
                query_ids,
                query_positions,
                past_key_values=active_cache,
                past_positions=active_positions,
                attention_mask="full",
            )
            active_positions.extend(query_positions)

        block_pos_start = query_rope_start + len(query_ids)
        full_cache, full_positions, _, first_token = self._initialize_fastdllm_suffix_cache(
            active_cache=active_cache,
            active_positions=active_positions,
            suffix_len=draft_len,
            suffix_pos_start=block_pos_start,
            last_context_logit=last_context_logit,
        )
        current_slot_start = len(active_positions)
        current_slot_end = current_slot_start + draft_len
        replace_position = torch.zeros(
            (1, len(full_positions)),
            device=self.device,
            dtype=torch.bool,
        )
        replace_position[:, current_slot_start:current_slot_end] = True

        block_ids = torch.full((1, draft_len), mask_token_id, device=self.device, dtype=torch.long)
        filled_target = min(draft_len, max(0, int(partial_steps)))
        filled_count = 0
        if filled_target > 0:
            block_ids[:, 0] = first_token[:1]
            filled_count = 1

        while filled_count < filled_target and (block_ids == mask_token_id).any():
            mask_index = block_ids == mask_token_id
            logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_positions,
                replace_position=replace_position,
            )
            shifted_logits = self._align_generation_logits(logits, last_context_logit)
            mask_logits = shifted_logits[mask_index]
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            candidate = torch.full_like(block_ids, mask_token_id, device=self.device)
            candidate[mask_index] = x0.clone()
            full_confidence = torch.full_like(block_ids, -torch.inf, device=self.device, dtype=shifted_logits.dtype)
            full_confidence[mask_index] = confidence
            transfer_count = min(filled_target - filled_count, int(mask_index.sum().item()))
            _, transfer_index_ids = torch.topk(full_confidence, transfer_count)
            row_indices = torch.arange(block_ids.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index_ids)
            block_ids[row_indices, transfer_index_ids] = candidate[row_indices, transfer_index_ids]
            filled_count += int(transfer_count)

        return block_ids[0].tolist(), filled_count

    def generate_partial_draft_rounds_from_cache(
        self,
        *,
        past_key_values: Cache,
        cache_positions: Sequence[int],
        query_ids: Sequence[int],
        query_rope_start: int,
        max_new_tokens: int,
        partial_rounds: int,
    ) -> Tuple[List[int], List[bool]]:
        mask_token_id = self._mask_token_id()
        query_ids = list(query_ids)
        cache_positions = list(cache_positions)
        draft_len = int(max_new_tokens)
        if draft_len <= 0:
            return [], []

        query_positions = self._range_positions(query_rope_start, len(query_ids))
        active_cache = past_key_values
        active_positions = list(cache_positions)
        last_context_logit: Optional[torch.Tensor] = None
        if query_ids:
            active_cache, last_context_logit = self._prefill_plain(
                query_ids,
                query_positions,
                past_key_values=active_cache,
                past_positions=active_positions,
                attention_mask="full",
            )
            active_positions.extend(query_positions)

        block_pos_start = query_rope_start + len(query_ids)
        full_cache, full_positions, _, first_token = self._initialize_fastdllm_suffix_cache(
            active_cache=active_cache,
            active_positions=active_positions,
            suffix_len=draft_len,
            suffix_pos_start=block_pos_start,
            last_context_logit=last_context_logit,
        )
        current_slot_start = len(active_positions)
        current_slot_end = current_slot_start + draft_len
        replace_position = torch.zeros(
            (1, len(full_positions)),
            device=self.device,
            dtype=torch.bool,
        )
        replace_position[:, current_slot_start:current_slot_end] = True

        block_ids = torch.full((1, draft_len), mask_token_id, device=self.device, dtype=torch.long)
        confirmed_mask = [False] * draft_len
        block_ids[:, 0] = first_token[:1]
        confirmed_mask[0] = True

        rounds = max(0, int(partial_rounds))
        for _ in range(rounds):
            mask_index = block_ids == mask_token_id
            if not mask_index.any():
                break
            logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_positions,
                replace_position=replace_position,
            )
            shifted_logits = self._align_generation_logits(logits, last_context_logit)
            mask_logits = shifted_logits[mask_index]
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            candidate = torch.full_like(block_ids, mask_token_id, device=self.device)
            candidate[mask_index] = x0.clone()
            full_confidence = torch.full_like(block_ids, -torch.inf, device=self.device, dtype=shifted_logits.dtype)
            full_confidence[mask_index] = confidence
            _, transfer_index_ids = torch.topk(full_confidence, 1)
            selected_pos = int(transfer_index_ids[0, 0].item())
            block_ids[0, selected_pos] = candidate[0, selected_pos]
            confirmed_mask[selected_pos] = True

        return block_ids[0].tolist(), confirmed_mask

    def generate_from_cache(
        self,
        *,
        past_key_values: Cache,
        cache_positions: Sequence[int],
        query_ids: Sequence[int],
        query_rope_start: int,
        max_new_tokens: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
        force_diffusion: bool = False,
    ) -> List[int]:
        mask_token_id = self._mask_token_id()
        query_ids = list(query_ids)
        cache_positions = list(cache_positions)
        max_new = int(max_new_tokens if max_new_tokens is not None else self.max_new_tokens)
        if max_new <= 0:
            return []

        query_positions = self._range_positions(query_rope_start, len(query_ids))
        active_cache = past_key_values
        active_positions = list(cache_positions)
        last_context_logit: Optional[torch.Tensor] = None
        if query_ids:
            active_cache, last_context_logit = self._prefill_plain(
                query_ids,
                query_positions,
                past_key_values=active_cache,
                past_positions=active_positions,
                attention_mask="full",
            )
            active_positions.extend(query_positions)

        total_steps = int(diffusion_steps if diffusion_steps is not None else self.diffusion_steps)
        num_blocks = max(1, math.ceil(max_new / self.block_length))
        steps_per_block = max(1, math.ceil(total_steps / num_blocks))

        generated: List[int] = []
        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_length
            block_len = min(self.block_length, max_new - block_start)
            if block_len <= 0:
                break
            if self.config.generation_position_mode != "after_query":
                raise ValueError(f"Unsupported generation_position_mode: {self.config.generation_position_mode}")
            block_pos_start = query_rope_start + len(query_ids) + block_start
            suffix_len = max_new - block_start
            full_cache, full_positions, _, first_token = self._initialize_fastdllm_suffix_cache(
                active_cache=active_cache,
                active_positions=active_positions,
                suffix_len=suffix_len,
                suffix_pos_start=block_pos_start,
                last_context_logit=last_context_logit,
            )
            current_slot_start = len(active_positions)
            current_slot_end = current_slot_start + block_len
            replace_position = torch.zeros(
                (1, len(full_positions)),
                device=self.device,
                dtype=torch.bool,
            )
            replace_position[:, current_slot_start:current_slot_end] = True
            block_ids = torch.full((1, block_len), mask_token_id, device=self.device, dtype=torch.long)
            block_ids[:, 0] = first_token[:1]

            if self.config.alg == "confidence_threshold" and not force_diffusion:
                block_ids, full_cache = self._denoise_block_confidence_threshold(
                    block_ids,
                    full_cache=full_cache,
                    full_positions=full_positions,
                    replace_position=replace_position,
                    last_context_logit=last_context_logit,
                )
            else:
                block_ids, full_cache = self._denoise_block_diffusion(
                    block_ids,
                    full_cache=full_cache,
                    full_positions=full_positions,
                    replace_position=replace_position,
                    last_context_logit=last_context_logit,
                    steps_per_block=steps_per_block,
                    confidence_alg="origin" if force_diffusion else None,
                )

            block_list = block_ids[0].tolist()
            generated.extend(block_list)

            final_logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_positions,
                replace_position=replace_position,
            )
            last_context_logit = final_logits[:, -1, :].detach()
            active_cache = self._slice_cache(full_cache, 0, current_slot_end)
            active_positions = list(full_positions[:current_slot_end])

        return generated[:max_new]

    @torch.inference_mode()
    def generate_full_prompt_mask(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        scoring_query_ids: Optional[Sequence[int]] = None,
    ) -> FastDLLMParallelCompResult:
        start = time.time()
        prefix_ids = list(prefix_ids)
        context_ids = list(context_ids)
        query_ids = list(query_ids)
        scoring_query_ids = list(scoring_query_ids if scoring_query_ids is not None else query_ids)

        candidate_chunks, selected_indices, chunk_scores, selection_query_ids, score_token_mask = self._prepare_candidate_chunks(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            scoring_query_ids=scoring_query_ids,
        )
        eviction_query_ids = self._token_eviction_query_ids(
            scoring_query_ids=scoring_query_ids,
            selection_query_ids=selection_query_ids,
            score_token_mask=score_token_mask,
        )

        prefix_positions = self._range_positions(0, len(prefix_ids))
        prompt_ids: List[int] = list(prefix_ids)
        prompt_positions: List[int] = list(prefix_positions)
        cache_positions: List[int] = list(prefix_positions)
        chunk_meta: List[FastDLLMChunkMeta] = []
        total_removed = 0
        compressed_cache_len = len(prefix_ids)
        per_head_chunk_spans: List[Tuple[int, int, List[torch.Tensor]]] = []

        for chunk_order, chunk_index in enumerate(selected_indices):
            original_chunk_ids = list(candidate_chunks[chunk_index])
            chunk_start = self._chunk_rope_start(len(prefix_ids), chunk_order, chunk_index)
            original_chunk_positions = self._range_positions(chunk_start, len(original_chunk_ids))
            full_span_start = len(prompt_ids)
            if self.config.token_eviction_granularity == "per_head":
                per_layer_per_head_keep = self._keep_positions_per_layer_per_head_for_chunk(
                    prefix_ids,
                    original_chunk_ids,
                    eviction_query_ids,
                )
                keep_positions = self._union_keep_positions_per_layer_per_head(per_layer_per_head_keep)
                kept_count = (
                    int(per_layer_per_head_keep[0].shape[1])
                    if per_layer_per_head_keep
                    else len(original_chunk_ids)
                )
                chunk_ids = list(original_chunk_ids)
                chunk_positions = list(original_chunk_positions)
                compressed_chunk_positions = original_chunk_positions[:kept_count]
            else:
                per_layer_per_head_keep = None
                keep_positions = self._keep_positions_for_chunk(prefix_ids, original_chunk_ids, eviction_query_ids)
                chunk_ids = [original_chunk_ids[idx] for idx in keep_positions]
                chunk_positions = [original_chunk_positions[idx] for idx in keep_positions]
                kept_count = len(chunk_ids)
                compressed_chunk_positions = chunk_positions
            prompt_ids.extend(chunk_ids)
            prompt_positions.extend(chunk_positions)
            full_span_end = len(prompt_ids)
            if per_layer_per_head_keep is not None:
                per_head_chunk_spans.append((full_span_start, full_span_end, per_layer_per_head_keep))
            cache_start = compressed_cache_len
            cache_positions.extend(compressed_chunk_positions)
            compressed_cache_len += kept_count
            cache_end = compressed_cache_len
            removed = max(0, len(original_chunk_ids) - kept_count)
            total_removed += removed
            chunk_meta.append(
                FastDLLMChunkMeta(
                    chunk_index=chunk_index,
                    original_tokens=len(original_chunk_ids),
                    kept_tokens=kept_count,
                    removed_tokens=removed,
                    cache_start=cache_start,
                    cache_end=cache_end,
                    rope_start=chunk_start,
                    rope_end=chunk_start + len(original_chunk_ids),
                    score=chunk_scores.get(chunk_index),
                    kept_positions=list(keep_positions),
                )
            )

        query_rope_start = self._final_query_rope_start(
            len(prefix_ids),
            cache_positions,
            selected_count=len(selected_indices),
        )
        full_query_rope_start = self._final_query_rope_start(
            len(prefix_ids),
            prompt_positions,
            selected_count=len(selected_indices),
        )
        query_positions = self._range_positions(query_rope_start, len(query_ids))
        full_query_positions = self._range_positions(full_query_rope_start, len(query_ids))
        max_new = int(self.max_new_tokens)
        suffix_pos_start = query_rope_start + len(query_ids)
        full_suffix_pos_start = full_query_rope_start + len(query_ids)
        mask_token_id = self._mask_token_id()

        full_prompt_len = len(prompt_ids) + len(query_ids)
        active_prompt_len = len(cache_positions) + len(query_ids)
        include_initial_masks = self.config.cache_build_mode == "full_prompt_mask"
        if include_initial_masks:
            suffix_positions = self._range_positions(suffix_pos_start, max_new)
            full_suffix_positions = self._range_positions(full_suffix_pos_start, max_new)
            suffix_ids = [mask_token_id] * max_new
            full_ids = list(prompt_ids) + list(query_ids) + suffix_ids
            full_positions = list(prompt_positions) + full_query_positions + full_suffix_positions
            active_positions_after_prefill = list(cache_positions) + query_positions + suffix_positions
        else:
            full_ids = list(prompt_ids) + list(query_ids)
            full_positions = list(prompt_positions) + full_query_positions
            active_positions_after_prefill = list(cache_positions) + query_positions

        outputs = self._model_forward(
            self._ids_tensor(full_ids),
            attention_mask="full",
            position_ids=self._position_ids_from_list(full_positions),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        active_cache = outputs.past_key_values
        active_positions = list(full_positions)
        if self.config.token_eviction_granularity == "per_head" and per_head_chunk_spans:
            keep_indices = self._full_prompt_keep_indices_per_layer_per_head(
                seq_len=len(full_ids),
                chunk_spans=per_head_chunk_spans,
            )
            active_cache = self._gather_cache_per_layer_per_head(active_cache, keep_indices)
            active_positions = active_positions_after_prefill
        first_token: Optional[torch.Tensor] = None
        if include_initial_masks:
            shifted_logits = self._align_generation_logits(outputs.logits)
            first_logits = shifted_logits[:, full_prompt_len:full_prompt_len + 1, :].reshape(-1, shifted_logits.shape[-1])
            _, first_token = sample_tokens(
                first_logits,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        last_context_logit: Optional[torch.Tensor] = (
            outputs.logits[:, full_prompt_len - 1, :].detach() if full_prompt_len > 0 else None
        )

        total_steps = int(self.diffusion_steps)
        num_blocks = max(1, math.ceil(max_new / self.block_length))
        steps_per_block = max(1, math.ceil(total_steps / num_blocks))

        generated: List[int] = []
        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_length
            block_len = min(self.block_length, max_new - block_start)
            if block_len <= 0:
                break
            if self.config.generation_position_mode != "after_query":
                raise ValueError(f"Unsupported generation_position_mode: {self.config.generation_position_mode}")

            if include_initial_masks and block_idx == 0:
                full_cache = active_cache
                full_block_positions = active_positions
                current_slot_start = active_prompt_len
                current_slot_end = current_slot_start + block_len
                assert first_token is not None
                current_first_token = first_token
            else:
                block_pos_start = suffix_pos_start + block_start
                suffix_len = max_new - block_start
                full_cache, full_block_positions, _, current_first_token = self._initialize_fastdllm_suffix_cache(
                    active_cache=active_cache,
                    active_positions=active_positions,
                    suffix_len=suffix_len,
                    suffix_pos_start=block_pos_start,
                    last_context_logit=last_context_logit,
                )
                current_slot_start = len(active_positions)
                current_slot_end = current_slot_start + block_len

            replace_position = torch.zeros(
                (1, len(full_block_positions)),
                device=self.device,
                dtype=torch.bool,
            )
            replace_position[:, current_slot_start:current_slot_end] = True
            block_ids = torch.full((1, block_len), mask_token_id, device=self.device, dtype=torch.long)
            block_ids[:, 0] = current_first_token[:1]

            if self.config.alg == "confidence_threshold":
                block_ids, full_cache = self._denoise_block_confidence_threshold(
                    block_ids,
                    full_cache=full_cache,
                    full_positions=full_block_positions,
                    replace_position=replace_position,
                    last_context_logit=last_context_logit,
                )
            else:
                block_ids, full_cache = self._denoise_block_diffusion(
                    block_ids,
                    full_cache=full_cache,
                    full_positions=full_block_positions,
                    replace_position=replace_position,
                    last_context_logit=last_context_logit,
                    steps_per_block=steps_per_block,
                )

            block_list = block_ids[0].tolist()
            generated.extend(block_list)

            final_logits, full_cache = self._forward_generation_block_replace(
                block_ids,
                full_cache=full_cache,
                full_positions=full_block_positions,
                replace_position=replace_position,
            )
            last_context_logit = final_logits[:, -1, :].detach()
            active_cache = self._slice_cache(full_cache, 0, current_slot_end)
            active_positions = list(full_block_positions[:current_slot_end])

        generated_ids = generated[:max_new]
        self.total_generation_time += time.time() - start
        self.generated_token_num += sum(1 for token_id in generated_ids if token_id != self.tokenizer.eos_token_id)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and eos in text:
            text = text.split(eos)[0]

        return FastDLLMParallelCompResult(
            text=text,
            sequences=generated_ids,
            selected_chunk_indices=selected_indices,
            chunk_scores=chunk_scores,
            cache_build_mode=self.config.cache_build_mode,
            prefix_tokens=len(prefix_ids),
            query_tokens=len(query_ids),
            raw_context_tokens=len(context_ids),
            candidate_chunks=len(candidate_chunks),
            cache_tokens=len(cache_positions),
            removed_tokens=total_removed,
            chunk_meta=chunk_meta,
            generation_blocks=num_blocks,
            generation_block_length=self.block_length,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        scoring_query_ids: Optional[Sequence[int]] = None,
    ) -> FastDLLMParallelCompResult:
        if self.config.cache_build_mode in {"full_prompt_mask", "full_prompt_query"}:
            return self.generate_full_prompt_mask(
                prefix_ids=prefix_ids,
                context_ids=context_ids,
                query_ids=query_ids,
                scoring_query_ids=scoring_query_ids,
            )
        if self.config.cache_build_mode not in {"chunk_query", "chunk_only"}:
            raise ValueError(f"Unsupported cache_build_mode: {self.config.cache_build_mode}")

        start = time.time()
        past_key_values, cache_positions, meta = self.build_prefill_cache(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            query_ids=query_ids,
            scoring_query_ids=scoring_query_ids,
        )
        generated_ids = self.generate_from_cache(
            past_key_values=past_key_values,
            cache_positions=cache_positions,
            query_ids=query_ids,
            query_rope_start=meta["query_rope_start"],
        )
        self.total_generation_time += time.time() - start
        self.generated_token_num += sum(1 for token_id in generated_ids if token_id != self.tokenizer.eos_token_id)

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and eos in text:
            text = text.split(eos)[0]

        return FastDLLMParallelCompResult(
            text=text,
            sequences=generated_ids,
            selected_chunk_indices=meta["selected_indices"],
            chunk_scores=meta["chunk_scores"],
            cache_build_mode=meta["cache_build_mode"],
            prefix_tokens=meta["prefix_tokens"],
            query_tokens=len(query_ids),
            raw_context_tokens=meta["raw_context_tokens"],
            candidate_chunks=meta["candidate_chunks"],
            cache_tokens=meta["cache_tokens"],
            removed_tokens=meta["removed_tokens"],
            chunk_meta=meta["chunk_meta"],
            generation_blocks=max(1, math.ceil(self.max_new_tokens / self.block_length)),
            generation_block_length=self.block_length,
        )

    def encode_fragment(self, text: str) -> List[int]:
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def bos_ids(self) -> List[int]:
        if not self.config.add_bos_token:
            return []
        return self._get_bos_token_ids()

    def generate_from_text_parts(
        self,
        *,
        prefix: str,
        context: str,
        query: str,
        scoring_query: Optional[str] = None,
    ) -> FastDLLMParallelCompResult:
        prefix_ids = self.bos_ids() + self.encode_fragment(prefix)
        context_ids = self.encode_fragment(context)
        query_ids = self.encode_fragment(query)
        scoring_query_ids = self.encode_fragment(scoring_query if scoring_query is not None else query)
        return self.generate(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            query_ids=query_ids,
            scoring_query_ids=scoring_query_ids,
        )


def load_fastdllm_parallelcomp(
    pretrained: str,
    fastdllm_dream_dir: Optional[str] = None,
    model_backend: str = "dream",
    fastdllm_llada_dir: Optional[str] = None,
    **kwargs,
) -> FastDLLMParallelComp:
    config = FastDLLMParallelCompConfig(
        fastdllm_dream_dir=fastdllm_dream_dir or default_fastdllm_dream_dir(),
        model_backend=model_backend,
        fastdllm_llada_dir=fastdllm_llada_dir,
        pretrained=pretrained,
        **kwargs,
    )
    return FastDLLMParallelComp(config)
