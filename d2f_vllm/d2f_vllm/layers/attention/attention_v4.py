import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import List
from functools import lru_cache, partial
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask 
from transformers.integrations.flex_attention import compile_friendly_flex_attention as flex_attention

from d2f_vllm.layers.attention.ops import (
    causal_lm_flash_decoding, diffusion_lm_flash_decoding, diffusion_lm_parallel_flash_decoding,
    store_kvcache_unified_layout, store_kvcache_distinct_layout, load_kvcache,
    CHECK_STORING, CHECK_LOADING, CHECK_ATTENTION
)
from d2f_vllm.utils.context import ContextForDiffusionLM, get_context_causal_lm, get_context_diffusion_lm


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        model_type='causal_lm'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.causal = model_type == 'causal_lm'
        self.model_type = model_type
        self.attention_backend = os.environ.get("D2F_VLLM_ATTENTION_BACKEND", "flex").lower()
        is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        } if is_rtx_xx90(torch.cuda.get_device_name(0)) else None
        self.attention = torch.compile(
            partial(flex_attention, kernel_options=kernel_options, enable_gqa=True, 
                    return_lse=False), dynamic=True)
        self._block_mask_cache = {}
        self.layer_idx = -1

    def _attention_forward(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        *,
        block_mask=None,
        dense_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention_backend == "sdpa":
            if q_t.shape[1] != k_t.shape[1]:
                if q_t.shape[1] % k_t.shape[1] != 0:
                    raise ValueError(
                        "Cannot expand KV heads to query heads."
                    )
                repeat = q_t.shape[1] // k_t.shape[1]
                k_t = k_t.repeat_interleave(repeat, dim=1)
                v_t = v_t.repeat_interleave(repeat, dim=1)
            attn_mask = None
            if dense_mask is not None:
                attn_mask = dense_mask.to(device=q_t.device, dtype=torch.bool)
                if attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            return F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=0.0)
        return self.attention(q_t, k_t, v_t, block_mask=block_mask)

    def dllm_block_mask(self, block_mask: torch.Tensor, 
                        B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        def _mask_mod(batch, head, token_q, token_kv):
            return block_mask[token_q, token_kv]
        return create_block_mask(_mask_mod, B, H, Q_LEN, KV_LEN, device=device)

    def _maybe_apply_decode_delta(
        self,
        o: torch.Tensor,
        q_t: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        context: ContextForDiffusionLM,
    ) -> torch.Tensor:
        state = getattr(context, "decode_delta_state", None)
        if not state or not state.get("enabled", False):
            return o
        if self.layer_idx < 0:
            return o
        full_k_by_layer = state.get("full_k")
        full_v_by_layer = state.get("full_v")
        if full_k_by_layer is None or full_v_by_layer is None:
            return o
        if self.layer_idx >= len(full_k_by_layer):
            return o
        full_prompt_k = full_k_by_layer[self.layer_idx]
        full_prompt_v = full_v_by_layer[self.layer_idx]
        if full_prompt_k is None or full_prompt_v is None or full_prompt_k.numel() == 0:
            return o

        _, num_heads, seq_len, _ = q_t.shape
        stride = max(1, int(state.get("stride", 4)))
        left = max(0, int(state.get("left", stride - 1)))
        anchor_offset = int(state.get("anchor_offset", stride - 1))
        if seq_len <= 0:
            return o
        if anchor_offset < 0:
            anchor_offset = stride - 1
        anchor_offset = min(anchor_offset, stride - 1)
        anchors = torch.arange(anchor_offset, seq_len, stride, device=q_t.device, dtype=torch.long)
        if anchors.numel() == 0:
            return o

        full_k = torch.cat([full_prompt_k.to(device=k_new.device, dtype=k_new.dtype), k_new], dim=0)
        full_v = torch.cat([full_prompt_v.to(device=v_new.device, dtype=v_new.dtype), v_new], dim=0)
        q_anchor = q_t.index_select(2, anchors)
        k_t, v_t = [rearrange(t, 's h d -> 1 h s d').contiguous() for t in (full_k, full_v)]

        full_prompt_len = int(full_prompt_k.shape[0])
        prompt_mask = torch.ones((anchors.numel(), full_prompt_len), dtype=torch.bool, device=q_t.device)
        if context.block_mask is not None:
            active_start = int(context.context_lens[0].item()) if context.context_lens is not None else 0
            active_mask = context.block_mask.index_select(0, anchors)[:, active_start:]
        else:
            active_mask = torch.ones((anchors.numel(), seq_len), dtype=torch.bool, device=q_t.device)
        delta_mask = torch.cat([prompt_mask, active_mask], dim=1)
        block_mask = self.dllm_block_mask(delta_mask, 1, num_heads, int(anchors.numel()), int(full_k.shape[0]), str(q_t.device))
        dense_anchor = self._attention_forward(q_anchor, k_t, v_t, block_mask=block_mask, dense_mask=delta_mask)
        base_anchor = o.index_select(2, anchors)
        delta = (dense_anchor - base_anchor) * float(state.get("scale", 1.0))
        if state.get("debug", False):
            delta_abs_max = float(delta.abs().max().item()) if delta.numel() else 0.0
            state["calls"] = int(state.get("calls", 0)) + 1
            state["nonzero_calls"] = int(state.get("nonzero_calls", 0)) + int(delta_abs_max > 0.0)
            state["max_abs"] = max(float(state.get("max_abs", 0.0)), delta_abs_max)
            state["anchor_count"] = int(state.get("anchor_count", 0)) + int(anchors.numel())

        correction = torch.zeros_like(o)
        for delta_idx, anchor in enumerate(anchors.tolist()):
            start = max(0, int(anchor) - left)
            rows = torch.arange(start, int(anchor) + 1, device=q_t.device, dtype=torch.long)
            correction[:, :, rows, :] += delta[:, :, delta_idx:delta_idx + 1, :]
        return o + correction
    
    @lru_cache(maxsize=32)
    def causal_lm_block_mask(self, cum_seq_lens: torch.Tensor, B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        document_ids = torch.zeros((cum_seq_lens[-1],), dtype=torch.int32, device=device)
        start_idx = 0
        for doc_idx, seq_len in enumerate(cum_seq_lens[1:]):
            end_idx = seq_len
            document_ids[start_idx:end_idx] = doc_idx
            start_idx = end_idx
        
        def _mask_mod(batch, head, token_q, token_kv):
            causal_mask = token_q >= token_kv
            document_mask = document_ids[token_q] == document_ids[token_kv]
            return causal_mask & document_mask
        
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: List[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context: ContextForDiffusionLM = get_context_causal_lm() if self.model_type == 'causal_lm' else get_context_diffusion_lm()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = context.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if not (self.model_type == 'diffusion_lm' and not context.need_kv_cache_store):
                store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type, context)
                # CHECK_STORING(k_cache, v_cache, k, v, context)

        transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d').contiguous()
        # Prefill / Decode logic
        if context.is_prefill:
            # Block PK
            if context.block_tables is not None and self.model_type == 'causal_lm':
                k, v = k_cache, v_cache
            elif context.block_tables is not None and self.model_type == 'diffusion_lm':
                # TODO: Implement Prefix Caching
                pass

            if not is_unified_layout and self.model_type == 'diffusion_lm' and k_cache.numel() > 0:
                config = context.seqs[0].config
                diffusion_block_size = config.diffusion_block_size
                o = torch.empty_like(q)
                diffusion_lm_parallel_flash_decoding(
                    q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
                    context.block_tables if context.block_tables is not None else torch.zeros((1, 1), dtype=torch.int32, device=q.device),
                    context.cu_seqlens_q,
                    torch.tensor([q.shape[0]], dtype=torch.int32, device=q.device),
                    0, q.shape[0], 1.0, 1.0,
                    diffusion_block_size, None, None, self.scale, context.block_mask,
                    bidirectional=True,
                )
            else:
                # Attention computation
                q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]

                B, H, S, _ = q_t.shape
                block_mask_fn = self.causal_lm_block_mask if self.model_type == 'causal_lm' else self.dllm_block_mask
                input_obj = context.cu_seqlens_q if self.model_type == 'causal_lm' else context.block_mask
                block_mask = block_mask_fn(input_obj, B, H, S, S, str(q.device))
                o = self._attention_forward(q_t, k_t, v_t, block_mask=block_mask, dense_mask=input_obj)
        else:
            if self.model_type == 'causal_lm':
                o = causal_lm_flash_decoding(
                    q, k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_tables=context.block_tables, 
                    softmax_scale=self.scale, page_size=256
                )
            else: 
                config = context.seqs[0].config
                diffusion_block_size = config.diffusion_block_size
                if is_unified_layout:
                    q_t = transpose_fn(q)
                    k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, context, k, v)
                    # k_comb, v_comb = CHECK_LOADING(k_comb, v_comb, k, v, k_cache, v_cache, context)``
                    k_t, v_t = transpose_fn(k_comb), transpose_fn(v_comb)

                    B, H, Sq, _ = q_t.shape
                    _, _, Skv, _ = k_t.shape
                    block_mask = self.dllm_block_mask(context.block_mask, B, H, Sq, Skv, str(q.device))

                    o = self._attention_forward(q_t, k_t, v_t, block_mask=block_mask, dense_mask=context.block_mask)
                    o = self._maybe_apply_decode_delta(o, q_t, k, v, context)
                else:
                    o = torch.empty_like(q)
                    diffusion_lm_parallel_flash_decoding(
                        q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
                        context.block_tables, context.cu_seqlens_q, context.total_lens,
                        max(context.total_lens), max(context.seq_lens), 1.0, 1.0,
                        diffusion_block_size, None, None, self.scale, context.block_mask
                    )
            
        # Final reshape
        if context.kv_cache_layout == "unified" and self.model_type == 'diffusion_lm':
            o = rearrange(o, '1 h s d -> s (h d)').contiguous()
        elif not is_unified_layout and self.model_type == 'diffusion_lm':
            o = o.view(-1, self.num_heads * self.head_dim).contiguous()
        else:
            if not context.is_prefill:
                o = o.view(-1, self.num_heads * self.head_dim).contiguous()
            elif context.is_prefill:
                o = rearrange(o, '1 h s d -> s (h d)').contiguous()

        return o
