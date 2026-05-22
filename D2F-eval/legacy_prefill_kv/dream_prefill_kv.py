#!/usr/bin/env python3
"""Generic Dream KV-prefill controller for ParallelComp-style experiments.

This module intentionally stays benchmark-agnostic. Adapters should prepare
token ids for prefix/context/query, then call `DreamPrefillKVController`.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributions as dists
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class DreamPrefillKVConfig:
    chunk_size: int = 1024
    topk_chunks: int = 3
    keep_first_chunk: bool = False
    split_from_tail: bool = False
    chunk_bos: bool = True
    chunk_cache_mode: str = "independent"

    score_mode: str = "self_information"
    score_query_window: int = 0
    score_disable_adapter: bool = False
    attention_score_layers: int = 4
    attention_query_window: int = 8
    score_draft_tokens: int = 0
    score_draft_steps: Optional[int] = None

    token_capacity: int = 0
    token_score_query_window: int = 8
    token_score_layers: int = 4
    token_score_layer_mode: str = "last"
    token_score_reduce: str = "sum"
    token_eviction_mode: str = "cache_slice"

    chunk_position_mode: str = "reuse"
    query_position_mode: str = "after_reused_window"
    prefix_position_mode: str = "continuous"
    replay_full_mask: bool = True

    max_new_tokens: int = 32
    steps: Optional[int] = None
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    alg: str = "entropy"
    alg_temp: Optional[float] = 0.0


@dataclass
class ChunkPrefillMeta:
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
class DreamPrefillKVResult:
    text: str
    sequences: List[int]
    selected_chunk_indices: List[int]
    chunk_scores: Dict[int, float]
    prefix_tokens: int
    query_tokens: int
    raw_context_tokens: int
    candidate_chunks: int
    cache_tokens: int
    removed_tokens: int
    chunk_meta: List[ChunkPrefillMeta]


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


class DreamPrefillKVController:
    """Build a compressed Dream KV cache, then replay query/generation over it."""

    def __init__(self, model, tokenizer, config: Optional[DreamPrefillKVConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DreamPrefillKVConfig()
        self.device = next(model.parameters()).device

    @property
    def _mask_dtype(self) -> torch.dtype:
        dtype = next(self.model.parameters()).dtype
        if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return dtype
        return torch.float32

    def _adapter_disabled(self):
        if self.config.score_disable_adapter and hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        return nullcontext()

    def _ids_tensor(self, ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(ids)], device=self.device, dtype=torch.long)

    def _full_visible_mask(self, query_length: int, key_length: Optional[int] = None) -> torch.Tensor:
        key_length = query_length if key_length is None else key_length
        return torch.zeros(
            (1, 1, query_length, key_length),
            device=self.device,
            dtype=self._mask_dtype,
        )

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.full(
            (1, 1, seq_len, seq_len),
            -torch.inf,
            device=self.device,
            dtype=self._mask_dtype,
        )
        tri = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        mask[0, 0, tri] = 0
        return mask

    def _cached_prefix_mask(self, cached_length: int, query_length: int) -> torch.Tensor:
        if self.config.replay_full_mask:
            return self._full_visible_mask(query_length, cached_length + query_length)

        mask = torch.full(
            (1, 1, query_length, cached_length + query_length),
            -torch.inf,
            device=self.device,
            dtype=self._mask_dtype,
        )
        if cached_length > 0:
            mask[:, :, :, :cached_length] = 0
        tri = torch.tril(
            torch.ones((query_length, query_length), device=self.device, dtype=torch.bool)
        )
        mask[0, 0, :, cached_length:][tri] = 0
        return mask

    def _window_query(self, query_ids: Sequence[int], window: int) -> List[int]:
        ids = list(query_ids)
        if window and window > 0:
            return ids[-window:]
        return ids

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

    def score_chunk_self_information(
        self,
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
    ) -> float:
        query_ids = self._window_query(query_ids, self.config.score_query_window)
        if not chunk_ids or not query_ids:
            return float("-inf")

        joint = self._ids_tensor(list(chunk_ids) + list(query_ids))
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        with torch.inference_mode(), self._adapter_disabled():
            outputs = self.model(
                joint,
                attention_mask=self._causal_mask(joint.shape[1]),
                return_dict=True,
                use_cache=False,
            )

        logits = outputs.logits
        if logits.shape[1] < chunk_len + query_len - 1:
            return float("-inf")

        query_logits = logits[:, chunk_len - 1:chunk_len + query_len - 1, :]
        query_labels = joint[:, chunk_len:chunk_len + query_len]
        if query_logits.shape[1] != query_labels.shape[1]:
            return float("-inf")

        log_probs = F.log_softmax(query_logits.float(), dim=-1)
        token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
        return float(-token_nll.mean().item())

    def query_attention_token_scores(
        self,
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
        *,
        query_window: int,
        layer_window: int,
        layer_mode: str = "last",
        reduce_mode: Optional[str] = None,
    ) -> torch.Tensor:
        query_ids = self._window_query(query_ids, query_window)
        if not chunk_ids or not query_ids:
            return torch.empty(0, device=self.device)

        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        joint = self._ids_tensor(list(chunk_ids) + list(query_ids))
        with torch.inference_mode(), self._adapter_disabled():
            outputs = self.model(
                joint,
                attention_mask=self._causal_mask(joint.shape[1]),
                return_dict=True,
                use_cache=False,
                output_attentions=True,
            )

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            return torch.empty(0, device=self.device)

        selected_layers = self._select_attention_layers(attentions, layer_window, layer_mode)
        layer_scores: List[torch.Tensor] = []
        for attn in selected_layers:
            if attn is None:
                continue
            query_to_chunk = attn[0, :, chunk_len:chunk_len + query_len, :chunk_len].float()
            if query_to_chunk.numel() == 0:
                continue
            if (reduce_mode or self.config.token_score_reduce) == "mean":
                layer_scores.append(query_to_chunk.mean(dim=(0, 1)))
            else:
                layer_scores.append(query_to_chunk.sum(dim=(0, 1)))

        if not layer_scores:
            return torch.empty(0, device=self.device)
        return torch.stack(layer_scores, dim=0).mean(dim=0)

    def _select_attention_layers(self, attentions, layer_window: int, layer_mode: str):
        if layer_mode == "all" or layer_window <= 0:
            return attentions
        if layer_mode == "first":
            return attentions[:layer_window]
        if layer_mode == "last":
            return attentions[-layer_window:]
        raise ValueError(f"Unsupported attention layer mode: {layer_mode}")

    def select_chunks(
        self,
        candidate_chunks: Sequence[Sequence[int]],
        scoring_query_ids: Sequence[int],
    ) -> Tuple[List[int], Dict[int, float]]:
        if not candidate_chunks:
            return [], {}

        if self.config.score_mode == "none" or not scoring_query_ids:
            selected = list(range(len(candidate_chunks)))
            if self.config.topk_chunks > 0:
                selected = selected[:self.config.topk_chunks]
            return selected, {}

        scores: Dict[int, float] = {}
        for idx, chunk_ids in enumerate(candidate_chunks):
            if self.config.score_mode == "attention":
                token_scores = self.query_attention_token_scores(
                    chunk_ids,
                    scoring_query_ids,
                    query_window=self.config.attention_query_window,
                    layer_window=self.config.attention_score_layers,
                    layer_mode="last",
                    reduce_mode=self.config.token_score_reduce,
                )
                score = float(token_scores.mean().item()) if token_scores.numel() else float("-inf")
            elif self.config.score_mode == "self_information":
                score = self.score_chunk_self_information(chunk_ids, scoring_query_ids)
            elif self.config.score_mode == "draft_self_information":
                score = self.score_chunk_self_information(chunk_ids, scoring_query_ids)
            else:
                raise ValueError(f"Unsupported score_mode: {self.config.score_mode}")
            scores[idx] = score if math.isfinite(score) else float("-inf")

        forced = [0] if self.config.keep_first_chunk and candidate_chunks else []
        remaining = [idx for idx in range(len(candidate_chunks)) if idx not in forced]
        ranked = sorted(remaining, key=lambda idx: scores.get(idx, float("-inf")), reverse=True)
        topk = len(ranked) if self.config.topk_chunks <= 0 else min(self.config.topk_chunks, len(ranked))
        return sorted(set(forced + ranked[:topk])), scores

    def _chunk_rope_start(self, prefix_len: int, chunk_order: int, chunk_len: int) -> int:
        mode = self.config.chunk_position_mode
        if mode == "reuse":
            return prefix_len
        if mode == "continuous":
            return prefix_len + chunk_order * max(1, self.config.chunk_size)
        if mode == "absolute":
            return prefix_len + chunk_order * chunk_len
        raise ValueError(f"Unsupported chunk_position_mode: {mode}")

    def _query_rope_start(self, prefix_len: int, selected_chunk_count: int, cache_len: int) -> int:
        mode = self.config.query_position_mode
        if mode == "after_reused_window":
            return prefix_len + max(1, self.config.chunk_size)
        if mode == "after_cache":
            return cache_len
        if mode == "after_selected_chunks":
            return prefix_len + selected_chunk_count * max(1, self.config.chunk_size)
        raise ValueError(f"Unsupported query_position_mode: {mode}")

    def _chunk_query_rope_start(self, chunk_rope_start: int, chunk_len: int) -> int:
        if self.config.chunk_position_mode in {"reuse", "continuous"}:
            return chunk_rope_start + max(1, self.config.chunk_size)
        return chunk_rope_start + chunk_len

    def _position_ids(self, start: int, length: int) -> torch.Tensor:
        return torch.arange(start, start + length, device=self.device, dtype=torch.long).unsqueeze(0)

    def _cache_positions(self, start: int, length: int) -> torch.Tensor:
        return torch.arange(start, start + length, device=self.device, dtype=torch.long)

    def _base_dream_model(self):
        model = self.model
        if hasattr(model, "get_base_model"):
            model = model.get_base_model()
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise RuntimeError("first_layer_recompute requires a DreamModel-like module with model.layers")
        return model

    def _concat_past_key_values(self, caches: Sequence[object]):
        caches = [cache for cache in caches if cache is not None]
        if not caches:
            return None
        combined = DynamicCache()
        num_layers = len(caches[0].key_cache)
        combined.key_cache = [
            torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=2)
            for layer_idx in range(num_layers)
        ]
        combined.value_cache = [
            torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=2)
            for layer_idx in range(num_layers)
        ]
        return combined

    def _slice_past_key_values(self, past_key_values, start: int, end: int):
        if past_key_values is None or end <= start:
            return None
        sliced = DynamicCache()
        sliced.key_cache = [
            layer_k[:, :, start:end, :].contiguous()
            for layer_k in past_key_values.key_cache
        ]
        sliced.value_cache = [
            layer_v[:, :, start:end, :].contiguous()
            for layer_v in past_key_values.value_cache
        ]
        return sliced

    def _prefill_ids(
        self,
        token_ids: Sequence[int],
        *,
        past_key_values,
        cached_length: int,
        rope_start: int,
    ):
        if not token_ids:
            return past_key_values
        ids = self._ids_tensor(token_ids)
        outputs = self.model(
            ids,
            attention_mask=self._cached_prefix_mask(cached_length, len(token_ids)),
            past_key_values=past_key_values,
            position_ids=self._position_ids(rope_start, len(token_ids)),
            cache_position=self._cache_positions(cached_length, len(token_ids)),
            use_cache=True,
            update_kvcache=len(token_ids),
            return_dict=True,
        )
        return outputs.past_key_values

    def _prefill_chunk_with_query(
        self,
        chunk_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
        *,
        past_key_values,
        cached_length: int,
        prefix_len: int,
        chunk_order: int,
    ):
        query_ids = self._window_query(scoring_query_ids, self.config.token_score_query_window)
        input_ids = list(chunk_ids) + list(query_ids)
        chunk_len = len(chunk_ids)
        seq_len = len(input_ids)
        rope_start = self._chunk_rope_start(prefix_len, chunk_order, chunk_len)
        query_start = self._chunk_query_rope_start(rope_start, chunk_len)
        chunk_positions = torch.arange(
            rope_start,
            rope_start + chunk_len,
            device=self.device,
            dtype=torch.long,
        )
        query_positions = torch.arange(
            query_start,
            query_start + len(query_ids),
            device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.cat([chunk_positions, query_positions], dim=0).unsqueeze(0)
        outputs = self.model(
            self._ids_tensor(input_ids),
            attention_mask=self._cached_prefix_mask(cached_length, seq_len),
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=self._cache_positions(cached_length, seq_len),
            use_cache=True,
            update_kvcache=chunk_len,
            return_dict=True,
        )
        return outputs.past_key_values, rope_start

    def _prefill_joint_selected_chunks_with_query(
        self,
        selected_chunks: Sequence[Sequence[int]],
        scoring_query_ids: Sequence[int],
        *,
        past_key_values,
        cached_length: int,
        prefix_len: int,
    ):
        query_ids = self._window_query(scoring_query_ids, self.config.token_score_query_window)
        flat_chunk_ids: List[int] = []
        chunk_positions_parts: List[torch.Tensor] = []
        chunk_spans: List[Tuple[int, int, int, int]] = []
        cursor = 0
        for chunk_order, chunk_ids in enumerate(selected_chunks):
            chunk_ids = list(chunk_ids)
            chunk_len = len(chunk_ids)
            rope_start = self._chunk_rope_start(prefix_len, chunk_order, chunk_len)
            positions = torch.arange(
                rope_start,
                rope_start + chunk_len,
                device=self.device,
                dtype=torch.long,
            )
            flat_chunk_ids.extend(chunk_ids)
            chunk_positions_parts.append(positions)
            chunk_spans.append((cursor, cursor + chunk_len, rope_start, rope_start + chunk_len))
            cursor += chunk_len

        if not flat_chunk_ids:
            return past_key_values, [], []

        if chunk_positions_parts:
            chunk_positions = torch.cat(chunk_positions_parts, dim=0)
        else:
            chunk_positions = torch.empty(0, device=self.device, dtype=torch.long)
        query_start = self._query_rope_start(
            prefix_len=prefix_len,
            selected_chunk_count=len(selected_chunks),
            cache_len=cached_length + len(flat_chunk_ids),
        )
        query_positions = torch.arange(
            query_start,
            query_start + len(query_ids),
            device=self.device,
            dtype=torch.long,
        )
        input_ids = flat_chunk_ids + list(query_ids)
        position_ids = torch.cat([chunk_positions, query_positions], dim=0).unsqueeze(0)
        outputs = self.model(
            self._ids_tensor(input_ids),
            attention_mask=self._cached_prefix_mask(cached_length, len(input_ids)),
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=self._cache_positions(cached_length, len(input_ids)),
            use_cache=True,
            update_kvcache=len(flat_chunk_ids),
            return_dict=True,
        )
        return outputs.past_key_values, chunk_spans, flat_chunk_ids

    def _prefill_chunk_first_layer_recompute(
        self,
        chunk_ids: Sequence[int],
        scoring_query_ids: Sequence[int],
        *,
        prefix_len: int,
        chunk_order: int,
        keep_positions: Sequence[int],
    ):
        chunk_ids = list(chunk_ids)
        query_ids = self._window_query(scoring_query_ids, self.config.token_score_query_window)
        input_ids = chunk_ids + list(query_ids)
        chunk_len = len(chunk_ids)
        query_len = len(query_ids)
        if chunk_len <= 0:
            return None, self._chunk_rope_start(prefix_len, chunk_order, 0), []
        if not input_ids:
            return None, self._chunk_rope_start(prefix_len, chunk_order, 0), []

        keep = torch.tensor(list(keep_positions), device=self.device, dtype=torch.long)
        if keep.numel() == 0:
            return None, self._chunk_rope_start(prefix_len, chunk_order, chunk_len), []

        dream_model = self._base_dream_model()
        base = dream_model.model
        cache = DynamicCache()
        rope_start = self._chunk_rope_start(prefix_len, chunk_order, chunk_len)
        query_start = self._chunk_query_rope_start(rope_start, chunk_len)
        chunk_positions = torch.arange(
            rope_start,
            rope_start + chunk_len,
            device=self.device,
            dtype=torch.long,
        )
        query_positions = torch.arange(
            query_start,
            query_start + query_len,
            device=self.device,
            dtype=torch.long,
        )
        joint_positions = torch.cat([chunk_positions, query_positions], dim=0).unsqueeze(0)
        joint_hidden = base.embed_tokens(self._ids_tensor(input_ids))
        joint_mask = self._full_visible_mask(len(input_ids))
        joint_cache_positions = torch.arange(len(input_ids), device=self.device, dtype=torch.long)

        layer0 = base.layers[0]
        layer0_outputs = layer0(
            joint_hidden,
            attention_mask=joint_mask,
            update_kvcache=len(input_ids),
            position_ids=joint_positions,
            past_key_value=cache,
            output_attentions=True,
            use_cache=True,
            cache_position=joint_cache_positions,
            position_embeddings=base.rotary_emb(joint_hidden, joint_positions),
        )
        layer0_hidden = layer0_outputs[0]
        attentions = layer0_outputs[1]
        if attentions is not None and query_len > 0 and self.config.token_capacity > 0:
            query_to_chunk = attentions[0, :, chunk_len:chunk_len + query_len, :chunk_len].float()
            if query_to_chunk.numel() > 0:
                if self.config.token_score_reduce == "mean":
                    token_scores = query_to_chunk.mean(dim=(0, 1))
                else:
                    token_scores = query_to_chunk.sum(dim=(0, 1))
                topk = min(int(self.config.token_capacity), chunk_len)
                keep = torch.topk(token_scores, k=topk, largest=True).indices.sort().values

        keep = keep.clamp(min=0, max=max(0, chunk_len - 1)).unique(sorted=True)
        if keep.numel() == 0:
            return None, rope_start, []

        cache.key_cache[0] = cache.key_cache[0].index_select(2, keep)
        cache.value_cache[0] = cache.value_cache[0].index_select(2, keep)
        hidden_states = layer0_hidden.index_select(1, keep)
        kept_positions = chunk_positions.index_select(0, keep).unsqueeze(0)
        kept_cache_positions = torch.arange(keep.numel(), device=self.device, dtype=torch.long)
        kept_mask = self._full_visible_mask(int(keep.numel()))

        for layer in base.layers[1:]:
            layer_outputs = layer(
                hidden_states,
                attention_mask=kept_mask,
                update_kvcache=int(keep.numel()),
                position_ids=kept_positions,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=kept_cache_positions,
                position_embeddings=base.rotary_emb(hidden_states, kept_positions),
            )
            hidden_states = layer_outputs[0]

        return cache, rope_start, keep.tolist()

    def _prune_recent_cache_span(
        self,
        past_key_values,
        start: int,
        end: int,
        keep_positions: Sequence[int],
    ) -> int:
        if past_key_values is None or start >= end:
            return 0
        keep = torch.tensor(list(keep_positions), device=self.device, dtype=torch.long)
        original = end - start
        if keep.numel() >= original:
            return 0

        new_keys = []
        new_values = []
        for layer_k, layer_v in zip(past_key_values.key_cache, past_key_values.value_cache):
            prefix_k = layer_k[:, :, :start, :]
            prefix_v = layer_v[:, :, :start, :]
            block_k = layer_k[:, :, start:end, :].index_select(2, keep)
            block_v = layer_v[:, :, start:end, :].index_select(2, keep)
            suffix_k = layer_k[:, :, end:, :]
            suffix_v = layer_v[:, :, end:, :]
            new_keys.append(torch.cat([prefix_k, block_k, suffix_k], dim=2))
            new_values.append(torch.cat([prefix_v, block_v, suffix_v], dim=2))

        past_key_values.key_cache = new_keys
        past_key_values.value_cache = new_values
        return original - int(keep.numel())

    def _keep_positions_for_chunk(
        self,
        chunk_ids: Sequence[int],
        query_ids: Sequence[int],
    ) -> List[int]:
        capacity = int(self.config.token_capacity or 0)
        chunk_len = len(chunk_ids)
        if capacity <= 0 or chunk_len <= capacity:
            return list(range(chunk_len))

        token_scores = self.query_attention_token_scores(
            chunk_ids,
            query_ids,
            query_window=self.config.token_score_query_window,
            layer_window=self.config.token_score_layers,
            layer_mode=self.config.token_score_layer_mode,
            reduce_mode=self.config.token_score_reduce,
        )
        if token_scores.numel() != chunk_len:
            head = capacity // 2
            tail = capacity - head
            return sorted(set(list(range(head)) + list(range(chunk_len - tail, chunk_len))))

        topk = min(capacity, chunk_len)
        return torch.topk(token_scores, k=topk, largest=True).indices.sort().values.tolist()

    def build_prefill_cache(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        scoring_query_ids: Optional[Sequence[int]] = None,
    ):
        scoring_query_ids = list(scoring_query_ids if scoring_query_ids is not None else query_ids)
        candidate_chunks = split_token_chunks(
            context_ids,
            self.config.chunk_size,
            split_from_tail=self.config.split_from_tail,
        )
        candidate_chunks = [
            self._maybe_prepend_bos_to_chunk(chunk_ids)
            for chunk_ids in candidate_chunks
        ]
        selection_query_ids = scoring_query_ids
        if self.config.score_mode == "draft_self_information" and self.config.score_draft_tokens > 0:
            draft_ids = self.generate_from_cache(
                past_key_values=None,
                cache_tokens=0,
                query_ids=scoring_query_ids,
                query_rope_start=0,
                max_new_tokens=int(self.config.score_draft_tokens),
                steps=int(self.config.score_draft_steps or self.config.score_draft_tokens),
            )
            selection_query_ids = scoring_query_ids + draft_ids
        selected_indices, chunk_scores = self.select_chunks(candidate_chunks, selection_query_ids)

        use_first_layer_recompute = self.config.token_eviction_mode == "first_layer_recompute"
        if self.config.token_eviction_mode not in {"cache_slice", "first_layer_recompute"}:
            raise ValueError(f"Unsupported token_eviction_mode: {self.config.token_eviction_mode}")
        chunk_cache_mode = str(self.config.chunk_cache_mode or "independent").lower()
        if chunk_cache_mode not in {"independent", "sequential", "joint_selected"}:
            raise ValueError(f"Unsupported chunk_cache_mode: {self.config.chunk_cache_mode}")
        if use_first_layer_recompute and chunk_cache_mode == "joint_selected":
            raise ValueError("joint_selected chunk_cache_mode does not support first_layer_recompute")

        past_key_values = None
        cache_parts = []
        cache_len = 0
        prefix_cache = None
        prefix_ids = list(prefix_ids)
        if prefix_ids:
            past_key_values = self._prefill_ids(
                prefix_ids,
                past_key_values=past_key_values,
                cached_length=cache_len,
                rope_start=0,
            )
            prefix_cache = self._slice_past_key_values(past_key_values, 0, len(prefix_ids))
            if use_first_layer_recompute or chunk_cache_mode in {"independent", "joint_selected"}:
                cache_parts.append(prefix_cache)
            cache_len += len(prefix_ids)

        chunk_meta: List[ChunkPrefillMeta] = []
        total_removed = 0
        if chunk_cache_mode == "joint_selected":
            selected_chunks = [list(candidate_chunks[idx]) for idx in selected_indices]
            joint_prefix_cache = (
                self._slice_past_key_values(prefix_cache, 0, len(prefix_ids))
                if prefix_cache is not None
                else None
            )
            joint_cache, chunk_spans, _ = self._prefill_joint_selected_chunks_with_query(
                selected_chunks,
                scoring_query_ids,
                past_key_values=joint_prefix_cache,
                cached_length=cache_len,
                prefix_len=len(prefix_ids),
            )
            for chunk_order, (chunk_index, chunk_ids) in enumerate(zip(selected_indices, selected_chunks)):
                keep_positions = self._keep_positions_for_chunk(chunk_ids, scoring_query_ids)
                rel_start, rel_end, rope_start, rope_end = chunk_spans[chunk_order]
                chunk_cache = self._slice_past_key_values(
                    joint_cache,
                    len(prefix_ids) + rel_start,
                    len(prefix_ids) + rel_end,
                )
                removed = self._prune_recent_cache_span(
                    chunk_cache,
                    0,
                    len(chunk_ids),
                    keep_positions,
                )
                kept_count = len(chunk_ids) - removed
                cache_start = cache_len
                if chunk_cache is not None and kept_count > 0:
                    cache_parts.append(chunk_cache)
                cache_len += kept_count
                total_removed += removed
                chunk_meta.append(
                    ChunkPrefillMeta(
                        chunk_index=chunk_index,
                        original_tokens=len(chunk_ids),
                        kept_tokens=kept_count,
                        removed_tokens=removed,
                        cache_start=cache_start,
                        cache_end=cache_len,
                        rope_start=rope_start,
                        rope_end=rope_end,
                        score=chunk_scores.get(chunk_index),
                        kept_positions=list(keep_positions),
                    )
                )
        else:
            for chunk_order, chunk_index in enumerate(selected_indices):
                chunk_ids = list(candidate_chunks[chunk_index])
                keep_positions = self._keep_positions_for_chunk(chunk_ids, scoring_query_ids)
                cache_start = cache_len
                if use_first_layer_recompute:
                    chunk_cache, rope_start, keep_positions = self._prefill_chunk_first_layer_recompute(
                        chunk_ids,
                        scoring_query_ids,
                        prefix_len=len(prefix_ids),
                        chunk_order=chunk_order,
                        keep_positions=keep_positions,
                    )
                    kept_count = len(keep_positions)
                    removed = max(0, len(chunk_ids) - kept_count)
                    if chunk_cache is not None:
                        cache_parts.append(chunk_cache)
                elif chunk_cache_mode == "independent":
                    chunk_prefix_cache = (
                        self._slice_past_key_values(prefix_cache, 0, len(prefix_ids))
                        if prefix_cache is not None
                        else None
                    )
                    full_chunk_cache, rope_start = self._prefill_chunk_with_query(
                        chunk_ids,
                        scoring_query_ids,
                        past_key_values=chunk_prefix_cache,
                        cached_length=len(prefix_ids),
                        prefix_len=len(prefix_ids),
                        chunk_order=chunk_order,
                    )
                    chunk_cache = self._slice_past_key_values(
                        full_chunk_cache,
                        len(prefix_ids),
                        len(prefix_ids) + len(chunk_ids),
                    )
                    removed = self._prune_recent_cache_span(
                        chunk_cache,
                        0,
                        len(chunk_ids),
                        keep_positions,
                    )
                    kept_count = len(chunk_ids) - removed
                    if chunk_cache is not None and kept_count > 0:
                        cache_parts.append(chunk_cache)
                else:
                    past_key_values, rope_start = self._prefill_chunk_with_query(
                        chunk_ids,
                        scoring_query_ids,
                        past_key_values=past_key_values,
                        cached_length=cache_len,
                        prefix_len=len(prefix_ids),
                        chunk_order=chunk_order,
                    )
                    removed = self._prune_recent_cache_span(
                        past_key_values,
                        cache_start,
                        cache_start + len(chunk_ids),
                        keep_positions,
                    )
                    kept_count = len(chunk_ids) - removed
                cache_len += len(chunk_ids) - removed
                total_removed += removed
                chunk_meta.append(
                    ChunkPrefillMeta(
                        chunk_index=chunk_index,
                        original_tokens=len(chunk_ids),
                        kept_tokens=kept_count,
                        removed_tokens=removed,
                        cache_start=cache_start,
                        cache_end=cache_len,
                        rope_start=rope_start,
                        rope_end=rope_start + len(chunk_ids),
                        score=chunk_scores.get(chunk_index),
                        kept_positions=list(keep_positions),
                    )
                )

        if use_first_layer_recompute or chunk_cache_mode in {"independent", "joint_selected"}:
            past_key_values = self._concat_past_key_values(cache_parts)

        query_rope_start = self._query_rope_start(
            prefix_len=len(prefix_ids),
            selected_chunk_count=len(selected_indices),
            cache_len=cache_len,
        )
        return past_key_values, {
            "selected_indices": selected_indices,
            "chunk_scores": chunk_scores,
            "prefix_tokens": len(prefix_ids),
            "query_rope_start": query_rope_start,
            "raw_context_tokens": len(context_ids),
            "candidate_chunks": len(candidate_chunks),
            "cache_tokens": cache_len,
            "removed_tokens": total_removed,
            "chunk_meta": chunk_meta,
        }

    def _mask_token_id(self) -> int:
        config = getattr(self.model, "config", None)
        mask_token_id = getattr(config, "mask_token_id", None)
        if mask_token_id is None:
            generation_config = getattr(self.model, "generation_config", None)
            mask_token_id = getattr(generation_config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Dream mask_token_id is not available")
        return int(mask_token_id)

    def generate_from_cache(
        self,
        *,
        past_key_values,
        cache_tokens: int,
        query_ids: Sequence[int],
        query_rope_start: int,
        max_new_tokens: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> List[int]:
        mask_token_id = self._mask_token_id()
        query_ids = list(query_ids)
        max_new = int(max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens)
        steps = int(
            steps
            if steps is not None
            else (self.config.steps if self.config.steps is not None else max_new)
        )
        x = torch.tensor(
            [query_ids + [mask_token_id] * max_new],
            device=self.device,
            dtype=torch.long,
        )
        timesteps = torch.linspace(1, 1e-3, steps + 1, device=self.device)
        position_ids = self._position_ids(query_rope_start, x.shape[1])
        cache_position = self._cache_positions(cache_tokens, x.shape[1])
        attention_mask = self._cached_prefix_mask(cache_tokens, x.shape[1])

        for step in range(steps):
            mask_index = x == mask_token_id
            if past_key_values is None or cache_tokens <= 0:
                outputs = self.model(
                    x,
                    attention_mask=self._full_visible_mask(x.shape[1]),
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            else:
                outputs = self.model(
                    x,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    update_kvcache=0,
                    return_dict=True,
                )
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]
            t = timesteps[step]
            s = timesteps[step + 1]

            if self.config.alg == "origin":
                p_transfer = 1 - s / t if step < steps - 1 else 1
                x0 = torch.full_like(x[mask_index], mask_token_id)
                transfer = torch.rand(*x0.shape, device=self.device) < p_transfer
                if transfer.any():
                    _, sampled = sample_tokens(
                        mask_logits[transfer],
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                    )
                    x0[transfer] = sampled
                x[mask_index] = x0.clone()
                continue

            if self.config.alg == "maskgit_plus":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                )
            elif self.config.alg == "topk_margin":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    margin_confidence=True,
                )
            elif self.config.alg == "entropy":
                confidence, x0 = sample_tokens(
                    mask_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"Unknown alg: {self.config.alg}")

            num_mask_token = mask_index.sum() / mask_index.shape[0]
            transfer_count = int(num_mask_token * (1 - s / t)) if step < steps - 1 else int(num_mask_token)
            full_confidence = torch.full_like(x, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            if transfer_count > 0:
                if self.config.alg_temp is None or self.config.alg_temp == 0:
                    _, transfer_index = torch.topk(full_confidence, transfer_count)
                else:
                    scaled = F.softmax(full_confidence / self.config.alg_temp, dim=-1)
                    transfer_index = torch.multinomial(scaled, num_samples=transfer_count)
                x_candidate = torch.full_like(x, mask_token_id, device=self.device)
                x_candidate[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=self.device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_candidate[row_indices, transfer_index]

        return x[0, len(query_ids):].tolist()

    def run(
        self,
        *,
        prefix_ids: Sequence[int],
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        scoring_query_ids: Optional[Sequence[int]] = None,
    ) -> DreamPrefillKVResult:
        with torch.inference_mode():
            past_key_values, meta = self.build_prefill_cache(
                prefix_ids=prefix_ids,
                context_ids=context_ids,
                query_ids=query_ids,
                scoring_query_ids=scoring_query_ids,
            )
            generated_ids = self.generate_from_cache(
                past_key_values=past_key_values,
                cache_tokens=meta["cache_tokens"],
                query_ids=query_ids,
                query_rope_start=meta["query_rope_start"],
            )

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and eos in text:
            text = text.split(eos)[0]

        return DreamPrefillKVResult(
            text=text,
            sequences=generated_ids,
            selected_chunk_indices=meta["selected_indices"],
            chunk_scores=meta["chunk_scores"],
            prefix_tokens=meta["prefix_tokens"],
            query_tokens=len(query_ids),
            raw_context_tokens=meta["raw_context_tokens"],
            candidate_chunks=meta["candidate_chunks"],
            cache_tokens=meta["cache_tokens"],
            removed_tokens=meta["removed_tokens"],
            chunk_meta=meta["chunk_meta"],
        )


def dataclass_to_jsonable(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {
            key: dataclass_to_jsonable(getattr(obj, key))
            for key in obj.__dataclass_fields__
        }
    if isinstance(obj, dict):
        return {key: dataclass_to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj
