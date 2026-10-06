"""Per-head token eviction for Dream's full-prompt generation path.

Ports Prefilling-dLLM's ``token_eviction_granularity=per_head`` with
``cache_build_mode=full_prompt_mask``: every oversized chunk is first scored by
a bidirectional ``prefix + chunk + scoring query`` forward, the full prompt is
then prefilled as usual, and each KV head of each layer keeps only its own top
``capacity`` chunk tokens before denoising starts.
"""

from __future__ import annotations

from array import array
from typing import Any, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.dllm.token_eviction import (
    DEFAULT_SCORE_QUERY_WINDOW,
    select_kept_token_positions,
)

STAGE_SCORE = "score"
STAGE_GENERATE = "generate"


class DllmHeadEvictionConfig(msgspec.Struct, frozen=True):
    capacity: int
    # Width of the per-head max pool over the chunk axis; 1 disables pooling.
    pool_kernel: int = 7
    # Chunks that start with BOS always retain it.
    force_keep_first: bool = True
    # Add chunk-to-query attention to the query-to-chunk score.
    bidirectional: bool = True


class DllmHeadEvictionState(msgspec.Struct):
    """Eviction progress carried by one generation request."""

    config: DllmHeadEvictionConfig
    prefix_len: int
    chunk_lens: List[int]
    query_len: int
    # Appended to each chunk in place of the full query while scoring.
    score_query_ids: List[int]
    stage: str = STAGE_GENERATE
    # Index of the chunk the next scoring forward covers.
    chunk_cursor: int = 0
    # Per chunk, [num_layers, num_kv_heads, capacity] ascending chunk-local
    # positions; None keeps the whole chunk.
    keep_positions: List[Optional[torch.Tensor]] = []
    # The request's input ids before compaction, restored on retraction.
    full_input_ids: Optional[array] = None

    def needs_scores(self, chunk_index: int) -> bool:
        return self.chunk_lens[chunk_index] > self.config.capacity

    def restart(self) -> None:
        self.keep_positions = [None] * len(self.chunk_lens)
        self.full_input_ids = None
        self.chunk_cursor = -1
        self.advance()

    def advance(self) -> None:
        """Move to the next chunk that needs scores, or on to generation."""
        cursor = self.chunk_cursor + 1
        while cursor < len(self.chunk_lens) and not self.needs_scores(cursor):
            cursor += 1
        self.chunk_cursor = cursor
        self.stage = STAGE_SCORE if cursor < len(self.chunk_lens) else STAGE_GENERATE

    @property
    def removed_per_chunk(self) -> List[int]:
        return [
            chunk_len - self.config.capacity if self.needs_scores(index) else 0
            for index, chunk_len in enumerate(self.chunk_lens)
        ]

    @property
    def compacted(self) -> bool:
        return self.full_input_ids is not None

    def full_token_index(self, compact_index: int) -> int:
        """Map an index of the compacted sequence back to the full prompt."""
        removed = 0
        chunk_end = self.prefix_len
        for chunk_len, chunk_removed in zip(self.chunk_lens, self.removed_per_chunk):
            chunk_end += chunk_len - chunk_removed
            if compact_index < chunk_end:
                break
            removed += chunk_removed
        return compact_index + removed


class DllmHeadEvictionRequest(msgspec.Struct):
    config: DllmHeadEvictionConfig
    prefix_len: int
    chunk_len: int
    query_len: int
    # Per captured layer, [num_kv_heads, capacity] kept chunk-local positions.
    layer_keeps: List[torch.Tensor] = []


class DllmHeadEvictionCapture(msgspec.Struct):
    # One entry per batch row; None for rows that are not scoring forwards.
    requests: List[Optional[DllmHeadEvictionRequest]]
    # Extend tokens per batch row, the row layout of the backend's q and k.
    extend_lens: List[int]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_head_token_eviction(
    config: Any, *, input_ids: Sequence[int]
) -> Optional[DllmHeadEvictionState]:
    """Validate the ``dllm_token_eviction`` custom param; None disables it."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("dllm_token_eviction must be an object")
    capacity = config.get("capacity")
    prefix_len = config.get("prefix_len")
    chunk_lens = config.get("chunk_lens")
    query_len = config.get("query_len")
    pool_kernel = config.get("pool_kernel", 7)
    score_query_window = config.get("score_query_window", DEFAULT_SCORE_QUERY_WINDOW)
    force_keep_first = config.get("force_keep_first", True)
    bidirectional = config.get("bidirectional", True)
    if (
        not _is_int(capacity)
        or capacity <= 0
        or not _is_int(prefix_len)
        or prefix_len < 0
        or not _is_int(query_len)
        or query_len <= 0
        or not isinstance(chunk_lens, list)
        or not chunk_lens
        or any(not _is_int(length) or length <= 0 for length in chunk_lens)
        or not _is_int(pool_kernel)
        or pool_kernel <= 0
        or not _is_int(score_query_window)
        or score_query_window <= 0
        or not isinstance(force_keep_first, bool)
        or not isinstance(bidirectional, bool)
    ):
        raise ValueError(
            "dllm_token_eviction requires a positive capacity, a non-negative "
            "prefix_len, a positive query_len, a non-empty list of positive "
            "chunk_lens, positive pool_kernel/score_query_window, and boolean "
            "force_keep_first/bidirectional"
        )
    if prefix_len + sum(chunk_lens) + query_len != len(input_ids):
        raise ValueError(
            "dllm_token_eviction boundaries do not cover the input: "
            f"prefix={prefix_len}, chunks={sum(chunk_lens)}, "
            f"query={query_len}, input={len(input_ids)}"
        )
    state = DllmHeadEvictionState(
        config=DllmHeadEvictionConfig(
            capacity=capacity,
            pool_kernel=pool_kernel,
            force_keep_first=force_keep_first,
            bidirectional=bidirectional,
        ),
        prefix_len=prefix_len,
        chunk_lens=list(chunk_lens),
        query_len=query_len,
        score_query_ids=list(input_ids[len(input_ids) - query_len :])[
            -score_query_window:
        ],
    )
    state.restart()
    return state


def score_stage_input_ids(state: DllmHeadEvictionState, input_ids: array) -> array:
    """``prefix + chunk + scoring query`` for the chunk under the cursor."""
    chunk_start = state.prefix_len + sum(state.chunk_lens[: state.chunk_cursor])
    chunk_end = chunk_start + state.chunk_lens[state.chunk_cursor]
    ids = array("q", input_ids[: state.prefix_len])
    ids.extend(input_ids[chunk_start:chunk_end])
    ids.extend(state.score_query_ids)
    return ids


def compact_input_ids(state: DllmHeadEvictionState, input_ids: array) -> array:
    """Shorten every evicted chunk to ``capacity`` ids, matching its KV span.

    Heads keep different tokens, so the surviving ids are placeholders; only
    their count is meaningful once the prompt KV is cached.
    """
    compact = array("q", input_ids[: state.prefix_len])
    cursor = state.prefix_len
    for chunk_len, removed in zip(state.chunk_lens, state.removed_per_chunk):
        compact.extend(input_ids[cursor : cursor + chunk_len - removed])
        cursor += chunk_len
    compact.extend(input_ids[cursor:])
    return compact


def build_head_eviction_capture(
    *, states: Sequence[Optional[DllmHeadEvictionState]], extend_lens: Sequence[int]
) -> Optional[DllmHeadEvictionCapture]:
    """Describe which rows of a forward are chunk-scoring forwards."""
    requests: List[Optional[DllmHeadEvictionRequest]] = []
    for state, extend_len in zip(states, extend_lens):
        if state is None or state.stage != STAGE_SCORE:
            requests.append(None)
            continue
        request = DllmHeadEvictionRequest(
            config=state.config,
            prefix_len=state.prefix_len,
            chunk_len=state.chunk_lens[state.chunk_cursor],
            query_len=len(state.score_query_ids),
        )
        row_len = request.prefix_len + request.chunk_len + request.query_len
        if extend_len != row_len:
            raise RuntimeError(
                "Dream token-eviction scoring needs the whole "
                f"prefix + chunk + query row in one forward: row={row_len}, "
                f"forward={extend_len}. Raise --chunked-prefill-size or pass "
                "--chunked-prefill-size -1."
            )
        requests.append(request)
    if all(request is None for request in requests):
        return None
    return DllmHeadEvictionCapture(requests=requests, extend_lens=list(extend_lens))


def head_keep_positions_for_layer(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    prefix_len: int,
    chunk_len: int,
    scaling: float,
    config: DllmHeadEvictionConfig,
) -> torch.Tensor:
    """Pick each KV head's kept chunk tokens from one unmasked attention layer.

    ``query`` is [seq_len, num_q_heads, head_dim] and ``keys`` is
    [seq_len, num_kv_heads, head_dim], both after RoPE, where the sequence is
    ``prefix + chunk + scoring query``. Returns [num_kv_heads, capacity]
    ascending chunk-local positions.
    """
    num_q_heads = query.shape[1]
    num_kv_heads = keys.shape[1]
    group_size = num_q_heads // num_kv_heads
    chunk_end = prefix_len + chunk_len
    kernel = max(1, min(config.pool_kernel, chunk_len))

    keeps = []
    for kv_head in range(num_kv_heads):
        q = query[:, kv_head * group_size : (kv_head + 1) * group_size]
        q = q.float().permute(1, 0, 2)
        k = keys[:, kv_head].float()
        probs = torch.softmax(torch.matmul(q, k.t()) * scaling, dim=-1)

        head_scores = probs[:, chunk_end:, prefix_len:chunk_end].sum(dim=1)
        if config.bidirectional:
            head_scores = head_scores + probs[:, prefix_len:chunk_end, chunk_end:].sum(
                dim=2
            )
        if kernel > 1:
            pooled = F.max_pool1d(
                head_scores.unsqueeze(1),
                kernel_size=kernel,
                padding=kernel // 2,
                stride=1,
            )
            head_scores = pooled.squeeze(1)[..., :chunk_len]
        keeps.append(
            select_kept_token_positions(
                head_scores.mean(dim=0),
                capacity=config.capacity,
                force_keep_first=config.force_keep_first,
            )
        )
    return torch.stack(keeps)


def accumulate_head_eviction_layer(
    capture: DllmHeadEvictionCapture,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scaling: float,
) -> None:
    """Record one attention layer's per-head keep table for every scoring row.

    ``q`` and ``k`` hold the forward's extend tokens in batch order.
    """
    q = q.view(-1, num_q_heads, head_dim)
    k = k.view(-1, num_kv_heads, head_dim)
    token_offset = 0
    for request, extend_len in zip(capture.requests, capture.extend_lens):
        if request is not None:
            request.layer_keeps.append(
                head_keep_positions_for_layer(
                    query=q[token_offset : token_offset + extend_len],
                    keys=k[token_offset : token_offset + extend_len],
                    prefix_len=request.prefix_len,
                    chunk_len=request.chunk_len,
                    scaling=scaling,
                    config=request.config,
                )
            )
        token_offset += extend_len


def stack_head_eviction_keep(
    capture: Optional[DllmHeadEvictionCapture], *, num_layers: int
) -> Optional[List[Optional[torch.Tensor]]]:
    """Per batch row, the [num_layers, num_kv_heads, capacity] keep table."""
    if capture is None:
        return None
    keep_per_request: List[Optional[torch.Tensor]] = []
    for request in capture.requests:
        if request is None:
            keep_per_request.append(None)
            continue
        if len(request.layer_keeps) != num_layers:
            raise RuntimeError(
                "Dream token eviction scored "
                f"{len(request.layer_keeps)} of {num_layers} attention layers; "
                "the attention backend did not score the chunk forward"
            )
        keep_per_request.append(torch.stack(request.layer_keeps))
    return keep_per_request


def compact_prompt_kv_per_head(
    *,
    kv_buffers: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    prompt_slots: torch.Tensor,
    state: DllmHeadEvictionState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move each KV head's kept chunk tokens into the chunk's leading slots.

    ``kv_buffers`` holds one (key, value) pool tensor per layer, each
    [num_slots, num_kv_heads, head_dim]. Keys carry their RoPE rotation, so a
    slot may hold a different token per head. Returns the prompt's surviving
    slots in order and the freed ones.
    """
    prompt_slots = prompt_slots.long()
    retained = [prompt_slots[: state.prefix_len]]
    evicted = []
    cursor = state.prefix_len
    for chunk_len, keep in zip(state.chunk_lens, state.keep_positions):
        chunk_slots = prompt_slots[cursor : cursor + chunk_len]
        cursor += chunk_len
        if keep is None:
            retained.append(chunk_slots)
            continue
        if keep.shape[0] != len(kv_buffers):
            raise RuntimeError(
                "Dream token eviction keep table does not cover the KV pool: "
                f"layers={keep.shape[0]}, pool={len(kv_buffers)}"
            )
        capacity = keep.shape[-1]
        target_slots = chunk_slots[:capacity]
        heads = torch.arange(keep.shape[1], device=keep.device).unsqueeze(1)
        for layer_keep, (key_buffer, value_buffer) in zip(keep, kv_buffers):
            # [num_kv_heads, capacity] source slot of every head's kept token.
            source_slots = chunk_slots[layer_keep]
            key_buffer[target_slots] = key_buffer[source_slots, heads].transpose(0, 1)
            value_buffer[target_slots] = value_buffer[source_slots, heads].transpose(
                0, 1
            )
        retained.append(target_slots)
        evicted.append(chunk_slots[capacity:])
    retained.append(prompt_slots[cursor:])
    return (
        torch.cat(retained),
        torch.cat(evicted) if evicted else prompt_slots[:0],
    )
