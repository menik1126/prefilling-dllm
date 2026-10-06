"""Query-attention token eviction for Dream ParallelComp chunk prefill.

Ports Prefilling-dLLM's global-granularity eviction: each chunk token is scored
by the attention it receives from the trailing scoring-query rows of the causal
``prefix + chunk + query`` forward, and only the top ``capacity`` tokens keep
their KV.
"""

from __future__ import annotations

from array import array
from typing import Any, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.nn.functional as F


class DllmTokenEvictionConfig(msgspec.Struct, frozen=True):
    capacity: int
    # Width of the per-head max pool over the chunk axis; 1 disables pooling.
    pool_kernel: int = 7
    # Chunks that start with BOS always retain it.
    force_keep_first: bool = True


class DllmTokenEvictionState(msgspec.Struct):
    """Eviction progress carried in a request's ParallelComp state."""

    config: DllmTokenEvictionConfig
    # Appended to every chunk in place of the official query while scoring.
    score_query_ids: List[int]
    # Per built chunk, its kept chunk-local positions in ascending order.
    kept_positions: List[List[int]] = []
    # The request's input ids before compaction, restored on retraction.
    full_input_ids: Optional[array] = None


class DllmTokenEvictionRequest(msgspec.Struct):
    config: DllmTokenEvictionConfig
    # KV slots of the shared prefix in sequence order; None without a prefix.
    prefix_kv_indices: Optional[torch.Tensor]
    chunk_lens: List[int]
    query_len: int
    # Per chunk, [chunk_len] float32 scores summed over the captured layers.
    score_sums: List[Optional[torch.Tensor]] = []


class DllmTokenEvictionCapture(msgspec.Struct):
    # One entry per batch row; None for rows that keep every chunk token.
    requests: List[Optional[DllmTokenEvictionRequest]]
    # Extend tokens per batch row, the row layout of the backend's q and k.
    extend_lens: List[int]
    num_layers: int = 0


# Prefilling-dLLM's token_score_query_window default.
DEFAULT_SCORE_QUERY_WINDOW = 8


def parse_token_eviction(
    config: Any, *, query_ids: Sequence[int]
) -> Optional[DllmTokenEvictionState]:
    """Validate ``dllm_parallelcomp.token_eviction``; None disables eviction."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("dllm_parallelcomp.token_eviction must be an object")
    capacity = config.get("capacity")
    pool_kernel = config.get("pool_kernel", 7)
    force_keep_first = config.get("force_keep_first", True)
    score_query_ids = config.get("score_query_ids")
    if (
        not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity <= 0
        or not isinstance(pool_kernel, int)
        or isinstance(pool_kernel, bool)
        or pool_kernel <= 0
        or not isinstance(force_keep_first, bool)
    ):
        raise ValueError(
            "dllm_parallelcomp.token_eviction requires a positive capacity, "
            "a positive pool_kernel, and a boolean force_keep_first"
        )
    if score_query_ids is None:
        score_query_ids = list(query_ids[-DEFAULT_SCORE_QUERY_WINDOW:])
    if (
        not isinstance(score_query_ids, list)
        or not score_query_ids
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool)
            for token_id in score_query_ids
        )
    ):
        raise ValueError(
            "dllm_parallelcomp.token_eviction needs a non-empty scoring query: "
            "pass score_query_ids or a non-empty query"
        )
    return DllmTokenEvictionState(
        config=DllmTokenEvictionConfig(
            capacity=capacity,
            pool_kernel=pool_kernel,
            force_keep_first=force_keep_first,
        ),
        score_query_ids=list(score_query_ids),
    )


def parallelcomp_chunk_query_len(state: dict) -> int:
    """Length of the temporary query appended to each chunk-stage item."""
    eviction = state["token_eviction"]
    if eviction is not None:
        return len(eviction.score_query_ids)
    return state["query_len"]


def build_token_eviction_capture(
    *, parallelcomp_states: Sequence[Optional[dict]], extend_lens: Sequence[int]
) -> Optional[DllmTokenEvictionCapture]:
    """Describe which chunk-stage rows of a forward need token scores."""
    requests: List[Optional[DllmTokenEvictionRequest]] = []
    for state in parallelcomp_states:
        eviction = None
        if state is not None and state["stage"] == "chunk":
            eviction = state["token_eviction"]
        if eviction is None:
            requests.append(None)
            continue
        cursor = state["chunk_cursor"]
        requests.append(
            DllmTokenEvictionRequest(
                config=eviction.config,
                prefix_kv_indices=state["common_prefix_indices"],
                chunk_lens=state["chunk_lens"][
                    cursor : cursor + state["chunk_batch_size"]
                ],
                query_len=len(eviction.score_query_ids),
            )
        )
    if all(request is None for request in requests):
        return None
    return DllmTokenEvictionCapture(requests=requests, extend_lens=list(extend_lens))


def split_kept_kv_indices(
    chunk_kv_indices: torch.Tensor, kept_positions: Sequence[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split one chunk's KV slots into (retained, evicted), both in order."""
    kept = torch.zeros(
        chunk_kv_indices.shape[0], dtype=torch.bool, device=chunk_kv_indices.device
    )
    kept[
        torch.as_tensor(
            kept_positions, dtype=torch.long, device=chunk_kv_indices.device
        )
    ] = True
    return chunk_kv_indices[kept], chunk_kv_indices[~kept]


def chunk_token_scores_for_layer(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    chunk_start: int,
    chunk_len: int,
    scaling: float,
    pool_kernel: int,
) -> torch.Tensor:
    """Score chunk tokens by the attention the trailing query rows give them.

    ``query`` is [query_len, num_q_heads, head_dim] and ``keys`` is
    [prefix_len + chunk_len + query_len, num_kv_heads, head_dim], both after
    RoPE. Returns [chunk_len] float32.
    """
    query_len, num_q_heads, _ = query.shape
    num_keys, num_kv_heads, _ = keys.shape
    q = query.float().permute(1, 0, 2)
    k = keys.float().permute(1, 0, 2)
    k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
    logits = torch.matmul(q, k.transpose(1, 2)) * scaling

    # Query row i is causal: it sees every key up to its own position.
    rows = torch.arange(query_len, device=query.device).unsqueeze(1)
    cols = torch.arange(num_keys, device=query.device).unsqueeze(0)
    logits = logits.masked_fill(cols > rows + (num_keys - query_len), float("-inf"))
    probs = torch.softmax(logits, dim=-1)

    head_scores = probs[:, :, chunk_start : chunk_start + chunk_len].sum(dim=1)
    kernel = max(1, min(pool_kernel, chunk_len))
    if kernel > 1:
        pooled = F.max_pool1d(
            head_scores.unsqueeze(1), kernel_size=kernel, padding=kernel // 2, stride=1
        )
        head_scores = pooled.squeeze(1)[..., :chunk_len]
    return head_scores.sum(dim=0)


def accumulate_token_eviction_layer(
    capture: DllmTokenEvictionCapture,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    key_buffer: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scaling: float,
) -> None:
    """Add one attention layer's chunk-token scores to the capture.

    ``q`` and ``k`` hold the forward's extend tokens in batch order;
    ``key_buffer`` is the layer's KV-pool key tensor, read for the prefix.
    """
    q = q.view(-1, num_q_heads, head_dim)
    k = k.view(-1, num_kv_heads, head_dim)
    token_offset = 0
    for request, extend_len in zip(capture.requests, capture.extend_lens):
        if request is not None:
            _accumulate_request_layer(
                request,
                q=q[token_offset : token_offset + extend_len],
                k=k[token_offset : token_offset + extend_len],
                key_buffer=key_buffer,
                scaling=scaling,
            )
        token_offset += extend_len
    capture.num_layers += 1


def _accumulate_request_layer(
    request: DllmTokenEvictionRequest,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    key_buffer: torch.Tensor,
    scaling: float,
) -> None:
    prefix_keys = k[:0]
    if request.prefix_kv_indices is not None:
        prefix_keys = key_buffer[request.prefix_kv_indices].to(k.dtype)
    prefix_len = prefix_keys.shape[0]

    if not request.score_sums:
        request.score_sums = [None] * len(request.chunk_lens)
    item_start = 0
    for chunk_order, chunk_len in enumerate(request.chunk_lens):
        item_end = item_start + chunk_len + request.query_len
        scores = chunk_token_scores_for_layer(
            query=q[item_start + chunk_len : item_end],
            keys=torch.cat([prefix_keys, k[item_start:item_end]], dim=0),
            chunk_start=prefix_len,
            chunk_len=chunk_len,
            scaling=scaling,
            pool_kernel=request.config.pool_kernel,
        )
        previous = request.score_sums[chunk_order]
        request.score_sums[chunk_order] = (
            scores if previous is None else previous + scores
        )
        item_start = item_end


def select_kept_token_positions(
    scores: torch.Tensor, *, capacity: int, force_keep_first: bool
) -> torch.Tensor:
    """Return the ascending chunk-local positions that survive eviction."""
    chunk_len = scores.shape[0]
    if chunk_len <= capacity:
        return torch.arange(chunk_len, device=scores.device)
    keep_count = max(1, capacity)
    if not force_keep_first:
        return torch.topk(scores, k=keep_count).indices.sort().values
    first = torch.zeros(1, dtype=torch.long, device=scores.device)
    if keep_count == 1:
        return first
    selected = torch.topk(scores[1:], k=keep_count - 1).indices + 1
    return torch.cat([first, selected]).sort().values


def select_token_eviction_keep(
    capture: Optional[DllmTokenEvictionCapture],
) -> Optional[List[Optional[List[List[int]]]]]:
    """Per batch row, per chunk in the forward, the kept chunk-local positions."""
    if capture is None:
        return None
    if capture.num_layers == 0:
        raise RuntimeError(
            "Dream token eviction captured no attention layer; the attention "
            "backend did not score the chunk forward"
        )
    keep_per_request: List[Optional[List[List[int]]]] = []
    for request in capture.requests:
        if request is None:
            keep_per_request.append(None)
            continue
        keep_per_request.append(
            [
                select_kept_token_positions(
                    score_sum / capture.num_layers,
                    capacity=request.config.capacity,
                    force_keep_first=request.config.force_keep_first,
                ).tolist()
                for score_sum in request.score_sums
            ]
        )
    return keep_per_request
