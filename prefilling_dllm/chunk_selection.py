"""Chunking, query-logprob chunk scoring, and top-k chunk selection."""

from __future__ import annotations

import math
from typing import Any, Sequence

from prefilling_dllm.client import SGLangClient


def split_token_chunks(token_ids: Sequence[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ids = list(token_ids)
    return [ids[start : start + chunk_size] for start in range(0, len(ids), chunk_size)]


def add_chunk_bos(
    chunks: Sequence[Sequence[int]], bos_token_id: int | None, chunk_size: int
) -> list[list[int]]:
    if bos_token_id is None:
        return [list(chunk) for chunk in chunks]
    result = []
    for chunk in chunks:
        values = list(chunk)
        if not values or values[0] != bos_token_id:
            values = [bos_token_id] + values
        result.append(values[:chunk_size])
    return result


def mean_query_logprob(
    values: Sequence[Any],
    expected_tokens: int,
    score_token_mask: Sequence[bool] | None = None,
) -> float:
    if expected_tokens <= 0:
        raise ValueError("expected_tokens must be positive")
    if len(values) < expected_tokens:
        raise ValueError(
            "prompt logprob row has "
            f"{len(values)} tokens, expected at least {expected_tokens}"
        )
    if score_token_mask is None:
        score_token_mask = [True] * expected_tokens
    elif len(score_token_mask) != expected_tokens:
        raise ValueError(
            "score token mask has "
            f"{len(score_token_mask)} slots, expected {expected_tokens}"
        )
    elif any(type(value) is not bool for value in score_token_mask):
        raise ValueError("score token mask must contain only booleans")
    logprobs = []
    for target_offset, (value, should_score) in enumerate(
        zip(values[-expected_tokens:], score_token_mask, strict=True)
    ):
        if not should_score:
            continue
        if not value or value[0] is None:
            raise ValueError(
                "prompt logprob row is missing a scored target at offset "
                f"{target_offset}"
            )
        logprobs.append(float(value[0]))
    if not logprobs:
        return float("-inf")
    return sum(logprobs) / len(logprobs)


def score_chunks(
    client: SGLangClient,
    prefix_ids: Sequence[int],
    chunks: Sequence[Sequence[int]],
    scoring_query_ids: Sequence[int],
    batch_size: int,
    score_token_mask: Sequence[bool] | None = None,
    draft_len: int = 0,
) -> list[float]:
    return score_chunk_groups(
        client,
        [(prefix_ids, chunks, scoring_query_ids)],
        batch_size,
        score_token_masks=(
            [score_token_mask] if score_token_mask is not None else None
        ),
        draft_lens=[draft_len],
    )[0]


def score_chunk_groups(
    client: SGLangClient,
    groups: Sequence[tuple[Sequence[int], Sequence[Sequence[int]], Sequence[int]]],
    batch_size: int,
    score_token_masks: Sequence[Sequence[bool] | None] | None = None,
    draft_lens: Sequence[int] | None = None,
) -> list[list[float]]:
    # Trailing draft tokens of each scoring query; the scorer hides them from
    # the prefix and chunk rows.
    group_draft_lens = [0] * len(groups) if draft_lens is None else list(draft_lens)
    if len(group_draft_lens) != len(groups) or any(
        draft_len < 0 or draft_len > len(scoring_query_ids)
        for (_, _, scoring_query_ids), draft_len in zip(groups, group_draft_lens)
    ):
        raise ValueError("draft lengths do not match the scoring query groups")
    if score_token_masks is None:
        group_score_token_masks: list[Sequence[bool] | None] = [None] * len(groups)
    else:
        if len(score_token_masks) != len(groups):
            raise ValueError(
                "score token masks have "
                f"{len(score_token_masks)} groups, expected {len(groups)}"
            )
        group_score_token_masks = list(score_token_masks)
    for (_, _, scoring_query_ids), score_token_mask in zip(
        groups, group_score_token_masks, strict=True
    ):
        if score_token_mask is not None and len(score_token_mask) != len(
            scoring_query_ids
        ):
            raise ValueError(
                "score token mask length does not match scoring query length"
            )
        if score_token_mask is not None and any(
            type(value) is not bool for value in score_token_mask
        ):
            raise ValueError("score token mask must contain only booleans")
    scores: list[list[float | None]] = [[None] * len(chunks) for _, chunks, _ in groups]
    coordinates = [
        (group_index, chunk_index)
        for group_index, (_, chunks, _) in enumerate(groups)
        for chunk_index in range(len(chunks))
    ]
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        rows = []
        logprob_starts = []
        expected_tokens = []
        batch_score_token_masks = []
        prefix_lens = []
        chunk_lens = []
        query_lens = []
        batch_draft_lens = []
        for group_index, chunk_index in batch_coordinates:
            prefix_ids, chunks, scoring_query_ids = groups[group_index]
            chunk = chunks[chunk_index]
            rows.append(list(prefix_ids) + list(chunk) + list(scoring_query_ids))
            # SGLang's prompt-logprob path consumes the hidden state at
            # logprob_start_len and then shifts labels by one position.  Start
            # one token earlier so the first scoring-query token is included.
            logprob_starts.append(len(prefix_ids) + len(chunk) - 1)
            expected_tokens.append(len(scoring_query_ids))
            batch_score_token_masks.append(group_score_token_masks[group_index])
            prefix_lens.append(len(prefix_ids))
            chunk_lens.append(len(chunk))
            draft_len = group_draft_lens[group_index]
            query_lens.append(len(scoring_query_ids) - draft_len)
            batch_draft_lens.append(draft_len)
        logprob_rows = client.prompt_logprobs(
            rows,
            logprob_starts,
            prefix_lens=prefix_lens,
            chunk_lens=chunk_lens,
            query_lens=query_lens,
            draft_lens=batch_draft_lens,
        )
        if len(logprob_rows) != len(batch_coordinates):
            raise RuntimeError(
                "Chunk selector returned "
                f"{len(logprob_rows)} rows, expected {len(batch_coordinates)}"
            )
        for coordinate, values, token_count, score_token_mask in zip(
            batch_coordinates,
            logprob_rows,
            expected_tokens,
            batch_score_token_masks,
            strict=True,
        ):
            group_index, chunk_index = coordinate
            scores[group_index][chunk_index] = mean_query_logprob(
                values,
                token_count,
                score_token_mask,
            )
    invalid = [
        (group_index, chunk_index)
        for group_index, group_scores in enumerate(scores)
        for chunk_index, score in enumerate(group_scores)
        if score is None or not math.isfinite(score)
    ]
    if invalid:
        raise RuntimeError(
            "Chunk selector received non-finite prompt logprobs for candidate "
            f"coordinates {invalid}; the scoring endpoint did not return input "
            "token logprobs."
        )
    return [[float(score) for score in group_scores] for group_scores in scores]


def select_chunk_indices(
    mode: str,
    chunk_count: int,
    top_k: int,
    scores: Sequence[float] | None = None,
) -> list[int]:
    if mode == "full" or top_k <= 0 or top_k >= chunk_count:
        return list(range(chunk_count))
    if mode == "head":
        return list(range(min(top_k, chunk_count)))
    if scores is None or len(scores) != chunk_count:
        raise ValueError("query_logprob selection requires one score per chunk")
    ranked = sorted(range(chunk_count), key=lambda index: scores[index], reverse=True)
    return sorted(ranked[:top_k])


def requires_chunk_scoring(mode: str, chunk_count: int, top_k: int) -> bool:
    """Return whether scores can change the selected chunk set."""
    return mode == "query_logprob" and 0 < top_k < chunk_count
