#!/usr/bin/env python3
"""Evaluate external query-aware chunk selection for Dream on MultiFieldQA-en.

The data format, prompt template, and QA F1 metric match the Prefilling-dLLM
LongBench evaluator. Chunk scoring is performed through a causal Dream
``/generate`` endpoint and can batch independent candidates. Generation may
use a separate PrefillingDream endpoint so its compressed prompt keeps the
reference engine's full-attention prefill semantics.

The evaluator supports both continuous and reused chunk positions. With
``--server-chunk-prefill``, it also sends exact chunk boundaries and RoPE starts
so SGLang can build every retained chunk independently from the common prefix.
Multiple isolated chunks can share one model forward without seeing each
other, controlled by ``--server-chunk-prefill-batch-size``. This experimental
causal construction is not equivalent to the reference ``full_prompt_mask``
mode; omit ``--server-chunk-prefill`` for accuracy-aligned evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

from transformers import AutoTokenizer

TASK = "multifieldqa_en"
PARTIAL_DRAFT_ROUNDS = 1


class PartialDraft(NamedTuple):
    token_ids: list[int]
    confirmed_mask: list[bool]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_selection_manifest(path: Path) -> dict[int, list[int]]:
    selections = {}
    for record in iter_jsonl(path):
        parallelcomp = record.get("parallelcomp", record)
        selected = parallelcomp.get("selected_chunk_indices")
        if selected is None:
            raise ValueError(f"Manifest record has no selected chunks: {record}")
        selections[int(record["index"])] = [int(index) for index in selected]
    return selections


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        punctuation = set(string.punctuation)
        return "".join(char for char in value if char not in punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, answers: Sequence[str]) -> float:
    return max((qa_f1_score(prediction, answer) for answer in answers), default=0.0)


def postprocess_prediction(
    prediction: str,
    max_words: int = 0,
    stop_at_answer_boundary: bool = False,
) -> str:
    if stop_at_answer_boundary:
        prediction = re.split(r"[;\n]", prediction, maxsplit=1)[0]
    if max_words > 0:
        prediction = " ".join(prediction.split()[:max_words])
    return prediction


def render_prompt_parts(template: str, example: dict[str, Any]) -> dict[str, str]:
    sentinel = "__LONGBENCH_CONTEXT_SENTINEL__"
    rendered = template.format(
        context=sentinel,
        input=example.get("input", ""),
    )
    if sentinel not in rendered:
        raise ValueError("LongBench prompt template is missing a {context} slot")
    prefix, query = rendered.split(sentinel, 1)
    return {
        "prefix": prefix,
        "context": example.get("context", ""),
        "query": query,
    }


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


class SGLangClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        causal_prompt_logprobs: bool = False,
    ):
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.generate_url = f"{base_url}/generate"
        self.timeout = timeout
        self.causal_prompt_logprobs = causal_prompt_logprobs

    def post(self, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang returned HTTP {error.code}: {body}") from error

    def prompt_logprobs(
        self,
        input_ids: Sequence[Sequence[int]],
        logprob_start_lens: Sequence[int],
    ) -> list[list[Any]]:
        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": 0,
        }
        if self.causal_prompt_logprobs:
            sampling_params["custom_params"] = {
                "dream_causal_prompt_logprob": True,
            }
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": sampling_params,
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": list(logprob_start_lens),
            }
        )
        rows = result if isinstance(result, list) else [result]
        return [row["meta_info"]["input_token_logprobs"] for row in rows]

    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        position_start: int | None = None,
        position_offset: int = 0,
        custom_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sampling_params = self._generation_sampling_params(
            max_new_tokens,
            position_start,
            position_offset,
            custom_params,
        )
        return self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": sampling_params,
            }
        )

    @staticmethod
    def _generation_sampling_params(
        max_new_tokens: int,
        position_start: int | None,
        position_offset: int,
        custom_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        }
        request_custom_params = dict(custom_params or {})
        if position_start is not None and position_offset:
            request_custom_params.update(
                {
                    "dllm_position_start": position_start,
                    "dllm_position_offset": position_offset,
                }
            )
        if request_custom_params:
            sampling_params["custom_params"] = request_custom_params
        return sampling_params

    def generate_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        *,
        position_starts: Sequence[int | None] | None = None,
        position_offsets: Sequence[int] | None = None,
        custom_params: Sequence[dict[str, Any] | None] | None = None,
    ) -> list[dict[str, Any]]:
        batch_size = len(input_ids)
        if batch_size == 0:
            return []
        if position_starts is None:
            position_starts = [None] * batch_size
        if position_offsets is None:
            position_offsets = [0] * batch_size
        if custom_params is None:
            custom_params = [None] * batch_size
        for name, values in (
            ("position_starts", position_starts),
            ("position_offsets", position_offsets),
            ("custom_params", custom_params),
        ):
            if len(values) != batch_size:
                raise ValueError(
                    f"{name} has {len(values)} rows, expected {batch_size}"
                )
        if batch_size == 1:
            return [
                self.generate(
                    input_ids[0],
                    max_new_tokens,
                    position_start=position_starts[0],
                    position_offset=position_offsets[0],
                    custom_params=custom_params[0],
                )
            ]

        sampling_params = [
            self._generation_sampling_params(
                max_new_tokens,
                position_start,
                position_offset,
                request_custom_params,
            )
            for position_start, position_offset, request_custom_params in zip(
                position_starts,
                position_offsets,
                custom_params,
                strict=True,
            )
        ]
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": sampling_params,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != batch_size:
            raise RuntimeError(
                "Dream final generation returned "
                f"{len(rows)} rows, expected {batch_size}"
            )
        invalid_rows = [
            index for index, row in enumerate(rows) if not isinstance(row, dict)
        ]
        if invalid_rows:
            raise RuntimeError(
                "Dream final generation returned non-object rows at indices "
                f"{invalid_rows}"
            )
        return rows

    def partial_draft(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        *,
        rounds: int = PARTIAL_DRAFT_ROUNDS,
    ) -> PartialDraft:
        if rounds < 0:
            raise ValueError("partial draft rounds must be non-negative")
        if max_new_tokens <= 0:
            return PartialDraft([], [])
        result = self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "custom_params": {
                        "dllm_partial_draft": {"rounds": rounds},
                    },
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        return self._partial_draft_from_result(result, max_new_tokens, rounds)

    def partial_draft_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        *,
        rounds: int = PARTIAL_DRAFT_ROUNDS,
    ) -> list[PartialDraft]:
        if rounds < 0:
            raise ValueError("partial draft rounds must be non-negative")
        if max_new_tokens <= 0:
            return [PartialDraft([], []) for _ in input_ids]
        if not input_ids:
            return []
        if len(input_ids) == 1:
            return [
                self.partial_draft(
                    input_ids[0],
                    max_new_tokens,
                    rounds=rounds,
                )
            ]
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                    "custom_params": {
                        "dllm_partial_draft": {"rounds": rounds},
                    },
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(input_ids):
            raise RuntimeError(
                "Dream partial draft generation returned "
                f"{len(rows)} rows, expected {len(input_ids)}"
            )
        return [
            self._partial_draft_from_result(row, max_new_tokens, rounds) for row in rows
        ]

    @staticmethod
    def _partial_draft_from_result(
        result: dict[str, Any], max_new_tokens: int, rounds: int
    ) -> PartialDraft:
        output_ids = result.get("output_ids", [])
        if output_ids and isinstance(output_ids[0], list):
            if len(output_ids) != 1:
                raise RuntimeError(
                    "Dream partial draft response contains multiple output-id rows"
                )
            output_ids = output_ids[0]
        token_ids = [int(token_id) for token_id in output_ids]
        if len(token_ids) != max_new_tokens:
            raise RuntimeError(
                "Dream partial draft generation returned "
                f"{len(token_ids)} slots, expected {max_new_tokens}"
            )

        meta_info = result.get("meta_info")
        if not isinstance(meta_info, dict) or "dllm_confirmed_mask" not in meta_info:
            raise RuntimeError(
                "Dream partial draft response is missing "
                "meta_info.dllm_confirmed_mask"
            )
        confirmed_mask = meta_info["dllm_confirmed_mask"]
        if not isinstance(confirmed_mask, list):
            raise RuntimeError("Dream partial draft confirmed mask must be a list")
        if len(confirmed_mask) != max_new_tokens:
            raise RuntimeError(
                "Dream partial draft confirmed mask has "
                f"{len(confirmed_mask)} slots, expected {max_new_tokens}"
            )
        if any(type(value) is not bool for value in confirmed_mask):
            raise RuntimeError(
                "Dream partial draft confirmed mask must contain only booleans"
            )
        expected_confirmed = min(max_new_tokens, 1 + rounds)
        if sum(confirmed_mask) != expected_confirmed:
            raise RuntimeError(
                "Dream partial draft confirmed "
                f"{sum(confirmed_mask)} slots, expected {expected_confirmed}"
            )
        if max_new_tokens and not confirmed_mask[0]:
            raise RuntimeError("Dream partial draft must confirm slot zero")
        return PartialDraft(token_ids, list(confirmed_mask))

    def draft_ids(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        generation_block_size: int,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        request_tokens = max(max_new_tokens, generation_block_size)
        result = self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": request_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        return self._draft_ids_from_result(result, max_new_tokens)

    def draft_ids_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        max_new_tokens: int,
        generation_block_size: int,
    ) -> list[list[int]]:
        if max_new_tokens <= 0:
            return [[] for _ in input_ids]
        if not input_ids:
            return []
        if len(input_ids) == 1:
            return [
                self.draft_ids(
                    input_ids[0],
                    max_new_tokens,
                    generation_block_size,
                )
            ]
        request_tokens = max(max_new_tokens, generation_block_size)
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": request_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
            }
        )
        rows = result if isinstance(result, list) else [result]
        if len(rows) != len(input_ids):
            raise RuntimeError(
                "Dream draft generation returned "
                f"{len(rows)} rows, expected {len(input_ids)}"
            )
        return [self._draft_ids_from_result(row, max_new_tokens) for row in rows]

    @staticmethod
    def _draft_ids_from_result(
        result: dict[str, Any], max_new_tokens: int
    ) -> list[int]:
        values = result["meta_info"].get("output_token_logprobs", [])
        token_ids = [
            int(value[1])
            for value in values[:max_new_tokens]
            if len(value) > 1 and value[1] is not None
        ]
        if not token_ids:
            # Diffusion generation returns the sampled ids at the top level but
            # currently leaves output_token_logprobs empty.  Falling back to
            # output_ids keeps draft-self-information scoring equivalent to
            # the reference engine instead of silently scoring without drafts.
            output_ids = result.get("output_ids", [])
            if output_ids and isinstance(output_ids[0], list):
                output_ids = output_ids[0]
            token_ids = [int(token_id) for token_id in output_ids[:max_new_tokens]]
        if len(token_ids) != max_new_tokens:
            raise RuntimeError(
                "Dream draft generation returned "
                f"{len(token_ids)} tokens, expected {max_new_tokens}"
            )
        return token_ids


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
) -> list[float]:
    return score_chunk_groups(
        client,
        [(prefix_ids, chunks, scoring_query_ids)],
        batch_size,
        score_token_masks=(
            [score_token_mask] if score_token_mask is not None else None
        ),
    )[0]


def score_chunk_groups(
    client: SGLangClient,
    groups: Sequence[tuple[Sequence[int], Sequence[Sequence[int]], Sequence[int]]],
    batch_size: int,
    score_token_masks: Sequence[Sequence[bool] | None] | None = None,
) -> list[list[float]]:
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
        for group_index, chunk_index in batch_coordinates:
            prefix_ids, chunks, scoring_query_ids = groups[group_index]
            chunk = chunks[chunk_index]
            rows.append(list(prefix_ids) + list(chunk) + list(scoring_query_ids))
            # SGLang's causal prompt-logprob path consumes the hidden state at
            # logprob_start_len and then shifts labels by one position.  Start
            # one token earlier so the first scoring-query token is included.
            logprob_starts.append(len(prefix_ids) + len(chunk) - 1)
            expected_tokens.append(len(scoring_query_ids))
            batch_score_token_masks.append(group_score_token_masks[group_index])
        logprob_rows = client.prompt_logprobs(rows, logprob_starts)
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
            f"coordinates {invalid}; Dream scoring requires a causal prompt-logprob "
            "server passed through --score-base-url."
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate external query-aware chunk selection on LongBench multifieldqa_en"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--score-base-url",
        help=(
            "Separate SGLang endpoint for causal Dream prompt logprobs; required "
            "by --selection-mode=query_logprob. The generation endpoint remains "
            "--base-url."
        ),
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--prompt-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--selection-mode",
        choices=["head", "query_logprob", "fixed", "manifest", "full"],
        default="query_logprob",
    )
    parser.add_argument(
        "--fixed-chunk-indices",
        default="",
        help="Comma-separated chunk indices used by --selection-mode=fixed.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="JSONL reference records containing index and selected_chunk_indices.",
    )
    parser.add_argument(
        "--position-mode", choices=["continuous", "reuse"], default="continuous"
    )
    parser.add_argument(
        "--query-position-mode",
        choices=[
            "after_compressed_context",
            "after_selected_chunks",
            "after_reused_window",
        ],
        default="after_compressed_context",
        help="Place query RoPE positions after real compressed tokens or fixed-size selected chunk slots.",
    )
    parser.add_argument(
        "--chunk-query-position-mode",
        choices=["after_reused_window", "after_chunk"],
        default="after_reused_window",
        help="Place each temporary chunk-conditioning query independently of the final query.",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--chunk-bos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument(
        "--selector-microbatch-size",
        type=int,
        default=1,
        help=(
            "Number of examples whose drafts and candidate chunks are batched "
            "together. One preserves the original per-example schedule."
        ),
    )
    parser.add_argument(
        "--generation-microbatch-size",
        type=int,
        default=1,
        help=(
            "Number of compressed prompts submitted together for final answer "
            "generation. One preserves the original scalar request shape. "
            "The effective batch is bounded by --selector-microbatch-size."
        ),
    )
    parser.add_argument(
        "--draft-tokens",
        type=int,
        default=0,
        help=(
            "Number of partial-draft slots appended to the scoring query. "
            "Positive values request one partial denoising round."
        ),
    )
    parser.add_argument("--generation-block-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prediction-max-words", type=int, default=0)
    parser.add_argument("--stop-at-answer-boundary", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--server-chunk-prefill",
        action="store_true",
        help=(
            "Build each selected chunk KV independently in SGLang, sharing only "
            "the template prefix and discarding temporary query KV. This causal "
            "experimental path is not equivalent to full_prompt_mask."
        ),
    )
    parser.add_argument(
        "--server-chunk-prefill-batch-size",
        type=int,
        default=4,
        help="Number of independently masked chunks packed into one server forward.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "Run draft generation and chunk scoring/selection, write their "
            "artifacts, and skip final answer generation."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.server_chunk_prefill_batch_size <= 0:
        raise ValueError("--server-chunk-prefill-batch-size must be positive")
    if args.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    if args.selector_microbatch_size <= 0:
        raise ValueError("selector_microbatch_size must be positive")
    if args.generation_microbatch_size <= 0:
        raise ValueError("generation_microbatch_size must be positive")
    if args.draft_tokens not in (0, 4):
        raise ValueError(
            "--draft-tokens must be 0 (disabled) or 4 for partial-draft selection"
        )
    if args.selection_only and args.dry_run:
        raise ValueError("--selection-only and --dry-run cannot be combined")
    if (
        args.selection_mode == "query_logprob"
        and not args.dry_run
        and args.score_base_url is None
    ):
        raise ValueError(
            "--selection-mode=query_logprob requires a dedicated "
            "--score-base-url for isolated causal scoring"
        )
    fixed_chunk_indices = [
        int(value) for value in args.fixed_chunk_indices.split(",") if value.strip()
    ]
    if args.selection_mode == "fixed" and not fixed_chunk_indices:
        raise ValueError("--selection-mode=fixed requires --fixed-chunk-indices")
    if args.selection_mode == "manifest" and args.selection_manifest is None:
        raise ValueError("--selection-mode=manifest requires --selection-manifest")
    selection_manifest = (
        load_selection_manifest(args.selection_manifest)
        if args.selection_manifest is not None
        else {}
    )

    with args.prompt_config.open("r", encoding="utf-8") as file:
        template = json.load(file)[TASK]
    examples = list(iter_jsonl(args.data_path))
    examples = examples[args.start_index :]
    if args.num_examples > 0:
        examples = examples[: args.num_examples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    client = None if args.dry_run else SGLangClient(args.base_url, args.timeout)
    score_client = (
        None
        if args.dry_run
        else SGLangClient(
            args.score_base_url or args.base_url,
            args.timeout,
            causal_prompt_logprobs=args.selection_mode == "query_logprob",
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"{TASK}_{args.selection_mode}_{args.position_mode}.jsonl"
    )
    metrics_path = (
        args.output_dir
        / f"{TASK}_{args.selection_mode}_{args.position_mode}_metrics.json"
    )
    prepared = []
    for local_index, example in enumerate(examples):
        index = args.start_index + local_index
        parts = render_prompt_parts(template, example)
        prefix_ids = tokenizer.encode(parts["prefix"], add_special_tokens=False)
        if tokenizer.bos_token_id is not None:
            prefix_ids = [tokenizer.bos_token_id] + prefix_ids
        context_ids = tokenizer.encode(parts["context"], add_special_tokens=False)
        query_ids = tokenizer.encode(parts["query"], add_special_tokens=False)
        chunks = split_token_chunks(context_ids, args.chunk_size)
        if args.chunk_bos:
            chunks = add_chunk_bos(chunks, tokenizer.bos_token_id, args.chunk_size)

        prepared.append(
            {
                "local_index": local_index,
                "index": index,
                "example": example,
                "prefix_ids": prefix_ids,
                "context_ids": context_ids,
                "query_ids": query_ids,
                "chunks": chunks,
            }
        )

    records = []
    total_draft_seconds = 0.0
    total_chunk_score_seconds = 0.0
    draft_request_count = 0
    score_request_count = 0
    shared_selector_timing = False
    active_microbatch_sizes = []
    generation_request_count = 0
    shared_generation_timing = False
    active_generation_microbatch_sizes = []
    total_generation_batch_seconds = 0.0
    for group_start in range(0, len(prepared), args.selector_microbatch_size):
        group = prepared[group_start : group_start + args.selector_microbatch_size]
        for state in group:
            state["draft_ids"] = []
            state["partial_draft_ids"] = []
            state["draft_confirmed_mask"] = []
            state["chunk_scores"] = None
            state["draft_seconds"] = 0.0
            state["chunk_score_seconds"] = 0.0
            state["score_seconds_are_attributed"] = False
            state["selector_active_microbatch_size"] = 0
            state["selector_scoring_skipped"] = None

        scoring_group = [
            state
            for state in group
            if requires_chunk_scoring(
                args.selection_mode,
                len(state["chunks"]),
                args.top_k,
            )
        ]
        if args.selection_mode == "query_logprob":
            for state in group:
                if state not in scoring_group:
                    state["selector_scoring_skipped"] = "top_k_covers_all"
        if scoring_group and client is not None:
            assert score_client is not None
            if args.draft_tokens > 0:
                draft_start = time.perf_counter()
                partial_drafts = client.partial_draft_batch(
                    [
                        state["prefix_ids"] + state["query_ids"]
                        for state in scoring_group
                    ],
                    args.draft_tokens,
                    rounds=PARTIAL_DRAFT_ROUNDS,
                )
                draft_groups = [draft.token_ids for draft in partial_drafts]
                draft_confirmed_masks = [
                    draft.confirmed_mask for draft in partial_drafts
                ]
                draft_seconds = time.perf_counter() - draft_start
                draft_request_count += 1
            else:
                draft_groups = [[] for _ in scoring_group]
                draft_confirmed_masks = [[] for _ in scoring_group]
                draft_seconds = 0.0

            scoring_query_groups = [
                state["query_ids"] + draft_ids
                for state, draft_ids in zip(scoring_group, draft_groups, strict=True)
            ]
            score_token_masks = (
                [
                    [True] * len(state["query_ids"]) + draft_confirmed_mask
                    for state, draft_confirmed_mask in zip(
                        scoring_group, draft_confirmed_masks, strict=True
                    )
                ]
                if args.draft_tokens > 0
                else None
            )
            chunk_score_start = time.perf_counter()
            chunk_score_groups = score_chunk_groups(
                score_client,
                [
                    (state["prefix_ids"], state["chunks"], scoring_query_ids)
                    for state, scoring_query_ids in zip(
                        scoring_group, scoring_query_groups, strict=True
                    )
                ],
                args.score_batch_size,
                score_token_masks=score_token_masks,
            )
            chunk_score_seconds = time.perf_counter() - chunk_score_start
            candidate_count = sum(len(state["chunks"]) for state in scoring_group)
            score_request_count += (
                candidate_count + args.score_batch_size - 1
            ) // args.score_batch_size
            total_draft_seconds += draft_seconds
            total_chunk_score_seconds += chunk_score_seconds

            active_microbatch_sizes.append(len(scoring_group))
            shared_selector_timing |= len(scoring_group) > 1
            draft_share = draft_seconds / len(scoring_group)
            for state, draft_ids, draft_confirmed_mask, chunk_scores in zip(
                scoring_group,
                draft_groups,
                draft_confirmed_masks,
                chunk_score_groups,
                strict=True,
            ):
                state["draft_ids"] = draft_ids
                state["partial_draft_ids"] = draft_ids
                state["draft_confirmed_mask"] = draft_confirmed_mask
                state["chunk_scores"] = chunk_scores
                state["draft_seconds"] = draft_share
                state["score_seconds_are_attributed"] = len(scoring_group) > 1
                state["selector_active_microbatch_size"] = len(scoring_group)
                state["chunk_score_seconds"] = (
                    chunk_score_seconds * len(state["chunks"]) / candidate_count
                    if candidate_count
                    else 0.0
                )

        for state in group:
            index = state["index"]
            prefix_ids = state["prefix_ids"]
            query_ids = state["query_ids"]
            chunks = state["chunks"]
            chunk_scores = state["chunk_scores"]

            if args.selection_mode in {"fixed", "manifest"}:
                requested_indices = (
                    fixed_chunk_indices
                    if args.selection_mode == "fixed"
                    else selection_manifest.get(index)
                )
                if requested_indices is None:
                    raise ValueError(
                        f"Selection manifest has no entry for index {index}"
                    )
                invalid = [value for value in requested_indices if value >= len(chunks)]
                if invalid:
                    raise ValueError(
                        f"Fixed chunk indices {invalid} exceed chunk count {len(chunks)}"
                    )
                selected = sorted(set(requested_indices))
            else:
                selected = select_chunk_indices(
                    args.selection_mode,
                    len(chunks),
                    args.top_k,
                    chunk_scores,
                )
            selected_context_ids = [
                token_id for chunk_index in selected for token_id in chunks[chunk_index]
            ]
            compressed_ids = prefix_ids + selected_context_ids + query_ids

            state["selected"] = selected
            state["compressed_ids"] = compressed_ids
            state["generation"] = None
            state["generation_seconds"] = 0.0
            state["generation_seconds_are_attributed"] = False
            state["generation_active_microbatch_size"] = 0
            state["query_position_offset"] = None
            state["generation_position_start"] = None
            state["generation_position_offset"] = 0
            state["generation_custom_params"] = None
            if client is None or args.selection_only:
                continue

            query_start = len(prefix_ids) + len(selected_context_ids)
            position_offset = 0
            if args.query_position_mode == "after_selected_chunks":
                query_rope_start = len(prefix_ids) + len(selected) * args.chunk_size
                position_offset = query_rope_start - query_start
                if position_offset < 0:
                    raise RuntimeError(
                        "Selected chunk slots end before the compressed query"
                    )
            elif args.query_position_mode == "after_reused_window":
                query_rope_start = len(prefix_ids) + args.chunk_size
                position_offset = query_rope_start - query_start
            else:
                query_rope_start = query_start

            custom_params = None
            if args.server_chunk_prefill:
                if args.position_mode == "reuse":
                    chunk_position_starts = [len(prefix_ids)] * len(selected)
                else:
                    chunk_position_starts = [
                        len(prefix_ids) + order * args.chunk_size
                        for order in range(len(selected))
                    ]
                if args.chunk_query_position_mode == "after_reused_window":
                    chunk_query_position_starts = [
                        len(prefix_ids) + args.chunk_size
                    ] * len(selected)
                else:
                    chunk_query_position_starts = [
                        start + len(chunks[chunk_index])
                        for start, chunk_index in zip(chunk_position_starts, selected)
                    ]
                custom_params = {
                    "dllm_parallelcomp": {
                        "prefix_len": len(prefix_ids),
                        "chunk_lens": [
                            len(chunks[chunk_index]) for chunk_index in selected
                        ],
                        "query_len": len(query_ids),
                        "chunk_batch_size": args.server_chunk_prefill_batch_size,
                        "chunk_position_starts": chunk_position_starts,
                        "chunk_query_position_starts": chunk_query_position_starts,
                        "query_position_start": query_rope_start,
                    }
                }
            state["query_position_offset"] = position_offset
            state["generation_position_start"] = (
                None if args.server_chunk_prefill else query_start
            )
            state["generation_position_offset"] = (
                0 if args.server_chunk_prefill else position_offset
            )
            state["generation_custom_params"] = custom_params

        if client is not None and not args.selection_only:
            for generation_start_index in range(
                0, len(group), args.generation_microbatch_size
            ):
                generation_group = group[
                    generation_start_index : generation_start_index
                    + args.generation_microbatch_size
                ]
                generation_start = time.perf_counter()
                generations = client.generate_batch(
                    [state["compressed_ids"] for state in generation_group],
                    args.max_new_tokens,
                    position_starts=[
                        state["generation_position_start"] for state in generation_group
                    ],
                    position_offsets=[
                        state["generation_position_offset"]
                        for state in generation_group
                    ],
                    custom_params=[
                        state["generation_custom_params"] for state in generation_group
                    ],
                )
                generation_seconds = time.perf_counter() - generation_start
                generation_request_count += 1
                total_generation_batch_seconds += generation_seconds
                active_generation_microbatch_sizes.append(len(generation_group))
                shared_generation_timing |= len(generation_group) > 1
                generation_share = generation_seconds / len(generation_group)
                for state, generation in zip(
                    generation_group, generations, strict=True
                ):
                    state["generation"] = generation
                    state["generation_seconds"] = generation_share
                    state["generation_seconds_are_attributed"] = (
                        len(generation_group) > 1
                    )
                    state["generation_active_microbatch_size"] = len(generation_group)

        for state in group:
            local_index = state["local_index"]
            index = state["index"]
            example = state["example"]
            prefix_ids = state["prefix_ids"]
            context_ids = state["context_ids"]
            query_ids = state["query_ids"]
            chunks = state["chunks"]
            draft_ids = state["draft_ids"]
            partial_draft_ids = state["partial_draft_ids"]
            draft_confirmed_mask = state["draft_confirmed_mask"]
            chunk_scores = state["chunk_scores"]
            score_seconds = state["draft_seconds"] + state["chunk_score_seconds"]
            selected = state["selected"]
            compressed_ids = state["compressed_ids"]
            generation = state["generation"]
            raw_prediction = (
                generation.get("text", "") if generation is not None else ""
            )
            prediction = postprocess_prediction(
                raw_prediction,
                max_words=args.prediction_max_words,
                stop_at_answer_boundary=args.stop_at_answer_boundary,
            )

            record = {
                "task": TASK,
                "example_id": example.get("_id", index),
                "index": index,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "answers": example.get("answers", []),
                "score": (
                    score_prediction(prediction, example.get("answers", []))
                    if client is not None and not args.selection_only
                    else None
                ),
                "selection_mode": args.selection_mode,
                "position_mode": args.position_mode,
                "query_position_mode": args.query_position_mode,
                "chunk_query_position_mode": args.chunk_query_position_mode,
                "query_position_offset": state["query_position_offset"],
                "chunk_size": args.chunk_size,
                "top_k": args.top_k,
                "score_batch_size": args.score_batch_size,
                "selector_microbatch_size": args.selector_microbatch_size,
                "selector_active_microbatch_size": state[
                    "selector_active_microbatch_size"
                ],
                "generation_microbatch_size": args.generation_microbatch_size,
                "generation_active_microbatch_size": state[
                    "generation_active_microbatch_size"
                ],
                "chunk_bos": args.chunk_bos,
                "server_chunk_prefill": args.server_chunk_prefill,
                "server_chunk_prefill_batch_size": (
                    args.server_chunk_prefill_batch_size
                ),
                "raw_context_tokens": len(context_ids),
                "candidate_chunks": len(chunks),
                "selected_chunk_indices": selected,
                "selected_original_position_starts": [
                    chunk_index * args.chunk_size for chunk_index in selected
                ],
                "selected_continuous_position_starts": [
                    order * args.chunk_size for order in range(len(selected))
                ],
                "chunk_scores": chunk_scores,
                "draft_ids": draft_ids,
                "partial_draft_ids": partial_draft_ids,
                "draft_confirmed_mask": draft_confirmed_mask,
                "selector_scoring_skipped": state["selector_scoring_skipped"],
                "prefix_tokens": len(prefix_ids),
                "query_tokens": len(query_ids),
                "compressed_prompt_tokens": len(compressed_ids),
                "generation_input_sha256": token_ids_sha256(compressed_ids),
                "score_seconds": score_seconds,
                "draft_seconds": state["draft_seconds"],
                "chunk_score_seconds": state["chunk_score_seconds"],
                "score_seconds_are_attributed": state["score_seconds_are_attributed"],
                "generation_seconds": state["generation_seconds"],
                "generation_seconds_are_attributed": state[
                    "generation_seconds_are_attributed"
                ],
                "generation_meta": (
                    generation.get("meta_info") if generation is not None else None
                ),
            }
            records.append(record)
            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            running_scores = [
                item["score"] for item in records if item["score"] is not None
            ]
            running = (
                100 * sum(running_scores) / len(running_scores)
                if running_scores
                else 0.0
            )
            print(
                f"[{local_index + 1}/{len(examples)}] index={index} "
                f"chunks={len(chunks)} selected={selected} score={running:.2f}",
                flush=True,
            )

    scores = [record["score"] for record in records if record["score"] is not None]
    metrics = {
        "task": TASK,
        "n": len(records),
        "score": round(100 * sum(scores) / len(scores), 2) if scores else None,
        "selection_mode": args.selection_mode,
        "position_mode": args.position_mode,
        "query_position_mode": args.query_position_mode,
        "chunk_query_position_mode": args.chunk_query_position_mode,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "score_batch_size": args.score_batch_size,
        "selector_microbatch_size": args.selector_microbatch_size,
        "average_selector_active_microbatch_size": (
            sum(active_microbatch_sizes) / len(active_microbatch_sizes)
            if active_microbatch_sizes
            else 0.0
        ),
        "max_selector_active_microbatch_size": (
            max(active_microbatch_sizes) if active_microbatch_sizes else 0
        ),
        "generation_microbatch_size": args.generation_microbatch_size,
        "average_generation_active_microbatch_size": (
            sum(active_generation_microbatch_sizes)
            / len(active_generation_microbatch_sizes)
            if active_generation_microbatch_sizes
            else 0.0
        ),
        "max_generation_active_microbatch_size": (
            max(active_generation_microbatch_sizes)
            if active_generation_microbatch_sizes
            else 0
        ),
        "generation_batch_size_histogram": {
            str(batch_size): count
            for batch_size, count in sorted(
                Counter(active_generation_microbatch_sizes).items()
            )
        },
        "server_chunk_prefill": args.server_chunk_prefill,
        "server_chunk_prefill_batch_size": args.server_chunk_prefill_batch_size,
        "draft_tokens": args.draft_tokens,
        "draft_partial_rounds": (
            PARTIAL_DRAFT_ROUNDS if args.draft_tokens > 0 else None
        ),
        "selection_only": args.selection_only,
        "prediction_max_words": args.prediction_max_words,
        "stop_at_answer_boundary": args.stop_at_answer_boundary,
        "average_raw_context_tokens": sum(
            record["raw_context_tokens"] for record in records
        )
        / len(records),
        "average_candidate_chunks": sum(
            record["candidate_chunks"] for record in records
        )
        / len(records),
        "average_compressed_prompt_tokens": sum(
            record["compressed_prompt_tokens"] for record in records
        )
        / len(records),
        "total_score_seconds": total_draft_seconds + total_chunk_score_seconds,
        "total_draft_seconds": total_draft_seconds,
        "total_chunk_score_seconds": total_chunk_score_seconds,
        "draft_request_count": draft_request_count,
        "score_request_count": score_request_count,
        "score_seconds_are_attributed": shared_selector_timing,
        "generation_request_count": generation_request_count,
        "generation_seconds_are_attributed": shared_generation_timing,
        "total_generation_seconds": total_generation_batch_seconds,
        "total_generation_batch_seconds": total_generation_batch_seconds,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
