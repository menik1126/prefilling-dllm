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
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer

# The pipeline package sits at the repository root, outside the sglang package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from prefilling_dllm.chunk_selection import (  # noqa: E402
    add_chunk_bos,
    requires_chunk_scoring,
    score_chunk_groups,
    select_chunk_indices,
    split_token_chunks,
)
from prefilling_dllm.client import (  # noqa: E402
    PARTIAL_DRAFT_ROUNDS,
    SGLangClient,
)

TASK = "multifieldqa_en"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate external query-aware chunk selection on LongBench multifieldqa_en"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--score-base-url",
        help=(
            "SGLang endpoint for chunk-scoring requests; defaults to --base-url, "
            "since a PrefillingDream server answers them itself. Point it at "
            "the prefill server (with --score-on-pd-server) when "
            "--base-url is a PD router."
        ),
    )
    parser.add_argument(
        "--score-attention-mask",
        choices=["full", "causal"],
        default="full",
        help=(
            "Attention used while scoring prefix+chunk+query+draft. full lets "
            "the prefix and chunk see prefix+chunk+query while the query and "
            "draft stay causal; it needs a scoring server launched with "
            "--disable-radix-cache. causal keeps the previous all-triangle path."
        ),
    )
    parser.add_argument(
        "--score-on-pd-server",
        action="store_true",
        help=(
            "--score-base-url is a PD prefill or decode server launched with "
            "a dLLM algorithm: send scoring requests with a bootstrap room so "
            "they finish on that server."
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
        "--token-capacity",
        type=int,
        default=0,
        help=(
            "Per-chunk KV budget for server-side token eviction; chunks longer "
            "than this keep only their highest query-attention tokens. 0 keeps "
            "every token. The default full-prompt path evicts per KV head; "
            "--server-chunk-prefill evicts whole tokens."
        ),
    )
    parser.add_argument(
        "--token-score-direction",
        choices=("bidirectional", "query_to_chunk"),
        default="bidirectional",
        help=(
            "Attention scored by per-head token eviction: query-to-chunk only, "
            "or query-to-chunk plus chunk-to-query. Ignored by "
            "--server-chunk-prefill."
        ),
    )
    parser.add_argument(
        "--token-score-query-window",
        type=int,
        default=8,
        help="Trailing query tokens whose attention scores chunk tokens.",
    )
    parser.add_argument(
        "--token-score-pool-kernel",
        type=int,
        default=7,
        help="Max-pool width over the chunk axis of each head's scores; 1 disables.",
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
    if args.token_capacity < 0:
        raise ValueError("--token-capacity must be non-negative")
    if args.token_score_query_window <= 0 or args.token_score_pool_kernel <= 0:
        raise ValueError(
            "--token-score-query-window and --token-score-pool-kernel must be positive"
        )
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
            causal_prompt_logprobs=args.score_attention_mask == "causal",
            score_attention_mask=args.score_attention_mask,
            score_on_pd_server=args.score_on_pd_server,
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
                draft_lens=[len(draft_ids) for draft_ids in draft_groups],
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
                if args.token_capacity > 0:
                    custom_params["dllm_parallelcomp"]["token_eviction"] = {
                        "capacity": args.token_capacity,
                        "score_query_ids": query_ids[-args.token_score_query_window :],
                        "pool_kernel": args.token_score_pool_kernel,
                        "force_keep_first": args.chunk_bos,
                    }
            elif args.token_capacity > 0:
                custom_params = {
                    "dllm_token_eviction": {
                        "capacity": args.token_capacity,
                        "prefix_len": len(prefix_ids),
                        "chunk_lens": [
                            len(chunks[chunk_index]) for chunk_index in selected
                        ],
                        "query_len": len(query_ids),
                        "score_query_window": args.token_score_query_window,
                        "pool_kernel": args.token_score_pool_kernel,
                        "force_keep_first": args.chunk_bos,
                        "bidirectional": (
                            args.token_score_direction == "bidirectional"
                        ),
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
                "token_capacity": args.token_capacity,
                "token_score_direction": args.token_score_direction,
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
        "token_capacity": args.token_capacity,
        "token_score_direction": args.token_score_direction,
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
