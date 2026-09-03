#!/usr/bin/env python3
"""Evaluate external query-aware chunk selection for Dream on MultiFieldQA-en.

The data format, prompt template, and QA F1 metric match the Prefilling-dLLM
LongBench evaluator. Chunk scoring is intentionally performed through an
already-running SGLang ``/generate`` endpoint, so the selector and generator
share one Dream model instance.

The evaluator supports both continuous and reused chunk positions. With
``--server-chunk-prefill``, it also sends exact chunk boundaries and RoPE starts
so SGLang can build every retained chunk independently from the common prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer

TASK = "multifieldqa_en"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


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
    def __init__(self, base_url: str, timeout: float):
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        self.generate_url = f"{base_url}/generate"
        self.timeout = timeout

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
        result = self.post(
            {
                "input_ids": [list(ids) for ids in input_ids],
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
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
        return self.post(
            {
                "input_ids": list(input_ids),
                "sampling_params": sampling_params,
            }
        )

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
        values = result["meta_info"].get("output_token_logprobs", [])
        return [
            int(value[1])
            for value in values[:max_new_tokens]
            if len(value) > 1 and value[1] is not None
        ]


def mean_query_logprob(values: Sequence[Any], expected_tokens: int) -> float:
    logprobs = []
    for value in values[-expected_tokens:]:
        if value and value[0] is not None:
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
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        rows = [
            list(prefix_ids) + list(chunk) + list(scoring_query_ids) for chunk in batch
        ]
        starts = [len(prefix_ids) + len(chunk) for chunk in batch]
        logprob_rows = client.prompt_logprobs(rows, starts)
        scores.extend(
            mean_query_logprob(values, len(scoring_query_ids))
            for values in logprob_rows
        )
    return scores


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate external query-aware chunk selection on LongBench multifieldqa_en"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
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
    parser.add_argument("--draft-tokens", type=int, default=0)
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
            "the template prefix and discarding temporary query KV."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"{TASK}_{args.selection_mode}_{args.position_mode}.jsonl"
    )
    metrics_path = (
        args.output_dir
        / f"{TASK}_{args.selection_mode}_{args.position_mode}_metrics.json"
    )

    records = []
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

        draft_ids = []
        chunk_scores = None
        scoring_query_ids = list(query_ids)
        score_seconds = 0.0
        if args.selection_mode == "query_logprob" and client is not None:
            score_start = time.perf_counter()
            draft_ids = client.draft_ids(
                prefix_ids + query_ids,
                args.draft_tokens,
                args.generation_block_size,
            )
            scoring_query_ids.extend(draft_ids)
            chunk_scores = score_chunks(
                client,
                prefix_ids,
                chunks,
                scoring_query_ids,
                args.score_batch_size,
            )
            score_seconds = time.perf_counter() - score_start

        if args.selection_mode in {"fixed", "manifest"}:
            requested_indices = (
                fixed_chunk_indices
                if args.selection_mode == "fixed"
                else selection_manifest.get(index)
            )
            if requested_indices is None:
                raise ValueError(f"Selection manifest has no entry for index {index}")
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

        generation = None
        raw_prediction = ""
        prediction = ""
        generation_seconds = 0.0
        if client is not None:
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
                        start + len(chunks[index])
                        for start, index in zip(chunk_position_starts, selected)
                    ]
                custom_params = {
                    "dllm_parallelcomp": {
                        "prefix_len": len(prefix_ids),
                        "chunk_lens": [len(chunks[index]) for index in selected],
                        "query_len": len(query_ids),
                        "chunk_position_starts": chunk_position_starts,
                        "chunk_query_position_starts": chunk_query_position_starts,
                        "query_position_start": query_rope_start,
                    }
                }
            generation_start = time.perf_counter()
            generation = client.generate(
                compressed_ids,
                args.max_new_tokens,
                position_start=None if args.server_chunk_prefill else query_start,
                position_offset=0 if args.server_chunk_prefill else position_offset,
                custom_params=custom_params,
            )
            generation_seconds = time.perf_counter() - generation_start
            raw_prediction = generation.get("text", "")
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
                if client
                else None
            ),
            "selection_mode": args.selection_mode,
            "position_mode": args.position_mode,
            "query_position_mode": args.query_position_mode,
            "chunk_query_position_mode": args.chunk_query_position_mode,
            "query_position_offset": position_offset if client is not None else None,
            "chunk_size": args.chunk_size,
            "top_k": args.top_k,
            "chunk_bos": args.chunk_bos,
            "server_chunk_prefill": args.server_chunk_prefill,
            "raw_context_tokens": len(context_ids),
            "candidate_chunks": len(chunks),
            "selected_chunk_indices": selected,
            "selected_original_position_starts": [
                index * args.chunk_size for index in selected
            ],
            "selected_continuous_position_starts": [
                order * args.chunk_size for order in range(len(selected))
            ],
            "chunk_scores": chunk_scores,
            "draft_ids": draft_ids,
            "prefix_tokens": len(prefix_ids),
            "query_tokens": len(query_ids),
            "compressed_prompt_tokens": len(compressed_ids),
            "score_seconds": score_seconds,
            "generation_seconds": generation_seconds,
            "generation_meta": generation.get("meta_info") if generation else None,
        }
        records.append(record)
        with output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        running_scores = [
            item["score"] for item in records if item["score"] is not None
        ]
        running = (
            100 * sum(running_scores) / len(running_scores) if running_scores else 0.0
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
        "server_chunk_prefill": args.server_chunk_prefill,
        "draft_tokens": args.draft_tokens,
        "prediction_max_words": args.prediction_max_words,
        "stop_at_answer_boundary": args.stop_at_answer_boundary,
        "average_raw_context_tokens": sum(
            record["raw_context_tokens"] for record in records
        )
        / len(records),
        "average_compressed_prompt_tokens": sum(
            record["compressed_prompt_tokens"] for record in records
        )
        / len(records),
        "total_score_seconds": sum(record["score_seconds"] for record in records),
        "total_generation_seconds": sum(
            record["generation_seconds"] for record in records
        ),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
