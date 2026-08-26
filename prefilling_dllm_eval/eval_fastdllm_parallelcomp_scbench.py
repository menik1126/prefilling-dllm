#!/usr/bin/env python3
"""Evaluate Fast-dLLM ParallelComp on flattened SCBench variable tracing data."""

from __future__ import annotations

import argparse
import json
import os
import re
from json import JSONDecodeError

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastdllm_parallelcomp import (
    dataclass_to_jsonable,
    default_fastdllm_dream_dir,
    default_fastdllm_llada_dir,
    load_fastdllm_parallelcomp,
)


VAR_RE = re.compile(r"\b[A-Z]{5}\b")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_examples(path, max_examples=None, start_index=0, limit=None):
    examples = []
    start_index = max(0, int(start_index or 0))
    limit = None if limit is None or limit <= 0 else int(limit)
    max_examples = None if max_examples is None or max_examples <= 0 else int(max_examples)
    target = limit if limit is not None else max_examples
    for source_idx, example in enumerate(iter_jsonl(path)):
        if source_idx < start_index:
            continue
        if target is not None and len(examples) >= target:
            break
        example = dict(example)
        example["_scbench_index"] = source_idx
        examples.append(example)
    return examples


def load_existing_predictions(out_file):
    predictions = []
    completed_ids = set()
    last_good_pos = 0
    saw_decode_error = False
    if not os.path.exists(out_file):
        return predictions, completed_ids
    with open(out_file, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                last_good_pos = pos
                break
            if not line.strip():
                last_good_pos = f.tell()
                continue
            try:
                record = json.loads(line)
            except JSONDecodeError:
                saw_decode_error = True
                last_good_pos = pos
                break
            predictions.append(record)
            completed_ids.add(record["example_id"])
            last_good_pos = f.tell()
    if saw_decode_error:
        with open(out_file, "rb+") as f:
            f.truncate(last_good_pos)
        print(f"Trimmed a partial JSONL tail while resuming: {out_file}")
    return predictions, completed_ids


def append_prediction_record(out_file, record):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def summarize_metrics(records):
    if not records:
        return {"accuracy": 0.0, "set_f1": 0.0, "recall": 0.0, "precision": 0.0, "n": 0}
    return {
        "accuracy": sum(float(r["correct"]) for r in records) / len(records),
        "set_f1": sum(float(r["set_f1"]) for r in records) / len(records),
        "recall": sum(float(r["recall"]) for r in records) / len(records),
        "precision": sum(float(r["precision"]) for r in records) / len(records),
        "n": len(records),
    }


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def trim_stop_tokens(text, stop_tokens):
    if not stop_tokens:
        return text
    cut = None
    for token in stop_tokens:
        if not token:
            continue
        pos = text.find(token)
        if pos >= 0:
            cut = pos if cut is None else min(cut, pos)
    return text if cut is None else text[:cut]


def normalize_vars(value):
    if isinstance(value, list):
        return [str(x).strip().upper() for x in value if str(x).strip()]
    return [match.group(0).upper() for match in VAR_RE.finditer(str(value))]


def score_variable_set(prediction, answer):
    gold = set(normalize_vars(answer))
    pred = set(normalize_vars(prediction))
    if not gold:
        correct = not pred
        return correct, 1.0 if correct else 0.0, 1.0 if correct else 0.0, 1.0 if correct else 0.0, sorted(pred)
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return pred == gold, precision, recall, f1, sorted(pred)


def build_token_parts(model, example, add_bos_token):
    prefix_ids = (model.bos_ids() if add_bos_token else []) + encode_fragment(
        model.tokenizer,
        example.get("prefix", ""),
    )
    context_ids = encode_fragment(model.tokenizer, example.get("context", ""))
    query = example.get("input", "")
    query_ids = encode_fragment(model.tokenizer, query)
    scoring_query_ids = encode_fragment(model.tokenizer, example.get("scoring_query", query))
    if not scoring_query_ids:
        scoring_query_ids = list(query_ids)
    return prefix_ids, context_ids, query_ids, scoring_query_ids


def add_parallelcomp_args(parser):
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk_chunks", type=int, default=4)
    parser.add_argument("--keep_first_chunk", action="store_true")
    parser.add_argument("--split_from_tail", action="store_true")
    parser.add_argument("--chunk_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_keep_chunk_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache_build_mode",
        choices=["chunk_query", "chunk_only", "full_prompt_mask", "full_prompt_query"],
        default="full_prompt_mask",
    )
    parser.add_argument(
        "--score_mode",
        choices=[
            "self_information",
            "draft_self_information",
            "per_chunk_draft_self_information",
            "next_block_logits",
            "attention",
            "none",
        ],
        default="draft_self_information",
    )
    parser.add_argument("--score_query_window", type=int, default=0)
    parser.add_argument("--score_draft_tokens", type=int, default=4)
    parser.add_argument("--score_draft_steps", type=int, default=None)
    parser.add_argument("--score_draft_partial_steps", type=int, default=None)
    parser.add_argument("--score_draft_partial_rounds", type=int, default=2)
    parser.add_argument("--score_draft_score_all_slots", action="store_true")
    parser.add_argument("--score_llada_shift_logits", action="store_true")
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument(
        "--score_attention_mask",
        choices=["causal", "full", "query_to_chunk"],
        default="full",
    )
    parser.add_argument("--attention_score_layers", type=int, default=4)
    parser.add_argument("--attention_query_window", type=int, default=0)
    parser.add_argument("--token_capacity", type=int, default=512)
    parser.add_argument("--token_score_query_window", type=int, default=0)
    parser.add_argument("--token_score_layers", type=int, default=0)
    parser.add_argument("--token_score_layer_mode", choices=["first", "last", "all"], default="all")
    parser.add_argument("--token_score_reduce", choices=["sum", "mean"], default="sum")
    parser.add_argument(
        "--token_score_direction",
        choices=["query_to_chunk", "chunk_to_query", "bidirectional"],
        default="bidirectional",
    )
    parser.add_argument("--token_score_include_prefix", dest="token_score_include_prefix", action="store_true", default=True)
    parser.add_argument("--no-token_score_include_prefix", dest="token_score_include_prefix", action="store_false")
    parser.add_argument("--token_score_use_generated", action="store_true")
    parser.add_argument(
        "--token_eviction_granularity",
        choices=["global", "per_head"],
        default="global",
    )
    parser.add_argument(
        "--token_attention_mask",
        choices=["full", "causal", "query_to_chunk"],
        default="full",
    )
    parser.add_argument(
        "--chunk_position_mode",
        choices=["reuse", "continuous", "absolute"],
        default="continuous",
    )
    parser.add_argument(
        "--chunk_query_position_mode",
        choices=["after_reused_window", "after_chunk"],
        default="after_reused_window",
    )
    parser.add_argument(
        "--query_position_mode",
        choices=["after_reused_window", "after_selected_chunks", "after_cache"],
        default="after_cache",
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Fast-dLLM ParallelComp on SCBench.")
    parser.add_argument("--pretrained", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--model_backend", choices=["dream", "llada"], default="dream")
    parser.add_argument("--fastdllm_dream_dir", default=default_fastdllm_dream_dir())
    parser.add_argument("--fastdllm_llada_dir", default=default_fastdllm_llada_dir())
    parser.add_argument("--llada_score_batch_size", type=int, default=8)
    parser.add_argument("--data_file", default="data_scbench/scbench_vt.jsonl")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="confidence_threshold")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--rope_scale_factor", type=float, default=64.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="./results_scbench_fastdllm_parallelcomp")
    parser.add_argument("--run_name", default="dream_scbench_vt")
    add_parallelcomp_args(parser)
    return parser


def main():
    args = build_arg_parser().parse_args()
    max_examples = None if args.max_examples <= 0 else args.max_examples
    limit = None if args.limit <= 0 else args.limit
    examples = load_examples(
        args.data_file,
        max_examples=max_examples,
        start_index=args.start_index,
        limit=limit,
    )
    if not examples:
        raise ValueError(f"No examples found in {args.data_file}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data file             : {args.data_file}")
    print(f"Loaded examples       : {len(examples)}")
    print(f"Start index/limit     : {args.start_index}/{args.limit or 'none'}")
    print(f"Model backend         : {args.model_backend}")
    print(f"Model path            : {args.pretrained}")
    print(f"Max length/new tokens : {args.max_length}/{args.max_new_tokens}")
    print(f"RoPE scale factor     : {args.rope_scale_factor}")
    print(f"Chunk size/top-k      : {args.chunk_size}/{args.topk_chunks}")
    print(f"Token capacity        : {args.token_capacity}")
    print(f"Cache build mode      : {args.cache_build_mode}")
    print(f"Score mode            : {args.score_mode}")
    print(f"Score batch size      : {args.score_batch_size}")
    print(f"Run name              : {args.run_name}")

    model = load_fastdllm_parallelcomp(
        pretrained=args.pretrained,
        fastdllm_dream_dir=args.fastdllm_dream_dir,
        model_backend=args.model_backend,
        fastdllm_llada_dir=args.fastdllm_llada_dir,
        llada_score_batch_size=args.llada_score_batch_size,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        block_length=args.block_length,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        threshold=args.threshold,
        rope_scale_factor=args.rope_scale_factor,
        dtype=args.dtype,
        add_bos_token=args.add_bos_token,
        chunk_size=args.chunk_size,
        topk_chunks=args.topk_chunks,
        keep_first_chunk=args.keep_first_chunk,
        split_from_tail=args.split_from_tail,
        chunk_bos=args.chunk_bos,
        force_keep_chunk_bos=args.force_keep_chunk_bos,
        cache_build_mode=args.cache_build_mode,
        score_mode=args.score_mode,
        score_query_window=args.score_query_window,
        score_draft_tokens=args.score_draft_tokens,
        score_draft_steps=args.score_draft_steps,
        score_draft_partial_steps=args.score_draft_partial_steps,
        score_draft_partial_rounds=args.score_draft_partial_rounds,
        score_draft_score_all_slots=args.score_draft_score_all_slots,
        score_llada_shift_logits=args.score_llada_shift_logits,
        score_batch_size=args.score_batch_size,
        score_attention_mask=args.score_attention_mask,
        attention_score_layers=args.attention_score_layers,
        attention_query_window=args.attention_query_window,
        token_capacity=args.token_capacity,
        token_score_query_window=args.token_score_query_window,
        token_score_layers=args.token_score_layers,
        token_score_layer_mode=args.token_score_layer_mode,
        token_score_reduce=args.token_score_reduce,
        token_score_direction=args.token_score_direction,
        token_score_include_prefix=args.token_score_include_prefix,
        token_attention_mask=args.token_attention_mask,
        token_score_use_generated=args.token_score_use_generated,
        token_eviction_granularity=args.token_eviction_granularity,
        chunk_position_mode=args.chunk_position_mode,
        chunk_query_position_mode=args.chunk_query_position_mode,
        query_position_mode=args.query_position_mode,
    )

    out_file = os.path.join(args.output_dir, f"{args.run_name}_predictions.jsonl")
    metrics_file = os.path.join(args.output_dir, f"{args.run_name}_metrics.json")
    predictions, completed_ids = load_existing_predictions(out_file)
    if predictions:
        print(f"Resuming from {len(predictions)} completed examples in {out_file}")

    for idx, example in enumerate(examples):
        source_idx = example.get("_scbench_index", idx)
        example_id = example.get("id", str(source_idx))
        if example_id in completed_ids:
            continue
        prefix_ids, context_ids, query_ids, scoring_query_ids = build_token_parts(
            model,
            example,
            args.add_bos_token,
        )
        result = model.generate(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            query_ids=query_ids,
            scoring_query_ids=scoring_query_ids,
        )
        prediction = trim_stop_tokens(result.text, args.stop_tokens)
        correct, precision, recall, f1, pred_vars = score_variable_set(prediction, example["answer"])
        record = {
            "task": "scbench_vt",
            "example_id": example_id,
            "index": source_idx,
            "row_id": example.get("row_id"),
            "turn_id": example.get("turn_id"),
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "set_f1": f1,
            "prediction": prediction,
            "pred_vars": pred_vars,
            "answer": example["answer"],
            "context_chars": len(example.get("context", "")),
            "input_chars": len(example.get("input", "")),
            "prompt_meta": {
                "prefix_tokens": len(prefix_ids),
                "context_tokens": len(context_ids),
                "query_tokens": len(query_ids),
                "scoring_query_tokens": len(scoring_query_ids),
            },
            "parallelcomp": dataclass_to_jsonable(result),
        }
        predictions.append(record)
        append_prediction_record(out_file, record)
        completed_ids.add(example_id)
        if (idx + 1) % 10 == 0:
            running = summarize_metrics(predictions)
            print(
                f"  [{idx + 1}/{len(examples)}] "
                f"acc={running['accuracy']:.4f} f1={running['set_f1']:.4f} recall={running['recall']:.4f}"
            )

    metrics = summarize_metrics(predictions)
    metrics.update(
        {
            "task": "scbench_vt",
            "data_file": args.data_file,
            "max_examples": len(examples),
            "start_index": args.start_index,
            "limit": args.limit or None,
            "model_backend": args.model_backend,
            "model_path": args.pretrained,
            "generation_max_new_tokens": args.max_new_tokens,
            "max_length": args.max_length,
            "block_length": args.block_length,
            "diffusion_steps": model.diffusion_steps,
            "rope_scale_factor": args.rope_scale_factor,
            "chunk_size": args.chunk_size,
            "topk_chunks": args.topk_chunks,
            "token_capacity": args.token_capacity,
            "chunk_bos": args.chunk_bos,
            "cache_build_mode": args.cache_build_mode,
            "score_mode": args.score_mode,
            "score_draft_tokens": args.score_draft_tokens,
            "score_draft_partial_rounds": args.score_draft_partial_rounds,
            "score_batch_size": args.score_batch_size,
            "token_score_direction": args.token_score_direction,
            "chunk_position_mode": args.chunk_position_mode,
            "query_position_mode": args.query_position_mode,
        }
    )
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to : {out_file}")
    print(
        f"SCBench VT accuracy   : {metrics['accuracy']:.4f} "
        f"F1={metrics['set_f1']:.4f} recall={metrics['recall']:.4f} (n={metrics['n']})"
    )
    print(f"Metrics saved to     : {metrics_file}")


if __name__ == "__main__":
    main()
