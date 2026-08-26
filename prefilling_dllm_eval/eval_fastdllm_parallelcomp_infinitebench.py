#!/usr/bin/env python3
import argparse
import json
import os
from json import JSONDecodeError

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastdllm_parallelcomp import (
    dataclass_to_jsonable,
    default_fastdllm_dream_dir,
    default_fastdllm_llada_dir,
    load_fastdllm_parallelcomp,
)
from infinitebench_tasks import (
    SUPPORTED_TASKS,
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt_parts,
    load_task_examples,
    normalize_answer_label,
    resolve_data_dir,
    score_prediction,
)


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
    scores = [int(record["correct"]) for record in records]
    return {
        "accuracy": (sum(scores) / len(scores)) if scores else 0.0,
        "n": len(scores),
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


def build_token_parts(model, parts, add_bos_token):
    prefix_ids = (model.bos_ids() if add_bos_token else []) + encode_fragment(
        model.tokenizer,
        parts.get("prefix", ""),
    )
    context_ids = encode_fragment(model.tokenizer, parts.get("context", ""))
    query_ids = encode_fragment(model.tokenizer, parts.get("query", ""))
    scoring_query_ids = encode_fragment(model.tokenizer, parts.get("scoring_query", ""))
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
        default="chunk_query",
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
    parser.add_argument("--score_draft_tokens", type=int, default=16)
    parser.add_argument("--score_draft_steps", type=int, default=None)
    parser.add_argument("--score_draft_partial_steps", type=int, default=None)
    parser.add_argument("--score_draft_partial_rounds", type=int, default=None)
    parser.add_argument("--score_draft_score_all_slots", action="store_true")
    parser.add_argument("--score_llada_shift_logits", action="store_true")
    parser.add_argument("--score_batch_size", type=int, default=8)
    parser.add_argument(
        "--score_attention_mask",
        choices=["causal", "full", "query_to_chunk"],
        default="causal",
    )
    parser.add_argument("--attention_score_layers", type=int, default=4)
    parser.add_argument("--attention_query_window", type=int, default=0)
    parser.add_argument("--token_capacity", type=int, default=0)
    parser.add_argument("--token_score_query_window", type=int, default=0)
    parser.add_argument("--token_score_layers", type=int, default=0)
    parser.add_argument("--token_score_layer_mode", choices=["first", "last", "all"], default="all")
    parser.add_argument("--token_score_reduce", choices=["sum", "mean"], default="sum")
    parser.add_argument(
        "--token_score_direction",
        choices=["query_to_chunk", "chunk_to_query", "bidirectional"],
        default="query_to_chunk",
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
        default="reuse",
    )
    parser.add_argument(
        "--chunk_query_position_mode",
        choices=["after_reused_window", "after_chunk"],
        default="after_reused_window",
    )
    parser.add_argument(
        "--query_position_mode",
        choices=["after_reused_window", "after_selected_chunks", "after_cache"],
        default="after_reused_window",
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Fast-dLLM v1 with full ParallelComp KV runtime on InfiniteBench."
    )
    parser.add_argument("--pretrained", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--model_backend", choices=["dream", "llada"], default="dream")
    parser.add_argument("--fastdllm_dream_dir", default=default_fastdllm_dream_dir())
    parser.add_argument("--fastdllm_llada_dir", default=default_fastdllm_llada_dir())
    parser.add_argument("--llada_score_batch_size", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument(
        "--prompt_style",
        choices=["parallelcomp_raw", "yarn-mistral", "gpt4", "slot_fill"],
        default="parallelcomp_raw",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="confidence_threshold")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", default="./results_infinitebench_fastdllm_parallelcomp")
    add_parallelcomp_args(parser)
    return parser


def main():
    args = build_arg_parser().parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}")

    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir              : {data_dir}")
    print(f"Tasks                 : {', '.join(args.tasks)}")
    print(f"Prompt style          : {args.prompt_style}")
    print(f"Model backend         : {args.model_backend}")
    print(f"Fast-dLLM Dream dir   : {args.fastdllm_dream_dir}")
    print(f"Fast-dLLM LLaDA dir   : {args.fastdllm_llada_dir}")
    print(f"Model max_length      : {args.max_length}")
    print(f"Generation max_new    : {args.max_new_tokens}")
    print(f"Block length          : {args.block_length}")
    print(f"Diffusion steps       : {args.diffusion_steps}")
    print(f"Score mode            : {args.score_mode}")
    print(f"Score draft steps     : {args.score_draft_steps}")
    print(f"Score partial steps   : {args.score_draft_partial_steps}")
    print(f"Score partial rounds  : {args.score_draft_partial_rounds}")
    print(f"Score all draft slots : {args.score_draft_score_all_slots}")
    print(f"Score batch size      : {args.score_batch_size}")
    print(f"LLaDA shifted score   : {args.score_llada_shift_logits}")
    print(f"Chunk size/top-k      : {args.chunk_size}/{args.topk_chunks}")
    print(f"Token capacity        : {args.token_capacity}")
    print(f"Token score generated : {args.token_score_use_generated}")
    print(f"Token score direction : {args.token_score_direction}")
    print(f"Token score prefix    : {args.token_score_include_prefix}")
    print(f"Token eviction gran.  : {args.token_eviction_granularity}")
    print(f"Chunk BOS             : {args.chunk_bos}")
    print(f"Cache build mode      : {args.cache_build_mode}")
    print(f"Chunk position mode   : {args.chunk_position_mode}")
    print(f"Query position mode   : {args.query_position_mode}")
    print(f"RoPE scale factor     : {args.rope_scale_factor}")

    print(f"Loading Fast-dLLM ParallelComp from {args.pretrained}...")
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

    all_results = {}
    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)
        examples = load_task_examples(task, data_dir, max_examples=args.max_examples)
        print(f"Loaded examples       : {len(examples)}")
        print(f"Official max_new      : {TASK_TO_MAX_NEW_TOKENS[task]}")

        out_file = os.path.join(
            args.output_dir,
            f"fastdllm_parallelcomp_infinitebench_{task}_n{len(examples)}_predictions.jsonl",
        )
        metrics_file = out_file.replace("_predictions.jsonl", "_metrics.json")
        if os.path.exists(metrics_file):
            print(f"Already done, skipping. ({metrics_file})")
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_results[task] = json.load(f)
            continue

        predictions, completed_ids = load_existing_predictions(out_file)
        if predictions:
            print(f"Resuming from {len(predictions)} completed examples in {out_file}")

        for idx, example in enumerate(examples):
            example_id = example.get("id", idx)
            if example_id in completed_ids:
                continue
            parts = create_prompt_parts(example, task, args.prompt_style)
            prefix_ids, context_ids, query_ids, scoring_query_ids = build_token_parts(
                model,
                parts,
                args.add_bos_token,
            )
            result = model.generate(
                prefix_ids=prefix_ids,
                context_ids=context_ids,
                query_ids=query_ids,
                scoring_query_ids=scoring_query_ids,
            )
            prediction = trim_stop_tokens(result.text, args.stop_tokens)
            answer_label = normalize_answer_label(task, example)
            correct = score_prediction(task, prediction, answer_label)
            record = {
                "task": task,
                "example_id": example_id,
                "index": idx,
                "correct": correct,
                "prediction": prediction,
                "answer": answer_label,
                "raw_answer": example["answer"],
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "prompt_chars": len(parts.get("prefix", "")) + len(parts.get("context", "")) + len(parts.get("query", "")),
                "prompt_meta": {
                    "prefix_tokens": len(prefix_ids),
                    "context_tokens": len(context_ids),
                    "query_tokens": len(query_ids),
                    "scoring_query_tokens": len(scoring_query_ids),
                },
                "parallelcomp": dataclass_to_jsonable(result),
            }
            if "options" in example:
                record["options"] = example["options"]
            predictions.append(record)
            append_prediction_record(out_file, record)
            completed_ids.add(example_id)
            if (idx + 1) % 10 == 0:
                running = summarize_metrics(predictions)
                print(f"  [{idx + 1}/{len(examples)}] running_acc={running['accuracy']:.4f}")

        metrics = summarize_metrics(predictions)
        metrics.update(
            {
                "task": task,
                "max_examples": len(examples),
                "prompt_style": args.prompt_style,
                "generation_max_new_tokens": args.max_new_tokens,
                "official_task_max_new_tokens": TASK_TO_MAX_NEW_TOKENS[task],
                "max_length": args.max_length,
                "block_length": args.block_length,
                "diffusion_steps": model.diffusion_steps,
                "alg": args.alg,
                "threshold": args.threshold,
                "rope_scale_factor": args.rope_scale_factor,
                "chunk_size": args.chunk_size,
                "topk_chunks": args.topk_chunks,
                "chunk_bos": args.chunk_bos,
                "cache_build_mode": args.cache_build_mode,
                "score_mode": args.score_mode,
                "score_draft_tokens": args.score_draft_tokens,
                "score_draft_steps": args.score_draft_steps,
                "score_draft_partial_steps": args.score_draft_partial_steps,
                "score_draft_partial_rounds": args.score_draft_partial_rounds,
                "score_draft_score_all_slots": args.score_draft_score_all_slots,
                "score_llada_shift_logits": args.score_llada_shift_logits,
                "score_batch_size": args.score_batch_size,
                "token_capacity": args.token_capacity,
                "token_score_use_generated": args.token_score_use_generated,
                "token_score_direction": args.token_score_direction,
                "token_score_include_prefix": args.token_score_include_prefix,
                "token_eviction_granularity": args.token_eviction_granularity,
                "token_score_layer_mode": args.token_score_layer_mode,
                "token_score_layers": args.token_score_layers,
                "chunk_position_mode": args.chunk_position_mode,
                "query_position_mode": args.query_position_mode,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to : {out_file}")
        print(f"Accuracy             : {metrics['accuracy']:.4f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")
        all_results[task] = metrics

    combined_file = os.path.join(args.output_dir, f"fastdllm_parallelcomp_infinitebench_all_n{args.max_examples}.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
