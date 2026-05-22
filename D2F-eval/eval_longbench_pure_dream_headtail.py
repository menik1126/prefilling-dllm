#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from eval_infinitebench_pure_dream_chunks import (
    apply_dream_rope_scaling,
    bos_ids,
    encode_fragment,
    generate_official_dream,
    maybe_load_generation_lora,
    resolve_dtype,
)
from eval_longbench_dream import (
    DATASET_TO_METRIC,
    SUPPORTED_TASKS,
    append_prediction_record,
    estimate_prompt_chars,
    load_existing_predictions,
    load_json,
    load_task_examples,
    render_prompt_parts,
    resolve_config_dir,
    resolve_data_dir,
    score_prediction,
    summarize_metrics,
)


def split_head_tail(ids, budget):
    if budget >= len(ids):
        return list(ids), len(ids), 0, len(ids)
    if budget <= 0:
        return [], 0, len(ids), 0
    head_len = budget // 2
    tail_len = budget - head_len
    return list(ids[:head_len]) + list(ids[-tail_len:]), head_len, len(ids) - budget, tail_len


def build_head_tail_prompt_ids(tokenizer, parts, args):
    prefix_ids = bos_ids(tokenizer, args.add_bos_token) + encode_fragment(tokenizer, parts.get("prefix", ""))
    query_ids = encode_fragment(tokenizer, parts.get("query", ""))
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))
    prompt_budget = max(1, args.max_length - args.max_new_tokens)

    context_budget = prompt_budget - len(prefix_ids) - len(query_ids)
    if context_budget < 0:
        keep_query = query_ids[-max(prompt_budget // 2, 1):]
        prefix_budget = max(prompt_budget - len(keep_query), 0)
        prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget else []
        query_ids = keep_query
        context_budget = 0

    context_budget = max(context_budget, 0)
    truncated_context_ids, head_len, dropped, tail_len = split_head_tail(context_ids, context_budget)
    prompt_ids = prefix_ids + truncated_context_ids + query_ids

    if len(prompt_ids) > prompt_budget:
        overflow = len(prompt_ids) - prompt_budget
        if overflow > 0 and truncated_context_ids:
            truncated_context_ids = truncated_context_ids[overflow:]
            prompt_ids = prefix_ids + truncated_context_ids + query_ids
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]

    return prompt_ids, {
        "raw_context_tokens": len(context_ids),
        "raw_prompt_tokens": len(prefix_ids) + len(context_ids) + len(query_ids),
        "prompt_tokens_after_truncation": len(prompt_ids),
        "prompt_budget_tokens": prompt_budget,
        "context_budget_tokens": context_budget,
        "context_head_tokens": head_len,
        "context_tail_tokens": tail_len,
        "context_dropped_tokens": dropped,
        "truncation_mode": "head_tail" if dropped else "none",
        "prefix_tokens": len(prefix_ids),
        "query_tokens": len(query_ids),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate official Dream generation on LongBench with raw prompt head-tail truncation."
    )
    parser.add_argument("--model_path", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--generation_lora_path", default=None)
    parser.add_argument("--tasks", nargs="+", default=["multifieldqa_en"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--output_dir", default="./results_longbench_pure_dream_headtail")
    parser.add_argument("--run_name", default="pure_dream_headtail_longbench")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--rope_scale_factor", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visibility_mode", choices=["full_prompt"], default="full_prompt")
    parser.add_argument("--longbench_e", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported LongBench tasks: {unknown_tasks}")
    missing_metrics = sorted(set(args.tasks) - set(DATASET_TO_METRIC))
    if missing_metrics:
        raise ValueError(f"Missing LongBench metric definitions: {missing_metrics}")

    data_dir = resolve_data_dir(args.data_dir)
    config_dir = resolve_config_dir(args.config_dir)
    prompt_templates = load_json(config_dir / "dataset2prompt_raw.json")
    dataset2maxlen = load_json(config_dir / "dataset2maxlen.json")
    max_examples = None if args.max_examples <= 0 else args.max_examples
    dtype = resolve_dtype(args.dtype)

    print(f"Model path       : {args.model_path}")
    print("Model mode       : Dream official diffusion_generate with raw prompt head-tail truncation")
    print(f"Generation LoRA  : {args.generation_lora_path or 'None'}")
    print(f"Data dir         : {data_dir}")
    print(f"Config dir       : {config_dir}")
    print(f"Tasks            : {', '.join(args.tasks)}")
    print(f"Run name         : {args.run_name}")
    print(f"Max length       : {args.max_length}")
    print(f"RoPE scale factor: {args.rope_scale_factor}")
    print(f"Max new tokens   : {args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device).eval()
    apply_dream_rope_scaling(model, args.rope_scale_factor)
    model = maybe_load_generation_lora(model, args.generation_lora_path)
    model = model.to(args.device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)
        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples  : {len(examples)}")
        print(f"Official max_new : {dataset2maxlen[task]}")

        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        out_file = task_dir / f"{args.run_name}.json"
        metrics_file = task_dir / f"{args.run_name}_metrics.json"

        if metrics_file.exists():
            print(f"Already done, skipping. ({metrics_file})")
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_results[task] = json.load(f)
            continue

        predictions, completed_ids = load_existing_predictions(out_file)
        if predictions:
            print(f"Resuming from {len(predictions)} completed examples in {out_file}")

        template = prompt_templates[task]
        for idx, example in enumerate(examples):
            example_id = example.get("_id", idx)
            if example_id in completed_ids:
                continue

            parts = render_prompt_parts(template, example, "\n\n")
            prompt_ids, prompt_meta = build_head_tail_prompt_ids(tokenizer, parts, args)
            print(
                f"[PureDreamHeadTail] label={task}:{example_id} "
                f"raw_prompt_tokens={prompt_meta['raw_prompt_tokens']} "
                f"prompt_tokens_after_truncation={prompt_meta['prompt_tokens_after_truncation']} "
                f"context_head={prompt_meta['context_head_tokens']} "
                f"context_tail={prompt_meta['context_tail_tokens']} "
                f"context_dropped={prompt_meta['context_dropped_tokens']}"
            )

            prediction = generate_official_dream(model, tokenizer, prompt_ids, {}, args)
            answers = example.get("answers", [])
            all_classes = example.get("all_classes")
            score = score_prediction(task, prediction, answers, all_classes)

            record = {
                "task": task,
                "example_id": example_id,
                "index": idx,
                "pred": prediction,
                "answers": answers,
                "all_classes": all_classes,
                "length": example.get("length"),
                "score": score,
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "prompt_chars": estimate_prompt_chars(parts),
                "prompt_meta": prompt_meta,
            }
            predictions.append(record)
            append_prediction_record(out_file, record)
            completed_ids.add(example_id)

            if (idx + 1) % 10 == 0:
                running = summarize_metrics(predictions, longbench_e=args.longbench_e)
                print(f"  [{idx + 1}/{len(examples)}] running_score={running['score']:.2f}")

        metrics = summarize_metrics(predictions, longbench_e=args.longbench_e)
        metrics.update(
            {
                "task": task,
                "max_examples": len(examples),
                "model_mode": (
                    "dream_official_headtail_generation_lora"
                    if args.generation_lora_path
                    else "pure_dream_official_headtail"
                ),
                "generation_lora_path": args.generation_lora_path,
                "truncation_strategy": "head_tail_context",
                "max_length": args.max_length,
                "rope_scale_factor": args.rope_scale_factor,
                "official_task_max_new_tokens": int(dataset2maxlen[task]),
                "generation_max_new_tokens": args.max_new_tokens,
                "run_name": args.run_name,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {out_file}")
        print(f"LongBench score      : {metrics['score']:.2f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")
        all_results[task] = metrics

    combined_file = output_dir / f"{args.run_name}_all_n{args.max_examples}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
