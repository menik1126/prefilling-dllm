#!/usr/bin/env python3
"""Evaluate vanilla native LLaDA on LongBench.

This is the non-Fast-DLLM baseline path: head-tail truncation, then native
masked-diffusion generation with the local LLaDA model implementation.
"""

import argparse
import json
import os
from pathlib import Path

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
from llada_native_model import LLaDANativeGenerator, build_native_config


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def split_head_tail(ids, budget):
    if budget >= len(ids):
        return list(ids), len(ids), 0, len(ids)
    if budget <= 0:
        return [], 0, len(ids), 0
    head_len = budget // 2
    tail_len = budget - head_len
    return list(ids[:head_len]) + list(ids[-tail_len:]), head_len, len(ids) - budget, tail_len


def render_prompt_text(tokenizer, parts, context_ids, args):
    context = tokenizer.decode(context_ids, skip_special_tokens=False)
    user_text = f"{parts.get('prefix', '')}{context}{parts.get('query', '')}"
    if not args.use_chat_template:
        return user_text
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_head_tail_prompt(tokenizer, template, example, args):
    parts = render_prompt_parts(template, example, args.segment_separator)
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))
    prompt_budget = args.max_length - args.max_new_tokens
    if args.reserve_boundary_token:
        prompt_budget -= 1
    if prompt_budget <= 0:
        raise ValueError(f"Prompt budget is non-positive: {prompt_budget}")

    def materialize(context_budget):
        truncated_context_ids, head_len, dropped, tail_len = split_head_tail(context_ids, context_budget)
        prompt_text = render_prompt_text(tokenizer, parts, truncated_context_ids, args)
        input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        return input_ids, {
            "context_budget_tokens": context_budget,
            "context_head_tokens": head_len,
            "context_tail_tokens": tail_len,
            "context_dropped_tokens": dropped,
            "truncation_mode": "head_tail" if dropped else "none",
            "prompt_chars_estimate": estimate_prompt_chars(
                {
                    "prefix": parts.get("prefix", ""),
                    "context": tokenizer.decode(truncated_context_ids, skip_special_tokens=False),
                    "query": parts.get("query", ""),
                }
            ),
        }

    lo, hi = 0, len(context_ids)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        input_ids, meta = materialize(mid)
        if len(input_ids) <= prompt_budget:
            best = (input_ids, meta)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        input_ids, meta = materialize(0)
        if len(input_ids) > prompt_budget:
            original_prompt_tokens = len(input_ids)
            input_ids, head_len, dropped, tail_len = split_head_tail(input_ids, prompt_budget)
            meta.update(
                {
                    "context_budget_tokens": 0,
                    "context_head_tokens": 0,
                    "context_tail_tokens": 0,
                    "context_dropped_tokens": len(context_ids),
                    "truncation_mode": "prompt_head_tail_no_context",
                    "non_context_prompt_original_tokens": original_prompt_tokens,
                    "non_context_prompt_head_tokens": head_len,
                    "non_context_prompt_tail_tokens": tail_len,
                    "non_context_prompt_dropped_tokens": dropped,
                }
            )
    else:
        input_ids, meta = best

    if args.align_prompt_to_block and input_ids:
        context_budget = meta["context_budget_tokens"]
        attempts = 0
        while (
            len(input_ids) % args.prompt_block_size != 0
            and context_budget > 0
            and attempts < args.prompt_block_size * 4
        ):
            context_budget -= 1
            input_ids, meta = materialize(context_budget)
            attempts += 1

    meta.update(
        {
            "raw_context_tokens": len(context_ids),
            "prompt_tokens": len(input_ids),
            "prompt_budget_tokens": prompt_budget,
            "prompt_mod_block_size": len(input_ids) % args.prompt_block_size,
            "max_length": args.max_length,
        }
    )
    return input_ids, meta


def add_common_args(parser):
    parser.add_argument("--model_path", default=os.environ.get("LLADA_MODEL", "GSAI-ML/LLaDA-8B-Instruct"))
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--run_name", default="llada_native_headtail")
    parser.add_argument("--output_dir", default="./results_longbench_llada_native")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--prompt_block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--mask_token_id", type=int, default=126336)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=os.environ.get("LLADA_DEVICE_MAP"))
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--segment_separator", default="\n\n")
    parser.add_argument("--align_prompt_to_block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reserve_boundary_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suffix_logits_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--longbench_e", action="store_true")
    parser.add_argument("--stop_tokens", nargs="*", default=[])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate native vanilla LLaDA on LongBench.")
    add_common_args(parser)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.steps is None:
        args.steps = args.max_new_tokens

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

    generator = LLaDANativeGenerator(build_native_config(args))
    tokenizer = generator.tokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model path          : {args.model_path}")
    print("Method              : llada_native")
    print(f"Data dir            : {data_dir}")
    print(f"Config dir          : {config_dir}")
    print(f"Tasks               : {', '.join(args.tasks)}")
    print(f"Run name            : {args.run_name}")
    print("Truncation          : head_tail context only")
    print(f"Max length          : {args.max_length}")
    print(f"Max new tokens      : {args.max_new_tokens}")
    print(f"RoPE scale factor   : {args.rope_scale_factor}")
    print(f"Steps               : {args.steps}")
    print(f"Block length        : {args.block_length}")
    print(f"Use chat template   : {args.use_chat_template}")
    print(f"Suffix logits only  : {args.suffix_logits_only}")

    all_results = {}
    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)
        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples     : {len(examples)}")
        print(f"Official max_new    : {dataset2maxlen[task]}")

        task_dir = Path(args.output_dir) / task
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

            input_ids, prompt_meta = build_head_tail_prompt(tokenizer, template, example, args)
            print(
                f"[LLaDA native LongBench] label={task}:{example_id} "
                f"prompt_tokens={prompt_meta['prompt_tokens']} "
                f"raw_context_tokens={prompt_meta['raw_context_tokens']} "
                f"context_head={prompt_meta['context_head_tokens']} "
                f"context_tail={prompt_meta['context_tail_tokens']} "
                f"context_dropped={prompt_meta['context_dropped_tokens']} "
                f"prompt_mod_block={prompt_meta['prompt_mod_block_size']}"
            )

            prediction, generated_ids, nfe = generator.generate_text(input_ids, stop_tokens=args.stop_tokens)
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
                "generated_tokens": len(generated_ids),
                "nfe": nfe,
                **prompt_meta,
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
                "official_task_max_new_tokens": int(dataset2maxlen[task]),
                "generation_max_new_tokens": args.max_new_tokens,
                "generation_steps": args.steps,
                "generation_block_length": args.block_length,
                "run_name": args.run_name,
                "model_mode": "llada_native_head_tail",
                "method": "native",
                "truncation_mode": "head_tail_context_only",
                "native_context_window": args.max_length,
                "no_ntk_or_rope_extrapolation": args.rope_scale_factor <= 1.0,
                "rope_scale_factor": args.rope_scale_factor,
                "max_length": args.max_length,
                "use_chat_template": args.use_chat_template,
                "align_prompt_to_block": args.align_prompt_to_block,
                "reserve_boundary_token": args.reserve_boundary_token,
                "suffix_logits_only": args.suffix_logits_only,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to: {out_file}")
        print(f"LongBench score      : {metrics['score']:.2f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")
        all_results[task] = metrics

    combined_file = Path(args.output_dir) / f"{args.run_name}_all_n{args.max_examples}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
