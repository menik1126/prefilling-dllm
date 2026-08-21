import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch

from eval_longbench_dream import (
    DATASET_TO_METRIC,
    SUPPORTED_TASKS,
    append_prediction_record,
    estimate_prompt_chars,
    load_existing_predictions,
    load_json,
    load_task_examples,
    render_prompt,
    render_prompt_parts,
    resolve_config_dir,
    resolve_data_dir,
    score_prediction,
    summarize_metrics,
)
from fastdllm_v1_model import default_fastdllm_dream_dir, load_fastdllm_v1_dream


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def bos_ids(tokenizer, add_bos_token=True):
    if not add_bos_token:
        return []
    if tokenizer.bos_token_id is not None:
        return [int(tokenizer.bos_token_id)]
    if tokenizer.bos_token:
        return tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
    return []


def split_context_ids(context_ids, budget, strategy):
    if budget >= len(context_ids):
        return list(context_ids), len(context_ids), 0, 0
    if budget <= 0:
        return [], 0, len(context_ids), 0
    if strategy == "left":
        return list(context_ids[-budget:]), 0, len(context_ids) - budget, budget
    if strategy == "head":
        return list(context_ids[:budget]), budget, len(context_ids) - budget, 0
    if strategy == "head_tail":
        head_keep = budget // 2
        tail_keep = budget - head_keep
        return (
            list(context_ids[:head_keep]) + list(context_ids[-tail_keep:]),
            head_keep,
            len(context_ids) - budget,
            tail_keep,
        )
    raise ValueError(f"Unsupported context truncation strategy: {strategy}")


def build_context_truncated_prompt_ids(tokenizer, template, example, args):
    parts = render_prompt_parts(template, example, "\n\n")
    prefix_ids = bos_ids(tokenizer, True) + encode_fragment(tokenizer, parts.get("prefix", ""))
    query_ids = encode_fragment(tokenizer, parts.get("query", ""))
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))
    prompt_budget = max(1, args.max_length - args.max_new_tokens)
    requested_context_budget = getattr(args, "context_budget_tokens", None)
    if requested_context_budget is not None:
        requested_context_budget = int(requested_context_budget)
        if requested_context_budget < 0:
            raise ValueError(f"context_budget_tokens must be non-negative: {requested_context_budget}")

    if requested_context_budget is None:
        context_budget = prompt_budget - len(prefix_ids) - len(query_ids)
    else:
        context_budget = min(requested_context_budget, len(context_ids))

    if context_budget < 0:
        keep_query = query_ids[-max(prompt_budget // 2, 1):]
        prefix_budget = max(prompt_budget - len(keep_query), 0)
        prefix_ids = prefix_ids[-prefix_budget:] if prefix_budget else []
        query_ids = keep_query
        context_budget = 0

    if requested_context_budget is not None and len(prefix_ids) + len(query_ids) + context_budget > prompt_budget:
        raise ValueError(
            f"Prompt with context_budget_tokens={requested_context_budget} exceeds budget: "
            f"{len(prefix_ids) + len(query_ids) + context_budget} > {prompt_budget}"
        )

    kept_context_ids, head_len, dropped, tail_len = split_context_ids(
        context_ids,
        max(context_budget, 0),
        args.truncation_strategy,
    )
    prompt_ids = prefix_ids + kept_context_ids + query_ids
    if len(prompt_ids) > prompt_budget:
        overflow = len(prompt_ids) - prompt_budget
        if overflow > 0 and kept_context_ids:
            kept_context_ids = kept_context_ids[:-overflow] if args.truncation_strategy == "head" else kept_context_ids[overflow:]
            prompt_ids = prefix_ids + kept_context_ids + query_ids
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]

    return prompt_ids, {
        "raw_context_tokens": len(context_ids),
        "raw_prompt_tokens": len(prefix_ids) + len(context_ids) + len(query_ids),
        "prompt_tokens_after_truncation": len(prompt_ids),
        "prompt_budget_tokens": prompt_budget,
        "requested_context_budget_tokens": requested_context_budget,
        "context_budget_tokens": max(context_budget, 0),
        "context_head_tokens": head_len,
        "context_tail_tokens": tail_len,
        "context_dropped_tokens": dropped,
        "truncation_mode": args.truncation_strategy if dropped else "none",
        "truncate_context_only": True,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Fast-dLLM v1 Dream on LongBench")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--fastdllm_dream_dir", default=default_fastdllm_dream_dir())
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--run_name", default="fastdllm_v1_longbench")
    parser.add_argument("--output_dir", default="./results_longbench_fastdllm_v1")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--context_budget_tokens", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="confidence_threshold")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--dual_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation_strategy", choices=["head_tail", "left", "head"], default="head_tail")
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument(
        "--truncate_context_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Truncate only the LongBench context slot while preserving the task query.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--stop_tokens", nargs="*", default=[])
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

    print(f"Data dir           : {data_dir}")
    print(f"Config dir         : {config_dir}")
    print(f"Tasks              : {', '.join(args.tasks)}")
    print(f"Run name           : {args.run_name}")
    print(f"Fast-dLLM Dream dir: {args.fastdllm_dream_dir}")
    print(f"Model max_length   : {args.max_length}")
    print(f"Context budget     : {args.context_budget_tokens}")
    print(f"Generation max_new : {args.max_new_tokens}")
    print(f"Block length       : {args.block_length}")
    print(f"Diffusion steps    : {args.diffusion_steps or max(1, args.max_new_tokens // args.block_length)}")
    print(f"Algorithm          : {args.alg}")
    print(f"Threshold          : {args.threshold}")
    print(f"Dual cache         : {args.dual_cache}")
    print(f"Truncation         : {args.truncation_strategy}")
    print(f"RoPE scale factor  : {args.rope_scale_factor}")
    print(f"Context-only trunc : {args.truncate_context_only}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading Fast-dLLM v1 Dream from {args.pretrained}...")
    model = load_fastdllm_v1_dream(
        pretrained=args.pretrained,
        fastdllm_dream_dir=args.fastdllm_dream_dir,
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
        dual_cache=args.dual_cache,
        truncation_strategy=args.truncation_strategy,
        rope_scale_factor=args.rope_scale_factor,
        dtype=args.dtype,
        add_bos_token=True,
    )

    all_results = {}
    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)
        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples    : {len(examples)}")
        print(f"Official max_new   : {dataset2maxlen[task]}")

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
            prompt_meta = None
            if args.truncate_context_only:
                prompt_ids, prompt_meta = build_context_truncated_prompt_ids(model.tokenizer, template, example, args)
                prediction = model.generate_one_ids(torch.tensor(prompt_ids, dtype=torch.long))
                for stop in args.stop_tokens or []:
                    if stop:
                        prediction = prediction.split(stop)[0]
                prompt_for_stats = render_prompt_parts(template, example, "\n\n")
            else:
                prompt = render_prompt(template, example)
                prediction = model.generate([prompt], stop_tokens=args.stop_tokens)[0]
                prompt_for_stats = prompt
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
                "prompt_chars": estimate_prompt_chars(prompt_for_stats),
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
                "official_task_max_new_tokens": int(dataset2maxlen[task]),
                "generation_max_new_tokens": args.max_new_tokens,
                "run_name": args.run_name,
                "max_length": args.max_length,
                "requested_context_budget_tokens": args.context_budget_tokens,
                "block_length": args.block_length,
                "diffusion_steps": model.diffusion_steps,
                "alg": args.alg,
                "threshold": args.threshold,
                "dual_cache": args.dual_cache,
                "truncation_strategy": args.truncation_strategy,
                "rope_scale_factor": args.rope_scale_factor,
                "truncate_context_only": args.truncate_context_only,
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
