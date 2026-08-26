#!/usr/bin/env python3
"""Evaluate vanilla native LLaDA on InfiniteBench.

This is the non-Fast-DLLM baseline path: head-tail truncation, then native
masked-diffusion generation with the local LLaDA model implementation.
"""

import argparse
import json
import os
from json import JSONDecodeError

from infinitebench_tasks import (
    SUPPORTED_TASKS,
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt_parts,
    load_task_examples,
    normalize_answer_label,
    resolve_data_dir,
    score_prediction,
)
from llada_native_model import LLaDANativeGenerator, build_native_config


def load_existing_predictions(out_file):
    predictions = []
    completed_ids = set()
    if not os.path.exists(out_file):
        return predictions, completed_ids

    last_good_pos = 0
    saw_decode_error = False
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
    return {"accuracy": (sum(scores) / len(scores)) if scores else 0.0, "n": len(scores)}


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


def build_head_tail_prompt(tokenizer, example, task, args):
    parts = create_prompt_parts(example, task, args.prompt_style)
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
            raise ValueError(
                f"Prompt without context still exceeds budget: {len(input_ids)} > {prompt_budget}"
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
    parser.add_argument("--output_dir", default="./results_infinitebench_llada_native")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument(
        "--prompt_style",
        choices=["parallelcomp_raw", "yarn-mistral", "gpt4", "slot_fill"],
        default="parallelcomp_raw",
    )
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
    parser.add_argument("--align_prompt_to_block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reserve_boundary_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suffix_logits_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop_tokens", nargs="*", default=[])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate native vanilla LLaDA on InfiniteBench.")
    add_common_args(parser)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.steps is None:
        args.steps = args.max_new_tokens

    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported InfiniteBench tasks: {unknown_tasks}")

    data_dir = resolve_data_dir(args.data_dir)
    generator = LLaDANativeGenerator(build_native_config(args))
    tokenizer = generator.tokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model path          : {args.model_path}")
    print("Method              : llada_native")
    print(f"Data dir            : {data_dir}")
    print(f"Tasks               : {', '.join(args.tasks)}")
    print("Truncation          : head_tail context only")
    print(f"Max length          : {args.max_length}")
    print(f"Max new tokens      : {args.max_new_tokens}")
    print(f"RoPE scale factor   : {args.rope_scale_factor}")
    print(f"Steps               : {args.steps}")
    print(f"Block length        : {args.block_length}")
    print(f"Use chat template   : {args.use_chat_template}")
    print(f"Suffix logits only  : {args.suffix_logits_only}")

    all_results = {}
    max_examples = None if args.max_examples <= 0 else args.max_examples
    for task in args.tasks:
        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        out_file = os.path.join(
            args.output_dir,
            f"llada_native_infinitebench_{task}_n{len(examples)}_predictions.jsonl",
        )
        metrics_file = out_file.replace("_predictions.jsonl", "_metrics.json")
        if os.path.exists(metrics_file):
            print(f"[{task}] Already done, skipping. ({metrics_file})")
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_results[task] = json.load(f)
            continue

        predictions, completed_ids = load_existing_predictions(out_file)
        if predictions:
            print(f"[{task}] Resuming from {len(predictions)} completed examples")

        for idx, example in enumerate(examples):
            example_id = example.get("id", idx)
            if example_id in completed_ids:
                continue

            input_ids, prompt_meta = build_head_tail_prompt(tokenizer, example, task, args)
            print(
                f"[LLaDA native InfiniteBench] label={task}:{example_id} "
                f"prompt_tokens={prompt_meta['prompt_tokens']} "
                f"raw_context_tokens={prompt_meta['raw_context_tokens']} "
                f"context_head={prompt_meta['context_head_tokens']} "
                f"context_tail={prompt_meta['context_tail_tokens']} "
                f"context_dropped={prompt_meta['context_dropped_tokens']} "
                f"prompt_mod_block={prompt_meta['prompt_mod_block_size']}"
            )

            answer_label = normalize_answer_label(task, example)
            prediction, generated_ids, nfe = generator.generate_text(input_ids, stop_tokens=args.stop_tokens)
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
                "generated_tokens": len(generated_ids),
                "nfe": nfe,
                **prompt_meta,
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
                "model_mode": "llada_native_head_tail",
                "method": "native",
                "truncation_mode": "head_tail_context_only",
                "native_context_window": args.max_length,
                "no_ntk_or_rope_extrapolation": args.rope_scale_factor <= 1.0,
                "rope_scale_factor": args.rope_scale_factor,
                "max_length": args.max_length,
                "generation_max_new_tokens": args.max_new_tokens,
                "generation_steps": args.steps,
                "generation_block_length": args.block_length,
                "official_task_max_new_tokens": TASK_TO_MAX_NEW_TOKENS[task],
                "use_chat_template": args.use_chat_template,
                "align_prompt_to_block": args.align_prompt_to_block,
                "reserve_boundary_token": args.reserve_boundary_token,
                "suffix_logits_only": args.suffix_logits_only,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[{task}] Accuracy: {metrics['accuracy']:.4f} (n={metrics['n']})")
        all_results[task] = metrics

    combined_file = os.path.join(args.output_dir, f"llada_native_all_n{args.max_examples}.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Combined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
