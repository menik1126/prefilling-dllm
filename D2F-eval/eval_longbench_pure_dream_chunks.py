#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from eval_infinitebench_pure_dream_chunks import (
    apply_dream_rope_scaling,
    bos_ids,
    encode_fragment,
    generate_official_dream,
    maybe_disable_adapter,
    maybe_load_generation_lora,
    pack_chunks,
    resolve_dtype,
    select_chunks,
    split_token_chunks,
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


def prune_chunk_by_query_attention(model, chunk_ids, query_ids, args):
    capacity = int(args.token_capacity or 0)
    if capacity <= 0 or len(chunk_ids) <= capacity:
        return list(chunk_ids), {
            "original_tokens": len(chunk_ids),
            "kept_tokens": len(chunk_ids),
            "removed_tokens": 0,
            "score_source": "none",
        }
    if not query_ids:
        return list(chunk_ids[:capacity]), {
            "original_tokens": len(chunk_ids),
            "kept_tokens": min(capacity, len(chunk_ids)),
            "removed_tokens": max(len(chunk_ids) - capacity, 0),
            "score_source": "head_fallback_no_query",
        }

    scoring_query_ids = query_ids
    if args.token_score_query_window and args.token_score_query_window > 0:
        scoring_query_ids = scoring_query_ids[-args.token_score_query_window:]

    joint_ids = torch.tensor([list(chunk_ids) + list(scoring_query_ids)], device=args.device, dtype=torch.long)
    seq_len = joint_ids.shape[1]
    attention_mask = torch.ones((1, 1, seq_len, seq_len), device=args.device, dtype=torch.bool)
    with torch.inference_mode(), maybe_disable_adapter(model, args.score_without_lora):
        outputs = model(
            joint_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
            output_attentions=True,
        )

    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        head_keep = capacity // 2
        tail_keep = capacity - head_keep
        kept = list(chunk_ids[:head_keep]) + list(chunk_ids[-tail_keep:])
        return kept, {
            "original_tokens": len(chunk_ids),
            "kept_tokens": len(kept),
            "removed_tokens": len(chunk_ids) - len(kept),
            "score_source": "head_tail_fallback_no_attentions",
        }

    chunk_len = len(chunk_ids)
    query_start = chunk_len
    layer_scores = []
    for attn in attentions:
        if attn is None:
            continue
        query_to_chunk = attn[0, :, query_start:, :chunk_len].float()
        if query_to_chunk.numel() == 0:
            continue
        layer_scores.append(query_to_chunk.mean(dim=(0, 1)))

    if not layer_scores:
        kept = list(chunk_ids[:capacity])
        return kept, {
            "original_tokens": len(chunk_ids),
            "kept_tokens": len(kept),
            "removed_tokens": len(chunk_ids) - len(kept),
            "score_source": "head_fallback_empty_attentions",
        }

    scores = torch.stack(layer_scores, dim=0).mean(dim=0)
    topk = min(capacity, scores.numel())
    kept_positions = torch.topk(scores, k=topk, largest=True).indices.sort().values.tolist()
    kept = [chunk_ids[pos] for pos in kept_positions]
    return kept, {
        "original_tokens": len(chunk_ids),
        "kept_tokens": len(kept),
        "removed_tokens": len(chunk_ids) - len(kept),
        "score_source": "query_attention_all_layers_heads",
        "score_query_tokens": len(scoring_query_ids),
    }


def prune_packed_chunks_by_attention(model, packed_chunks, packed_indices, scoring_query_ids, args):
    if not args.token_capacity or args.token_capacity <= 0:
        return packed_chunks, []

    pruned_chunks = []
    prune_meta = []
    for chunk_ids, chunk_index in zip(packed_chunks, packed_indices):
        pruned_ids, meta = prune_chunk_by_query_attention(model, chunk_ids, scoring_query_ids, args)
        meta["chunk_index"] = chunk_index
        pruned_chunks.append(pruned_ids)
        prune_meta.append(meta)
    return pruned_chunks, prune_meta


def build_selected_prompt_ids(tokenizer, model, parts, args):
    prefix_ids = bos_ids(tokenizer, args.add_bos_token) + encode_fragment(tokenizer, parts.get("prefix", ""))
    query_ids = encode_fragment(tokenizer, parts.get("query", ""))
    scoring_query_ids = encode_fragment(tokenizer, parts.get("scoring_query", ""))
    separator_ids = encode_fragment(tokenizer, args.segment_separator)
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))

    candidate_chunks = split_token_chunks(
        context_ids,
        chunk_size=args.chunk_size,
        split_from_tail=args.split_from_tail,
    )
    selected_indices, scores = select_chunks(model, candidate_chunks, scoring_query_ids, args)

    prompt_budget = max(1, args.max_length - args.max_new_tokens)
    available_context_tokens = prompt_budget - len(prefix_ids) - len(query_ids)
    packed_chunks, packed_indices, context_tokens_after_pack = pack_chunks(
        candidate_chunks=candidate_chunks,
        selected_indices=selected_indices,
        separator_ids=separator_ids,
        available_context_tokens=available_context_tokens,
    )
    packed_chunks, token_prune_meta = prune_packed_chunks_by_attention(
        model=model,
        packed_chunks=packed_chunks,
        packed_indices=packed_indices,
        scoring_query_ids=scoring_query_ids,
        args=args,
    )
    context_tokens_after_pack = sum(len(chunk_ids) for chunk_ids in packed_chunks)

    prompt_ids = list(prefix_ids)
    prefix_span = [0, len(prompt_ids)]
    chunk_spans = []
    for idx, chunk_ids in enumerate(packed_chunks):
        if idx > 0:
            prompt_ids.extend(separator_ids)
        start = len(prompt_ids)
        prompt_ids.extend(chunk_ids)
        chunk_spans.append([start, len(prompt_ids)])

    query_start = len(prompt_ids)
    prompt_ids.extend(query_ids)
    query_span = [query_start, len(prompt_ids)]

    if len(prompt_ids) > prompt_budget:
        trim = len(prompt_ids) - prompt_budget
        prompt_ids = prompt_ids[trim:]
        prefix_span = [max(prefix_span[0] - trim, 0), max(prefix_span[1] - trim, 0)]
        chunk_spans = [
            [max(start - trim, 0), max(end - trim, 0)]
            for start, end in chunk_spans
            if end > trim
        ]
        query_span = [max(query_span[0] - trim, 0), max(query_span[1] - trim, 0)]

    finite_scores = [scores[idx] for idx in packed_indices if idx in scores and math.isfinite(scores[idx])]
    return prompt_ids, {
        "raw_context_tokens": len(context_ids),
        "raw_prompt_tokens": len(prefix_ids) + len(context_ids) + len(query_ids),
        "candidate_chunks": len(candidate_chunks),
        "selected_chunk_indices": packed_indices,
        "selected_chunks": len(packed_indices),
        "context_tokens_after_pack": context_tokens_after_pack,
        "score_min": min(finite_scores) if finite_scores else None,
        "score_max": max(finite_scores) if finite_scores else None,
        "token_capacity": args.token_capacity,
        "token_prune_meta": token_prune_meta,
        "token_removed_total": sum(item.get("removed_tokens", 0) for item in token_prune_meta),
        "prefix_span": prefix_span,
        "chunk_spans": chunk_spans,
        "query_span": query_span,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate official Dream generation on LongBench after text-level chunk selection."
    )
    parser.add_argument("--model_path", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument(
        "--generation_lora_path",
        default=None,
        help="Optional LoRA adapter used only for official diffusion_generate. Chunk scoring can still disable it.",
    )
    parser.add_argument("--tasks", nargs="+", default=["multifieldqa_en"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--output_dir", default="./results_longbench_pure_dream_chunks")
    parser.add_argument("--run_name", default="pure_dream_chunks_longbench")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Number of examples per task. Use 0 or a negative value for the full file.",
    )
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
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk_chunks", type=int, default=3)
    parser.add_argument(
        "--token_capacity",
        type=int,
        default=0,
        help="If >0, prune each selected chunk to this many tokens by query-to-chunk attention before official generation.",
    )
    parser.add_argument("--token_score_query_window", type=int, default=0)
    parser.add_argument("--score_mode", choices=["self_information", "none"], default="self_information")
    parser.add_argument("--score_query_window", type=int, default=0)
    parser.add_argument(
        "--score_without_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --generation_lora_path is set, disable the adapter during chunk scoring by default.",
    )
    parser.add_argument("--keep_first_chunk", action="store_true")
    parser.add_argument("--split_from_tail", action="store_true")
    parser.add_argument("--segment_separator", default="\n\n")
    parser.add_argument(
        "--visibility_mode",
        choices=["full_prompt", "query_to_chunks", "query_chunk_mutual"],
        default="full_prompt",
    )
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
    print("Model mode       : Dream official diffusion_generate after text-level chunk selection")
    print(f"Generation LoRA  : {args.generation_lora_path or 'None'}")
    print(f"Score without LoRA: {args.score_without_lora}")
    print(f"Data dir         : {data_dir}")
    print(f"Config dir       : {config_dir}")
    print(f"Tasks            : {', '.join(args.tasks)}")
    print(f"Run name         : {args.run_name}")
    print(f"Chunk selection  : mode={args.score_mode}, chunk_size={args.chunk_size}, topk={args.topk_chunks}")
    print(f"Token pruning    : capacity={args.token_capacity}, query_window={args.token_score_query_window}")
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

            parts = render_prompt_parts(template, example, args.segment_separator)
            prompt_ids, prompt_meta = build_selected_prompt_ids(tokenizer, model, parts, args)
            print(
                f"[PureDreamLongBench] label={task}:{example_id} "
                f"raw_prompt_tokens={prompt_meta['raw_prompt_tokens']} "
                f"candidate_chunks={prompt_meta['candidate_chunks']} "
                f"kept_chunks={prompt_meta['selected_chunks']} "
                f"kept_chunk_indices={prompt_meta['selected_chunk_indices']} "
                f"context_tokens_after_pack={prompt_meta['context_tokens_after_pack']} "
                f"token_removed={prompt_meta['token_removed_total']}"
            )

            prediction = generate_official_dream(model, tokenizer, prompt_ids, prompt_meta, args)
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
                "prompt_tokens_after_pack": len(prompt_ids),
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
                    "dream_official_chunks_generation_lora"
                    if args.generation_lora_path
                    else "pure_dream_official_chunks"
                ),
                "generation_lora_path": args.generation_lora_path,
                "score_without_lora": args.score_without_lora,
                "chunk_size": args.chunk_size,
                "topk_chunks": args.topk_chunks,
                "token_capacity": args.token_capacity,
                "token_score_query_window": args.token_score_query_window,
                "score_mode": args.score_mode,
                "visibility_mode": args.visibility_mode,
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
