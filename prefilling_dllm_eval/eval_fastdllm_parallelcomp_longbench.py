#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
from fastdllm_parallelcomp import (
    dataclass_to_jsonable,
    default_fastdllm_dream_dir,
    default_fastdllm_llada_dir,
    load_fastdllm_parallelcomp,
)


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


def apply_chat_template_to_parts(tokenizer, parts):
    sentinel = "__PARALLELCOMP_CONTEXT_SENTINEL__"
    prefix = parts.get("prefix", "")
    query = parts.get("query", "")
    if sentinel in prefix or sentinel in query or sentinel in parts.get("context", ""):
        raise ValueError("Chat template sentinel unexpectedly appears in prompt text")
    user_content = f"{prefix}{sentinel}{query}"
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support apply_chat_template")
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if sentinel not in chat_text:
        raise ValueError("Chat template removed the context sentinel")
    chat_prefix, chat_query = chat_text.split(sentinel, 1)
    templated = dict(parts)
    templated["prefix"] = chat_prefix
    templated["query"] = chat_query
    templated["scoring_query"] = chat_query
    return templated


def build_token_parts(model, parts, add_bos_token, use_chat_template=False):
    effective_parts = apply_chat_template_to_parts(model.tokenizer, parts) if use_chat_template else parts
    # Chat templates already include their own BOS / role-control tokens.
    external_bos_ids = [] if use_chat_template else (model.bos_ids() if add_bos_token else [])
    prefix_ids = external_bos_ids + encode_fragment(
        model.tokenizer,
        effective_parts.get("prefix", ""),
    )
    context_ids = encode_fragment(model.tokenizer, effective_parts.get("context", ""))
    query_ids = encode_fragment(model.tokenizer, effective_parts.get("query", ""))
    scoring_query_ids = encode_fragment(model.tokenizer, effective_parts.get("scoring_query", ""))
    if not scoring_query_ids:
        scoring_query_ids = list(query_ids)
    return prefix_ids, context_ids, query_ids, scoring_query_ids, effective_parts


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
    parser.add_argument("--score_draft_fixed_steps", action="store_true")
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
    parser.add_argument(
        "--score_context_mode",
        choices=["single_chunk", "joint_chunks_target_last"],
        default="single_chunk",
    )
    parser.add_argument("--attention_score_layers", type=int, default=4)
    parser.add_argument("--attention_query_window", type=int, default=0)
    parser.add_argument("--token_capacity", type=int, default=0)
    parser.add_argument("--token_score_query_window", type=int, default=8)
    parser.add_argument("--token_score_layers", type=int, default=0)
    parser.add_argument("--token_score_layer_mode", choices=["first", "last", "all"], default="all")
    parser.add_argument("--token_score_reduce", choices=["sum", "mean"], default="sum")
    parser.add_argument("--token_score_pooling", choices=["none", "avgpool", "maxpool"], default="maxpool")
    parser.add_argument("--token_score_pool_kernel", type=int, default=7)
    parser.add_argument("--token_score_head_reduce", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--token_score_layer_reduce", choices=["sum", "mean", "max"], default="mean")
    parser.add_argument(
        "--token_score_direction",
        choices=["query_to_chunk", "chunk_to_query", "bidirectional"],
        default="query_to_chunk",
    )
    parser.add_argument("--token_score_keep", choices=["high", "low"], default="high")
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
        default="causal",
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
        description="Evaluate Fast-dLLM v1 with full ParallelComp KV runtime on LongBench."
    )
    parser.add_argument("--pretrained", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--model_backend", choices=["dream", "llada"], default="dream")
    parser.add_argument("--fastdllm_dream_dir", default=default_fastdllm_dream_dir())
    parser.add_argument("--fastdllm_llada_dir", default=default_fastdllm_llada_dir())
    parser.add_argument("--llada_score_batch_size", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--run_name", default="fastdllm_parallelcomp_longbench")
    parser.add_argument("--output_dir", default="./results_longbench_fastdllm_parallelcomp")
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
    parser.add_argument("--rope_scaling_type", choices=["yarn", "linear"], default="yarn")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--longbench_e", action="store_true")
    parser.add_argument("--segment_separator", default="\n\n")
    add_parallelcomp_args(parser)
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

    print(f"Data dir              : {data_dir}")
    print(f"Config dir            : {config_dir}")
    print(f"Tasks                 : {', '.join(args.tasks)}")
    print(f"Run name              : {args.run_name}")
    print(f"Model backend         : {args.model_backend}")
    print(f"Fast-dLLM Dream dir   : {args.fastdllm_dream_dir}")
    print(f"Fast-dLLM LLaDA dir   : {args.fastdllm_llada_dir}")
    print(f"Model max_length      : {args.max_length}")
    print(f"Generation max_new    : {args.max_new_tokens}")
    print(f"Block length          : {args.block_length}")
    print(f"Diffusion steps       : {args.diffusion_steps}")
    print(f"Score mode            : {args.score_mode}")
    print(f"Score draft steps     : {args.score_draft_steps}")
    print(f"Score fixed steps     : {args.score_draft_fixed_steps}")
    print(f"Score partial steps   : {args.score_draft_partial_steps}")
    print(f"Score partial rounds  : {args.score_draft_partial_rounds}")
    print(f"Score all draft slots : {args.score_draft_score_all_slots}")
    print(f"Score batch size      : {args.score_batch_size}")
    print(f"LLaDA shifted score   : {args.score_llada_shift_logits}")
    print(f"Score attention mask  : {args.score_attention_mask}")
    print(f"Score context mode    : {args.score_context_mode}")
    print(f"Chunk size/top-k      : {args.chunk_size}/{args.topk_chunks}")
    print(f"Token capacity        : {args.token_capacity}")
    print(f"Token score generated : {args.token_score_use_generated}")
    print(f"Token eviction gran.  : {args.token_eviction_granularity}")
    print(
        "Token score config    : "
        f"mask={args.token_attention_mask}, query_window={args.token_score_query_window}, "
        f"layers={args.token_score_layer_mode}:{args.token_score_layers}, "
        f"pool={args.token_score_pooling}/{args.token_score_pool_kernel}, "
        f"head_reduce={args.token_score_head_reduce}, layer_reduce={args.token_score_layer_reduce}, "
        f"direction={args.token_score_direction}, keep={args.token_score_keep}, "
        f"include_prefix={args.token_score_include_prefix}"
    )
    print(f"Chunk BOS             : {args.chunk_bos}")
    print(f"Cache build mode      : {args.cache_build_mode}")
    print(f"Chunk position mode   : {args.chunk_position_mode}")
    print(f"Query position mode   : {args.query_position_mode}")
    print(f"Use chat template     : {args.use_chat_template}")
    print(f"RoPE scale factor     : {args.rope_scale_factor}")
    print(f"RoPE scaling type     : {args.rope_scaling_type}")

    os.makedirs(args.output_dir, exist_ok=True)
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
        rope_scaling_type=args.rope_scaling_type,
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
        score_draft_fixed_steps=args.score_draft_fixed_steps,
        score_draft_partial_steps=args.score_draft_partial_steps,
        score_draft_partial_rounds=args.score_draft_partial_rounds,
        score_draft_score_all_slots=args.score_draft_score_all_slots,
        score_llada_shift_logits=args.score_llada_shift_logits,
        score_batch_size=args.score_batch_size,
        score_attention_mask=args.score_attention_mask,
        score_context_mode=args.score_context_mode,
        attention_score_layers=args.attention_score_layers,
        attention_query_window=args.attention_query_window,
        token_capacity=args.token_capacity,
        token_score_query_window=args.token_score_query_window,
        token_score_layers=args.token_score_layers,
        token_score_layer_mode=args.token_score_layer_mode,
        token_score_reduce=args.token_score_reduce,
        token_score_pooling=args.token_score_pooling,
        token_score_pool_kernel=args.token_score_pool_kernel,
        token_score_head_reduce=args.token_score_head_reduce,
        token_score_layer_reduce=args.token_score_layer_reduce,
        token_score_direction=args.token_score_direction,
        token_score_keep=args.token_score_keep,
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
        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples       : {len(examples)}")
        print(f"Official max_new      : {dataset2maxlen[task]}")

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
            parts = render_prompt_parts(template, example, args.segment_separator)
            prefix_ids, context_ids, query_ids, scoring_query_ids, effective_parts = build_token_parts(
                model,
                parts,
                args.add_bos_token,
                args.use_chat_template,
            )
            result = model.generate(
                prefix_ids=prefix_ids,
                context_ids=context_ids,
                query_ids=query_ids,
                scoring_query_ids=scoring_query_ids,
            )
            prediction = trim_stop_tokens(result.text, args.stop_tokens)
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
                "prompt_chars": estimate_prompt_chars(effective_parts),
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
                "block_length": args.block_length,
                "diffusion_steps": model.diffusion_steps,
                "alg": args.alg,
                "threshold": args.threshold,
                "rope_scale_factor": args.rope_scale_factor,
                "rope_scaling_type": args.rope_scaling_type,
                "use_chat_template": args.use_chat_template,
                "chunk_size": args.chunk_size,
                "topk_chunks": args.topk_chunks,
                "chunk_bos": args.chunk_bos,
                "cache_build_mode": args.cache_build_mode,
                "score_mode": args.score_mode,
                "score_draft_tokens": args.score_draft_tokens,
                "score_draft_steps": args.score_draft_steps,
                "score_draft_fixed_steps": args.score_draft_fixed_steps,
                "score_draft_partial_steps": args.score_draft_partial_steps,
                "score_draft_partial_rounds": args.score_draft_partial_rounds,
                "score_draft_score_all_slots": args.score_draft_score_all_slots,
                "score_llada_shift_logits": args.score_llada_shift_logits,
                "score_batch_size": args.score_batch_size,
                "score_attention_mask": args.score_attention_mask,
                "score_context_mode": args.score_context_mode,
                "token_capacity": args.token_capacity,
                "token_score_use_generated": args.token_score_use_generated,
                "token_score_layer_mode": args.token_score_layer_mode,
                "token_score_layers": args.token_score_layers,
                "token_score_query_window": args.token_score_query_window,
                "token_score_reduce": args.token_score_reduce,
                "token_score_pooling": args.token_score_pooling,
                "token_score_pool_kernel": args.token_score_pool_kernel,
                "token_score_head_reduce": args.token_score_head_reduce,
                "token_score_layer_reduce": args.token_score_layer_reduce,
                "token_score_direction": args.token_score_direction,
                "token_score_keep": args.token_score_keep,
                "token_score_include_prefix": args.token_score_include_prefix,
                "token_attention_mask": args.token_attention_mask,
                "chunk_position_mode": args.chunk_position_mode,
                "token_eviction_granularity": args.token_eviction_granularity,
                "query_position_mode": args.query_position_mode,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to : {out_file}")
        print(f"LongBench score      : {metrics['score']:.2f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")
        all_results[task] = metrics

    combined_file = Path(args.output_dir) / f"{args.run_name}_all_n{args.max_examples}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
