#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for path in (THIS_DIR, PARENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import torch
import transformers
from dream_prefill_kv import (
    DreamPrefillKVConfig,
    DreamPrefillKVController,
    dataclass_to_jsonable,
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


def resolve_dtype(dtype):
    if dtype == "auto" or dtype is None:
        return dtype
    return getattr(torch, dtype)


def apply_dream_rope_scaling(model, factor):
    factor = float(factor or 1.0)
    if factor <= 1.0:
        return

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot apply RoPE scaling: model has no config")
    original_max_pos = getattr(config, "max_position_embeddings", None)
    if original_max_pos is None:
        raise ValueError("Cannot apply RoPE scaling: config has no max_position_embeddings")

    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_pos,
    }
    from model_cache.dream.model_dream import DreamRotaryEmbedding

    device = next(model.parameters()).device
    for module in model.modules():
        if isinstance(module, DreamRotaryEmbedding):
            module.__init__(config=config)
            module.to(device)


def load_dream_model_and_tokenizer(pretrained, lora_path, dtype, device, rope_scale_factor):
    from model_cache.dream.configuration_dream import DreamConfig
    from model_cache.dream.model_dream import DreamModel

    target_dtype = resolve_dtype(dtype)
    model_config = DreamConfig.from_pretrained(pretrained)
    model = DreamModel.from_pretrained(
        pretrained,
        config=model_config,
        torch_dtype=target_dtype,
        trust_remote_code=False,
    ).eval()

    if lora_path:
        from peft import PeftConfig, PeftModel

        PeftConfig.from_pretrained(lora_path)
        model = PeftModel.from_pretrained(model, lora_path)

    if target_dtype is not None and target_dtype != "auto":
        model = model.to(target_dtype)
    model = model.to(device).eval()
    apply_dream_rope_scaling(model, rope_scale_factor)
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
    return model, tokenizer


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def bos_ids(tokenizer, add_bos_token):
    if not add_bos_token:
        return []
    if tokenizer.bos_token_id is not None:
        return [int(tokenizer.bos_token_id)]
    if tokenizer.bos_token:
        return tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
    return []


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


def build_token_parts(tokenizer, parts, add_bos_token):
    prefix_ids = bos_ids(tokenizer, add_bos_token) + encode_fragment(tokenizer, parts.get("prefix", ""))
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))
    query_ids = encode_fragment(tokenizer, parts.get("query", ""))
    scoring_query_ids = encode_fragment(tokenizer, parts.get("scoring_query", ""))
    return prefix_ids, context_ids, query_ids, scoring_query_ids


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Dream with a generic ParallelComp-style prefill KV controller on LongBench."
    )
    parser.add_argument("--pretrained", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--tasks", nargs="+", default=["multifieldqa_en"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--output_dir", default="./results_longbench_prefill_kv")
    parser.add_argument("--run_name", default="dream_prefill_kv_longbench")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Number of examples per task. Use 0 or a negative value for the full file.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--longbench_e", action="store_true")

    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk_chunks", type=int, default=3)
    parser.add_argument("--keep_first_chunk", action="store_true")
    parser.add_argument("--split_from_tail", action="store_true")
    parser.add_argument("--chunk_bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--chunk_cache_mode",
        choices=["independent", "sequential", "joint_selected"],
        default="independent",
        help=(
            "How selected chunk KV is built: independent is the corrected "
            "ParallelComp-style path; sequential preserves the old incremental "
            "path; joint_selected full-forwards all selected chunks together."
        ),
    )
    parser.add_argument(
        "--score_mode",
        choices=["self_information", "draft_self_information", "attention", "none"],
        default="self_information",
    )
    parser.add_argument("--score_query_window", type=int, default=0)
    parser.add_argument("--score_disable_adapter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention_score_layers", type=int, default=4)
    parser.add_argument("--attention_query_window", type=int, default=8)
    parser.add_argument("--score_draft_tokens", type=int, default=0)
    parser.add_argument("--score_draft_steps", type=int, default=None)

    parser.add_argument(
        "--token_capacity",
        type=int,
        default=0,
        help="If >0, prune each selected chunk inside KV cache to this many tokens.",
    )
    parser.add_argument("--token_score_query_window", type=int, default=8)
    parser.add_argument("--token_score_layers", type=int, default=4)
    parser.add_argument("--token_score_layer_mode", choices=["first", "last", "all"], default="last")
    parser.add_argument("--token_score_reduce", choices=["sum", "mean"], default="sum")
    parser.add_argument(
        "--token_eviction_mode",
        choices=["cache_slice", "first_layer_recompute"],
        default="cache_slice",
    )
    parser.add_argument(
        "--chunk_position_mode",
        choices=["reuse", "continuous", "absolute"],
        default="reuse",
    )
    parser.add_argument(
        "--query_position_mode",
        choices=["after_reused_window", "after_cache", "after_selected_chunks"],
        default="after_reused_window",
    )
    parser.add_argument("--replay_full_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment_separator", default="\n\n")
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

    print(f"Model path          : {args.pretrained}")
    print(f"LoRA path           : {args.lora_path or 'None'}")
    print("Model mode          : Dream prefill KV controller")
    print(f"Data dir            : {data_dir}")
    print(f"Config dir          : {config_dir}")
    print(f"Tasks               : {', '.join(args.tasks)}")
    print(f"Run name            : {args.run_name}")
    print(f"Chunk selection     : mode={args.score_mode}, chunk_size={args.chunk_size}, topk={args.topk_chunks}")
    print(f"Context chunk BOS   : {args.chunk_bos}")
    print(f"Chunk cache mode    : {args.chunk_cache_mode}")
    print(
        "KV token eviction   : "
        f"capacity={args.token_capacity}, query_window={args.token_score_query_window}, "
        f"layers={args.token_score_layer_mode}:{args.token_score_layers}, mode={args.token_eviction_mode}"
    )
    print(f"Position modes      : chunk={args.chunk_position_mode}, query={args.query_position_mode}")
    print(f"Replay full mask    : {args.replay_full_mask}")
    print(f"Max new tokens      : {args.max_new_tokens}")

    model, tokenizer = load_dream_model_and_tokenizer(
        pretrained=args.pretrained,
        lora_path=args.lora_path,
        dtype=args.dtype,
        device=args.device,
        rope_scale_factor=args.rope_scale_factor,
    )

    controller_config = DreamPrefillKVConfig(
        chunk_size=args.chunk_size,
        topk_chunks=args.topk_chunks,
        keep_first_chunk=args.keep_first_chunk,
        split_from_tail=args.split_from_tail,
        chunk_bos=args.chunk_bos,
        chunk_cache_mode=args.chunk_cache_mode,
        score_mode=args.score_mode,
        score_query_window=args.score_query_window,
        score_disable_adapter=args.score_disable_adapter,
        attention_score_layers=args.attention_score_layers,
        attention_query_window=args.attention_query_window,
        score_draft_tokens=args.score_draft_tokens,
        score_draft_steps=args.score_draft_steps,
        token_capacity=args.token_capacity,
        token_score_query_window=args.token_score_query_window,
        token_score_layers=args.token_score_layers,
        token_score_layer_mode=args.token_score_layer_mode,
        token_score_reduce=args.token_score_reduce,
        token_eviction_mode=args.token_eviction_mode,
        chunk_position_mode=args.chunk_position_mode,
        query_position_mode=args.query_position_mode,
        replay_full_mask=args.replay_full_mask,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
    )
    controller = DreamPrefillKVController(model=model, tokenizer=tokenizer, config=controller_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)

        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples     : {len(examples)}")
        print(f"Official max_new    : {dataset2maxlen[task]}")

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
            prefix_ids, context_ids, query_ids, scoring_query_ids = build_token_parts(
                tokenizer,
                parts,
                add_bos_token=args.add_bos_token,
            )
            result = controller.run(
                prefix_ids=prefix_ids,
                context_ids=context_ids,
                query_ids=query_ids,
                scoring_query_ids=scoring_query_ids,
            )
            prediction = trim_stop_tokens(result.text, args.stop_tokens)
            answers = example.get("answers", [])
            all_classes = example.get("all_classes")
            score = score_prediction(task, prediction, answers, all_classes)

            meta = dataclass_to_jsonable(result)
            meta.pop("text", None)
            meta.pop("sequences", None)
            print(
                f"[PrefillKV LongBench] label={task}:{example_id} "
                f"raw_context_tokens={result.raw_context_tokens} "
                f"candidate_chunks={result.candidate_chunks} "
                f"kept_chunks={len(result.selected_chunk_indices)} "
                f"kept_chunk_indices={result.selected_chunk_indices} "
                f"cache_tokens={result.cache_tokens} "
                f"removed_tokens={result.removed_tokens}"
            )

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
                "prefill_kv_meta": meta,
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
                "model_mode": "dream_prefill_kv_controller",
                "pretrained": args.pretrained,
                "lora_path": args.lora_path,
                "chunk_size": args.chunk_size,
                "topk_chunks": args.topk_chunks,
                "keep_first_chunk": args.keep_first_chunk,
                "chunk_bos": args.chunk_bos,
                "chunk_cache_mode": args.chunk_cache_mode,
                "score_mode": args.score_mode,
                "score_draft_tokens": args.score_draft_tokens,
                "score_draft_steps": args.score_draft_steps,
                "token_capacity": args.token_capacity,
                "token_score_query_window": args.token_score_query_window,
                "token_score_layers": args.token_score_layers,
                "token_score_layer_mode": args.token_score_layer_mode,
                "token_score_reduce": args.token_score_reduce,
                "token_eviction_mode": args.token_eviction_mode,
                "chunk_position_mode": args.chunk_position_mode,
                "query_position_mode": args.query_position_mode,
                "replay_full_mask": args.replay_full_mask,
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
