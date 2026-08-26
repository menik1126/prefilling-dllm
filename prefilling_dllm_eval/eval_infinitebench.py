import argparse
import json
import os
from json import JSONDecodeError

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from prefilling_model import generate, load_model
from infinitebench_tasks import (
    SUPPORTED_TASKS,
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt,
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


def estimate_prompt_chars(prompt):
    if isinstance(prompt, str):
        return len(prompt)
    if isinstance(prompt, dict):
        total = 0
        total += len(prompt.get("prefix", ""))
        total += len(prompt.get("context", ""))
        total += len(prompt.get("query", ""))
        total += sum(len(segment) for segment in prompt.get("context_segments", []) or [])
        return total
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Prefilling-dLLM with Dream on InfiniteBench")
    parser.add_argument("--model_type", choices=["llada", "dream"], required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument(
        "--prompt_style",
        choices=["parallelcomp_raw", "yarn-mistral", "gpt4", "slot_fill"],
        default="parallelcomp_raw",
        help="Official InfiniteBench prompt family to reuse.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Global generation cap. Defaults to the max official cap across selected tasks.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32768,
        help="Model total context window (tokens) passed into Dream/LLaDA runtime.",
    )
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--stop_tokens",
        nargs="*",
        default=[],
        help="Optional stop strings. Leave empty to rely on max_new_tokens only.",
    )
    parser.add_argument("--output_dir", default="./results_infinitebench")
    parser.add_argument(
        "--ntk_context_tail_truncation",
        action="store_true",
        help=(
            "For raw NTK Dream runs, pass structured prompts so the runtime can "
            "truncate the context tail while preserving the query."
        ),
    )
    parser.add_argument(
        "--ntk_head_tail_truncation",
        action="store_true",
        help=(
            "For raw NTK Dream runs, keep the beginning and end of the prompt "
            "and remove the middle when the prompt exceeds the NTK budget."
        ),
    )

    parser.add_argument("--parallelcomp_mode", action="store_true")
    parser.add_argument("--parallelcomp_pre_runtime_mode", action="store_true")
    parser.add_argument("--parallelcomp_cache_compress_mode", action="store_true")
    parser.add_argument("--parallelcomp_chunk_size", type=int, default=256)
    parser.add_argument("--parallelcomp_query_tokens", type=int, default=0)
    parser.add_argument("--parallelcomp_topk_chunks", type=int, default=4)
    parser.add_argument("--parallelcomp_min_prompt_tokens", type=int, default=1024)
    parser.add_argument("--parallelcomp_keep_first_chunk", action="store_true")
    parser.add_argument("--parallelcomp_split_from_tail", action="store_true")
    parser.add_argument(
        "--parallelcomp_chunk_score_query_window",
        type=int,
        default=0,
        help="Query-token window for chunk self-information scoring. 0 means full query.",
    )
    parser.add_argument(
        "--parallelcomp_chunk_score_attention_mask",
        choices=["causal", "full", "full_visible", "query_to_chunk", "prefix_full"],
        default="query_to_chunk",
        help="Local [chunk, query] attention mask used for one-forward chunk scoring and token eviction.",
    )
    parser.add_argument(
        "--parallelcomp_recent_token_window",
        type=int,
        default=0,
        help="Query-token window used for token eviction attention scoring. 0 means full query.",
    )
    parser.add_argument("--parallelcomp_hidden_topk", type=int, default=32)
    parser.add_argument("--parallelcomp_token_capacity", type=int, default=128)
    parser.add_argument("--parallelcomp_token_keep_min", type=int, default=32)
    parser.add_argument("--parallelcomp_high_score_threshold", type=float, default=None)
    parser.add_argument("--parallelcomp_select_low_score_chunks", action="store_true")
    parser.add_argument(
        "--parallelcomp_fixed_query_text",
        default="Please answer the question using the long context above.",
    )
    parser.add_argument(
        "--parallelcomp_tail_replay_full_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parallelcomp_score_mode", type=str, default="self_information")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}")
    if args.ntk_context_tail_truncation and args.ntk_head_tail_truncation:
        raise ValueError("Choose only one NTK truncation strategy.")

    data_dir = resolve_data_dir(args.data_dir)
    generation_max_new_tokens = args.max_new_tokens or max(
        TASK_TO_MAX_NEW_TOKENS[task] for task in args.tasks
    )

    print(f"Data dir           : {data_dir}")
    print(f"Tasks              : {', '.join(args.tasks)}")
    print(f"Prompt style       : {args.prompt_style}")
    print(f"Model max_length   : {args.max_length}")
    print(f"Generation max_new : {generation_max_new_tokens}")
    print(f"NTK tail truncation: {args.ntk_context_tail_truncation}")
    print(f"NTK head/tail trunc: {args.ntk_head_tail_truncation}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_type} model from {args.pretrained}...")
    model = load_model(
        args.model_type,
        args.pretrained,
        args.lora_path,
        rope_scale_factor=args.rope_scale_factor,
        max_new_tokens=generation_max_new_tokens,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=args.temperature,
        add_bos_token=True,
        parallelcomp_mode=args.parallelcomp_mode,
        parallelcomp_pre_runtime_mode=args.parallelcomp_pre_runtime_mode,
        parallelcomp_cache_compress_mode=args.parallelcomp_cache_compress_mode,
        parallelcomp_chunk_size=args.parallelcomp_chunk_size,
        parallelcomp_query_tokens=args.parallelcomp_query_tokens,
        parallelcomp_topk_chunks=args.parallelcomp_topk_chunks,
        parallelcomp_min_prompt_tokens=args.parallelcomp_min_prompt_tokens,
        parallelcomp_keep_first_chunk=args.parallelcomp_keep_first_chunk,
        parallelcomp_split_from_tail=args.parallelcomp_split_from_tail,
        parallelcomp_chunk_score_query_window=args.parallelcomp_chunk_score_query_window,
        parallelcomp_chunk_score_attention_mask=args.parallelcomp_chunk_score_attention_mask,
        parallelcomp_recent_token_window=args.parallelcomp_recent_token_window,
        parallelcomp_hidden_topk=args.parallelcomp_hidden_topk,
        parallelcomp_token_capacity=args.parallelcomp_token_capacity,
        parallelcomp_token_keep_min=args.parallelcomp_token_keep_min,
        parallelcomp_high_score_threshold=args.parallelcomp_high_score_threshold,
        parallelcomp_select_low_score_chunks=args.parallelcomp_select_low_score_chunks,
        parallelcomp_fixed_query_text=args.parallelcomp_fixed_query_text,
        parallelcomp_tail_replay_full_mask=args.parallelcomp_tail_replay_full_mask,
        parallelcomp_score_mode=args.parallelcomp_score_mode,
    )

    all_results = {}

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)

        examples = load_task_examples(task, data_dir, max_examples=args.max_examples)
        print(f"Loaded examples    : {len(examples)}")
        print(f"Official max_new   : {TASK_TO_MAX_NEW_TOKENS[task]}")

        out_file = os.path.join(
            args.output_dir,
            f"{args.model_type}_infinitebench_{task}_n{len(examples)}_predictions.jsonl",
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

            if (
                args.model_type == "dream"
                and (
                    args.parallelcomp_pre_runtime_mode
                    or args.ntk_context_tail_truncation
                    or args.ntk_head_tail_truncation
                )
            ):
                prompt = create_prompt_parts(example, task, args.prompt_style)
                prompt["metadata_label"] = f"{task}:{example_id}"
                if args.ntk_context_tail_truncation:
                    prompt["ntk_truncation_strategy"] = "context_tail"
                elif args.ntk_head_tail_truncation:
                    prompt["ntk_truncation_strategy"] = "head_tail"
            else:
                prompt = create_prompt(example, task, args.prompt_style)
            answer_label = normalize_answer_label(task, example)
            prediction = generate(model, [prompt], stop_tokens=args.stop_tokens)[0]
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
                "prompt_chars": estimate_prompt_chars(prompt),
            }
            if "options" in example:
                record["options"] = example["options"]

            predictions.append(record)
            append_prediction_record(out_file, record)
            completed_ids.add(example_id)

            if (idx + 1) % 10 == 0:
                running = summarize_metrics(predictions)
                print(
                    f"  [{idx + 1}/{len(examples)}] "
                    f"running_acc={running['accuracy']:.4f}"
                )

        metrics = summarize_metrics(predictions)
        metrics["task"] = task
        metrics["max_examples"] = len(examples)
        metrics["prompt_style"] = args.prompt_style
        metrics["ntk_context_tail_truncation"] = bool(args.ntk_context_tail_truncation)
        metrics["ntk_head_tail_truncation"] = bool(args.ntk_head_tail_truncation)
        metrics["generation_max_new_tokens"] = generation_max_new_tokens
        metrics["official_task_max_new_tokens"] = TASK_TO_MAX_NEW_TOKENS[task]

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {out_file}")
        print(f"Accuracy             : {metrics['accuracy']:.4f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")

        all_results[task] = metrics

    combined_file = os.path.join(
        args.output_dir,
        f"{args.model_type}_infinitebench_all_n{args.max_examples}.json",
    )
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
