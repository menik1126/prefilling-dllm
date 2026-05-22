import argparse
import json
import os
from json import JSONDecodeError

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastdllm_v1_model import default_fastdllm_dream_dir, load_fastdllm_v1_dream
from infinitebench_tasks import (
    SUPPORTED_TASKS,
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt,
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Fast-dLLM v1 Dream on InfiniteBench")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--fastdllm_dream_dir", default=default_fastdllm_dream_dir())
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
    parser.add_argument("--dual_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation_strategy", choices=["head_tail", "left"], default="head_tail")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--output_dir", default="./results_infinitebench_fastdllm_v1")
    return parser


def main():
    args = build_arg_parser().parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}")

    data_dir = resolve_data_dir(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data dir           : {data_dir}")
    print(f"Tasks              : {', '.join(args.tasks)}")
    print(f"Prompt style       : {args.prompt_style}")
    print(f"Fast-dLLM Dream dir: {args.fastdllm_dream_dir}")
    print(f"Model max_length   : {args.max_length}")
    print(f"Generation max_new : {args.max_new_tokens}")
    print(f"Block length       : {args.block_length}")
    print(f"Diffusion steps    : {args.diffusion_steps or max(1, args.max_new_tokens // args.block_length)}")
    print(f"Algorithm          : {args.alg}")
    print(f"Threshold          : {args.threshold}")
    print(f"Dual cache         : {args.dual_cache}")
    print(f"Truncation         : {args.truncation_strategy}")

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
        dtype=args.dtype,
        add_bos_token=True,
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
            f"fastdllm_v1_infinitebench_{task}_n{len(examples)}_predictions.jsonl",
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
            prompt = create_prompt(example, task, args.prompt_style)
            answer_label = normalize_answer_label(task, example)
            prediction = model.generate([prompt], stop_tokens=args.stop_tokens)[0]
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
                "prompt_chars": len(prompt),
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
                "dual_cache": args.dual_cache,
                "truncation_strategy": args.truncation_strategy,
            }
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to: {out_file}")
        print(f"Accuracy             : {metrics['accuracy']:.4f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")
        all_results[task] = metrics

    combined_file = os.path.join(args.output_dir, f"fastdllm_v1_infinitebench_all_n{args.max_examples}.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
