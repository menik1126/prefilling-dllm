import argparse
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from d2f_model import generate, load_model
from infinitebench_tasks import (
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt_parts,
    load_task_examples,
    normalize_answer_label,
    resolve_data_dir,
    score_prediction,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate DREAM on InfiniteBench with oracle context chunks")
    parser.add_argument("--model_type", choices=["dream"], default="dream")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--prompt_style", default="parallelcomp_raw")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument(
        "--oracle_mode",
        choices=["chunk", "answer_only", "pair_only"],
        default="chunk",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--query_instruction", default="")
    parser.add_argument(
        "--query_format",
        choices=["default", "kv_value_stub"],
        default="default",
    )
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument("--output_dir", required=True)
    return parser


def summarize_metrics(records):
    scores = [int(record["correct"]) for record in records]
    return {
        "accuracy": (sum(scores) / len(scores)) if scores else 0.0,
        "n": len(scores),
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    max_new_tokens = args.max_new_tokens or TASK_TO_MAX_NEW_TOKENS[args.task]
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(
        args.model_type,
        args.pretrained,
        args.lora_path,
        max_new_tokens=max_new_tokens,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=args.temperature,
        add_bos_token=True,
        parallelcomp_pre_runtime_mode=True,
        parallelcomp_chunk_size=args.chunk_size,
        parallelcomp_topk_chunks=1,
        parallelcomp_min_prompt_tokens=1,
    )
    inner = model._inner

    examples = load_task_examples(args.task, data_dir, max_examples=args.max_examples)
    predictions = []

    pred_path = os.path.join(
        args.output_dir,
        f"{args.model_type}_oracle_chunk_{args.task}_n{len(examples)}_predictions.jsonl",
    )
    metrics_path = pred_path.replace("_predictions.jsonl", "_metrics.json")

    for idx, example in enumerate(examples):
        prompt_parts = create_prompt_parts(example, args.task, args.prompt_style)
        context_ids = inner.tokenizer.encode(example["context"], add_special_tokens=False)
        answer_label = normalize_answer_label(args.task, example)
        answer_text = answer_label[0] if isinstance(answer_label, list) else answer_label
        answer_text = str(answer_text)

        answer_char_pos = example["context"].find(answer_text)
        if answer_char_pos < 0:
            raise ValueError(f"Could not find oracle answer text in context for idx={idx}")

        answer_prefix_ids = inner.tokenizer.encode(
            example["context"][:answer_char_pos],
            add_special_tokens=False,
        )
        oracle_chunk_idx = len(answer_prefix_ids) // args.chunk_size
        chunk_start = oracle_chunk_idx * args.chunk_size
        chunk_end = min(chunk_start + args.chunk_size, len(context_ids))
        oracle_chunk_text = inner.tokenizer.decode(
            context_ids[chunk_start:chunk_end],
            skip_special_tokens=False,
        )

        oracle_context_text = oracle_chunk_text
        if args.oracle_mode == "answer_only":
            oracle_context_text = answer_text
        elif args.oracle_mode == "pair_only":
            if args.task != "kv_retrieval":
                raise ValueError("--oracle_mode pair_only is currently only supported for kv_retrieval")
            lines = example["input"].splitlines()
            key_line = next((line for line in lines if line.startswith("Key:")), example["input"].strip())
            key = key_line.split(":", 1)[1].strip() if ":" in key_line else key_line
            oracle_context_text = f"JSON data:\n{{{key}: \"{answer_text}\"}}"

        prompt_query = prompt_parts["query"]
        if args.query_format == "kv_value_stub":
            if args.task != "kv_retrieval":
                raise ValueError("--query_format kv_value_stub is currently only supported for kv_retrieval")
            lines = example["input"].splitlines()
            key_line = next((line for line in lines if line.startswith("Key:")), example["input"].strip())
            prompt_query = f"{key_line}\nValue:"

        prompt = {
            "prefix": prompt_parts["prefix"],
            "context": oracle_context_text,
            "query": (
                (args.query_instruction.rstrip() + "\n\n") if args.query_instruction else ""
            ) + prompt_query,
            "metadata_label": f"{args.task}:{idx}:oracle_{args.oracle_mode}_{oracle_chunk_idx}",
        }

        prediction = generate(model, [prompt], stop_tokens=args.stop_tokens)[0]
        correct = score_prediction(args.task, prediction, answer_label)

        record = {
            "task": args.task,
            "example_id": example.get("id", idx),
            "index": idx,
            "correct": correct,
            "prediction": prediction,
            "answer": answer_label,
            "oracle_chunk_idx": oracle_chunk_idx,
            "oracle_chunk_token_span": [chunk_start, chunk_end],
            "oracle_mode": args.oracle_mode,
            "answer_char_pos": answer_char_pos,
            "answer_in_oracle_chunk": answer_text in oracle_chunk_text,
            "input_chars": len(example.get("input", "")),
            "oracle_chunk_chars": len(oracle_context_text),
        }
        predictions.append(record)

        with open(pred_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0:
            running = summarize_metrics(predictions)
            print(f"[{idx + 1}/{len(examples)}] running_acc={running['accuracy']:.4f}")

    metrics = summarize_metrics(predictions)
    metrics["task"] = args.task
    metrics["max_examples"] = len(examples)
    metrics["prompt_style"] = args.prompt_style
    metrics["max_length"] = args.max_length
    metrics["max_new_tokens"] = max_new_tokens
    metrics["chunk_size"] = args.chunk_size
    metrics["oracle_mode"] = args.oracle_mode
    metrics["query_instruction"] = args.query_instruction
    metrics["query_format"] = args.query_format

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to: {pred_path}")
    print(f"Accuracy             : {metrics['accuracy']:.4f} (n={metrics['n']})")
    print(f"Metrics saved to     : {metrics_path}")


if __name__ == "__main__":
    main()
