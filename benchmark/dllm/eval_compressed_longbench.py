"""Evaluate reference-selected LongBench prompts through an SGLang server."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import requests


TASK = "multifieldqa_en"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def save_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    args = parse_args()
    sys.path.insert(0, args.eval_dir)
    from eval_fastdllm_parallelcomp_longbench import (
        load_task_examples,
        score_prediction,
        summarize_metrics,
        trim_stop_tokens,
    )

    with open(args.compressed, encoding="utf-8") as f:
        records = json.load(f)["records"]
    records = records[args.start :]
    if args.limit is not None:
        records = records[: args.limit]
    examples = load_task_examples(TASK, args.data_dir, max_examples=0)

    output_path = Path(args.output)
    results = []
    started = time.time()
    for offset, record in enumerate(records):
        index = record["idx"]
        request_started = time.time()
        response = requests.post(
            f"{args.base_url}/generate",
            json={
                "input_ids": record["prompt_ids"],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                },
            },
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        prediction = trim_stop_tokens(payload["text"], ["</s>", "<|im_end|>"])
        example = examples[index]
        score = score_prediction(
            TASK,
            prediction,
            example.get("answers", []),
            example.get("all_classes"),
        )
        results.append(
            {
                "index": index,
                "example_id": record["example_id"],
                "pred": prediction,
                "answers": example.get("answers", []),
                "score": score,
                "output_ids": payload.get("output_ids"),
                "latency_seconds": time.time() - request_started,
                "meta_info": payload.get("meta_info"),
                "selected_chunk_indices": record["selected_chunk_indices"],
            }
        )
        metrics = summarize_metrics(results, longbench_e=False)
        save_json(
            output_path,
            {
                "completed": len(results),
                "requested": len(records),
                "start": args.start,
                "elapsed_seconds": time.time() - started,
                "metrics": metrics,
                "results": results,
            },
        )
        print(
            f"DONE completed={len(results)}/{len(records)} index={index} "
            f"score={metrics['score']:.2f} latency={results[-1]['latency_seconds']:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
