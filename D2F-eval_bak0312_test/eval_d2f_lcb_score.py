"""
Score pre-generated LCB outputs using LiveCodeBench's evaluation pipeline.
Run with the LiveCodeBench venv:
  source /home/hq/LiveCodeBench/.venv/bin/activate
  python eval_d2f_lcb_score.py --input_file results_d2f/llada_lcb_...json
"""
import argparse
import json
import sys
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/home/hq/LiveCodeBench")
os.chdir("/home/hq/LiveCodeBench")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--release_version", default="release_v5")
    parser.add_argument("--num_process_evaluate", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()

    from lcb_runner.benchmarks import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

    print("Loading dataset for evaluation...")
    dataset = load_code_generation_dataset(release_version=args.release_version)
    benchmark_map = {p.question_id: p for p in dataset}

    print(f"Loading generations from: {args.input_file}")
    with open(args.input_file) as f:
        results = json.load(f)

    # Match generations to benchmark problems
    eval_samples = []
    generations = []
    for r in results:
        qid = r["question_id"]
        if qid in benchmark_map:
            eval_samples.append(benchmark_map[qid].get_evaluation_sample())
            generations.append(r["code_list"])
        else:
            print(f"Warning: question_id {qid} not found in dataset")

    print(f"Matched {len(eval_samples)} problems")

    # Run evaluation
    summary, per_problem, _ = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )

    print("\n=== LCB Results ===")
    print(f"  pass@1: {summary.get('pass@1', 0):.4f}")
    print(f"  pass@5: {summary.get('pass@5', 0):.4f}")

    out_file = args.input_file.replace(".json", "_eval.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
