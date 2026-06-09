#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_longbench_dream import load_json, load_task_examples as load_longbench, render_prompt_parts
from infinitebench_tasks import create_prompt_parts, load_task_examples as load_infinitebench


MODEL = "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/UltraLLaDA"
LONGBENCH_DATA = Path("/home/ma-user/work/ParallelComp_official/datasets/LongBench")
LONGBENCH_CONFIG = Path("/home/ma-user/work/ParallelComp_official/longbench_config")
INFINITEBENCH_DATA = Path("/home/ma-user/work/InfiniteBench/data")
OUT_DIR = Path("/home/ma-user/work/task/query_length_stats_20260524")

LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "trec",
    "triviaqa",
    "passage_count",
    "passage_retrieval_en",
    "qmsum",
    "samsum",
    "lcc",
    "multi_news",
    "repobench-p",
    "gov_report",
]

INFINITEBENCH_TASKS = [
    "passkey",
    "number_string",
    "kv_retrieval",
    "longbook_choice_eng",
    "math_find",
    "code_debug",
]


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def summarize(values):
    values = list(values)
    return {
        "min": min(values),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "mean": sum(values) / len(values),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def rounded(value):
    return "" if value is None else int(round(value))


def encode_len(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    longbench_templates = load_json(LONGBENCH_CONFIG / "dataset2prompt_raw.json")

    rows = []
    details = []

    for task in LONGBENCH_TASKS:
        examples = load_longbench(task, LONGBENCH_DATA, None)
        query_lengths = []
        scoring_query_lengths = []
        for idx, example in enumerate(examples):
            parts = render_prompt_parts(longbench_templates[task], example, "\n\n")
            query_len = encode_len(tokenizer, parts.get("query", ""))
            scoring_query_len = encode_len(tokenizer, parts.get("scoring_query", ""))
            query_lengths.append(query_len)
            scoring_query_lengths.append(scoring_query_len)
            details.append(
                {
                    "benchmark": "LongBench",
                    "task": task,
                    "idx": idx,
                    "query_tokens": query_len,
                    "scoring_query_tokens": scoring_query_len,
                    "query_chars": len(parts.get("query", "")),
                    "scoring_query_chars": len(parts.get("scoring_query", "")),
                }
            )
        rows.append(
            {
                "benchmark": "LongBench",
                "task": task,
                "n": len(examples),
                "query": summarize(query_lengths),
                "scoring_query": summarize(scoring_query_lengths),
            }
        )

    for task in INFINITEBENCH_TASKS:
        examples = load_infinitebench(task, INFINITEBENCH_DATA, None)
        query_lengths = []
        scoring_query_lengths = []
        for idx, example in enumerate(examples):
            parts = create_prompt_parts(example, task, "parallelcomp_raw")
            query_len = encode_len(tokenizer, parts.get("query", ""))
            scoring_query_len = encode_len(tokenizer, parts.get("scoring_query", ""))
            query_lengths.append(query_len)
            scoring_query_lengths.append(scoring_query_len)
            details.append(
                {
                    "benchmark": "InfiniteBench",
                    "task": task,
                    "idx": idx,
                    "query_tokens": query_len,
                    "scoring_query_tokens": scoring_query_len,
                    "query_chars": len(parts.get("query", "")),
                    "scoring_query_chars": len(parts.get("scoring_query", "")),
                }
            )
        rows.append(
            {
                "benchmark": "InfiniteBench",
                "task": task,
                "n": len(examples),
                "query": summarize(query_lengths),
                "scoring_query": summarize(scoring_query_lengths),
            }
        )

    payload = {
        "model": MODEL,
        "tokenizer": tokenizer.__class__.__name__,
        "longbench_prompt": "dataset2prompt_raw.json",
        "infinitebench_prompt_style": "parallelcomp_raw",
        "rows": rows,
        "details": details,
    }
    json_path = OUT_DIR / "query_length_stats_ultrallada_nochat_parallelcomp_raw.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "query_length_stats_ultrallada_nochat_parallelcomp_raw.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "benchmark,task,n,query_mean,query_p50,query_p90,query_p95,query_max,"
            "scoring_query_mean,scoring_query_p50,scoring_query_p90,scoring_query_p95,scoring_query_max\n"
        )
        for row in rows:
            q = row["query"]
            sq = row["scoring_query"]
            fields = [
                row["benchmark"],
                row["task"],
                row["n"],
                rounded(q["mean"]),
                rounded(q["p50"]),
                rounded(q["p90"]),
                rounded(q["p95"]),
                rounded(q["max"]),
                rounded(sq["mean"]),
                rounded(sq["p50"]),
                rounded(sq["p90"]),
                rounded(sq["p95"]),
                rounded(sq["max"]),
            ]
            f.write(",".join(map(str, fields)) + "\n")

    print(f"model={MODEL}")
    print(f"tokenizer={tokenizer.__class__.__name__}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(
        "benchmark\ttask\tn\tq_mean\tq_p50\tq_p90\tq_p95\tq_max\t"
        "sq_mean\tsq_p50\tsq_p90\tsq_p95\tsq_max"
    )
    for row in rows:
        q = row["query"]
        sq = row["scoring_query"]
        fields = [
            row["benchmark"],
            row["task"],
            row["n"],
            rounded(q["mean"]),
            rounded(q["p50"]),
            rounded(q["p90"]),
            rounded(q["p95"]),
            rounded(q["max"]),
            rounded(sq["mean"]),
            rounded(sq["p50"]),
            rounded(sq["p90"]),
            rounded(sq["p95"]),
            rounded(sq["max"]),
        ]
        print("\t".join(map(str, fields)))


if __name__ == "__main__":
    main()
