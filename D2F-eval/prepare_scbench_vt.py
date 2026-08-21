#!/usr/bin/env python3
"""Download and flatten SCBench Retr.MultiHop (scbench_vt)."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://datasets-server.huggingface.co"


def fetch_json(endpoint: str, **params):
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def split_prefix_context(raw_input: str):
    marker = "\n\n"
    if marker not in raw_input:
        return "", raw_input
    prefix, context = raw_input.split(marker, 1)
    return prefix.strip() + "\n\n", context


def iter_rows(dataset: str, config: str, split: str):
    offset = 0
    page_size = 100
    while True:
        payload = fetch_json(
            "rows",
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=page_size,
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item["row"]
        offset += len(rows)
        total = payload.get("num_rows_total")
        if total is not None and offset >= total:
            break
        if len(rows) < page_size:
            break


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="microsoft/SCBench")
    parser.add_argument("--config", default="scbench_vt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="data_scbench")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    original_rows = []
    flat_records = []
    first_turn_records = []
    context_lengths = []
    answer_lengths = []

    for row in iter_rows(args.dataset, args.config, args.split):
        row_id = int(row["index"])
        prefix, context = split_prefix_context(row["input"])
        turns = row.get("multi_turns") or []
        original_rows.append(
            {
                "row_id": row_id,
                "length": row.get("length"),
                "prefix": prefix,
                "context": context,
                "multi_turns": turns,
            }
        )
        context_lengths.append(len(context))
        for turn_id, turn in enumerate(turns):
            answer = turn["answer"]
            record = {
                "id": f"scbench_vt_{row_id:03d}_turn_{turn_id}",
                "row_id": row_id,
                "turn_id": turn_id,
                "task": "scbench_vt",
                "source_dataset": args.dataset,
                "source_config": args.config,
                "source_split": args.split,
                "length": row.get("length"),
                "prefix": prefix,
                "context": context,
                "input": turn["input"],
                "answer": answer,
                "answers": [answer],
            }
            flat_records.append(record)
            if turn_id == 0:
                first_turn_records.append(record)
            if isinstance(answer, list):
                answer_lengths.append(len(answer))
            else:
                answer_lengths.append(1)

    flat_path = out_dir / "scbench_vt.jsonl"
    rows_path = out_dir / "scbench_vt_rows.jsonl"
    first_path = out_dir / "scbench_vt_first_turn.jsonl"
    manifest_path = out_dir / "scbench_vt_manifest.json"

    n_flat = write_jsonl(flat_path, flat_records)
    n_rows = write_jsonl(rows_path, original_rows)
    n_first = write_jsonl(first_path, first_turn_records)

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "rows": n_rows,
        "flat_queries": n_flat,
        "first_turn_queries": n_first,
        "turns_per_row": sorted({len(r["multi_turns"]) for r in original_rows}),
        "context_chars_min": min(context_lengths) if context_lengths else 0,
        "context_chars_max": max(context_lengths) if context_lengths else 0,
        "context_chars_avg": round(statistics.mean(context_lengths), 2) if context_lengths else 0,
        "answer_items_min": min(answer_lengths) if answer_lengths else 0,
        "answer_items_max": max(answer_lengths) if answer_lengths else 0,
        "files": {
            "flat": str(flat_path),
            "rows": str(rows_path),
            "first_turn": str(first_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
