#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREFILLING_DLLM_ENGINE_DIR="$REPO_DIR/prefilling_dllm_engine"
PREFILLING_DLLM_EVAL_DIR="${PREFILLING_DLLM_EVAL_DIR:-$REPO_DIR/prefilling_dllm_engine_eval}"

PYTHON="${PYTHON:-/home/ma-user/work/conda-envs/prefilling_dllm_eval_parallelcomp/bin/python}"
DREAM_BASE="${DREAM_BASE:-$PREFILLING_DLLM_EVAL_DIR/model_weights/Dream-v0-Base-7B}"
FASTDLLM_DREAM="${FASTDLLM_DREAM:-/home/ma-user/work/Fast-dLLM/v1/dream}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20}"
WARMUP="${WARMUP:-3}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$PREFILLING_DLLM_EVAL_DIR:${PYTHONPATH:-}"

LOG_DIR="$PREFILLING_DLLM_ENGINE_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/parallelcomp_hf_profiling_${RUN_TS}.log"
RESULT_FILE="$LOG_DIR/parallelcomp_hf_profiling_${RUN_TS}.json"

echo "============================================"
echo "  ParallelComp HF Profiling (Selection + Decode)"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "Num examples        : $NUM_EXAMPLES"
echo "Warmup examples     : $WARMUP"
echo "Run timestamp       : $RUN_TS"
echo "Log file            : $LOG_FILE"
echo "Result file         : $RESULT_FILE"
echo "============================================"

cd "$PREFILLING_DLLM_EVAL_DIR"

export REPO_DIR PREFILLING_DLLM_ENGINE_DIR PREFILLING_DLLM_EVAL_DIR DREAM_BASE FASTDLLM_DREAM DATA_DIR CONFIG_DIR
export RUN_TS NUM_EXAMPLES WARMUP RESULT_FILE

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_FILE"
import json
import os
import sys
import time
from pathlib import Path

import torch

prefilling_dllm_eval_dir = os.environ["PREFILLING_DLLM_EVAL_DIR"]
if prefilling_dllm_eval_dir not in sys.path:
    sys.path.insert(0, prefilling_dllm_eval_dir)

from eval_fastdllm_parallelcomp_longbench import (
    build_token_parts,
    load_json,
    load_task_examples,
    render_prompt_parts,
    trim_stop_tokens,
)
from fastdllm_parallelcomp import load_fastdllm_parallelcomp

TASK = "multifieldqa_en"
STOP_STRINGS = ["</s>", "<|im_end|>", "\n"]

dream_base = os.environ["DREAM_BASE"]
fastdllm_dream = os.environ["FASTDLLM_DREAM"]
data_dir = os.environ["DATA_DIR"]
config_dir = os.environ["CONFIG_DIR"]
run_ts = os.environ["RUN_TS"]
result_file = os.environ["RESULT_FILE"]
num_examples = int(os.environ.get("NUM_EXAMPLES", "20"))
warmup = int(os.environ.get("WARMUP", "3"))

prompt_templates = load_json(Path(config_dir) / "dataset2prompt_raw.json")

print("Loading LongBench examples...", flush=True)
examples = load_task_examples(TASK, data_dir, max_examples=0)
examples = examples[:num_examples + warmup]
print(f"Loaded {len(examples)} examples ({warmup} warmup + {num_examples} measured)", flush=True)

# ============================================================
# Load ParallelComp HF model (selection + decode in one)
# ============================================================
print("\nLoading ParallelComp HF model...", flush=True)
model = load_fastdllm_parallelcomp(
    pretrained=dream_base,
    fastdllm_dream_dir=fastdllm_dream,
    max_new_tokens=32,
    max_length=4096,
    block_length=32,
    temperature=0.0,
    alg="confidence_threshold",
    threshold=0.9,
    rope_scale_factor=1.0,
    dtype="bfloat16",
    add_bos_token=True,
    chunk_size=1024,
    topk_chunks=4,
    chunk_bos=True,
    force_keep_chunk_bos=True,
    cache_build_mode="full_prompt_mask",
    score_mode="draft_self_information",
    score_draft_tokens=4,
    score_draft_partial_rounds=1,
    score_attention_mask="causal",
    score_context_mode="single_chunk",
    token_capacity=0,
    token_score_layer_mode="first",
    token_score_layers=1,
    chunk_position_mode="continuous",
    query_position_mode="after_selected_chunks",
)
print("Model loaded.", flush=True)

# ============================================================
# Profile: end-to-end (selection + decode)
# ============================================================
template = prompt_templates[TASK]
per_sample = []

print(f"\nRunning: {warmup} warmup + {num_examples} measured...", flush=True)
for idx, example in enumerate(examples):
    parts = render_prompt_parts(template, example, "\n")
    prefix_ids, context_ids, query_ids, scoring_query_ids = build_token_parts(
        model, parts, add_bos_token=True,
    )

    torch.cuda.synchronize()
    t0 = time.time()

    result = model.generate(
        prefix_ids=prefix_ids,
        context_ids=context_ids,
        query_ids=query_ids,
        scoring_query_ids=scoring_query_ids,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    prediction = trim_stop_tokens(result.text, STOP_STRINGS)
    n_tokens = len([t for t in result.sequences if t != 0])

    if idx >= warmup:
        per_sample.append({
            "idx": idx - warmup,
            "latency_s": elapsed,
            "tokens_generated": n_tokens,
            "prompt_tokens": result.cache_tokens,
            "context_tokens": result.raw_context_tokens,
            "selected_chunks": len(result.selected_chunk_indices),
            "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0,
        })
    label = "warmup" if idx < warmup else "measured"
    print(f"  [{label}] {idx+1}/{len(examples)} tokens={n_tokens} "
          f"latency={elapsed:.3f}s tps={n_tokens/elapsed:.1f}", flush=True)

del model
torch.cuda.empty_cache()

total_tokens = sum(s["tokens_generated"] for s in per_sample)
total_time = sum(s["latency_s"] for s in per_sample)
avg_latency = total_time / len(per_sample)
tps = total_tokens / total_time if total_time > 0 else 0

print(f"\n--- ParallelComp HF Results ({num_examples} samples) ---", flush=True)
print(f"  Total tokens:     {total_tokens}", flush=True)
print(f"  Total time:       {total_time:.2f}s", flush=True)
print(f"  Avg latency:      {avg_latency:.3f}s/sample", flush=True)
print(f"  Throughput:       {tps:.2f} tokens/s", flush=True)

result_data = {
    "run_ts": run_ts,
    "task": TASK,
    "num_examples": num_examples,
    "warmup": warmup,
    "method": "parallelcomp_hf",
    "config": {
        "chunk_size": 1024,
        "topk_chunks": 4,
        "block_length": 32,
        "max_new_tokens": 32,
        "threshold": 0.9,
        "score_mode": "draft_self_information",
        "score_draft_tokens": 4,
        "cache_build_mode": "full_prompt_mask",
    },
    "avg_latency_s": avg_latency,
    "throughput_tps": tps,
    "total_tokens": total_tokens,
    "total_time_s": total_time,
    "per_sample": per_sample,
}

with open(result_file, "w") as f:
    json.dump(result_data, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to: {result_file}", flush=True)
PY
