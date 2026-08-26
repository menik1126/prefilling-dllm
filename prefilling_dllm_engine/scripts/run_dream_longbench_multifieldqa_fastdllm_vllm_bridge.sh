#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREFILLING_DLLM_ENGINE_DIR="$REPO_DIR/prefilling_dllm_engine"
PREFILLING_DLLM_EVAL_DIR="${PREFILLING_DLLM_EVAL_DIR:-$REPO_DIR/prefilling_dllm_engine_eval}"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
DREAM_BASE="${DREAM_BASE:-$PREFILLING_DLLM_EVAL_DIR/model_weights/Dream-v0-Base-7B}"
FASTDLLM_DREAM="${FASTDLLM_DREAM:-/home/ma-user/work/Fast-dLLM/v1/dream}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
SAMPLE_BATCH_CHUNK_SIZE="${SAMPLE_BATCH_CHUNK_SIZE:-50}"

# Current best multifieldqa_en selector setting.
PC_CHUNK_SIZE="${PC_CHUNK_SIZE:-1024}"
TOPK_CHUNKS="${TOPK_CHUNKS:-4}"
SCORE_MODE="${SCORE_MODE:-draft_self_information}"
SCORE_DRAFT_TOKENS="${SCORE_DRAFT_TOKENS:-4}"
SCORE_DRAFT_PARTIAL_ROUNDS="${SCORE_DRAFT_PARTIAL_ROUNDS:-1}"
CACHE_BUILD_MODE="${CACHE_BUILD_MODE:-full_prompt_mask}"
CHUNK_POSITION_MODE="${CHUNK_POSITION_MODE:-continuous}"
QUERY_POSITION_MODE="${QUERY_POSITION_MODE:-after_selected_chunks}"
TOKEN_CAPACITY="${TOKEN_CAPACITY:-0}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
ACCEPT_THRESHOLD="${ACCEPT_THRESHOLD:-0.9}"
COMPLETE_THRESHOLD="${COMPLETE_THRESHOLD:-0.0}"
ADD_NEW_BLOCK_THRESHOLD="${ADD_NEW_BLOCK_THRESHOLD:-1.0}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_DIR:$PREFILLING_DLLM_EVAL_DIR:${PYTHONPATH:-}"

export REPO_DIR PREFILLING_DLLM_ENGINE_DIR PREFILLING_DLLM_EVAL_DIR DREAM_BASE FASTDLLM_DREAM DATA_DIR CONFIG_DIR
export START_INDEX LIMIT RUN_TS CHECK_ONLY DRY_RUN SAMPLE_BATCH_CHUNK_SIZE
export PC_CHUNK_SIZE TOPK_CHUNKS SCORE_MODE SCORE_DRAFT_TOKENS SCORE_DRAFT_PARTIAL_ROUNDS
export CACHE_BUILD_MODE CHUNK_POSITION_MODE QUERY_POSITION_MODE TOKEN_CAPACITY
export MAX_NEW_TOKENS BLOCK_LENGTH MAX_MODEL_LEN TENSOR_PARALLEL_SIZE MAX_NUM_SEQS
export GPU_MEMORY_UTILIZATION ACCEPT_THRESHOLD COMPLETE_THRESHOLD ADD_NEW_BLOCK_THRESHOLD

LOG_DIR="$PREFILLING_DLLM_ENGINE_DIR/log"
mkdir -p "$LOG_DIR"
RESULTS_TAG="longbench_multifieldqa_en_fastdllm_vllm_bridge"
LOG_FILE="$LOG_DIR/${RESULTS_TAG}_${RUN_TS}.log"

echo "============================================"
echo "  Fast-DLLM selector + vLLM engine - LongBench multifieldqa_en"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "Fast-DLLM Dream dir : $FASTDLLM_DREAM"
echo "CUDA devices        : $CUDA_VISIBLE_DEVICES"
echo "Run timestamp       : $RUN_TS"
echo "Selector setting    : topk=$TOPK_CHUNKS chunk=$PC_CHUNK_SIZE score=$SCORE_MODE draft=$SCORE_DRAFT_TOKENS partial_rounds=$SCORE_DRAFT_PARTIAL_ROUNDS cache=$CACHE_BUILD_MODE positions=$CHUNK_POSITION_MODE/$QUERY_POSITION_MODE"
echo "Decode setting      : prefilling_dllm LLM engine block=$BLOCK_LENGTH max_new=$MAX_NEW_TOKENS max_model_len=$MAX_MODEL_LEN"
echo "Log file            : $LOG_FILE"
echo "============================================"

cd "$PREFILLING_DLLM_ENGINE_DIR"

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_FILE"
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

repo_dir = os.environ["REPO_DIR"]
prefilling_dllm_eval_dir = os.environ["PREFILLING_DLLM_EVAL_DIR"]
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if prefilling_dllm_eval_dir not in sys.path:
    sys.path.insert(0, prefilling_dllm_eval_dir)

from prefilling_dllm import LLM, SamplingParams
from eval_fastdllm_parallelcomp_longbench import (
    build_token_parts,
    load_json,
    load_task_examples,
    render_prompt_parts,
    score_prediction,
    summarize_metrics,
    trim_stop_tokens,
)
from fastdllm_parallelcomp import load_fastdllm_parallelcomp


TASK = "multifieldqa_en"
STOP_STRINGS = ["</s>", "<|im_end|>"]


def env_int(name, default):
    value = os.environ.get(name, "")
    return int(value) if value != "" else int(default)


def env_float(name, default):
    value = os.environ.get(name, "")
    return float(value) if value != "" else float(default)


def save_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


dream_base = os.environ["DREAM_BASE"]
fastdllm_dream = os.environ["FASTDLLM_DREAM"]
data_dir = os.environ["DATA_DIR"]
config_dir = os.environ["CONFIG_DIR"]
log_dir = Path(os.environ["PREFILLING_DLLM_ENGINE_DIR"]) / "log"
run_ts = os.environ["RUN_TS"]
prompt_templates = load_json(Path(config_dir) / "dataset2prompt_raw.json")
dataset2maxlen = load_json(Path(config_dir) / "dataset2maxlen.json")

start_index = env_int("START_INDEX", 0)
limit_env = os.environ.get("LIMIT", "")
limit = int(limit_env) if limit_env else None
sample_batch_chunk_size = env_int("SAMPLE_BATCH_CHUNK_SIZE", 50)
check_only = env_int("CHECK_ONLY", 0)
dry_run = env_int("DRY_RUN", 0)

pc_chunk_size = env_int("PC_CHUNK_SIZE", 1024)
topk_chunks = env_int("TOPK_CHUNKS", 4)
score_draft_tokens = env_int("SCORE_DRAFT_TOKENS", 4)
score_draft_partial_rounds = env_int("SCORE_DRAFT_PARTIAL_ROUNDS", 1)
token_capacity = env_int("TOKEN_CAPACITY", 0)
max_new_tokens = env_int("MAX_NEW_TOKENS", 32)
block_length = env_int("BLOCK_LENGTH", 32)
max_model_len = env_int("MAX_MODEL_LEN", 8192)
tensor_parallel_size = env_int("TENSOR_PARALLEL_SIZE", 1)
max_num_seqs = env_int("MAX_NUM_SEQS", 2)
gpu_memory_utilization = env_float("GPU_MEMORY_UTILIZATION", 0.60)
accept_threshold = env_float("ACCEPT_THRESHOLD", 0.9)
complete_threshold = env_float("COMPLETE_THRESHOLD", 0.9)
add_new_block_threshold = env_float("ADD_NEW_BLOCK_THRESHOLD", 0.1)

if check_only:
    print("CHECK_ONLY=1: imports and paths are valid; no model is loaded.", flush=True)
    raise SystemExit(0)
if token_capacity != 0:
    raise ValueError("This bridge intentionally supports TOKEN_CAPACITY=0 only.")

result_path = log_dir / f"longbench_multifieldqa_en_fastdllm_vllm_bridge_results_{run_ts}.json"
compressed_path = log_dir / f"longbench_multifieldqa_en_fastdllm_vllm_bridge_compressed_{run_ts}.json"

print("Loading LongBench examples...", flush=True)
examples = load_task_examples(TASK, data_dir, max_examples=0)
examples = examples[start_index:]
if limit is not None:
    examples = examples[:limit]
print(f"Loaded examples: {len(examples)}", flush=True)
print(f"Official max_new for {TASK}: {dataset2maxlen[TASK]}", flush=True)

print("Loading Fast-DLLM ParallelComp selector...", flush=True)
selector = load_fastdllm_parallelcomp(
    pretrained=dream_base,
    fastdllm_dream_dir=fastdllm_dream,
    max_new_tokens=max_new_tokens,
    max_length=4096,
    block_length=block_length,
    temperature=0.0,
    alg="confidence_threshold",
    threshold=0.9,
    rope_scale_factor=1.0,
    dtype="bfloat16",
    add_bos_token=True,
    chunk_size=pc_chunk_size,
    topk_chunks=topk_chunks,
    chunk_bos=True,
    force_keep_chunk_bos=True,
    cache_build_mode=os.environ["CACHE_BUILD_MODE"],
    score_mode=os.environ["SCORE_MODE"],
    score_draft_tokens=score_draft_tokens,
    score_draft_partial_rounds=score_draft_partial_rounds,
    score_attention_mask="causal",
    score_context_mode="single_chunk",
    token_capacity=token_capacity,
    token_score_layer_mode="first",
    token_score_layers=1,
    chunk_position_mode=os.environ["CHUNK_POSITION_MODE"],
    query_position_mode=os.environ["QUERY_POSITION_MODE"],
)

compressed_records = []
selection_t0 = time.time()
template = prompt_templates[TASK]
for idx, example in enumerate(examples):
    parts = render_prompt_parts(template, example, "\n")
    prefix_ids, context_ids, query_ids, scoring_query_ids = build_token_parts(
        selector,
        parts,
        add_bos_token=True,
    )
    candidate_chunks, selected_indices, chunk_scores, selection_query_ids, score_token_mask = (
        selector._prepare_candidate_chunks(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            scoring_query_ids=scoring_query_ids,
        )
    )
    eviction_query_ids = selector._token_eviction_query_ids(
        scoring_query_ids=scoring_query_ids,
        selection_query_ids=selection_query_ids,
        score_token_mask=score_token_mask,
    )
    prompt_ids = list(prefix_ids)
    prompt_positions = selector._range_positions(0, len(prefix_ids))
    cache_positions = list(prompt_positions)
    for chunk_order, chunk_idx in enumerate(selected_indices):
        original_chunk_ids = list(candidate_chunks[chunk_idx])
        chunk_start = selector._chunk_rope_start(len(prefix_ids), chunk_order, chunk_idx)
        original_chunk_positions = selector._range_positions(chunk_start, len(original_chunk_ids))
        keep_positions = selector._keep_positions_for_chunk(prefix_ids, original_chunk_ids, eviction_query_ids)
        chunk_ids = [original_chunk_ids[pos] for pos in keep_positions]
        chunk_positions = [original_chunk_positions[pos] for pos in keep_positions]
        prompt_ids.extend(chunk_ids)
        prompt_positions.extend(chunk_positions)
        cache_positions.extend(chunk_positions)
    query_rope_start = selector._final_query_rope_start(
        len(prefix_ids),
        cache_positions,
        selected_count=len(selected_indices),
    )
    query_positions = selector._range_positions(query_rope_start, len(query_ids))
    prompt_ids.extend(query_ids)
    prompt_positions.extend(query_positions)
    if len(prompt_ids) + max_new_tokens > max_model_len:
        raise ValueError(
            f"Compressed prompt too long for max_model_len={max_model_len}: "
            f"idx={idx}, prompt={len(prompt_ids)}, max_new={max_new_tokens}"
        )
    compressed_records.append(
        {
            "idx": start_index + idx,
            "example_id": example.get("_id", idx),
            "prompt_ids": prompt_ids,
            "prompt_positions": prompt_positions,
            "selected_chunk_indices": selected_indices,
            "chunk_scores": {str(k): float(v) for k, v in chunk_scores.items()},
            "prompt_meta": {
                "prefix_tokens": len(prefix_ids),
                "context_tokens": len(context_ids),
                "query_tokens": len(query_ids),
                "compressed_prompt_tokens": len(prompt_ids),
                "max_position": max(prompt_positions) if prompt_positions else -1,
                "query_rope_start": query_rope_start,
            },
        }
    )
    if (idx + 1) % 10 == 0:
        print(f"  selected {idx + 1}/{len(examples)}", flush=True)

selection_seconds = time.time() - selection_t0
save_json(
    compressed_path,
    {
        "task": TASK,
        "run_ts": run_ts,
        "bridge_mode": "fastdllm_selector_then_prefilling_dllm_LLM_engine",
        "selection_seconds": selection_seconds,
        "setting": {
            "topk_chunks": topk_chunks,
            "parallelcomp_chunk_size": pc_chunk_size,
            "score_mode": os.environ["SCORE_MODE"],
            "score_draft_tokens": score_draft_tokens,
            "score_draft_partial_rounds": score_draft_partial_rounds,
            "cache_build_mode_label": os.environ["CACHE_BUILD_MODE"],
            "chunk_position_mode": os.environ["CHUNK_POSITION_MODE"],
            "query_position_mode": os.environ["QUERY_POSITION_MODE"],
            "token_capacity": token_capacity,
        },
        "records": compressed_records,
    },
)
print(f"Compressed prompts saved to: {compressed_path}", flush=True)
print(f"Selection time: {selection_seconds:.2f}s", flush=True)

del selector
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if dry_run:
    print("DRY_RUN=1, stopping after prompt compression.", flush=True)
    raise SystemExit(0)

print("Loading prefilling_dllm LLM engine...", flush=True)
llm = LLM(
    dream_base,
    model_name="dream",
    model_type="diffusion_lm",
    enforce_eager=True,
    data_parallel_size=1,
    tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=gpu_memory_utilization,
    max_num_batched_tokens=max_model_len,
    max_num_seqs=max_num_seqs,
    max_model_len=max_model_len,
    diffusion_block_size=block_length,
    accept_threshold=accept_threshold,
    complete_threshold=complete_threshold,
    add_new_block_threshold=add_new_block_threshold,
    kv_cache_layout="unified",
)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=max_new_tokens,
    stop=STOP_STRINGS,
)

results = []
decode_t0 = time.time()
for chunk_start in range(0, len(compressed_records), sample_batch_chunk_size):
    chunk_end = min(chunk_start + sample_batch_chunk_size, len(compressed_records))
    chunk_records = compressed_records[chunk_start:chunk_end]
    chunk_examples = examples[chunk_start:chunk_end]
    chunk_t0 = time.time()
    print(f"\nRunning engine chunk {chunk_start}:{chunk_end}...", flush=True)
    outputs = llm.generate(
        [r["prompt_ids"] for r in chunk_records],
        sampling_params,
        use_tqdm=True,
        prompt_positions=[r["prompt_positions"] for r in chunk_records],
    )
    for local_idx, (output, example, compressed) in enumerate(zip(outputs, chunk_examples, chunk_records)):
        idx = chunk_start + local_idx
        prediction = trim_stop_tokens(output["text"], STOP_STRINGS)
        answers = example.get("answers", [])
        all_classes = example.get("all_classes")
        score = score_prediction(TASK, prediction, answers, all_classes)
        results.append(
            {
                "task": TASK,
                "example_id": compressed["example_id"],
                "index": start_index + idx,
                "pred": prediction,
                "answers": answers,
                "all_classes": all_classes,
                "score": score,
                "length": example.get("length"),
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "token_count": len(output["token_ids"]),
                "n_diff_steps": output.get("n_diff_steps"),
                "parallelcomp_bridge": {
                    "selected_chunk_indices": compressed["selected_chunk_indices"],
                    "chunk_scores": compressed["chunk_scores"],
                    "prompt_meta": compressed["prompt_meta"],
                },
            }
        )
    metrics = summarize_metrics(results, longbench_e=False)
    payload = {
        "task": TASK,
        "run_ts": run_ts,
        "bridge_mode": "fastdllm_selector_then_prefilling_dllm_LLM_engine",
        "completed": len(results),
        "total": len(examples),
        "selection_seconds": selection_seconds,
        "decode_seconds": time.time() - decode_t0,
        "metrics": metrics,
        "setting": {
            "topk_chunks": topk_chunks,
            "parallelcomp_chunk_size": pc_chunk_size,
            "score_mode": os.environ["SCORE_MODE"],
            "score_draft_tokens": score_draft_tokens,
            "score_draft_partial_rounds": score_draft_partial_rounds,
            "cache_build_mode_label": os.environ["CACHE_BUILD_MODE"],
            "chunk_position_mode": os.environ["CHUNK_POSITION_MODE"],
            "query_position_mode": os.environ["QUERY_POSITION_MODE"],
            "token_capacity": token_capacity,
            "max_model_len": max_model_len,
            "max_new_tokens": max_new_tokens,
            "block_length": block_length,
            "accept_threshold": accept_threshold,
            "complete_threshold": complete_threshold,
            "add_new_block_threshold": add_new_block_threshold,
        },
        "results": results,
    }
    save_json(result_path, payload)
    print(
        f"Chunk {chunk_start}:{chunk_end} score={metrics['score']:.2f} "
        f"completed={len(results)}/{len(examples)} chunk_time={time.time() - chunk_t0:.2f}s",
        flush=True,
    )

final_metrics = summarize_metrics(results, longbench_e=False)
print()
print("=" * 60)
print("LongBench multifieldqa_en Fast-DLLM selector + vLLM engine")
print("=" * 60)
print(f"Score             : {final_metrics['score']:.2f}")
print(f"Completed         : {len(results)}/{len(examples)}")
print(f"Selection seconds : {selection_seconds:.2f}")
print(f"Decode seconds    : {time.time() - decode_t0:.2f}")
print(f"Results saved to  : {result_path}")
print("=" * 60)
PY
