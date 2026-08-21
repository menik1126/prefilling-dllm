#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
D2F_VLLM_DIR="$REPO_DIR/d2f_vllm"
D2F_EVAL_DIR="${D2F_EVAL_DIR:-$REPO_DIR/D2F-eval}"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/d2f/bin/python}"
DREAM_BASE="${DREAM_BASE:-$D2F_EVAL_DIR/model_weights/Dream-v0-Base-7B}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"
TASK_NAME="${LONGBENCH_TASK:-multifieldqa_en}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-2333}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

PC_CHUNK_SIZE="${PC_CHUNK_SIZE:-1024}"
TOPK_CHUNKS="${TOPK_CHUNKS:-4}"
SCORE_MODE="${SCORE_MODE:-draft_self_information}"
SCORE_DRAFT_TOKENS="${SCORE_DRAFT_TOKENS:-4}"
SCORE_DRAFT_PARTIAL_ROUNDS="${SCORE_DRAFT_PARTIAL_ROUNDS:-1}"
SCORE_DRAFT_SCORE_ALL_SLOTS="${SCORE_DRAFT_SCORE_ALL_SLOTS:-0}"
SCORE_ATTENTION_MASK="${SCORE_ATTENTION_MASK:-causal}"
SCORE_CONTEXT_MODE="${SCORE_CONTEXT_MODE:-single_chunk}"
CACHE_BUILD_MODE="${CACHE_BUILD_MODE:-full_prompt_mask}"
CHUNK_POSITION_MODE="${CHUNK_POSITION_MODE:-continuous}"
QUERY_POSITION_MODE="${QUERY_POSITION_MODE:-after_selected_chunks}"
KEEP_FIRST_CHUNK="${KEEP_FIRST_CHUNK:-0}"
SPLIT_FROM_TAIL="${SPLIT_FROM_TAIL:-0}"
CHUNK_BOS="${CHUNK_BOS:-1}"
FORCE_KEEP_CHUNK_BOS="${FORCE_KEEP_CHUNK_BOS:-1}"

TOKEN_CAPACITY="${TOKEN_CAPACITY:-512}"
TOKEN_SCORE_QUERY_WINDOW="${TOKEN_SCORE_QUERY_WINDOW:-8}"
TOKEN_SCORE_LAYERS="${TOKEN_SCORE_LAYERS:-0}"
TOKEN_SCORE_LAYER_MODE="${TOKEN_SCORE_LAYER_MODE:-all}"
TOKEN_SCORE_REDUCE="${TOKEN_SCORE_REDUCE:-sum}"
TOKEN_SCORE_POOLING="${TOKEN_SCORE_POOLING:-maxpool}"
TOKEN_SCORE_POOL_KERNEL="${TOKEN_SCORE_POOL_KERNEL:-7}"
TOKEN_SCORE_HEAD_REDUCE="${TOKEN_SCORE_HEAD_REDUCE:-sum}"
TOKEN_SCORE_LAYER_REDUCE="${TOKEN_SCORE_LAYER_REDUCE:-mean}"
TOKEN_SCORE_DIRECTION="${TOKEN_SCORE_DIRECTION:-query_to_chunk}"
TOKEN_SCORE_KEEP="${TOKEN_SCORE_KEEP:-high}"
TOKEN_SCORE_INCLUDE_PREFIX="${TOKEN_SCORE_INCLUDE_PREFIX:-1}"
TOKEN_SCORE_USE_GENERATED="${TOKEN_SCORE_USE_GENERATED:-0}"
TOKEN_ATTENTION_MASK="${TOKEN_ATTENTION_MASK:-causal}"
TOKEN_SCORE_BACKEND="${TOKEN_SCORE_BACKEND:-torch}"
TOKEN_EVICTION_GRANULARITY="${TOKEN_EVICTION_GRANULARITY:-per_head}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
THRESHOLD="${THRESHOLD:-0.9}"
KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-unified}"
DECODE_DELTA_MODE="${DECODE_DELTA_MODE:-none}"
DECODE_DELTA_STRIDE="${DECODE_DELTA_STRIDE:-4}"
DECODE_DELTA_LEFT="${DECODE_DELTA_LEFT:-3}"
DECODE_DELTA_SCALE="${DECODE_DELTA_SCALE:-1.0}"
DECODE_DELTA_DEBUG="${DECODE_DELTA_DEBUG:-0}"
PREFILL_SPARSE_MODE="${PREFILL_SPARSE_MODE:-none}"
PREFILL_DELTA_MODE="${PREFILL_DELTA_MODE:-none}"
PREFILL_DELTA_STRIDE="${PREFILL_DELTA_STRIDE:-8}"
PREFILL_DELTA_LEFT="${PREFILL_DELTA_LEFT:-7}"
PREFILL_DELTA_SCALE="${PREFILL_DELTA_SCALE:-1.0}"
PREFILL_DELTA_DEBUG="${PREFILL_DELTA_DEBUG:-0}"
PD_REMOTE_ENGINE="${PD_REMOTE_ENGINE:-0}"
PD_PIPELINE_OVERLAP="${PD_PIPELINE_OVERLAP:-0}"
PD_DECODE_DEVICE_START="${PD_DECODE_DEVICE_START:-1}"
PD_DECODE_MASTER_PORT="${PD_DECODE_MASTER_PORT:-$((MASTER_PORT + 1))}"
PD_DECODE_SHM_NAME="${PD_DECODE_SHM_NAME:-d2f_vllm_pd_decode}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export MASTER_PORT
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_DIR:$D2F_EVAL_DIR:${PYTHONPATH:-}"

export REPO_DIR D2F_VLLM_DIR D2F_EVAL_DIR DREAM_BASE DATA_DIR CONFIG_DIR TASK_NAME
export START_INDEX LIMIT RUN_TS CHECK_ONLY DRY_RUN
export PC_CHUNK_SIZE TOPK_CHUNKS SCORE_MODE SCORE_DRAFT_TOKENS SCORE_DRAFT_PARTIAL_ROUNDS SCORE_DRAFT_SCORE_ALL_SLOTS
export SCORE_ATTENTION_MASK SCORE_CONTEXT_MODE CACHE_BUILD_MODE CHUNK_POSITION_MODE QUERY_POSITION_MODE KEEP_FIRST_CHUNK
export SPLIT_FROM_TAIL CHUNK_BOS FORCE_KEEP_CHUNK_BOS
export TOKEN_CAPACITY TOKEN_SCORE_QUERY_WINDOW TOKEN_SCORE_LAYERS TOKEN_SCORE_LAYER_MODE TOKEN_SCORE_REDUCE
export TOKEN_SCORE_POOLING TOKEN_SCORE_POOL_KERNEL TOKEN_SCORE_HEAD_REDUCE TOKEN_SCORE_LAYER_REDUCE
export TOKEN_SCORE_DIRECTION TOKEN_SCORE_KEEP TOKEN_SCORE_INCLUDE_PREFIX TOKEN_SCORE_USE_GENERATED
export TOKEN_ATTENTION_MASK TOKEN_SCORE_BACKEND TOKEN_EVICTION_GRANULARITY MAX_NEW_TOKENS BLOCK_LENGTH MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION THRESHOLD KV_CACHE_LAYOUT
export DECODE_DELTA_MODE DECODE_DELTA_STRIDE DECODE_DELTA_LEFT DECODE_DELTA_SCALE DECODE_DELTA_DEBUG
export PREFILL_SPARSE_MODE PREFILL_DELTA_MODE PREFILL_DELTA_STRIDE PREFILL_DELTA_LEFT PREFILL_DELTA_SCALE PREFILL_DELTA_DEBUG
export PD_REMOTE_ENGINE PD_PIPELINE_OVERLAP PD_DECODE_DEVICE_START PD_DECODE_MASTER_PORT PD_DECODE_SHM_NAME

LOG_DIR="$D2F_VLLM_DIR/log"
mkdir -p "$LOG_DIR"
RESULTS_TAG="longbench_${TASK_NAME}_fastdllm_engine_full_bridge"
LOG_FILE="$LOG_DIR/${RESULTS_TAG}_${RUN_TS}.log"

echo "============================================"
echo "  Full engine bridge - LongBench ${TASK_NAME}"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "CUDA devices        : $CUDA_VISIBLE_DEVICES"
echo "Master port         : $MASTER_PORT"
echo "Run timestamp       : $RUN_TS"
echo "Chunk selection     : backend=engine topk=$TOPK_CHUNKS chunk=$PC_CHUNK_SIZE score=$SCORE_MODE draft=$SCORE_DRAFT_TOKENS partial_rounds=$SCORE_DRAFT_PARTIAL_ROUNDS mask=$SCORE_ATTENTION_MASK"
echo "Token eviction      : capacity=$TOKEN_CAPACITY granularity=$TOKEN_EVICTION_GRANULARITY backend=engine score_backend=$TOKEN_SCORE_BACKEND layers=$TOKEN_SCORE_LAYER_MODE:$TOKEN_SCORE_LAYERS pool=$TOKEN_SCORE_POOLING/$TOKEN_SCORE_POOL_KERNEL"
echo "Prefill sparse      : mode=$PREFILL_SPARSE_MODE delta=$PREFILL_DELTA_MODE stride=$PREFILL_DELTA_STRIDE left=$PREFILL_DELTA_LEFT scale=$PREFILL_DELTA_SCALE debug=$PREFILL_DELTA_DEBUG"
echo "Decode setting      : FastDLLMDreamEngine block=$BLOCK_LENGTH max_new=$MAX_NEW_TOKENS delta=$DECODE_DELTA_MODE stride=$DECODE_DELTA_STRIDE left=$DECODE_DELTA_LEFT scale=$DECODE_DELTA_SCALE debug=$DECODE_DELTA_DEBUG"
echo "PD remote engine    : enabled=$PD_REMOTE_ENGINE overlap=$PD_PIPELINE_OVERLAP decode_device_start=$PD_DECODE_DEVICE_START decode_master_port=$PD_DECODE_MASTER_PORT"
echo "Log file            : $LOG_FILE"
echo "============================================"

cd "$D2F_VLLM_DIR"

"$PYTHON" <<'PYCODE' 2>&1 | tee "$LOG_FILE"
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence
import torch

repo_dir = os.environ["REPO_DIR"]
d2f_eval_dir = os.environ["D2F_EVAL_DIR"]
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if d2f_eval_dir not in sys.path:
    sys.path.insert(0, d2f_eval_dir)

from d2f_vllm import FastDLLMDreamEngine
from d2f_vllm.pd_pipeline import ordered_prefetch_map
from eval_fastdllm_parallelcomp_longbench import (
    load_json,
    load_task_examples,
    render_prompt_parts,
    score_prediction,
    summarize_metrics,
    trim_stop_tokens,
)

STOP_STRINGS = ["</s>", "<|im_end|>"]


def env_int(name, default):
    value = os.environ.get(name, "")
    return int(value) if value != "" else int(default)


def env_float(name, default):
    value = os.environ.get(name, "")
    return float(value) if value != "" else float(default)


def env_bool(name, default):
    value = os.environ.get(name, "")
    if value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def save_json(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def split_token_chunks(token_ids: Sequence[int], chunk_size: int, split_from_tail: bool = False) -> List[List[int]]:
    ids = list(int(x) for x in token_ids)
    if not ids:
        return []
    if chunk_size <= 0 or len(ids) <= chunk_size:
        return [ids]
    chunks = []
    cursor = 0
    if split_from_tail:
        leading_remainder = len(ids) % chunk_size
        if leading_remainder > 0:
            chunks.append(ids[:leading_remainder])
            cursor = leading_remainder
    while cursor < len(ids):
        chunks.append(ids[cursor:cursor + chunk_size])
        cursor += chunk_size
    return chunks


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def bos_ids(tokenizer):
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return [int(bos_token_id)]
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token:
        return tokenizer.encode(bos_token, add_special_tokens=False)
    return []


def maybe_prepend_bos_to_chunk(tokenizer, chunk_ids, chunk_size, chunk_bos=True):
    ids = list(int(x) for x in chunk_ids)
    if not chunk_bos:
        return ids
    bos = bos_ids(tokenizer)
    if not bos or ids[:len(bos)] == bos:
        return ids
    with_bos = bos + ids
    if chunk_size > 0 and len(with_bos) > chunk_size:
        with_bos = with_bos[:chunk_size]
    return with_bos


def range_positions(start, length):
    return list(range(int(start), int(start) + int(length)))


def chunk_rope_start(prefix_len, chunk_order, chunk_index, chunk_size, mode):
    if mode == "reuse":
        return prefix_len
    if mode == "continuous":
        return prefix_len + chunk_order * max(1, chunk_size)
    if mode == "absolute":
        return prefix_len + chunk_index * max(1, chunk_size)
    raise ValueError(f"Unsupported chunk_position_mode: {mode}")


def final_query_rope_start(prefix_len, cache_positions, selected_count, chunk_size, mode):
    if mode == "after_reused_window":
        return prefix_len + max(1, chunk_size)
    if mode == "after_selected_chunks":
        return prefix_len + selected_count * max(1, chunk_size)
    if mode == "after_cache":
        return (max(cache_positions) + 1) if cache_positions else prefix_len
    raise ValueError(f"Unsupported query_position_mode: {mode}")


def token_eviction_query_ids(scoring_query_ids, selection_query_ids, score_token_mask, token_score_use_generated):
    if not token_score_use_generated:
        return list(scoring_query_ids)
    if score_token_mask is not None and len(score_token_mask) == len(selection_query_ids):
        return [int(token_id) for token_id, keep in zip(selection_query_ids, score_token_mask) if keep]
    return list(selection_query_ids)


dream_base = os.environ["DREAM_BASE"]
data_dir = os.environ["DATA_DIR"]
config_dir = os.environ["CONFIG_DIR"]
task = os.environ["TASK_NAME"]
log_dir = Path(os.environ["D2F_VLLM_DIR"]) / "log"
run_ts = os.environ["RUN_TS"]

check_only = env_int("CHECK_ONLY", 0)
if check_only:
    print("CHECK_ONLY=1: imports and paths are valid; no model is loaded.", flush=True)
    raise SystemExit(0)

start_index = env_int("START_INDEX", 0)
limit_env = os.environ.get("LIMIT", "")
limit = int(limit_env) if limit_env else None
pc_chunk_size = env_int("PC_CHUNK_SIZE", 1024)
topk_chunks = env_int("TOPK_CHUNKS", 4)
score_mode = os.environ.get("SCORE_MODE", "draft_self_information")
score_draft_tokens = env_int("SCORE_DRAFT_TOKENS", 4)
score_draft_partial_rounds = env_int("SCORE_DRAFT_PARTIAL_ROUNDS", 1)
score_draft_score_all_slots = env_bool("SCORE_DRAFT_SCORE_ALL_SLOTS", False)
score_attention_mask = os.environ.get("SCORE_ATTENTION_MASK", "causal")
score_context_mode = os.environ.get("SCORE_CONTEXT_MODE", "single_chunk")
chunk_position_mode = os.environ.get("CHUNK_POSITION_MODE", "continuous")
query_position_mode = os.environ.get("QUERY_POSITION_MODE", "after_selected_chunks")
keep_first_chunk = env_bool("KEEP_FIRST_CHUNK", False)
split_from_tail = env_bool("SPLIT_FROM_TAIL", False)
chunk_bos = env_bool("CHUNK_BOS", True)
token_capacity = env_int("TOKEN_CAPACITY", 512)
token_score_query_window = env_int("TOKEN_SCORE_QUERY_WINDOW", 8)
token_score_layers = env_int("TOKEN_SCORE_LAYERS", 0)
token_score_layer_mode = os.environ.get("TOKEN_SCORE_LAYER_MODE", "all")
token_score_reduce = os.environ.get("TOKEN_SCORE_REDUCE", "sum")
token_score_pooling = os.environ.get("TOKEN_SCORE_POOLING", "maxpool")
token_score_pool_kernel = env_int("TOKEN_SCORE_POOL_KERNEL", 7)
token_score_direction = os.environ.get("TOKEN_SCORE_DIRECTION", "query_to_chunk")
token_score_keep = os.environ.get("TOKEN_SCORE_KEEP", "high")
token_score_include_prefix = env_bool("TOKEN_SCORE_INCLUDE_PREFIX", True)
token_score_use_generated = env_bool("TOKEN_SCORE_USE_GENERATED", False)
token_attention_mask = os.environ.get("TOKEN_ATTENTION_MASK", "causal")
token_score_backend = os.environ.get("TOKEN_SCORE_BACKEND", "torch")
token_eviction_granularity = os.environ.get("TOKEN_EVICTION_GRANULARITY", "per_head")
max_new_tokens = env_int("MAX_NEW_TOKENS", 32)
block_length = env_int("BLOCK_LENGTH", 32)
max_model_len = env_int("MAX_MODEL_LEN", 8192)
gpu_memory_utilization = env_float("GPU_MEMORY_UTILIZATION", 0.60)
threshold = env_float("THRESHOLD", 0.9)
kv_cache_layout = os.environ.get("KV_CACHE_LAYOUT", "unified")
master_port = env_int("MASTER_PORT", 2333)
pd_remote_engine = env_bool("PD_REMOTE_ENGINE", False)
pd_pipeline_overlap = env_bool("PD_PIPELINE_OVERLAP", False)
pd_decode_device_start = env_int("PD_DECODE_DEVICE_START", 1)
pd_decode_master_port = env_int("PD_DECODE_MASTER_PORT", master_port + 1)
pd_decode_shm_name = os.environ.get("PD_DECODE_SHM_NAME", "d2f_vllm_pd_decode")
decode_delta_mode = os.environ.get("DECODE_DELTA_MODE", "none")
decode_delta_stride = env_int("DECODE_DELTA_STRIDE", 4)
decode_delta_left = env_int("DECODE_DELTA_LEFT", 3)
decode_delta_scale = env_float("DECODE_DELTA_SCALE", 1.0)
decode_delta_debug = env_bool("DECODE_DELTA_DEBUG", False)
prefill_sparse_mode = os.environ.get("PREFILL_SPARSE_MODE", "none")
prefill_delta_mode = os.environ.get("PREFILL_DELTA_MODE", "none")
prefill_delta_stride = env_int("PREFILL_DELTA_STRIDE", 8)
prefill_delta_left = env_int("PREFILL_DELTA_LEFT", 7)
prefill_delta_scale = env_float("PREFILL_DELTA_SCALE", 1.0)
prefill_delta_debug = env_bool("PREFILL_DELTA_DEBUG", False)
dry_run = env_int("DRY_RUN", 0)

if token_eviction_granularity != "per_head":
    raise ValueError("full-engine bridge currently targets per_head token eviction.")
if pd_pipeline_overlap and not pd_remote_engine:
    raise ValueError("PD_PIPELINE_OVERLAP=1 requires PD_REMOTE_ENGINE=1.")

bridge_mode = (
    "fastdllm_full_engine_selection_keep_decode_pd_overlap"
    if pd_remote_engine and pd_pipeline_overlap
    else "fastdllm_full_engine_selection_keep_decode"
)

prompt_templates = load_json(Path(config_dir) / "dataset2prompt_raw.json")
dataset2maxlen = load_json(Path(config_dir) / "dataset2maxlen.json")
examples = load_task_examples(task, data_dir, max_examples=0)
examples = examples[start_index:]
if limit is not None:
    examples = examples[:limit]
print(f"Loaded examples: {len(examples)}", flush=True)
print(f"Official max_new for {task}: {dataset2maxlen[task]}", flush=True)

print("Loading FastDLLMDreamEngine for selection + keep-index + decode...", flush=True)
engine = FastDLLMDreamEngine(
    dream_base,
    max_model_len=max_model_len,
    block_length=block_length,
    gpu_memory_utilization=gpu_memory_utilization,
    threshold=threshold,
    temperature=0.0,
    max_num_batched_tokens=max_model_len,
    max_num_seqs=1,
    kv_cache_layout=kv_cache_layout,
    master_port=master_port,
    device_start=0,
)
decode_engine = None
if pd_remote_engine:
    print("Loading Decode FastDLLMDreamEngine for PD remote decode...", flush=True)
    torch.cuda.set_device(pd_decode_device_start)
    decode_engine = FastDLLMDreamEngine(
        dream_base,
        max_model_len=max_model_len,
        block_length=block_length,
        gpu_memory_utilization=gpu_memory_utilization,
        threshold=threshold,
        temperature=0.0,
        max_num_batched_tokens=max_model_len,
        max_num_seqs=1,
        kv_cache_layout=kv_cache_layout,
        master_port=pd_decode_master_port,
        shm_name=pd_decode_shm_name,
        device_start=pd_decode_device_start,
    )
    torch.cuda.set_device(0)


def generation_kwargs(compressed):
    return {
        "max_new_tokens": max_new_tokens,
        "prompt_positions": compressed["prompt_positions"],
        "active_prompt_positions": compressed["active_prompt_positions"],
        "prompt_keep_indices_per_layer_per_head": compressed["prompt_keep_indices_per_layer_per_head"],
        "decode_delta_mode": decode_delta_mode,
        "decode_delta_stride": decode_delta_stride,
        "decode_delta_left": decode_delta_left,
        "decode_delta_scale": decode_delta_scale,
        "decode_delta_debug": decode_delta_debug,
        "prefill_sparse_mode": prefill_sparse_mode,
        "prefill_delta_mode": prefill_delta_mode,
        "prefill_delta_stride": prefill_delta_stride,
        "prefill_delta_left": prefill_delta_left,
        "prefill_delta_scale": prefill_delta_scale,
        "prefill_delta_debug": prefill_delta_debug,
    }


def stop_token_ids():
    return [engine.tokenizer.eos_token_id] if engine.tokenizer.eos_token_id is not None else None


def prepare_pd_decode_record(compressed):
    kwargs = generation_kwargs(compressed)
    source_record = None
    try:
        torch.cuda.set_device(0)
        source_record = engine.prefill_to_pd_record(compressed["prompt_ids"], **kwargs)
        torch.cuda.set_device(pd_decode_device_start)
        decode_record = decode_engine.make_pd_record_from_remote_engine(engine, source_record)
        torch.cuda.synchronize(pd_decode_device_start)
        return decode_record
    finally:
        if source_record is not None:
            torch.cuda.set_device(0)
            engine.release_pd_record(source_record)


def decode_prepared_pd_record(record):
    torch.cuda.set_device(pd_decode_device_start)
    return decode_engine.decode_from_pd_record(
        record,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids(),
    )


def release_prepared_pd_record(record):
    torch.cuda.set_device(pd_decode_device_start)
    decode_engine.release_pd_record(record)


def generate_with_optional_pd(compressed):
    kwargs = generation_kwargs(compressed)
    if not pd_remote_engine:
        torch.cuda.set_device(0)
        return engine.generate_token_ids(
            compressed["prompt_ids"],
            stop_token_ids=stop_token_ids(),
            **kwargs,
        )
    return decode_prepared_pd_record(prepare_pd_decode_record(compressed))

result_path = log_dir / f"longbench_{task}_fastdllm_engine_full_bridge_results_{run_ts}.json"
compressed_path = log_dir / f"longbench_{task}_fastdllm_engine_full_bridge_compressed_{run_ts}.json"
compressed_records = []
selection_t0 = time.time()
template = prompt_templates[task]

try:
    for idx, example in enumerate(examples):
        torch.cuda.set_device(0)
        parts = render_prompt_parts(template, example, "\n")
        prefix_ids = bos_ids(engine.tokenizer) + encode_fragment(engine.tokenizer, parts.get("prefix", ""))
        context_ids = encode_fragment(engine.tokenizer, parts.get("context", ""))
        query_ids = encode_fragment(engine.tokenizer, parts.get("query", ""))
        scoring_query_ids = encode_fragment(engine.tokenizer, parts.get("scoring_query", "")) or list(query_ids)

        candidate_chunks = split_token_chunks(context_ids, pc_chunk_size, split_from_tail=split_from_tail)
        candidate_chunks = [maybe_prepend_bos_to_chunk(engine.tokenizer, chunk, pc_chunk_size, chunk_bos=chunk_bos) for chunk in candidate_chunks]
        selected_indices, chunk_scores, selection_query_ids, score_token_mask = engine.select_chunks_by_engine(
            prefix_ids,
            candidate_chunks,
            scoring_query_ids,
            topk_chunks=topk_chunks,
            score_mode=score_mode,
            score_draft_tokens=score_draft_tokens,
            score_draft_partial_rounds=score_draft_partial_rounds,
            score_draft_score_all_slots=score_draft_score_all_slots,
            score_attention_mask=score_attention_mask,
            score_context_mode=score_context_mode,
            keep_first_chunk=keep_first_chunk,
        )
        eviction_query_ids = token_eviction_query_ids(
            scoring_query_ids,
            selection_query_ids,
            score_token_mask,
            token_score_use_generated,
        )

        prompt_ids = list(prefix_ids)
        prompt_positions = range_positions(0, len(prefix_ids))
        active_prompt_positions = list(prompt_positions)
        kept_context_tokens = 0
        removed_context_tokens = 0
        chunk_keep_counts = []
        engine_per_head_chunk_spans = []
        for chunk_order, chunk_idx in enumerate(selected_indices):
            original_chunk_ids = list(candidate_chunks[chunk_idx])
            chunk_start = chunk_rope_start(len(prefix_ids), chunk_order, chunk_idx, pc_chunk_size, chunk_position_mode)
            original_chunk_positions = range_positions(chunk_start, len(original_chunk_ids))
            full_span_start = len(prompt_ids)
            kept_count = min(max(1, token_capacity), len(original_chunk_ids)) if token_capacity > 0 and original_chunk_ids else len(original_chunk_ids)
            chunk_ids = list(original_chunk_ids)
            chunk_positions = list(original_chunk_positions)
            active_chunk_positions = original_chunk_positions[:kept_count]
            full_span_end = full_span_start + len(chunk_ids)
            engine_per_head_chunk_spans.append({"start": full_span_start, "end": full_span_end, "chunk_ids": original_chunk_ids})
            kept_context_tokens += kept_count
            removed_context_tokens += max(0, len(original_chunk_ids) - kept_count)
            chunk_keep_counts.append(
                {
                    "chunk_index": int(chunk_idx),
                    "original_tokens": len(original_chunk_ids),
                    "kept_tokens": kept_count,
                    "removed_tokens": max(0, len(original_chunk_ids) - kept_count),
                    "union_kept_tokens": None,
                    "chunk_selection_backend": "engine",
                    "token_score_backend": f"engine:{token_score_backend}",
                }
            )
            prompt_ids.extend(chunk_ids)
            prompt_positions.extend(chunk_positions)
            active_prompt_positions.extend(active_chunk_positions)

        active_query_rope_start = final_query_rope_start(
            len(prefix_ids), active_prompt_positions, len(selected_indices), pc_chunk_size, query_position_mode
        )
        full_query_rope_start = final_query_rope_start(
            len(prefix_ids), prompt_positions, len(selected_indices), pc_chunk_size, query_position_mode
        )
        query_positions = range_positions(active_query_rope_start, len(query_ids))
        full_query_positions = range_positions(full_query_rope_start, len(query_ids))
        prompt_ids.extend(query_ids)
        prompt_positions.extend(full_query_positions)
        active_prompt_positions.extend(query_positions)

        if len(prompt_ids) + max_new_tokens > max_model_len:
            raise ValueError(
                f"Compressed prompt too long for max_model_len={max_model_len}: idx={idx}, prompt={len(prompt_ids)}, max_new={max_new_tokens}"
            )
        keep_indices, engine_chunk_meta = engine.compute_prompt_keep_indices_per_layer_per_head(
            full_prompt_len=len(prompt_ids),
            prefix_ids=prefix_ids,
            chunk_spans=engine_per_head_chunk_spans,
            query_ids=eviction_query_ids,
            token_capacity=token_capacity,
            token_score_query_window=token_score_query_window,
            token_score_layers=token_score_layers,
            token_score_layer_mode=token_score_layer_mode,
            token_score_reduce=token_score_reduce,
            token_score_pooling=token_score_pooling,
            token_score_pool_kernel=token_score_pool_kernel,
            token_score_direction=token_score_direction,
            token_score_keep=token_score_keep,
            token_score_include_prefix=token_score_include_prefix,
            token_attention_mask=token_attention_mask,
            token_score_backend=token_score_backend,
        )
        active_len = len(keep_indices[0][0]) if keep_indices and keep_indices[0] else 0
        if active_len != len(active_prompt_positions):
            raise ValueError(f"engine active length mismatch: keep={active_len}, positions={len(active_prompt_positions)}")
        for chunk_count, engine_meta in zip(chunk_keep_counts, engine_chunk_meta):
            chunk_count["union_kept_tokens"] = int(engine_meta["union_kept_tokens"])

        compressed_records.append(
            {
                "idx": start_index + idx,
                "example_id": example.get("_id", idx),
                "prompt_ids": prompt_ids,
                "prompt_positions": prompt_positions,
                "active_prompt_positions": active_prompt_positions,
                "prompt_keep_indices_per_layer_per_head": keep_indices,
                "engine_token_eviction": None,
                "selected_chunk_indices": selected_indices,
                "chunk_scores": {str(k): float(v) for k, v in chunk_scores.items()},
                "prompt_meta": {
                    "prefix_tokens": len(prefix_ids),
                    "context_tokens": len(context_ids),
                    "kept_context_tokens": kept_context_tokens,
                    "removed_context_tokens": removed_context_tokens,
                    "query_tokens": len(query_ids),
                    "scoring_query_tokens": len(scoring_query_ids),
                    "candidate_chunks": len(candidate_chunks),
                    "chunk_keep_counts": chunk_keep_counts,
                    "compressed_prompt_tokens": len(prompt_ids),
                    "active_prompt_tokens": len(active_prompt_positions),
                    "max_position": max(prompt_positions) if prompt_positions else -1,
                    "query_rope_start": active_query_rope_start,
                    "full_query_rope_start": full_query_rope_start,
                    "selection_query_tokens": len(selection_query_ids),
                    "score_token_mask_true": sum(1 for x in score_token_mask if x) if score_token_mask is not None else None,
                    "chunk_selection_backend": "engine",
                    "token_score_backend": f"engine:{token_score_backend}",
                },
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(examples):
            print(f"  selected+kept {idx + 1}/{len(examples)}", flush=True)

    selection_seconds = time.time() - selection_t0
    save_json(
        compressed_path,
        {
            "task": task,
            "run_ts": run_ts,
            "bridge_mode": bridge_mode,
            "selection_seconds": selection_seconds,
            "setting": {
                "chunk_selection_backend": "engine",
                "token_score_backend": token_score_backend,
                "topk_chunks": topk_chunks,
                "parallelcomp_chunk_size": pc_chunk_size,
                "score_mode": score_mode,
                "score_draft_tokens": score_draft_tokens,
                "score_draft_partial_rounds": score_draft_partial_rounds,
                "score_attention_mask": score_attention_mask,
                "score_context_mode": score_context_mode,
                "cache_build_mode_label": os.environ["CACHE_BUILD_MODE"],
                "chunk_position_mode": chunk_position_mode,
                "query_position_mode": query_position_mode,
                "token_capacity": token_capacity,
                "token_score_query_window": token_score_query_window,
                "token_score_layers": token_score_layers,
                "token_score_layer_mode": token_score_layer_mode,
                "token_score_reduce": token_score_reduce,
                "token_score_pooling": token_score_pooling,
                "token_score_pool_kernel": token_score_pool_kernel,
                "token_score_direction": token_score_direction,
                "token_score_keep": token_score_keep,
                "token_attention_mask": token_attention_mask,
                "token_score_backend": token_score_backend,
                "token_eviction_granularity": token_eviction_granularity,
                "pd_remote_engine": pd_remote_engine,
                "pd_pipeline_overlap": pd_pipeline_overlap,
                "pd_decode_device_start": pd_decode_device_start,
                "prefill_sparse_mode": prefill_sparse_mode,
                "prefill_delta_mode": prefill_delta_mode,
                "prefill_delta_stride": prefill_delta_stride,
                "prefill_delta_left": prefill_delta_left,
                "prefill_delta_scale": prefill_delta_scale,
                "prefill_delta_debug": prefill_delta_debug,
                "decode_delta_mode": decode_delta_mode,
                "decode_delta_stride": decode_delta_stride,
                "decode_delta_left": decode_delta_left,
                "decode_delta_scale": decode_delta_scale,
                "decode_delta_debug": decode_delta_debug,
            },
            "records": compressed_records,
        },
    )
    print(f"Compressed prompts saved to: {compressed_path}", flush=True)
    print(f"Engine selection+keep time: {selection_seconds:.2f}s", flush=True)

    if dry_run:
        print("DRY_RUN=1, stopping after engine selection+keep.", flush=True)
        raise SystemExit(0)

    results = []
    decode_t0 = time.time()

    def append_result(idx, example, compressed, output):
        raw_prediction = output.text
        prediction = trim_stop_tokens(raw_prediction, STOP_STRINGS)
        answers = example.get("answers", [])
        all_classes = example.get("all_classes")
        score = score_prediction(task, prediction, answers, all_classes)
        results.append(
            {
                "task": task,
                "example_id": compressed["example_id"],
                "index": start_index + idx,
                "pred": prediction,
                "raw_pred": raw_prediction,
                "answers": answers,
                "all_classes": all_classes,
                "score": score,
                "length": example.get("length"),
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "token_count": len(output.token_ids),
                "n_diff_steps": output.n_diff_steps,
                "parallelcomp_bridge": {
                    "selected_chunk_indices": compressed["selected_chunk_indices"],
                    "chunk_scores": compressed["chunk_scores"],
                    "prompt_meta": compressed["prompt_meta"],
                },
            }
        )

    def save_progress(idx):
        if (idx + 1) % 10 == 0 or idx + 1 == len(examples):
            metrics = summarize_metrics(results, longbench_e=False)
            payload = {
                "task": task,
                "run_ts": run_ts,
                "bridge_mode": bridge_mode,
                "completed": len(results),
                "total": len(examples),
                "selection_seconds": selection_seconds,
                "decode_seconds": time.time() - decode_t0,
                "metrics": metrics,
                "setting": {
                    "chunk_selection_backend": "engine",
                    "token_score_backend": token_score_backend,
                    "topk_chunks": topk_chunks,
                    "parallelcomp_chunk_size": pc_chunk_size,
                    "score_mode": score_mode,
                    "score_draft_tokens": score_draft_tokens,
                    "score_draft_partial_rounds": score_draft_partial_rounds,
                    "score_attention_mask": score_attention_mask,
                    "score_context_mode": score_context_mode,
                    "token_capacity": token_capacity,
                    "token_score_layer_mode": token_score_layer_mode,
                    "token_score_layers": token_score_layers,
                    "max_model_len": max_model_len,
                    "max_new_tokens": max_new_tokens,
                    "block_length": block_length,
                    "threshold": threshold,
                    "pd_remote_engine": pd_remote_engine,
                    "pd_pipeline_overlap": pd_pipeline_overlap,
                    "pd_decode_device_start": pd_decode_device_start,
                    "prefill_sparse_mode": prefill_sparse_mode,
                    "prefill_delta_mode": prefill_delta_mode,
                    "prefill_delta_stride": prefill_delta_stride,
                    "prefill_delta_left": prefill_delta_left,
                    "prefill_delta_scale": prefill_delta_scale,
                    "prefill_delta_debug": prefill_delta_debug,
                    "decode_delta_mode": decode_delta_mode,
                    "decode_delta_stride": decode_delta_stride,
                    "decode_delta_left": decode_delta_left,
                    "decode_delta_scale": decode_delta_scale,
                    "decode_delta_debug": decode_delta_debug,
                },
                "results": results,
            }
            save_json(result_path, payload)
            print(f"completed={len(results)}/{len(examples)} score={metrics['score']:.2f} decode_seconds={time.time() - decode_t0:.2f}", flush=True)

    if pd_remote_engine and pd_pipeline_overlap:
        def prepare_item(item):
            _idx, _example, compressed = item
            return prepare_pd_decode_record(compressed)

        def consume_item(item, record):
            _idx, _example, _compressed = item
            return decode_prepared_pd_record(record)

        decode_items = list(zip(range(len(examples)), examples, compressed_records))
        for item, output in zip(
            decode_items,
            ordered_prefetch_map(
                decode_items,
                prepare=prepare_item,
                consume=consume_item,
                release_prepared=release_prepared_pd_record,
            ),
        ):
            idx, example, compressed = item
            append_result(idx, example, compressed, output)
            save_progress(idx)
    else:
        for idx, (example, compressed) in enumerate(zip(examples, compressed_records)):
            output = generate_with_optional_pd(compressed)
            append_result(idx, example, compressed, output)
            save_progress(idx)
finally:
    if decode_engine is not None:
        torch.cuda.set_device(pd_decode_device_start)
        decode_engine.close()
    torch.cuda.set_device(0)
    engine.close()

final_metrics = summarize_metrics(results, longbench_e=False) if 'results' in locals() else {"score": 0.0}
print()
print("=" * 60)
print("LongBench task full-engine bridge")
print("=" * 60)
print(f"Score             : {final_metrics['score']:.2f}")
print(f"Completed         : {len(results) if 'results' in locals() else 0}/{len(examples)}")
print(f"Results saved to  : {result_path}")
print("=" * 60)
PYCODE
