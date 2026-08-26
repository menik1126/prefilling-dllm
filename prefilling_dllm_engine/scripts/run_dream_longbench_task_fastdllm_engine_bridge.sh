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
TOKEN_EVICTION_GRANULARITY="${TOKEN_EVICTION_GRANULARITY:-global}"
TOKEN_SCORE_BACKEND="${TOKEN_SCORE_BACKEND:-auto}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
THRESHOLD="${THRESHOLD:-0.9}"
KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-unified}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_DIR:$PREFILLING_DLLM_EVAL_DIR:${PYTHONPATH:-}"

export REPO_DIR PREFILLING_DLLM_ENGINE_DIR PREFILLING_DLLM_EVAL_DIR DREAM_BASE FASTDLLM_DREAM DATA_DIR CONFIG_DIR
export START_INDEX LIMIT RUN_TS CHECK_ONLY DRY_RUN
export PC_CHUNK_SIZE TOPK_CHUNKS SCORE_MODE SCORE_DRAFT_TOKENS SCORE_DRAFT_PARTIAL_ROUNDS
export CACHE_BUILD_MODE CHUNK_POSITION_MODE QUERY_POSITION_MODE TOKEN_CAPACITY
export TOKEN_SCORE_QUERY_WINDOW TOKEN_SCORE_LAYERS TOKEN_SCORE_LAYER_MODE TOKEN_SCORE_REDUCE
export TOKEN_SCORE_POOLING TOKEN_SCORE_POOL_KERNEL TOKEN_SCORE_HEAD_REDUCE TOKEN_SCORE_LAYER_REDUCE
export TOKEN_SCORE_DIRECTION TOKEN_SCORE_KEEP TOKEN_SCORE_INCLUDE_PREFIX TOKEN_SCORE_USE_GENERATED
export TOKEN_ATTENTION_MASK TOKEN_EVICTION_GRANULARITY TOKEN_SCORE_BACKEND
export MAX_NEW_TOKENS BLOCK_LENGTH MAX_MODEL_LEN GPU_MEMORY_UTILIZATION THRESHOLD KV_CACHE_LAYOUT

LOG_DIR="$PREFILLING_DLLM_ENGINE_DIR/log"
mkdir -p "$LOG_DIR"
TASK_NAME="${LONGBENCH_TASK:-multifieldqa_en}"
RESULTS_TAG="longbench_${TASK_NAME}_fastdllm_engine_bridge"
LOG_FILE="$LOG_DIR/${RESULTS_TAG}_${RUN_TS}.log"

echo "============================================"
echo "  Fast-DLLM semantic engine bridge - LongBench ${TASK_NAME}"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "Fast-DLLM Dream dir : $FASTDLLM_DREAM"
echo "CUDA devices        : $CUDA_VISIBLE_DEVICES"
echo "Run timestamp       : $RUN_TS"
echo "Selector setting    : topk=$TOPK_CHUNKS chunk=$PC_CHUNK_SIZE score=$SCORE_MODE draft=$SCORE_DRAFT_TOKENS partial_rounds=$SCORE_DRAFT_PARTIAL_ROUNDS cache=$CACHE_BUILD_MODE positions=$CHUNK_POSITION_MODE/$QUERY_POSITION_MODE"
echo "Token eviction      : capacity=$TOKEN_CAPACITY granularity=$TOKEN_EVICTION_GRANULARITY backend=$TOKEN_SCORE_BACKEND direction=$TOKEN_SCORE_DIRECTION keep=$TOKEN_SCORE_KEEP layers=$TOKEN_SCORE_LAYER_MODE:$TOKEN_SCORE_LAYERS query_window=$TOKEN_SCORE_QUERY_WINDOW"
echo "Decode setting      : FastDLLMDreamEngine full_prompt_mask block=$BLOCK_LENGTH max_new=$MAX_NEW_TOKENS"
echo "Log file            : $LOG_FILE"
echo "============================================"

cd "$PREFILLING_DLLM_ENGINE_DIR"

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_FILE"
import json
import os
import sys
import time
from pathlib import Path

repo_dir = os.environ["REPO_DIR"]
prefilling_dllm_eval_dir = os.environ["PREFILLING_DLLM_EVAL_DIR"]
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if prefilling_dllm_eval_dir not in sys.path:
    sys.path.insert(0, prefilling_dllm_eval_dir)

from prefilling_dllm import FastDLLMDreamEngine
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


TASK = os.environ.get("LONGBENCH_TASK", "multifieldqa_en")
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
check_only = env_int("CHECK_ONLY", 0)
dry_run = env_int("DRY_RUN", 0)

pc_chunk_size = env_int("PC_CHUNK_SIZE", 1024)
topk_chunks = env_int("TOPK_CHUNKS", 4)
score_draft_tokens = env_int("SCORE_DRAFT_TOKENS", 4)
score_draft_partial_rounds = env_int("SCORE_DRAFT_PARTIAL_ROUNDS", 1)
token_capacity = env_int("TOKEN_CAPACITY", 0)
token_score_query_window = env_int("TOKEN_SCORE_QUERY_WINDOW", 8)
token_score_layers = env_int("TOKEN_SCORE_LAYERS", 0)
token_score_layer_mode = os.environ.get("TOKEN_SCORE_LAYER_MODE", "all")
token_score_reduce = os.environ.get("TOKEN_SCORE_REDUCE", "sum")
token_score_pooling = os.environ.get("TOKEN_SCORE_POOLING", "maxpool")
token_score_pool_kernel = env_int("TOKEN_SCORE_POOL_KERNEL", 7)
token_score_head_reduce = os.environ.get("TOKEN_SCORE_HEAD_REDUCE", "sum")
token_score_layer_reduce = os.environ.get("TOKEN_SCORE_LAYER_REDUCE", "mean")
token_score_direction = os.environ.get("TOKEN_SCORE_DIRECTION", "query_to_chunk")
token_score_keep = os.environ.get("TOKEN_SCORE_KEEP", "high")
token_score_include_prefix = env_bool("TOKEN_SCORE_INCLUDE_PREFIX", True)
token_score_use_generated = env_bool("TOKEN_SCORE_USE_GENERATED", False)
token_attention_mask = os.environ.get("TOKEN_ATTENTION_MASK", "causal")
token_eviction_granularity = os.environ.get("TOKEN_EVICTION_GRANULARITY", "global")
token_score_backend = os.environ.get("TOKEN_SCORE_BACKEND", "auto").lower()
max_new_tokens = env_int("MAX_NEW_TOKENS", 32)
block_length = env_int("BLOCK_LENGTH", 32)
max_model_len = env_int("MAX_MODEL_LEN", 8192)
gpu_memory_utilization = env_float("GPU_MEMORY_UTILIZATION", 0.60)
threshold = env_float("THRESHOLD", 0.9)
kv_cache_layout = os.environ.get("KV_CACHE_LAYOUT", "unified")

if check_only:
    print("CHECK_ONLY=1: imports and paths are valid; no model is loaded.", flush=True)
    raise SystemExit(0)
if token_eviction_granularity not in {"global", "per_head"}:
    raise ValueError(
        f"Unsupported token_eviction_granularity={token_eviction_granularity!r}; "
        "expected 'global' or 'per_head'."
    )
if token_score_backend == "auto":
    token_score_backend = "engine" if token_eviction_granularity == "per_head" else "selector"
if token_score_backend not in {"selector", "engine"}:
    raise ValueError(f"Unsupported TOKEN_SCORE_BACKEND={token_score_backend!r}; expected selector/engine/auto.")
if token_score_backend == "engine" and token_eviction_granularity != "per_head":
    raise ValueError("TOKEN_SCORE_BACKEND=engine is currently implemented for per_head token eviction only.")

result_path = log_dir / f"longbench_{TASK}_fastdllm_engine_bridge_results_{run_ts}.json"
compressed_path = log_dir / f"longbench_{TASK}_fastdllm_engine_bridge_compressed_{run_ts}.json"
print(f"Effective token score backend: {token_score_backend}", flush=True)

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
    threshold=threshold,
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
    token_score_query_window=token_score_query_window,
    token_score_layers=token_score_layers,
    token_score_layer_mode=token_score_layer_mode,
    token_score_reduce=token_score_reduce,
    token_score_pooling=token_score_pooling,
    token_score_pool_kernel=token_score_pool_kernel,
    token_score_head_reduce=token_score_head_reduce,
    token_score_layer_reduce=token_score_layer_reduce,
    token_score_direction=token_score_direction,
    token_score_keep=token_score_keep,
    token_score_include_prefix=token_score_include_prefix,
    token_score_use_generated=token_score_use_generated,
    token_attention_mask=token_attention_mask,
    token_eviction_granularity=token_eviction_granularity,
    chunk_position_mode=os.environ["CHUNK_POSITION_MODE"],
    query_position_mode=os.environ["QUERY_POSITION_MODE"],
)

compressed_records = []
selection_t0 = time.time()
template = prompt_templates[TASK]
for idx, example in enumerate(examples):
    parts = render_prompt_parts(template, example, "\n")
    token_parts = build_token_parts(
        selector,
        parts,
        add_bos_token=True,
    )
    prefix_ids, context_ids, query_ids, scoring_query_ids = token_parts[:4]
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
    active_prompt_positions = list(prompt_positions)
    kept_context_tokens = 0
    removed_context_tokens = 0
    chunk_keep_counts = []
    selector_per_head_chunk_spans = []
    engine_per_head_chunk_spans = []
    for chunk_order, chunk_idx in enumerate(selected_indices):
        original_chunk_ids = list(candidate_chunks[chunk_idx])
        chunk_start = selector._chunk_rope_start(len(prefix_ids), chunk_order, chunk_idx)
        original_chunk_positions = selector._range_positions(chunk_start, len(original_chunk_ids))
        full_span_start = len(prompt_ids)
        if token_eviction_granularity == "per_head" and token_score_backend == "engine":
            per_layer_per_head_keep = None
            keep_positions = []
            kept_count = (
                min(max(1, token_capacity), len(original_chunk_ids))
                if token_capacity > 0 and original_chunk_ids
                else len(original_chunk_ids)
            )
            chunk_ids = list(original_chunk_ids)
            chunk_positions = list(original_chunk_positions)
            active_chunk_positions = original_chunk_positions[:kept_count]
        elif token_eviction_granularity == "per_head":
            per_layer_per_head_keep = selector._keep_positions_per_layer_per_head_for_chunk(
                prefix_ids,
                original_chunk_ids,
                eviction_query_ids,
            )
            keep_positions = selector._union_keep_positions_per_layer_per_head(per_layer_per_head_keep)
            kept_count = (
                int(per_layer_per_head_keep[0].shape[1])
                if per_layer_per_head_keep
                else len(original_chunk_ids)
            )
            chunk_ids = list(original_chunk_ids)
            chunk_positions = list(original_chunk_positions)
            active_chunk_positions = original_chunk_positions[:kept_count]
        else:
            per_layer_per_head_keep = None
            keep_positions = selector._keep_positions_for_chunk(prefix_ids, original_chunk_ids, eviction_query_ids)
            chunk_ids = [original_chunk_ids[pos] for pos in keep_positions]
            chunk_positions = [original_chunk_positions[pos] for pos in keep_positions]
            kept_count = len(chunk_ids)
            active_chunk_positions = list(chunk_positions)
        full_span_end = full_span_start + len(chunk_ids)
        if token_eviction_granularity == "per_head" and token_score_backend == "engine":
            engine_per_head_chunk_spans.append(
                {
                    "start": full_span_start,
                    "end": full_span_end,
                    "chunk_ids": original_chunk_ids,
                }
            )
        elif per_layer_per_head_keep is not None:
            selector_per_head_chunk_spans.append((full_span_start, full_span_end, per_layer_per_head_keep))
        kept_context_tokens += kept_count
        removed_context_tokens += max(0, len(original_chunk_ids) - kept_count)
        chunk_keep_counts.append(
            {
                "chunk_index": int(chunk_idx),
                "original_tokens": len(original_chunk_ids),
                "kept_tokens": kept_count,
                "removed_tokens": max(0, len(original_chunk_ids) - kept_count),
                "union_kept_tokens": (len(keep_positions) if keep_positions else None),
                "token_score_backend": token_score_backend if token_eviction_granularity == "per_head" else "selector",
            }
        )
        prompt_ids.extend(chunk_ids)
        prompt_positions.extend(chunk_positions)
        active_prompt_positions.extend(active_chunk_positions)
    active_query_rope_start = selector._final_query_rope_start(
        len(prefix_ids),
        active_prompt_positions,
        selected_count=len(selected_indices),
    )
    full_query_rope_start = selector._final_query_rope_start(
        len(prefix_ids),
        prompt_positions,
        selected_count=len(selected_indices),
    )
    query_positions = selector._range_positions(active_query_rope_start, len(query_ids))
    full_query_positions = selector._range_positions(full_query_rope_start, len(query_ids))
    prompt_ids.extend(query_ids)
    prompt_positions.extend(full_query_positions)
    active_prompt_positions.extend(query_positions)
    prompt_keep_indices_per_layer_per_head = None
    engine_token_eviction = None
    if token_eviction_granularity == "per_head":
        if token_score_backend == "engine":
            engine_token_eviction = {
                "prefix_ids": prefix_ids,
                "query_ids": eviction_query_ids,
                "chunk_spans": engine_per_head_chunk_spans,
            }
        else:
            per_head_keep = selector._full_prompt_keep_indices_per_layer_per_head(
                seq_len=len(prompt_ids),
                chunk_spans=selector_per_head_chunk_spans,
            )
            prompt_keep_indices_per_layer_per_head = [layer_keep.tolist() for layer_keep in per_head_keep]
            if prompt_keep_indices_per_layer_per_head:
                per_head_active_len = len(prompt_keep_indices_per_layer_per_head[0][0])
                if per_head_active_len != len(active_prompt_positions):
                    raise ValueError(
                        f"per-head active prompt length mismatch: keep={per_head_active_len}, "
                        f"positions={len(active_prompt_positions)}"
                    )
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
            "active_prompt_positions": (
                active_prompt_positions if token_eviction_granularity == "per_head" else None
            ),
            "prompt_keep_indices_per_layer_per_head": prompt_keep_indices_per_layer_per_head,
            "engine_token_eviction": engine_token_eviction,
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
                "score_token_mask_true": (
                    sum(1 for x in score_token_mask if x) if score_token_mask is not None else None
                ),
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
        "bridge_mode": "fastdllm_selector_then_fastdllm_prefilling_dllm_engine_decode",
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
            "token_score_query_window": token_score_query_window,
            "token_score_layers": token_score_layers,
            "token_score_layer_mode": token_score_layer_mode,
            "token_score_reduce": token_score_reduce,
            "token_score_pooling": token_score_pooling,
            "token_score_pool_kernel": token_score_pool_kernel,
            "token_score_head_reduce": token_score_head_reduce,
            "token_score_layer_reduce": token_score_layer_reduce,
            "token_score_direction": token_score_direction,
            "token_score_keep": token_score_keep,
            "token_score_include_prefix": token_score_include_prefix,
            "token_score_use_generated": token_score_use_generated,
            "token_attention_mask": token_attention_mask,
            "token_eviction_granularity": token_eviction_granularity,
            "token_score_backend": token_score_backend,
        },
        "records": compressed_records,
    },
)
print(f"Compressed prompts saved to: {compressed_path}", flush=True)
print(f"Selection time: {selection_seconds:.2f}s", flush=True)

del selector
if dry_run:
    print("DRY_RUN=1, stopping after prompt compression.", flush=True)
    raise SystemExit(0)

print("Loading FastDLLMDreamEngine...", flush=True)
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
)

if token_eviction_granularity == "per_head" and token_score_backend == "engine":
    print("Computing per-head token keep indices with FastDLLMDreamEngine...", flush=True)
    for compressed in compressed_records:
        engine_token_eviction = compressed.get("engine_token_eviction")
        if not engine_token_eviction:
            continue
        keep_indices, engine_chunk_meta = engine.compute_prompt_keep_indices_per_layer_per_head(
            full_prompt_len=len(compressed["prompt_ids"]),
            prefix_ids=engine_token_eviction["prefix_ids"],
            chunk_spans=engine_token_eviction["chunk_spans"],
            query_ids=engine_token_eviction["query_ids"],
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
        )
        compressed["prompt_keep_indices_per_layer_per_head"] = keep_indices
        active_len = len(keep_indices[0][0]) if keep_indices and keep_indices[0] else 0
        if active_len != len(compressed.get("active_prompt_positions") or []):
            raise ValueError(
                f"engine token scorer active length mismatch: keep={active_len}, "
                f"positions={len(compressed.get('active_prompt_positions') or [])}"
            )
        chunk_counts = compressed["prompt_meta"].get("chunk_keep_counts") or []
        for chunk_count, engine_meta in zip(chunk_counts, engine_chunk_meta):
            chunk_count["union_kept_tokens"] = int(engine_meta["union_kept_tokens"])
            chunk_count["token_score_backend"] = "engine"
        compressed["prompt_meta"]["token_score_backend"] = "engine"
        compressed["engine_token_eviction"] = None
    with open(compressed_path, "r", encoding="utf-8") as f:
        compressed_payload = json.load(f)
    compressed_payload["records"] = compressed_records
    compressed_payload.setdefault("setting", {})["token_score_backend"] = token_score_backend
    save_json(compressed_path, compressed_payload)
    print("Engine token keep indices written to compressed prompts.", flush=True)

results = []
decode_t0 = time.time()
try:
    for idx, (example, compressed) in enumerate(zip(examples, compressed_records)):
        output = engine.generate_token_ids(
            compressed["prompt_ids"],
            max_new_tokens=max_new_tokens,
            prompt_positions=compressed["prompt_positions"],
            active_prompt_positions=compressed.get("active_prompt_positions"),
            prompt_keep_indices_per_layer_per_head=compressed.get("prompt_keep_indices_per_layer_per_head"),
            stop_token_ids=[engine.tokenizer.eos_token_id] if engine.tokenizer.eos_token_id is not None else None,
        )
        raw_prediction = output.text
        prediction = trim_stop_tokens(raw_prediction, STOP_STRINGS)
        answers = example.get("answers", [])
        all_classes = example.get("all_classes")
        score = score_prediction(TASK, prediction, answers, all_classes)
        results.append(
            {
                "task": TASK,
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
        if (idx + 1) % 10 == 0 or idx + 1 == len(examples):
            metrics = summarize_metrics(results, longbench_e=False)
            payload = {
                "task": TASK,
                "run_ts": run_ts,
                "bridge_mode": "fastdllm_selector_then_fastdllm_prefilling_dllm_engine_decode",
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
                    "token_score_query_window": token_score_query_window,
                    "token_score_layers": token_score_layers,
                    "token_score_layer_mode": token_score_layer_mode,
                    "token_score_reduce": token_score_reduce,
                    "token_score_pooling": token_score_pooling,
                    "token_score_pool_kernel": token_score_pool_kernel,
                    "token_score_head_reduce": token_score_head_reduce,
                    "token_score_layer_reduce": token_score_layer_reduce,
                    "token_score_direction": token_score_direction,
                    "token_score_keep": token_score_keep,
                    "token_score_include_prefix": token_score_include_prefix,
                    "token_score_use_generated": token_score_use_generated,
                    "token_attention_mask": token_attention_mask,
                    "token_eviction_granularity": token_eviction_granularity,
                    "token_score_backend": token_score_backend,
                    "max_model_len": max_model_len,
                    "max_new_tokens": max_new_tokens,
                    "block_length": block_length,
                    "threshold": threshold,
                },
                "results": results,
            }
            save_json(result_path, payload)
            print(
                f"completed={len(results)}/{len(examples)} score={metrics['score']:.2f} "
                f"decode_seconds={time.time() - decode_t0:.2f}",
                flush=True,
            )
finally:
    engine.close()

final_metrics = summarize_metrics(results, longbench_e=False)
print()
print("=" * 60)
print("LongBench task Fast-DLLM semantic engine bridge")
print("=" * 60)
print(f"Score             : {final_metrics['score']:.2f}")
print(f"Completed         : {len(results)}/{len(examples)}")
print(f"Selection seconds : {selection_seconds:.2f}")
print(f"Decode seconds    : {time.time() - decode_t0:.2f}")
print(f"Results saved to  : {result_path}")
print("=" * 60)
PY
