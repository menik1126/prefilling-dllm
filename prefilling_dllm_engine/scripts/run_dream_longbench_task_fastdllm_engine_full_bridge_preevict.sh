#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREFILLING_DLLM_ENGINE_DIR="$REPO_DIR/prefilling_dllm_engine"
PREFILLING_DLLM_EVAL_DIR="${PREFILLING_DLLM_EVAL_DIR:-$REPO_DIR/prefilling_dllm_engine_eval}"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
DREAM_BASE="${DREAM_BASE:-$PREFILLING_DLLM_EVAL_DIR/model_weights/Dream-v0-Base-7B}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"
TASK_NAME="${LONGBENCH_TASK:-multifieldqa_en}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
TOKEN_EVICTION_GRANULARITY="${TOKEN_EVICTION_GRANULARITY:-per_head}"

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

export REPO_DIR PREFILLING_DLLM_ENGINE_DIR PREFILLING_DLLM_EVAL_DIR DREAM_BASE DATA_DIR CONFIG_DIR TASK_NAME
export START_INDEX LIMIT RUN_TS CHECK_ONLY DRY_RUN
export PC_CHUNK_SIZE TOPK_CHUNKS SCORE_MODE SCORE_DRAFT_TOKENS SCORE_DRAFT_PARTIAL_ROUNDS SCORE_DRAFT_SCORE_ALL_SLOTS
export SCORE_ATTENTION_MASK SCORE_CONTEXT_MODE CACHE_BUILD_MODE CHUNK_POSITION_MODE QUERY_POSITION_MODE KEEP_FIRST_CHUNK
export SPLIT_FROM_TAIL CHUNK_BOS FORCE_KEEP_CHUNK_BOS
export TOKEN_CAPACITY TOKEN_SCORE_QUERY_WINDOW TOKEN_SCORE_LAYERS TOKEN_SCORE_LAYER_MODE TOKEN_SCORE_REDUCE
export TOKEN_SCORE_POOLING TOKEN_SCORE_POOL_KERNEL TOKEN_SCORE_HEAD_REDUCE TOKEN_SCORE_LAYER_REDUCE
export TOKEN_SCORE_DIRECTION TOKEN_SCORE_KEEP TOKEN_SCORE_INCLUDE_PREFIX TOKEN_SCORE_USE_GENERATED
export TOKEN_ATTENTION_MASK TOKEN_EVICTION_GRANULARITY MAX_NEW_TOKENS BLOCK_LENGTH MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION THRESHOLD KV_CACHE_LAYOUT

LOG_DIR="$PREFILLING_DLLM_ENGINE_DIR/log"
mkdir -p "$LOG_DIR"
RESULTS_TAG="longbench_${TASK_NAME}_fastdllm_engine_full_bridge_preevict"
LOG_FILE="$LOG_DIR/${RESULTS_TAG}_${RUN_TS}.log"

echo "============================================"
echo "  Full engine bridge PRE-EVICT - LongBench ${TASK_NAME}"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "CUDA devices        : $CUDA_VISIBLE_DEVICES"
echo "Run timestamp       : $RUN_TS"
echo "Chunk selection     : backend=engine topk=$TOPK_CHUNKS chunk=$PC_CHUNK_SIZE score=$SCORE_MODE draft=$SCORE_DRAFT_TOKENS partial_rounds=$SCORE_DRAFT_PARTIAL_ROUNDS mask=$SCORE_ATTENTION_MASK"
echo "Token eviction      : capacity=$TOKEN_CAPACITY granularity=$TOKEN_EVICTION_GRANULARITY backend=engine layers=$TOKEN_SCORE_LAYER_MODE:$TOKEN_SCORE_LAYERS pool=$TOKEN_SCORE_POOLING/$TOKEN_SCORE_POOL_KERNEL"
echo "Decode setting      : FastDLLMDreamEngine block=$BLOCK_LENGTH max_new=$MAX_NEW_TOKENS"
echo "Log file            : $LOG_FILE"
echo "============================================"

cd "$PREFILLING_DLLM_ENGINE_DIR"

"$PYTHON" <<'PYCODE' 2>&1 | tee "$LOG_FILE"
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Sequence

repo_dir = os.environ["REPO_DIR"]
prefilling_dllm_eval_dir = os.environ["PREFILLING_DLLM_EVAL_DIR"]
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if prefilling_dllm_eval_dir not in sys.path:
    sys.path.insert(0, prefilling_dllm_eval_dir)

from prefilling_dllm import FastDLLMDreamEngine
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


def union_keep_indices_per_prompt(keep_indices, full_prompt_len):
    keep = set()
    for layer in keep_indices or []:
        for head in layer or []:
            for idx in head or []:
                idx = int(idx)
                if 0 <= idx < full_prompt_len:
                    keep.add(idx)
    if not keep:
        keep = set(range(full_prompt_len))
    return sorted(keep)


dream_base = os.environ["DREAM_BASE"]
data_dir = os.environ["DATA_DIR"]
config_dir = os.environ["CONFIG_DIR"]
task = os.environ["TASK_NAME"]
log_dir = Path(os.environ["PREFILLING_DLLM_ENGINE_DIR"]) / "log"
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
token_eviction_granularity = os.environ.get("TOKEN_EVICTION_GRANULARITY", "per_head")
max_new_tokens = env_int("MAX_NEW_TOKENS", 32)
block_length = env_int("BLOCK_LENGTH", 32)
max_model_len = env_int("MAX_MODEL_LEN", 8192)
gpu_memory_utilization = env_float("GPU_MEMORY_UTILIZATION", 0.60)
threshold = env_float("THRESHOLD", 0.9)
kv_cache_layout = os.environ.get("KV_CACHE_LAYOUT", "unified")
dry_run = env_int("DRY_RUN", 0)

if token_eviction_granularity != "per_head":
    raise ValueError("full-engine bridge currently targets per_head token eviction.")

prompt_templates = load_json(Path(config_dir) / "dataset2prompt_raw.json")
dataset2maxlen = load_json(Path(config_dir) / "dataset2maxlen.json")
examples = load_task_examples(task, data_dir, max_examples=0)
examples = examples[start_index:]
if limit is not None:
    examples = examples[:limit]
print(f"Loaded examples: {len(examples)}", flush=True)
print(f"Official max_new for {task}: {dataset2maxlen[task]}", flush=True)

print("Loading FastDLLMDreamEngine for selection + keep-index + pre-evict prefill decode...", flush=True)
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

result_path = log_dir / f"longbench_{task}_fastdllm_engine_full_bridge_results_{run_ts}.json"
compressed_path = log_dir / f"longbench_{task}_fastdllm_engine_full_bridge_compressed_{run_ts}.json"
compressed_records = []
selection_t0 = time.time()
template = prompt_templates[task]

try:
    for idx, example in enumerate(examples):
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
                    "token_score_backend": "engine",
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
        )
        preevict_indices = union_keep_indices_per_prompt(keep_indices, len(prompt_ids))
        preevict_prompt_ids = [prompt_ids[i] for i in preevict_indices]
        preevict_prompt_positions = [prompt_positions[i] for i in preevict_indices]
        preevict_index_set = set(preevict_indices)
        for chunk_count, engine_meta, span in zip(chunk_keep_counts, engine_chunk_meta, engine_per_head_chunk_spans):
            chunk_count["union_kept_tokens"] = int(engine_meta["union_kept_tokens"])
            chunk_count["preevict_prompt_kept_tokens"] = sum(1 for pos in range(int(span["start"]), int(span["end"])) if pos in preevict_index_set)

        compressed_records.append(
            {
                "idx": start_index + idx,
                "example_id": example.get("_id", idx),
                "prompt_ids": preevict_prompt_ids,
                "prompt_positions": preevict_prompt_positions,
                "active_prompt_positions": None,
                "prompt_keep_indices_per_layer_per_head": None,
                "original_prompt_tokens_before_preevict": len(prompt_ids),
                "preevict_prompt_tokens": len(preevict_prompt_ids),
                "preevict_keep_indices_union": preevict_indices,
                "engine_token_eviction": "pre_full_prefill_union_keep",
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
                    "compressed_prompt_tokens_before_preevict": len(prompt_ids),
                    "compressed_prompt_tokens": len(preevict_prompt_ids),
                    "preevict_prompt_tokens": len(preevict_prompt_ids),
                    "preevict_removed_prompt_tokens": len(prompt_ids) - len(preevict_prompt_ids),
                    "active_prompt_tokens": len(preevict_prompt_ids),
                    "max_position": max(preevict_prompt_positions) if preevict_prompt_positions else -1,
                    "query_rope_start": active_query_rope_start,
                    "full_query_rope_start": full_query_rope_start,
                    "selection_query_tokens": len(selection_query_ids),
                    "score_token_mask_true": sum(1 for x in score_token_mask if x) if score_token_mask is not None else None,
                    "chunk_selection_backend": "engine",
                    "token_score_backend": "engine",
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
            "bridge_mode": "fastdllm_full_engine_selection_keep_preevict_decode",
            "selection_seconds": selection_seconds,
            "setting": {
                "chunk_selection_backend": "engine",
                "token_score_backend": "engine",
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
                "token_eviction_granularity": token_eviction_granularity,
                "preevict_before_full_prefill": True,
                "preevict_keep_strategy": "union_across_layers_and_kv_heads",
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
    for idx, (example, compressed) in enumerate(zip(examples, compressed_records)):
        output = engine.generate_token_ids(
            compressed["prompt_ids"],
            max_new_tokens=max_new_tokens,
            prompt_positions=compressed["prompt_positions"],
            stop_token_ids=[engine.tokenizer.eos_token_id] if engine.tokenizer.eos_token_id is not None else None,
        )
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
        if (idx + 1) % 10 == 0 or idx + 1 == len(examples):
            metrics = summarize_metrics(results, longbench_e=False)
            payload = {
                "task": task,
                "run_ts": run_ts,
                "bridge_mode": "fastdllm_full_engine_selection_keep_preevict_decode",
                "completed": len(results),
                "total": len(examples),
                "selection_seconds": selection_seconds,
                "decode_seconds": time.time() - decode_t0,
                "metrics": metrics,
                "setting": {
                    "chunk_selection_backend": "engine",
                    "token_score_backend": "engine",
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
                },
                "results": results,
            }
            save_json(result_path, payload)
            print(f"completed={len(results)}/{len(examples)} score={metrics['score']:.2f} decode_seconds={time.time() - decode_t0:.2f}", flush=True)
finally:
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
