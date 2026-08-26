#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${WAIT_PID:-707720}"
RUN_TS="${RUN_TS:-engine_full_longbench_cap512_perhead_all0_20260606_after_current}"
PREFILLING_DLLM_ENGINE_DIR="/home/ma-user/work/prefilling-dllm/prefilling_dllm_engine"
LOG_DIR="${PREFILLING_DLLM_ENGINE_DIR}/log"
mkdir -p "${LOG_DIR}"

echo "[$(date '+%F %T')] queued full-engine LongBench; waiting for PID ${WAIT_PID}; run_ts=${RUN_TS}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  echo "[$(date '+%F %T')] current run PID ${WAIT_PID} still active; sleeping 300s"
  sleep 300
done

echo "[$(date '+%F %T')] PID ${WAIT_PID} finished; starting full-engine LongBench"
cd "${PREFILLING_DLLM_ENGINE_DIR}"
export RUN_TS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKEN_CAPACITY=512
export TOKEN_EVICTION_GRANULARITY=per_head
export TOKEN_SCORE_LAYER_MODE=all
export TOKEN_SCORE_LAYERS=0
export TOKEN_SCORE_POOLING=maxpool
export TOKEN_SCORE_POOL_KERNEL=7
export TOKEN_SCORE_QUERY_WINDOW=8
export TOKEN_SCORE_REDUCE=sum
export TOKEN_SCORE_HEAD_REDUCE=sum
export TOKEN_SCORE_LAYER_REDUCE=mean
export TOKEN_SCORE_DIRECTION=query_to_chunk
export TOKEN_SCORE_KEEP=high
export CACHE_BUILD_MODE=full_prompt_mask
export CHUNK_POSITION_MODE=continuous
export QUERY_POSITION_MODE=after_selected_chunks
export SCORE_MODE=draft_self_information
export SCORE_DRAFT_TOKENS=4
export SCORE_DRAFT_PARTIAL_ROUNDS=1
export SCORE_ATTENTION_MASK=causal
export SCORE_CONTEXT_MODE=single_chunk
export MAX_NEW_TOKENS=32
export BLOCK_LENGTH=32
export MAX_MODEL_LEN=8192
export GPU_MEMORY_UTILIZATION=0.60
exec "${PREFILLING_DLLM_ENGINE_DIR}/scripts/run_dream_longbench_all_fastdllm_engine_full_bridge.sh"
