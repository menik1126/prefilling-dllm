#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ma-user/work/Discrete-Diffusion-Forcing
REPO="$ROOT/d2f_vllm"
cd "$REPO"
mkdir -p log/mllm

export PYTHON=/home/ma-user/work/venvs/d2f/bin/python
export D2F_EVAL_DIR="$ROOT/D2F-eval"
export DREAM_BASE="$ROOT/D2F-eval/model_weights/Dream-v0-Base-7B"
export LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
export LONGBENCH_CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export LONGBENCH_TASK="${LONGBENCH_TASK:-multifieldqa_en}"
export LIMIT="${LIMIT:-10}"
export START_INDEX="${START_INDEX:-0}"
export TOKEN_SCORE_BACKEND="${TOKEN_SCORE_BACKEND:-torch}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export BLOCK_LENGTH="${BLOCK_LENGTH:-32}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-32000}"

run_mode() {
  local label="$1"
  local pd_remote="$2"
  local pd_overlap="$3"
  local offset="$4"

  export PD_REMOTE_ENGINE="$pd_remote"
  export PD_PIPELINE_OVERLAP="$pd_overlap"
  export MASTER_PORT="$((BASE_PORT + offset))"
  export PD_DECODE_MASTER_PORT="$((MASTER_PORT + 1))"
  export PD_DECODE_SHM_NAME="d2f_mllm_pd_${label}_${RUN_ID}"
  export RUN_TS="mllm_pd_ab_${label}_${LONGBENCH_TASK}_limit${LIMIT}_${RUN_ID}"

  echo "===== mllm PD A/B mode=$label task=$LONGBENCH_TASK limit=$LIMIT master_port=$MASTER_PORT ====="
  bash scripts/run_dream_longbench_task_fastdllm_engine_full_bridge.sh
}

run_mode nonpd 0 0 0
run_mode pd_serial 1 0 2
run_mode pd_overlap 1 1 4

echo "MLLM_PD_OVERLAP_AB_DONE"
