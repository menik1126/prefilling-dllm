#!/usr/bin/env bash
#SBATCH --job-name=dream-fe-lb
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15%5
#SBATCH --chdir=/home/xiongjing/prefilling-dllm
#SBATCH --output=/home/xiongjing/prefilling-dllm/prefilling_dllm_engine/log/%x_%A_%a.out
#SBATCH --error=/home/xiongjing/prefilling-dllm/prefilling_dllm_engine/log/%x_%A_%a.err

set -euo pipefail

REPO_DIR="/home/xiongjing/prefilling-dllm"
PREFILLING_DLLM_ENGINE_DIR="${REPO_DIR}/prefilling_dllm_engine"
TASK_SCRIPT="${PREFILLING_DLLM_ENGINE_DIR}/scripts/run_dream_longbench_task_fastdllm_engine_full_bridge.sh"

TASKS=(
  narrativeqa
  qasper
  multifieldqa_en
  hotpotqa
  2wikimqa
  musique
  trec
  triviaqa
  passage_count
  passage_retrieval_en
  qmsum
  samsum
  lcc
  multi_news
  repobench-p
  gov_report
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -ge "${#TASKS[@]}" ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; task count=${#TASKS[@]}" >&2
  exit 2
fi

TASK="${TASKS[$TASK_ID]}"
TASK_SAFE="${TASK//-/_}"
TASK_SAFE="${TASK_SAFE//./_}"

mkdir -p "${PREFILLING_DLLM_ENGINE_DIR}/log"
cd "${REPO_DIR}"

if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS%%,*}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

RUN_TS_BASE="${RUN_TS:-lab_engine_full_cap512_perhead_all0_${SLURM_ARRAY_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
export RUN_TS="${RUN_TS_BASE}_${TASK_SAFE}"
export LONGBENCH_TASK="${TASK}"

export PYTHON="${PYTHON:-/home/xiongjing/resource_dir/envs/mmsearch-r1/bin/python}"
export DREAM_BASE="${DREAM_BASE:-/mnt/Data/xiongjing/models/Dream-v0-Base-7B}"
export LONGBENCH_DATA_DIR="${LONGBENCH_DATA_DIR:-/mnt/Data/xiongjing/ParallelComp/datasets/LongBench}"
export LONGBENCH_CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/mnt/Data/xiongjing/ParallelComp/longbench_config}"
export HF_HOME="${HF_HOME:-/mnt/Data/xiongjing/huggingface_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PREFILLING_DLLM_ENGINE_ATTENTION_BACKEND="${PREFILLING_DLLM_ENGINE_ATTENTION_BACKEND:-sdpa}"

export TOKEN_CAPACITY="${TOKEN_CAPACITY:-512}"
export TOKEN_EVICTION_GRANULARITY="${TOKEN_EVICTION_GRANULARITY:-per_head}"
export TOKEN_SCORE_LAYER_MODE="${TOKEN_SCORE_LAYER_MODE:-all}"
export TOKEN_SCORE_LAYERS="${TOKEN_SCORE_LAYERS:-0}"
export TOKEN_SCORE_POOLING="${TOKEN_SCORE_POOLING:-maxpool}"
export TOKEN_SCORE_POOL_KERNEL="${TOKEN_SCORE_POOL_KERNEL:-7}"
export TOKEN_SCORE_QUERY_WINDOW="${TOKEN_SCORE_QUERY_WINDOW:-8}"
export TOKEN_SCORE_REDUCE="${TOKEN_SCORE_REDUCE:-sum}"
export TOKEN_SCORE_HEAD_REDUCE="${TOKEN_SCORE_HEAD_REDUCE:-sum}"
export TOKEN_SCORE_LAYER_REDUCE="${TOKEN_SCORE_LAYER_REDUCE:-mean}"
export TOKEN_SCORE_DIRECTION="${TOKEN_SCORE_DIRECTION:-query_to_chunk}"
export TOKEN_SCORE_KEEP="${TOKEN_SCORE_KEEP:-high}"
export TOKEN_SCORE_INCLUDE_PREFIX="${TOKEN_SCORE_INCLUDE_PREFIX:-1}"
export TOKEN_SCORE_USE_GENERATED="${TOKEN_SCORE_USE_GENERATED:-0}"
export TOKEN_ATTENTION_MASK="${TOKEN_ATTENTION_MASK:-causal}"

export CACHE_BUILD_MODE="${CACHE_BUILD_MODE:-full_prompt_mask}"
export CHUNK_POSITION_MODE="${CHUNK_POSITION_MODE:-continuous}"
export QUERY_POSITION_MODE="${QUERY_POSITION_MODE:-after_selected_chunks}"
export SCORE_MODE="${SCORE_MODE:-draft_self_information}"
export SCORE_DRAFT_TOKENS="${SCORE_DRAFT_TOKENS:-4}"
export SCORE_DRAFT_PARTIAL_ROUNDS="${SCORE_DRAFT_PARTIAL_ROUNDS:-1}"
export SCORE_ATTENTION_MASK="${SCORE_ATTENTION_MASK:-causal}"
export SCORE_CONTEXT_MODE="${SCORE_CONTEXT_MODE:-single_chunk}"
export PC_CHUNK_SIZE="${PC_CHUNK_SIZE:-1024}"
export TOPK_CHUNKS="${TOPK_CHUNKS:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
export BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
export KV_CACHE_LAYOUT="${KV_CACHE_LAYOUT:-unified}"

echo "START $(date '+%F %T')"
echo "Host                 : $(hostname)"
echo "SLURM job/array      : ${SLURM_JOB_ID:-manual}/${TASK_ID}"
echo "Task                 : ${TASK}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python               : ${PYTHON}"
echo "Dream model          : ${DREAM_BASE}"
echo "LongBench data       : ${LONGBENCH_DATA_DIR}"
echo "LongBench config     : ${LONGBENCH_CONFIG_DIR}"
echo "Run timestamp        : ${RUN_TS}"
echo "Setting              : full-engine bridge, cap512 per-head, all:0, query_to_chunk, maxpool7"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

exec "${TASK_SCRIPT}"
