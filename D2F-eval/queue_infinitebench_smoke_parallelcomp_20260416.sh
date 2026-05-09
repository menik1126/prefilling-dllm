#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval

TASK="$1"
MAX_NEW_TOKENS="$2"
RUN_TAG="parallelcomp_20260416"
LOG="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/infinitebench_${TASK}_${RUN_TAG}.log"
PIDFILE="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/infinitebench_${TASK}_${RUN_TAG}.pid"
OUTDIR="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_infinitebench_smoke_${RUN_TAG}"
CONDA_BIN=/home/ma-user/miniconda3/bin/conda
ENV_PREFIX=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp

mkdir -p /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs
mkdir -p "$OUTDIR"
exec >> "$LOG" 2>&1

echo $$ > "$PIDFILE"
echo "[$(date)] queue worker started for task=${TASK} max_new_tokens=${MAX_NEW_TOKENS}"

launch() {
  local gpu="$1"
  echo "[$(date)] launching task=${TASK} on GPU ${gpu}"
  export HF_HOME=/home/ma-user/work/hf-cache
  export HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -p "$ENV_PREFIX" python eval_infinitebench.py \
    --model_type dream \
    --pretrained /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B \
    --lora_path /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora \
    --data_dir /home/ma-user/work/InfiniteBench/data \
    --tasks "$TASK" \
    --max_examples 5 \
    --max_length 32768 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --block_size 32 \
    --temperature 0 \
    --prompt_style parallelcomp_raw \
    --parallelcomp_mode \
    --parallelcomp_cache_compress_mode \
    --parallelcomp_chunk_size 1024 \
    --parallelcomp_topk_chunks 3 \
    --parallelcomp_min_prompt_tokens 1 \
    --parallelcomp_token_capacity 256 \
    --parallelcomp_token_keep_min 32 \
    --parallelcomp_fixed_query_text "Please answer the question using the long context above." \
    --output_dir "$OUTDIR"
}

while true; do
  if [ -n "${FORCE_GPU:-}" ]; then
    GPU="$FORCE_GPU"
  else
    GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) < 10000) {print $1; exit}}')
  fi
  if [ -n "${GPU:-}" ]; then
    launch "$GPU"
    status=$?
    echo "[$(date)] task=${TASK} finished with status ${status}"
    exit "$status"
  fi
  echo "[$(date)] task=${TASK} waiting for a free GPU"
  sleep 60
done
