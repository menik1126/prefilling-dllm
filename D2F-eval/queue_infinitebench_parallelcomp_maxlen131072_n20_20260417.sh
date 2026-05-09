#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval

TASK="$1"
MAX_NEW_TOKENS="$2"
RUN_TAG="${3:-parallelcomp_maxlen131072_n20_20260417}"
PROMPT_STYLE="${4:-parallelcomp_raw}"
MAX_EXAMPLES="${5:-20}"
MAX_LENGTH="${6:-131072}"
ALLOWED_GPUS="${ALLOWED_GPUS:-0 2 3}"

LOG_DIR="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs"
OUTDIR="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_infinitebench_${RUN_TAG}"
LOG="${LOG_DIR}/infinitebench_${TASK}_${RUN_TAG}.log"
PIDFILE="${LOG_DIR}/infinitebench_${TASK}_${RUN_TAG}.pid"
CONDA_BIN=/home/ma-user/miniconda3/bin/conda
ENV_PREFIX=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp

mkdir -p "$LOG_DIR"
mkdir -p "$OUTDIR"
exec >> "$LOG" 2>&1

echo $$ > "$PIDFILE"
echo "[$(date)] queue worker started for task=${TASK} max_new_tokens=${MAX_NEW_TOKENS} max_examples=${MAX_EXAMPLES} max_length=${MAX_LENGTH} prompt_style=${PROMPT_STYLE}"
echo "[$(date)] allowed_gpus=${ALLOWED_GPUS}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' -v allowed="$ALLOWED_GPUS" '
      BEGIN {
        split(allowed, arr, " ");
        for (i in arr) {
          if (arr[i] != "") {
            ok[arr[i]] = 1;
          }
        }
      }
      {
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        if (($1 in ok) && (($2 + 0) < 10000)) {
          print $1;
          exit;
        }
      }
    '
}

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
    --max_examples "$MAX_EXAMPLES" \
    --max_length "$MAX_LENGTH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --block_size 32 \
    --temperature 0 \
    --prompt_style "$PROMPT_STYLE" \
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
    GPU="$(pick_gpu)"
  fi
  if [ -n "${GPU:-}" ]; then
    launch "$GPU"
    status=$?
    echo "[$(date)] task=${TASK} finished with status ${status}"
    exit "$status"
  fi
  echo "[$(date)] task=${TASK} waiting for a free GPU from: ${ALLOWED_GPUS}"
  sleep 60
done
