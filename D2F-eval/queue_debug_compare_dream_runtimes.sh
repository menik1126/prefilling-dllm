#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval

LOG=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/debug_compare_dream_runtimes_queue_20260406.log
PIDFILE=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/debug_compare_dream_runtimes_queue_20260406.pid
JSON_OUT=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/debug_compare_dream_runtimes_20260406.json
RUN_LOG=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/debug_compare_dream_runtimes_run_20260406.log
CONDA_BIN=/home/ma-user/miniconda3/bin/conda
ENV_PREFIX=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp

mkdir -p /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs
exec >> "$LOG" 2>&1

echo $$ > "$PIDFILE"
echo "[$(date '+%F %T')] queue worker started"

launch() {
  local gpu="$1"
  echo "[$(date '+%F %T')] launching debug compare on GPU ${gpu}"
  export HF_HOME=/home/ma-user/work/hf-cache
  export HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -p "$ENV_PREFIX" python debug_compare_dream_runtimes.py > "$JSON_OUT" 2> "$RUN_LOG"
}

while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) < 10000) {print $1; exit}}')
  if [ -n "${GPU:-}" ]; then
    launch "$GPU"
    status=$?
    echo "[$(date '+%F %T')] finished with status ${status}"
    exit "$status"
  fi
  echo "[$(date '+%F %T')] waiting for a free GPU"
  sleep 60
done
