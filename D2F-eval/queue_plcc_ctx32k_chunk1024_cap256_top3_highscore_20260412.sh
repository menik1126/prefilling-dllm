#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval

LOG=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/plcc_ctx32k_chunk1024_cap256_top3_highscore_20260412.log
PIDFILE=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs/plcc_ctx32k_chunk1024_cap256_top3_highscore_20260412.pid
OUTDIR=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_plcc_ctx32k_chunk1024_cap256_top3_highscore_20260412
CONDA_BIN=/home/ma-user/miniconda3/bin/conda
ENV_PREFIX=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp

mkdir -p /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/logs
mkdir -p "$OUTDIR"
exec >> "$LOG" 2>&1

echo $$ > "$PIDFILE"
echo "[$(date '+%F %T')] queue worker started"

launch() {
  local gpu="$1"
  echo "[$(date '+%F %T')] launching on GPU ${gpu}"
  export HF_HOME=/home/ma-user/work/hf-cache
  export HF_ENDPOINT=https://hf-mirror.com
  CUDA_VISIBLE_DEVICES="$gpu" "$CONDA_BIN" run -p "$ENV_PREFIX" python eval_d2f_plcc.py \
    --model_type dream \
    --pretrained /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B \
    --lora_path /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora \
    --configs medium_context \
    --max_length 32768 \
    --max_new_tokens 128 \
    --top_percent 30 \
    --block_size 32 \
    --temperature 0 \
    --rope_scale_factor 1.0 \
    --parallelcomp_mode \
    --parallelcomp_cache_compress_mode \
    --parallelcomp_chunk_size 1024 \
    --parallelcomp_topk_chunks 3 \
    --parallelcomp_min_prompt_tokens 1 \
    --parallelcomp_token_capacity 256 \
    --parallelcomp_token_keep_min 32 \
    --parallelcomp_fixed_query_text "Please complete the preceding code." \
    --output_dir "$OUTDIR"
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
