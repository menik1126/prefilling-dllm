#!/bin/bash

#SBATCH -J d2f_qlocal_8k_top100
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=LocalQ
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/Data/xiongjing/d2f_results/logs/%x_%j.out
#SBATCH --error=/mnt/Data/xiongjing/d2f_results/logs/%x_%j.out

set -euo pipefail

source /mnt/Data/xiongjing/venvs/d2f/bin/activate

export HF_HOME=/mnt/Data/xiongjing/hf_cache
export HF_DATASETS_CACHE=/mnt/Data/xiongjing/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/Data/xiongjing/hf_cache/transformers
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /home/xiongjing/Discrete-Diffusion-Forcing/D2F-eval

echo "Starting local_prefix query ablation on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

python eval_d2f_plcc.py \
  --model_type dream \
  --pretrained /mnt/Data/xiongjing/models/Dream-v0-Base-7B \
  --lora_path /mnt/Data/xiongjing/models/D2F_Dream_Base_7B_Lora \
  --configs medium_context \
  --top_percent 100 \
  --max_length 8192 \
  --max_new_tokens 128 \
  --rope_scale_factor 1.0 \
  --parallelcomp_mode \
  --parallelcomp_cache_compress_mode \
  --parallelcomp_chunk_size 1024 \
  --parallelcomp_query_tokens 512 \
  --parallelcomp_topk_chunks 6 \
  --parallelcomp_min_prompt_tokens 1024 \
  --parallelcomp_token_capacity 256 \
  --parallelcomp_token_keep_min 32 \
  --parallelcomp_query_strategy local_prefix \
  --parallelcomp_local_query_lines 16 \
  --output_dir /mnt/Data/xiongjing/d2f_results/plcc_8k_medium_hf_top100_qlocal
