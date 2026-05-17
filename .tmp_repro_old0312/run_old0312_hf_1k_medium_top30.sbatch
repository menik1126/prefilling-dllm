#!/bin/bash

#SBATCH -J d2f_old0312_1k
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

cd /home/xiongjing/Discrete-Diffusion-Forcing/D2F-eval_repro_old0312_20260401

nvidia-smi

python eval_d2f_plcc.py \
  --model_type dream \
  --pretrained /mnt/Data/xiongjing/models/Dream-v0-Base-7B \
  --lora_path /mnt/Data/xiongjing/models/D2F_Dream_Base_7B_Lora \
  --configs medium_context \
  --top_percent 30 \
  --max_length 1024 \
  --output_dir /mnt/Data/xiongjing/d2f_results/plcc_1k_medium_hf_oldwrapper0312
