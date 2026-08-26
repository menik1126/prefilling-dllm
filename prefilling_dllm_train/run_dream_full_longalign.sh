#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
"${PYTHON}" -m accelerate.commands.launch \
  --mixed_precision bf16 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29577}" \
  train.py \
  --config config/dream_full_longalign.yaml
