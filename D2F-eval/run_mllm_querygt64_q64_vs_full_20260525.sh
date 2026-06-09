#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python}"
ULTRA_MODEL="${ULTRA_MODEL:-${SCRIPT_DIR}/model_weights/UltraLLaDA}"
FASTDLLM_DREAM_DIR="${FASTDLLM_DREAM_DIR:-/home/ma-user/work/Fast-dLLM/v1/dream}"
FASTDLLM_LLADA_DIR="${FASTDLLM_LLADA_DIR:-/home/ma-user/work/Fast-dLLM/v1/llada}"
LONGBENCH_CONFIG="${LONGBENCH_CONFIG:-/home/ma-user/work/ParallelComp_official/longbench_config}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/tmp_longbench_querygt64_n5_20260524}"
TASKS=(qasper samsum triviaqa passage_retrieval_en repobench-p)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

cd "${SCRIPT_DIR}"
mkdir -p logs

echo "START $(date '+%F %T')"
echo "host=$(hostname)"
echo "cuda_visible=${CUDA_VISIBLE_DEVICES}"
echo "data_dir=${DATA_DIR}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

run_one () {
  local label="$1"
  local query_window="$2"
  local run_name="ultrallada_pc_top2_chunk1024_bos_noevict_querygt64_n5_${label}_mllm_20260525"
  local output_dir="${SCRIPT_DIR}/results_longbench_fastdllm_parallelcomp_${run_name}"

  echo "============================================================"
  echo "RUN ${label} score_query_window=${query_window} $(date '+%F %T')"
  echo "run_name=${run_name}"
  echo "output_dir=${output_dir}"
  echo "============================================================"

  "${PYTHON_BIN}" eval_fastdllm_parallelcomp_longbench.py \
    --pretrained "${ULTRA_MODEL}" \
    --model_backend llada \
    --fastdllm_dream_dir "${FASTDLLM_DREAM_DIR}" \
    --fastdllm_llada_dir "${FASTDLLM_LLADA_DIR}" \
    --max_new_tokens 32 \
    --max_length 131072 \
    --block_length 32 \
    --temperature 0 \
    --alg confidence_threshold \
    --threshold 0.9 \
    --rope_scale_factor 1.0 \
    --rope_scaling_type yarn \
    --dtype bfloat16 \
    --no-use_chat_template \
    --score_mode self_information \
    --score_draft_tokens 0 \
    --score_query_window "${query_window}" \
    --score_attention_mask causal \
    --score_context_mode single_chunk \
    --chunk_size 1024 \
    --topk_chunks 2 \
    --chunk_bos \
    --cache_build_mode full_prompt_mask \
    --chunk_position_mode continuous \
    --query_position_mode after_selected_chunks \
    --token_capacity 0 \
    --data_dir "${DATA_DIR}" \
    --config_dir "${LONGBENCH_CONFIG}" \
    --tasks "${TASKS[@]}" \
    --max_examples 0 \
    --run_name "${run_name}" \
    --output_dir "${output_dir}"
}

run_one q64 64
run_one full 0

echo "DONE $(date '+%F %T')"
