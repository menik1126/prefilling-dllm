#!/usr/bin/env bash
set -u

SCRIPT_DIR=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
PYTHON_BIN=${PYTHON_BIN:-/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python}
FASTDLLM_DREAM_DIR=${FASTDLLM_DREAM_DIR:-/home/ma-user/work/Fast-dLLM/v1/dream}
FASTDLLM_LLADA_DIR=${FASTDLLM_LLADA_DIR:-/home/ma-user/work/Fast-dLLM/v1/llada}
LONGBENCH_DATA=${LONGBENCH_DATA:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}
LONGBENCH_CONFIG=${LONGBENCH_CONFIG:-/home/ma-user/work/ParallelComp_official/longbench_config}
ULTRA_MODEL=${ULTRA_MODEL:-${SCRIPT_DIR}/model_weights/UltraLLaDA}
LLADA_MODEL=${LLADA_MODEL:-/home/ma-user/work/models/LLaDA-8B-Instruct}
STAMP=full150_serial_noevict_gt_evict_probe_20260603
LOG_DIR=${SCRIPT_DIR}/logs/${STAMP}
mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}" || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

COMMON_ARGS=(
  --model_backend llada
  --fastdllm_dream_dir "${FASTDLLM_DREAM_DIR}"
  --fastdllm_llada_dir "${FASTDLLM_LLADA_DIR}"
  --max_new_tokens 32
  --block_length 32
  --temperature 0
  --alg confidence_threshold
  --threshold 0.9
  --rope_scaling_type yarn
  --dtype bfloat16
  --no-use_chat_template
  --score_mode draft_self_information
  --score_draft_tokens 4
  --score_draft_partial_rounds 1
  --score_attention_mask causal
  --score_context_mode single_chunk
  --chunk_bos
  --cache_build_mode full_prompt_mask
  --chunk_position_mode continuous
  --query_position_mode after_selected_chunks
  --token_score_query_window 8
  --token_score_layers 0
  --token_score_layer_mode all
  --token_score_reduce sum
  --token_score_pooling maxpool
  --token_score_pool_kernel 7
  --token_score_head_reduce sum
  --token_score_layer_reduce mean
  --token_score_direction bidirectional
  --token_score_keep high
  --token_eviction_granularity per_head
  --token_attention_mask full
  --data_dir "${LONGBENCH_DATA}"
  --config_dir "${LONGBENCH_CONFIG}"
  --tasks multifieldqa_en
  --max_examples 0
)

run_one() {
  local model_label="$1"
  local pretrained="$2"
  local max_length="$3"
  local rope_scale="$4"
  local topk="$5"
  local chunk="$6"
  local cap="$7"
  local cap_label="$8"
  local run_name="${model_label}_pc_top${topk}_chunk${chunk}_draft4_pr1_${cap_label}_${STAMP}"
  local out_dir="${SCRIPT_DIR}/results_longbench_fastdllm_parallelcomp_${run_name}"
  local log="${LOG_DIR}/${run_name}.log"
  local err="${LOG_DIR}/${run_name}.err"

  {
    echo "============================================================"
    echo "START $(date +%F %T) run=${run_name}"
    echo "pretrained=${pretrained} max_length=${max_length} rope=${rope_scale} topk=${topk} chunk=${chunk} cap=${cap}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
  } | tee -a "${LOG_DIR}/controller.log" "${log}"

  "${PYTHON_BIN}" eval_fastdllm_parallelcomp_longbench.py \
    --pretrained "${pretrained}" \
    --max_length "${max_length}" \
    --rope_scale_factor "${rope_scale}" \
    --topk_chunks "${topk}" \
    --chunk_size "${chunk}" \
    --token_capacity "${cap}" \
    --run_name "${run_name}" \
    --output_dir "${out_dir}" \
    "${COMMON_ARGS[@]}" >>"${log}" 2>>"${err}"
  local status=$?
  echo "END $(date +%F %T) run=${run_name} status=${status}" | tee -a "${LOG_DIR}/controller.log" "${log}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee -a "${LOG_DIR}/controller.log" || true
  return 0
}

{
  echo "BATCH START $(date +%F %T) host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES}"
  echo "log_dir=${LOG_DIR}"
} | tee -a "${LOG_DIR}/controller.log"

# UltraLLaDA full n=150 candidates. Same flow, only eviction cap changes.
run_one ultrallada "${ULTRA_MODEL}" 131072 1.0 2 1024 0 noevict
run_one ultrallada "${ULTRA_MODEL}" 131072 1.0 2 1024 512 evict_bidir_high_cap512
run_one ultrallada "${ULTRA_MODEL}" 131072 1.0 2 1024 256 evict_bidir_high_cap256

# LLaDA full n=150 candidates. Mirrors the existing full no-evict top4/draft4/pr1 flow.
run_one llada "${LLADA_MODEL}" 4096 1.0 4 1024 0 noevict
run_one llada "${LLADA_MODEL}" 4096 1.0 4 1024 512 evict_bidir_high_cap512
run_one llada "${LLADA_MODEL}" 4096 1.0 4 1024 256 evict_bidir_high_cap256

echo "BATCH END $(date +%F %T)" | tee -a "${LOG_DIR}/controller.log"
