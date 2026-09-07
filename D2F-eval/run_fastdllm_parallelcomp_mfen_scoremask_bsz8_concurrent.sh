#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python}"
MODEL="${MODEL:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B}"
FASTDLLM_DREAM="${FASTDLLM_DREAM:-/home/ma-user/work/Fast-dLLM/v1/dream}"
LONGBENCH_DATA="${LONGBENCH_DATA:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
LONGBENCH_CONFIG="${LONGBENCH_CONFIG:-/home/ma-user/work/ParallelComp_official/longbench_config}"

GPU_CAUSAL="${GPU_CAUSAL:-0}"
GPU_FULL="${GPU_FULL:-1}"
WAIT="${WAIT:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

cd "${SCRIPT_DIR}"
mkdir -p logs

COMMON_ARGS=(
  eval_fastdllm_parallelcomp_longbench.py
  --pretrained "${MODEL}"
  --fastdllm_dream_dir "${FASTDLLM_DREAM}"
  --data_dir "${LONGBENCH_DATA}"
  --config_dir "${LONGBENCH_CONFIG}"
  --tasks multifieldqa_en
  --max_examples 0
  --max_new_tokens 32
  --max_length 4096
  --block_length 32
  --temperature 0
  --alg confidence_threshold
  --threshold 0.9
  --rope_scale_factor 1.0
  --dtype bfloat16
  --chunk_size 1024
  --topk_chunks 4
  --chunk_bos
  --score_mode draft_self_information
  --score_draft_tokens 4
  --score_draft_partial_rounds 1
  --score_batch_size 8
  --score_context_mode single_chunk
  --cache_build_mode full_prompt_mask
  --token_capacity 0
  --token_score_layer_mode all
  --token_score_layers 0
  --chunk_position_mode continuous
  --query_position_mode after_selected_chunks
)

write_cmd_file() {
  local gpu="$1"
  local cmd_file="$2"
  shift 2

  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    printf '%q ' "${PYTHON_BIN}" "$@"
    printf '\n'
  } > "${cmd_file}"
}

launch_one() {
  local mask="$1"
  local gpu="$2"
  local run_name="fastdllm_parallelcomp_mfen_best47_${mask}_scoremask_bsz8_${RUN_TAG}"
  local output_dir="${SCRIPT_DIR}/results_longbench_fastdllm_parallelcomp_${run_name}"
  local log_file="${SCRIPT_DIR}/logs/${run_name}.log"
  local cmd_file="${SCRIPT_DIR}/logs/${run_name}.cmd"
  local cmd=(
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${output_dir}"
    --score_attention_mask "${mask}"
  )

  write_cmd_file "${gpu}" "${cmd_file}" "${cmd[@]}"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" "${cmd[@]}"
  ) > "${log_file}" 2>&1 &

  local pid=$!
  echo "${mask} pid=${pid} gpu=${gpu}"
  echo "${mask} log=${log_file}"
  echo "${mask} cmd=${cmd_file}"
  echo "${mask} output=${output_dir}"
  LAST_PID="${pid}"
}

echo "Running LongBench multifieldqa_en best47 score-mask comparison, score_batch_size=8"
echo "RUN_TAG=${RUN_TAG}"
echo "causal uses GPU ${GPU_CAUSAL}; full uses GPU ${GPU_FULL}"

launch_one causal "${GPU_CAUSAL}"
CAUSAL_PID="${LAST_PID}"

launch_one full "${GPU_FULL}"
FULL_PID="${LAST_PID}"

if [[ "${WAIT}" != "1" ]]; then
  echo "WAIT=${WAIT}; jobs are running in the background."
  exit 0
fi

set +e
wait "${CAUSAL_PID}"
CAUSAL_STATUS=$?
wait "${FULL_PID}"
FULL_STATUS=$?
set -e

echo "causal exit_status=${CAUSAL_STATUS}"
echo "full exit_status=${FULL_STATUS}"

echo "Recent metric lines:"
grep -R "score\|multifieldqa_en" "${SCRIPT_DIR}/results_longbench_fastdllm_parallelcomp_fastdllm_parallelcomp_mfen_best47_"*"_scoremask_bsz8_${RUN_TAG}" 2>/dev/null | tail -n 40 || true

if [[ "${CAUSAL_STATUS}" -ne 0 || "${FULL_STATUS}" -ne 0 ]]; then
  exit 1
fi
