#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
mkdir -p logs

PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
LORA=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora
DATA=/home/ma-user/work/InfiniteBench/data
HF_HOME=/home/ma-user/work/hf-cache
HF_ENDPOINT=https://hf-mirror.com
PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME HF_ENDPOINT PYTORCH_ALLOC_CONF
unset PARALLELCOMP_CHUNK_BOS_ABLATION PARALLELCOMP_GENERATION_BLOCK_BOS_ABLATION

WAIT_TAG=ntk128k_head_tail_full_20260515_0525

wait_for_current_jobs() {
  while true; do
    running="$(ps -eo pid,cmd | grep -E "eval_infinitebench.py|${WAIT_TAG}" | grep -v grep || true)"
    if [[ -z "${running}" ]]; then
      echo "[$(date '+%F %T')] no current eval jobs; starting queued ablations"
      break
    fi
    echo "[$(date '+%F %T')] waiting for current eval jobs to finish"
    echo "${running}" | head -20
    sleep 300
  done
}

run_task() {
  local gpu="$1"
  local tag="$2"
  local setting="$3"
  local task="$4"
  local rope="$5"
  local block="$6"
  local extra_flags="$7"
  local out="results_infinitebench_${task}_n0_cap128_${tag}_${setting}_${task}"
  local log="logs/${tag}_${setting}_${task}.log"

  {
    echo "Task                : ${task}"
    echo "Run tag             : ${tag}"
    echo "Setting             : ${setting}"
    echo "Token capacity      : 128"
    echo "Max examples        : 0"
    echo "Local attention mask: query_to_chunk"
    echo "Score mode          : next_block_logits"
    echo "Generation block    : ${block}"
    echo "RoPE scale factor   : ${rope}"
    echo "Extra flags         : ${extra_flags}"
    echo "CUDA_VISIBLE_DEVICES: ${gpu}"
    echo "Output dir          : ${out}"
  } > "${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_infinitebench.py \
    --model_type dream \
    --pretrained "${MODEL}" \
    --lora_path "${LORA}" \
    --data_dir "${DATA}" \
    --tasks "${task}" \
    --max_examples 0 \
    --max_length 131072 \
    --rope_scale_factor "${rope}" \
    --max_new_tokens 32 \
    --block_size "${block}" \
    --temperature 0 \
    --prompt_style parallelcomp_raw \
    --parallelcomp_mode \
    --parallelcomp_pre_runtime_mode \
    --parallelcomp_cache_compress_mode \
    --parallelcomp_chunk_size 1024 \
    --parallelcomp_token_capacity 128 \
    --parallelcomp_token_keep_min 128 \
    --parallelcomp_topk_chunks 3 \
    --parallelcomp_min_prompt_tokens 1024 \
    --parallelcomp_chunk_score_attention_mask query_to_chunk \
    --parallelcomp_chunk_score_query_window 0 \
    --parallelcomp_recent_token_window 0 \
    --parallelcomp_score_mode next_block_logits \
    ${extra_flags} \
    --output_dir "${out}" >> "${log}" 2>&1
}

run_lane() {
  local gpu="$1"
  local tag="$2"
  local setting="$3"
  local rope="$4"
  local block="$5"
  local extra_flags="$6"
  shift 6
  for task in "$@"; do
    run_task "${gpu}" "${tag}" "${setting}" "${task}" "${rope}" "${block}" "${extra_flags}"
  done
}

run_suite() {
  local tag="$1"
  local setting="$2"
  local rope="$3"
  local block="$4"
  local extra_flags="$5"

  echo "[$(date '+%F %T')] starting suite ${tag} ${setting}"
  run_lane 0 "${tag}" "${setting}" "${rope}" "${block}" "${extra_flags}" passkey &
  p0=$!
  run_lane 1 "${tag}" "${setting}" "${rope}" "${block}" "${extra_flags}" number_string &
  p1=$!
  run_lane 2 "${tag}" "${setting}" "${rope}" "${block}" "${extra_flags}" kv_retrieval code_debug &
  p2=$!
  run_lane 3 "${tag}" "${setting}" "${rope}" "${block}" "${extra_flags}" longbook_choice_eng math_find &
  p3=$!
  wait "${p0}" "${p1}" "${p2}" "${p3}"
  echo "[$(date '+%F %T')] completed suite ${tag} ${setting}"
}

wait_for_current_jobs
run_suite "full_nextlogits_b64_20260515_1955" "nextlogits_b64" "1.0" "64" ""
run_suite "full_ntk128k_contpos_nextlogits_b32_20260515_1955" "ntk_contpos_nextlogits_b32" "64.0" "32" "--parallelcomp_continuous_chunk_positions"
echo "[$(date '+%F %T')] queued ablations finished"
