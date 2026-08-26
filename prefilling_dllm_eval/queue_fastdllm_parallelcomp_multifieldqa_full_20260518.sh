#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ma-user/work/prefilling-dllm/prefilling_dllm_eval
PY=/home/ma-user/work/conda-envs/prefilling_dllm_eval_parallelcomp/bin/python
MODEL=${ROOT}/model_weights/Dream-v0-Base-7B
FASTDLLM_DREAM=/home/ma-user/work/Fast-dLLM/v1/dream
LB_DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
LB_CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config

TAG=fastdllm_parallelcomp_multifieldqa_top4_draft4_partialrounds1_fullpromptmask_contpos_noevict_full_20260520_$(date +%H%M)
OUT_DIR=${ROOT}/results_longbench_fastdllm_parallelcomp_${TAG}
RUN_NAME=${TAG}
LOG_FILE=${ROOT}/logs/${TAG}.log
LAUNCH_LOG=${ROOT}/logs/${TAG}.launcher.log

cd "${ROOT}"
mkdir -p logs "${OUT_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LAUNCH_LOG}"
}

find_stably_free_gpu() {
  local threshold_mb=${1:-2000}
  local needed_hits=${2:-3}
  local sleep_sec=${3:-30}
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
  local hits
  hits=$(printf '0 %.0s' $(seq 1 "${gpu_count}"))

  while true; do
    local picked=""
    local new_hits=()
    while IFS=, read -r idx used; do
      idx=$(echo "${idx}" | tr -d ' ')
      used=$(echo "${used}" | tr -d ' ')
      local old_hit
      old_hit=$(echo "${hits}" | awk -v n=$((idx + 1)) '{print $n}')
      if [ "${used}" -lt "${threshold_mb}" ]; then
        old_hit=$((old_hit + 1))
      else
        old_hit=0
      fi
      new_hits[$idx]=${old_hit}
      if [ "${old_hit}" -ge "${needed_hits}" ] && [ -z "${picked}" ]; then
        picked=${idx}
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    hits="${new_hits[*]}"
    if [ -n "${picked}" ]; then
      echo "${picked}"
      return 0
    fi
    log "waiting for a stably free GPU; memory threshold=${threshold_mb}MB hits=${hits}"
    sleep "${sleep_sec}"
  done
}

log "queued Fast-DLLM ParallelComp LongBench multifieldqa_en full"
log "run_name=${RUN_NAME}"
log "setting: topk_chunks=4, chunk_size=1024, score_mode=draft_self_information, score_draft_tokens=4, score_draft_partial_rounds=1, chunk_bos=true, cache_build_mode=full_prompt_mask, chunk_position_mode=continuous, query_position_mode=after_selected_chunks, token_capacity=0"

GPU=$(find_stably_free_gpu 2000 3 30)
log "starting on gpu=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" eval_fastdllm_parallelcomp_longbench.py \
  --pretrained "${MODEL}" \
  --fastdllm_dream_dir "${FASTDLLM_DREAM}" \
  --data_dir "${LB_DATA}" \
  --config_dir "${LB_CONFIG}" \
  --tasks multifieldqa_en \
  --max_examples 0 \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --max_new_tokens 32 \
  --max_length 4096 \
  --block_length 32 \
  --temperature 0 \
  --alg confidence_threshold \
  --threshold 0.9 \
  --rope_scale_factor 1.0 \
  --dtype bfloat16 \
  --chunk_size 1024 \
  --topk_chunks 4 \
  --chunk_bos \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_rounds 1 \
  --score_attention_mask causal \
  --score_context_mode single_chunk \
  --cache_build_mode full_prompt_mask \
  --token_capacity 0 \
  --token_score_layer_mode all \
  --token_score_layers 0 \
  --chunk_position_mode continuous \
  --query_position_mode after_selected_chunks \
  > "${LOG_FILE}" 2>&1

log "finished run_name=${RUN_NAME}"
