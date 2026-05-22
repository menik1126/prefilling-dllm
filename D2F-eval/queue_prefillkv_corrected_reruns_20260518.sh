#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=${ROOT}/model_weights/Dream-v0-Base-7B
LB_DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
LB_CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config

TAG=prefillkv_corrected_chunkindependent_full_20260518_$(date +%H%M)
OUT_BASE=${ROOT}/results_longbench_prefill_kv_corrected_${TAG}

cd "${ROOT}"
mkdir -p logs "${OUT_BASE}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_case() {
  local gpu=$1
  local run_name=$2
  shift 2
  local log_file="${ROOT}/logs/${run_name}.log"
  log "start gpu=${gpu} run=${run_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" legacy_prefill_kv/eval_longbench_prefill_kv.py \
    --pretrained "${MODEL}" \
    --data_dir "${LB_DATA}" \
    --config_dir "${LB_CONFIG}" \
    --tasks multifieldqa_en \
    --max_examples 0 \
    --run_name "${run_name}" \
    --output_dir "${OUT_BASE}" \
    --dtype bfloat16 \
    --max_new_tokens 32 \
    --chunk_size 1024 \
    --chunk_position_mode reuse \
    --query_position_mode after_reused_window \
    "$@" \
    > "${log_file}" 2>&1
  log "done gpu=${gpu} run=${run_name}"
}

log "queued corrected pure-Dream prefill-KV reruns tag=${TAG}"
log "output=${OUT_BASE}"

(
  run_case 0 "${TAG}_jointselected_top4_draft16_noevict_nobos" \
    --chunk_cache_mode joint_selected --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --no-chunk_bos
  run_case 0 "${TAG}_independent_top2_selfinfo_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 2 --score_mode self_information --score_draft_tokens 0 --token_capacity 0 --no-chunk_bos
  run_case 0 "${TAG}_independent_top3_selfinfo_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 3 --score_mode self_information --score_draft_tokens 0 --token_capacity 0 --no-chunk_bos
  run_case 0 "${TAG}_independent_top4_selfinfo_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode self_information --score_draft_tokens 0 --token_capacity 0 --no-chunk_bos
) &

(
  run_case 1 "${TAG}_independent_top5_selfinfo_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 5 --score_mode self_information --score_draft_tokens 0 --token_capacity 0 --no-chunk_bos
  run_case 1 "${TAG}_independent_top4_draft16_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --no-chunk_bos
  run_case 1 "${TAG}_independent_top4_draft16_cap256_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 256 --token_eviction_mode cache_slice --no-chunk_bos
  run_case 1 "${TAG}_independent_top4_draft16_cap512_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 512 --token_eviction_mode cache_slice --no-chunk_bos
) &

(
  run_case 2 "${TAG}_independent_top4_draft16_firstlayer_cap256_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 256 --token_eviction_mode first_layer_recompute --no-chunk_bos
  run_case 2 "${TAG}_independent_top4_draft16_firstlayer_cap512_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 512 --token_eviction_mode first_layer_recompute --no-chunk_bos
  run_case 2 "${TAG}_independent_ntk4k_top4_draft16_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --rope_scale_factor 2.0 --no-chunk_bos
  run_case 2 "${TAG}_independent_ntk8k_top4_queryonly_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode self_information --score_draft_tokens 0 --token_capacity 0 --rope_scale_factor 4.0 --no-chunk_bos
) &

(
  run_case 3 "${TAG}_independent_ntk8k_top4_draft16_noevict_nobos" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --rope_scale_factor 4.0 --no-chunk_bos
  run_case 3 "${TAG}_independent_top4_draft16_chunkbos_noevict" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --chunk_bos
  run_case 3 "${TAG}_independent_ntk4k_top4_draft16_chunkbos_noevict" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --rope_scale_factor 2.0 --chunk_bos
  run_case 3 "${TAG}_independent_ntk8k_top4_draft16_chunkbos_noevict" \
    --chunk_cache_mode independent --topk_chunks 4 --score_mode draft_self_information --score_draft_tokens 16 --token_capacity 0 --rope_scale_factor 4.0 --chunk_bos
) &

wait
log "all corrected prefill-KV reruns finished tag=${TAG}"
