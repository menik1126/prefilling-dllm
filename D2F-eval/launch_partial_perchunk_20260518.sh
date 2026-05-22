#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=${ROOT}/model_weights/Dream-v0-Base-7B
FASTDLLM_DREAM=/home/ma-user/work/Fast-dLLM/v1/dream
LB_DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
LB_CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config
RUN_TS=${RUN_TS:-$(date +%H%M)}

cd "${ROOT}"
mkdir -p logs

run_one() {
  local gpu=$1
  local run_name=$2
  shift 2
  local out_dir="${ROOT}/results_longbench_fastdllm_parallelcomp_${run_name}"
  local log_file="${ROOT}/logs/${run_name}.log"

  echo "[$(date '+%F %T')] start gpu=${gpu} run=${run_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_fastdllm_parallelcomp_longbench.py \
    --pretrained "${MODEL}" \
    --fastdllm_dream_dir "${FASTDLLM_DREAM}" \
    --data_dir "${LB_DATA}" \
    --config_dir "${LB_CONFIG}" \
    --tasks multifieldqa_en \
    --max_examples 0 \
    --run_name "${run_name}" \
    --output_dir "${out_dir}" \
    --max_new_tokens 32 \
    --block_length 32 \
    --temperature 0 \
    --alg confidence_threshold \
    --threshold 0.9 \
    --dtype bfloat16 \
    --chunk_size 1024 \
    --topk_chunks 4 \
    --chunk_bos \
    --cache_build_mode full_prompt_mask \
    --token_capacity 0 \
    --token_score_layer_mode all \
    --token_score_layers 0 \
    --chunk_position_mode continuous \
    --query_position_mode after_selected_chunks \
    "$@" \
    > "${log_file}" 2>&1
  echo "[$(date '+%F %T')] done gpu=${gpu} run=${run_name}"
}

(
  run_one 0 "fastdllm_parallelcomp_multifieldqa_top4_draft4_partialsteps1_chunkbos_noevict_cache_fullpromptmask_contpos_full_20260518_${RUN_TS}" \
    --score_mode draft_self_information \
    --score_draft_tokens 4 \
    --score_draft_partial_steps 1
  run_one 0 "fastdllm_parallelcomp_multifieldqa_top4_perchunkdraft4_chunkbos_noevict_cache_fullpromptmask_contpos_full_20260518_${RUN_TS}" \
    --score_mode per_chunk_draft_self_information \
    --score_draft_tokens 4
) &

run_one 1 "fastdllm_parallelcomp_multifieldqa_top4_draft4_partialsteps2_chunkbos_noevict_cache_fullpromptmask_contpos_full_20260518_${RUN_TS}" \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_steps 2 &

run_one 2 "fastdllm_parallelcomp_multifieldqa_top4_draft4_partialsteps3_chunkbos_noevict_cache_fullpromptmask_contpos_full_20260518_${RUN_TS}" \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_steps 3 &

run_one 3 "fastdllm_parallelcomp_multifieldqa_top4_draft4_partialsteps4_chunkbos_noevict_cache_fullpromptmask_contpos_full_20260518_${RUN_TS}" \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_steps 4 &

wait
echo "[$(date '+%F %T')] all launched runs finished"
