#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=${ROOT}/model_weights/Dream-v0-Base-7B
FASTDLLM_DREAM=/home/ma-user/work/Fast-dLLM/v1/dream
INF_DATA=/home/ma-user/work/InfiniteBench/data
LB_DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
LB_CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config

TAG=fastdllm_v1_2k_headtail_full_20260517_$(date +%H%M)
WAIT_TAG=settingA_nextlogits_b32_full_20260516_1705

cd "${ROOT}"
mkdir -p logs

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_previous_run() {
  log "waiting for previous run tag ${WAIT_TAG} to finish"
  while ps -eo pid,cmd | grep "${WAIT_TAG}" | grep -v grep >/dev/null 2>&1; do
    sleep 60
  done
  log "previous run finished"
}

run_inf_lane() {
  local gpu=$1
  shift
  local tasks=("$@")
  local task_label="${tasks[*]}"
  task_label="${task_label// /_}"
  local out_dir="${ROOT}/results_infinitebench_fastdllm_v1_2k_headtail_n0_${TAG}_gpu${gpu}_${task_label}"
  local log_file="${ROOT}/logs/${TAG}_infinitebench_gpu${gpu}_${task_label}.log"
  log "starting InfiniteBench gpu=${gpu} tasks=${tasks[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_fastdllm_v1_infinitebench.py \
    --pretrained "${MODEL}" \
    --fastdllm_dream_dir "${FASTDLLM_DREAM}" \
    --data_dir "${INF_DATA}" \
    --tasks "${tasks[@]}" \
    --max_examples 0 \
    --prompt_style parallelcomp_raw \
    --max_length 2048 \
    --max_new_tokens 32 \
    --block_length 32 \
    --temperature 0 \
    --alg confidence_threshold \
    --threshold 0.9 \
    --dual_cache \
    --truncation_strategy head_tail \
    --output_dir "${out_dir}" \
    > "${log_file}" 2>&1
  log "completed InfiniteBench gpu=${gpu} tasks=${tasks[*]}"
}

run_lb_lane() {
  local gpu=$1
  shift
  local tasks=("$@")
  local task_label="${tasks[*]}"
  task_label="${task_label// /_}"
  local out_dir="${ROOT}/results_longbench_fastdllm_v1_2k_headtail_n0_${TAG}"
  local run_name="fastdllm_v1_2k_headtail_${TAG}"
  local log_file="${ROOT}/logs/${TAG}_longbench_gpu${gpu}_${task_label}.log"
  log "starting LongBench gpu=${gpu} tasks=${tasks[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_fastdllm_v1_longbench.py \
    --pretrained "${MODEL}" \
    --fastdllm_dream_dir "${FASTDLLM_DREAM}" \
    --data_dir "${LB_DATA}" \
    --config_dir "${LB_CONFIG}" \
    --tasks "${tasks[@]}" \
    --max_examples 0 \
    --run_name "${run_name}" \
    --max_length 2048 \
    --max_new_tokens 32 \
    --block_length 32 \
    --temperature 0 \
    --alg confidence_threshold \
    --threshold 0.9 \
    --dual_cache \
    --truncation_strategy head_tail \
    --output_dir "${out_dir}" \
    > "${log_file}" 2>&1
  log "completed LongBench gpu=${gpu} tasks=${tasks[*]}"
}

log "queued Fast-dLLM v1 2K head-tail full baseline tag=${TAG}"
log "pretrained=${MODEL}"
log "fastdllm_dream_dir=${FASTDLLM_DREAM}"
log "config: no LoRA, no ParallelComp, no NTK/YaRN, max_length=2048, max_new_tokens=32, block_length=32, diffusion_steps=1, truncation=head_tail"

wait_for_previous_run

log "starting InfiniteBench full ${TAG}"
run_inf_lane 0 passkey &
pid0=$!
run_inf_lane 1 number_string &
pid1=$!
run_inf_lane 2 kv_retrieval code_debug &
pid2=$!
run_inf_lane 3 longbook_choice_eng math_find &
pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
log "completed InfiniteBench full ${TAG}"

log "starting LongBench full ${TAG}"
run_lb_lane 0 narrativeqa qasper multifieldqa_en trec &
pid0=$!
run_lb_lane 1 hotpotqa 2wikimqa musique triviaqa &
pid1=$!
run_lb_lane 2 passage_count passage_retrieval_en qmsum samsum &
pid2=$!
run_lb_lane 3 lcc multi_news repobench-p gov_report &
pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
log "completed LongBench full ${TAG}"
log "all done ${TAG}"
