#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
mkdir -p logs

PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
LORA=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora
LB_DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
LB_CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config
WAIT_TAG=settingA_nextlogits_b16_infinitebench_full_20260517_1113
TAG=longbench_nextlogits_b32_cap256_full_20260517_$(date +%H%M)

export HF_HOME=/home/ma-user/work/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PARALLELCOMP_CHUNK_BOS_ABLATION PARALLELCOMP_GENERATION_BLOCK_BOS_ABLATION

log() {
  echo "[$(date '+%F %T')] $*"
}

wait_for_previous_run() {
  log "waiting for previous run tag ${WAIT_TAG} to finish"
  while ps -eo pid,cmd | grep "${WAIT_TAG}" | grep -v grep >/dev/null 2>&1; do
    sleep 60
  done
  log "previous run finished"
}

run_longbench_lane() {
  local gpu="$1"
  shift
  local out="/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_longbench_all_n0_cap256_${TAG}"
  local run_name="dream_${TAG}"
  local log_file="logs/${TAG}_longbench_gpu${gpu}.log"
  {
    echo "Suite               : LongBench"
    echo "Run tag             : ${TAG}"
    echo "Run name            : ${run_name}"
    echo "CUDA_VISIBLE_DEVICES: ${gpu}"
    echo "Tasks               : $*"
    echo "Score mode          : next_block_logits"
    echo "Generation block    : 32"
    echo "Token capacity      : 256"
    echo "Token keep min      : 256"
    echo "Chunk size          : 1024"
    echo "Top-k chunks        : 3"
    echo "Query               : full query"
    echo "Local attention mask: query_to_chunk"
    echo "Output dir          : ${out}"
  } > "${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_longbench_dream.py \
    --model_type dream \
    --pretrained "${MODEL}" \
    --lora_path "${LORA}" \
    --data_dir "${LB_DATA}" \
    --config_dir "${LB_CONFIG}" \
    --tasks "$@" \
    --max_examples 0 \
    --max_length 131072 \
    --max_new_tokens 32 \
    --block_size 32 \
    --temperature 0 \
    --run_name "${run_name}" \
    --segment_separator $'\n\n' \
    --parallelcomp_mode \
    --parallelcomp_pre_runtime_mode \
    --parallelcomp_cache_compress_mode \
    --parallelcomp_chunk_size 1024 \
    --parallelcomp_topk_chunks 3 \
    --parallelcomp_chunk_score_query_window 0 \
    --parallelcomp_chunk_score_attention_mask query_to_chunk \
    --parallelcomp_recent_token_window 0 \
    --parallelcomp_min_prompt_tokens 1 \
    --parallelcomp_token_capacity 256 \
    --parallelcomp_token_keep_min 256 \
    --parallelcomp_tail_replay_full_mask \
    --parallelcomp_fixed_query_text "Please answer the question using the long context above." \
    --parallelcomp_score_mode next_block_logits \
    --output_dir "${out}" >> "${log_file}" 2>&1
}

log "queued LongBench nextlogits_b32 cap256 full tag=${TAG}"
log "config: score_mode=next_block_logits, block_size=32, token_capacity=256, token_keep_min=256, chunk_size=1024, topk=3"

wait_for_previous_run

log "starting LongBench full ${TAG}"
run_longbench_lane 0 narrativeqa qasper multifieldqa_en trec &
p0=$!
run_longbench_lane 1 hotpotqa 2wikimqa musique triviaqa &
p1=$!
run_longbench_lane 2 passage_count passage_retrieval_en qmsum samsum &
p2=$!
run_longbench_lane 3 lcc multi_news repobench-p gov_report &
p3=$!
wait "${p0}" "${p1}" "${p2}" "${p3}"
log "completed LongBench full ${TAG}"
echo "[$(date '+%F %T')] ${TAG} finished" | tee -a "logs/${TAG}.done"
