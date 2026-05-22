#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
mkdir -p logs

PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
LORA=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora
IB_DATA=/home/ma-user/work/InfiniteBench/data
TAG=settingA_nextlogits_b16_infinitebench_full_20260517_$(date +%H%M)

export HF_HOME=/home/ma-user/work/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PARALLELCOMP_CHUNK_BOS_ABLATION PARALLELCOMP_GENERATION_BLOCK_BOS_ABLATION

run_infinite_task() {
  local gpu="$1"
  local task="$2"
  local out="results_infinitebench_${task}_n0_cap128_${TAG}_${task}"
  local log="logs/${TAG}_infinitebench_${task}.log"
  {
    echo "Suite               : InfiniteBench"
    echo "Task                : ${task}"
    echo "Run tag             : ${TAG}"
    echo "CUDA_VISIBLE_DEVICES: ${gpu}"
    echo "Score mode          : next_block_logits"
    echo "Generation block    : 16"
    echo "Token capacity      : 128"
    echo "Chunk size          : 1024"
    echo "Top-k chunks        : 3"
    echo "Query               : full query"
    echo "Local attention mask: query_to_chunk"
    echo "Output dir          : ${out}"
  } > "${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_infinitebench.py \
    --model_type dream \
    --pretrained "${MODEL}" \
    --lora_path "${LORA}" \
    --data_dir "${IB_DATA}" \
    --tasks "${task}" \
    --max_examples 0 \
    --max_length 131072 \
    --max_new_tokens 32 \
    --block_size 16 \
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
    --output_dir "${out}" >> "${log}" 2>&1
}

run_infinite_lane() {
  local gpu="$1"
  shift
  for task in "$@"; do
    run_infinite_task "${gpu}" "${task}"
  done
}

echo "[$(date '+%F %T')] starting InfiniteBench ${TAG}"
run_infinite_lane 0 passkey &
p0=$!
run_infinite_lane 1 number_string &
p1=$!
run_infinite_lane 2 kv_retrieval code_debug &
p2=$!
run_infinite_lane 3 longbook_choice_eng math_find &
p3=$!
wait "${p0}" "${p1}" "${p2}" "${p3}"
echo "[$(date '+%F %T')] completed InfiniteBench ${TAG}"
echo "[$(date '+%F %T')] ${TAG} finished" | tee -a "logs/${TAG}.done"
