#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
mkdir -p logs

PY=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
MODEL=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/Dream-v0-Base-7B
LORA=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora
DATA=/home/ma-user/work/ParallelComp_official/datasets/LongBench
CONFIG=/home/ma-user/work/ParallelComp_official/longbench_config
TAG=longbench_ntk128k_raw_lefttrunc_full_20260516_1545
RUN_NAME=dream_${TAG}
OUT=/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/results_longbench_all_n0_${TAG}

export HF_HOME=/home/ma-user/work/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_lane() {
  local gpu="$1"
  shift
  local log="logs/${TAG}_gpu${gpu}.log"
  {
    echo "Run tag             : ${TAG}"
    echo "Run name            : ${RUN_NAME}"
    echo "CUDA_VISIBLE_DEVICES: ${gpu}"
    echo "Tasks               : $*"
    echo "Model max_length    : 131072"
    echo "RoPE scale factor   : 64.0"
    echo "Generation max_new  : 32"
    echo "Truncation strategy : raw prompt default left truncation"
    echo "ParallelComp        : runtime entry only; pre-runtime/cache compression disabled"
    echo "Output dir          : ${OUT}"
  } > "${log}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" eval_longbench_dream.py \
    --model_type dream \
    --pretrained "${MODEL}" \
    --lora_path "${LORA}" \
    --data_dir "${DATA}" \
    --config_dir "${CONFIG}" \
    --tasks "$@" \
    --max_examples 0 \
    --max_length 131072 \
    --rope_scale_factor 64.0 \
    --max_new_tokens 32 \
    --block_size 32 \
    --temperature 0 \
    --run_name "${RUN_NAME}" \
    --segment_separator $'\n\n' \
    --parallelcomp_mode \
    --output_dir "${OUT}" >> "${log}" 2>&1
}

run_lane 0 narrativeqa qasper multifieldqa_en trec &
p0=$!
run_lane 1 hotpotqa 2wikimqa musique triviaqa &
p1=$!
run_lane 2 passage_count passage_retrieval_en qmsum samsum &
p2=$!
run_lane 3 lcc multi_news repobench-p gov_report &
p3=$!

wait "${p0}" "${p1}" "${p2}" "${p3}"
echo "[$(date '+%F %T')] ${TAG} completed" | tee -a "logs/${TAG}.done"
