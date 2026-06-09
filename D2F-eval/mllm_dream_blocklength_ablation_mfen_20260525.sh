#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval}"
PY="${PY:-/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python}"
MODEL="${MODEL:-${ROOT}/model_weights/Dream-v0-Base-7B}"
FASTDLLM_DREAM="${FASTDLLM_DREAM:-/home/ma-user/work/Fast-dLLM/v1/dream}"
LB_DATA="${LB_DATA:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
LB_CONFIG="${LB_CONFIG:-/home/ma-user/work/ParallelComp_official/longbench_config}"
RUN_STAMP="${RUN_STAMP:-20260525_dream_blocklen_mfen}"
BLOCKS=(${BLOCKS:-8 16 4})

cd "${ROOT}"
mkdir -p logs

pick_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return
  fi
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '$2 + 0 < 2000 {gsub(/ /, "", $1); print $1; exit}'
}

GPU="$(pick_gpu)"
if [[ -z "${GPU}" ]]; then
  echo "No GPU with <2GB memory used is available." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

echo "START dream block-length ablation $(date '+%F %T')"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "blocks=${BLOCKS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

for BLOCK in "${BLOCKS[@]}"; do
  TAG="dream_pc_best47_mfen_block${BLOCK}_${RUN_STAMP}"
  OUT_DIR="${ROOT}/results_longbench_fastdllm_parallelcomp_${TAG}"
  LOG_FILE="${ROOT}/logs/${TAG}.log"

  echo "RUN block_length=${BLOCK} tag=${TAG} $(date '+%F %T')"
  "${PY}" eval_fastdllm_parallelcomp_longbench.py \
    --pretrained "${MODEL}" \
    --model_backend dream \
    --fastdllm_dream_dir "${FASTDLLM_DREAM}" \
    --data_dir "${LB_DATA}" \
    --config_dir "${LB_CONFIG}" \
    --tasks multifieldqa_en \
    --max_examples 0 \
    --run_name "${TAG}" \
    --output_dir "${OUT_DIR}" \
    --max_new_tokens 32 \
    --max_length 4096 \
    --block_length "${BLOCK}" \
    --diffusion_steps 1 \
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
  echo "DONE block_length=${BLOCK} $(date '+%F %T')"
done

echo "DONE_ALL dream block-length ablation $(date '+%F %T')"
