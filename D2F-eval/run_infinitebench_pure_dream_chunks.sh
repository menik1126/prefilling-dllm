#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:-passkey}"
MAX_EXAMPLES="${2:-20}"

PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python}"
DREAM_BASE="${DREAM_BASE:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B}"
GENERATION_LORA_PATH="${PURE_DREAM_GENERATION_LORA_PATH:-${GENERATION_LORA_PATH:-}}"
INFINITEBENCH_DATA="${INFINITEBENCH_DATA:-/home/ma-user/work/InfiniteBench/data}"

MAX_LENGTH="${MAX_LENGTH:-2048}"
ROPE_SCALE_FACTOR="${ROPE_SCALE_FACTOR:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CHUNK_SIZE="${PURE_DREAM_CHUNK_SIZE:-1024}"
TOPK_CHUNKS="${PURE_DREAM_TOPK_CHUNKS:-3}"
SCORE_MODE="${PURE_DREAM_SCORE_MODE:-self_information}"
SCORE_QUERY_WINDOW="${PURE_DREAM_SCORE_QUERY_WINDOW:-0}"
RUN_TAG="${RUN_TAG:-pure_dream_chunks_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_infinitebench_${TASK}_n${MAX_EXAMPLES}_pure_dream_chunks_${RUN_TAG}}"

export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$SCRIPT_DIR"

echo "Task              : ${TASK}"
echo "Model             : pure Dream official diffusion_generate"
echo "Generation LoRA   : ${GENERATION_LORA_PATH:-<none>}"
echo "Max examples      : ${MAX_EXAMPLES}"
echo "Max length        : ${MAX_LENGTH}"
echo "RoPE scale factor : ${ROPE_SCALE_FACTOR}"
echo "Max new tokens    : ${MAX_NEW_TOKENS}"
echo "Chunk size        : ${CHUNK_SIZE}"
echo "Top-k chunks      : ${TOPK_CHUNKS}"
echo "Score mode        : ${SCORE_MODE}"
echo "Score query window: ${SCORE_QUERY_WINDOW}"
echo "Output dir        : ${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${GENERATION_LORA_PATH}" ]]; then
  EXTRA_ARGS+=(--generation_lora_path "$GENERATION_LORA_PATH")
fi

"$PYTHON_BIN" eval_infinitebench_pure_dream_chunks.py \
  --model_path "$DREAM_BASE" \
  --data_dir "$INFINITEBENCH_DATA" \
  --tasks "$TASK" \
  --max_examples "$MAX_EXAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --prompt_style parallelcomp_raw \
  --max_length "$MAX_LENGTH" \
  --rope_scale_factor "$ROPE_SCALE_FACTOR" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --chunk_size "$CHUNK_SIZE" \
  --topk_chunks "$TOPK_CHUNKS" \
  --score_mode "$SCORE_MODE" \
  --score_query_window "$SCORE_QUERY_WINDOW" \
  "${EXTRA_ARGS[@]}" \
  --temperature 0
