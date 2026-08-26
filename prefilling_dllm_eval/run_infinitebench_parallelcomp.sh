#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:-passkey}"
TOKEN_CAPACITY="${2:-128}"
MAX_EXAMPLES="${3:-5}"

PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/conda-envs/prefilling_dllm_eval_parallelcomp/bin/python}"
DREAM_BASE="${DREAM_BASE:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B}"
DREAM_LORA="${DREAM_LORA:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B-LoRA}"
INFINITEBENCH_DATA="${INFINITEBENCH_DATA:-/home/ma-user/work/InfiniteBench/data}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
CHUNK_SIZE="${PARALLELCOMP_CHUNK_SIZE:-1024}"
TOPK_CHUNKS="${PARALLELCOMP_TOPK_CHUNKS:-3}"
CHUNK_SCORE_QUERY_WINDOW="${PARALLELCOMP_CHUNK_SCORE_QUERY_WINDOW:-${PARALLELCOMP_QUERY_WINDOW:-0}}"
RECENT_TOKEN_WINDOW="${PARALLELCOMP_RECENT_TOKEN_WINDOW:-${PARALLELCOMP_QUERY_WINDOW:-0}}"
CHUNK_SCORE_ATTENTION_MASK="${PARALLELCOMP_LOCAL_ATTENTION_MASK:-${PARALLELCOMP_CHUNK_SCORE_ATTENTION_MASK:-query_to_chunk}}"
TOKEN_KEEP_MIN="${PARALLELCOMP_TOKEN_KEEP_MIN:-32}"
SCORE_MODE="${PARALLELCOMP_SCORE_MODE:-self_information}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_infinitebench_${TASK}_n${MAX_EXAMPLES}_cap${TOKEN_CAPACITY}_${RUN_TAG}}"

export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES

cd "$SCRIPT_DIR"

echo "Task                : ${TASK}"
echo "Token capacity      : ${TOKEN_CAPACITY}"
echo "Max examples        : ${MAX_EXAMPLES}"
echo "Local attention mask: ${CHUNK_SCORE_ATTENTION_MASK}"
echo "Query windows       : score=${CHUNK_SCORE_QUERY_WINDOW}, token=${RECENT_TOKEN_WINDOW}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Output dir          : ${OUTPUT_DIR}"

"$PYTHON_BIN" eval_infinitebench.py \
  --model_type dream \
  --pretrained "$DREAM_BASE" \
  --lora_path "$DREAM_LORA" \
  --data_dir "$INFINITEBENCH_DATA" \
  --tasks "$TASK" \
  --max_examples "$MAX_EXAMPLES" \
  --max_length "$MAX_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --block_size "$BLOCK_SIZE" \
  --temperature 0 \
  --prompt_style parallelcomp_raw \
  --parallelcomp_mode \
  --parallelcomp_pre_runtime_mode \
  --parallelcomp_cache_compress_mode \
  --parallelcomp_chunk_size "$CHUNK_SIZE" \
  --parallelcomp_topk_chunks "$TOPK_CHUNKS" \
  --parallelcomp_chunk_score_query_window "$CHUNK_SCORE_QUERY_WINDOW" \
  --parallelcomp_chunk_score_attention_mask "$CHUNK_SCORE_ATTENTION_MASK" \
  --parallelcomp_recent_token_window "$RECENT_TOKEN_WINDOW" \
  --parallelcomp_min_prompt_tokens 1 \
  --parallelcomp_token_capacity "$TOKEN_CAPACITY" \
  --parallelcomp_token_keep_min "$TOKEN_KEEP_MIN" \
  --parallelcomp_tail_replay_full_mask \
  --parallelcomp_fixed_query_text "Please answer the question using the long context above." \
  --parallelcomp_score_mode "$SCORE_MODE" \
  --output_dir "$OUTPUT_DIR"
