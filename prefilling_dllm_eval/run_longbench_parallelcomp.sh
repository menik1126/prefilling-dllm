#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK="${1:-narrativeqa}"
TOKEN_CAPACITY="${2:-128}"
MAX_EXAMPLES="${3:-20}"

PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/work/conda-envs/prefilling_dllm_eval_parallelcomp/bin/python}"
DREAM_BASE="${DREAM_BASE:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B}"
DREAM_LORA="${DREAM_LORA:-${SCRIPT_DIR}/model_weights/Dream-v0-Base-7B-LoRA}"
LONGBENCH_DATA="${LONGBENCH_DATA:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
LONGBENCH_CONFIG="${LONGBENCH_CONFIG:-/home/ma-user/work/ParallelComp_official/longbench_config}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
CHUNK_SIZE="${PARALLELCOMP_CHUNK_SIZE:-1024}"
TOPK_CHUNKS="${PARALLELCOMP_TOPK_CHUNKS:-3}"
CHUNK_SCORE_QUERY_WINDOW="${PARALLELCOMP_CHUNK_SCORE_QUERY_WINDOW:-${PARALLELCOMP_QUERY_WINDOW:-0}}"
RECENT_TOKEN_WINDOW="${PARALLELCOMP_RECENT_TOKEN_WINDOW:-${PARALLELCOMP_QUERY_WINDOW:-0}}"
CHUNK_SCORE_ATTENTION_MASK="${PARALLELCOMP_LOCAL_ATTENTION_MASK:-${PARALLELCOMP_CHUNK_SCORE_ATTENTION_MASK:-query_to_chunk}}"
TOKEN_KEEP_MIN="${PARALLELCOMP_TOKEN_KEEP_MIN:-32}"
SCORE_MODE="${PARALLELCOMP_SCORE_MODE:-self_information}"
SEGMENT_SEPARATOR="${PARALLELCOMP_SEGMENT_SEPARATOR:-$'\n\n'}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-dream_longbench_${TASK}_cap${TOKEN_CAPACITY}_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_longbench_${TASK}_n${MAX_EXAMPLES}_cap${TOKEN_CAPACITY}_${RUN_TAG}}"

if [[ "$TASK" == "all" ]]; then
  TASK_ARGS=(
    narrativeqa
    qasper
    multifieldqa_en
    hotpotqa
    2wikimqa
    musique
    trec
    triviaqa
    passage_count
    passage_retrieval_en
    qmsum
    samsum
    lcc
    multi_news
    repobench-p
    gov_report
  )
else
  TASK_ARGS=("$TASK")
fi

export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES

cd "$SCRIPT_DIR"

echo "Task                : ${TASK}"
echo "Token capacity      : ${TOKEN_CAPACITY}"
echo "Max examples        : ${MAX_EXAMPLES}"
echo "Local attention mask: ${CHUNK_SCORE_ATTENTION_MASK}"
echo "Query windows       : score=${CHUNK_SCORE_QUERY_WINDOW}, token=${RECENT_TOKEN_WINDOW}"
echo "Score mode          : ${SCORE_MODE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Run name            : ${RUN_NAME}"
echo "Output dir          : ${OUTPUT_DIR}"

cmd=(
  "$PYTHON_BIN" eval_longbench_dream.py
  --model_type dream
  --pretrained "$DREAM_BASE"
  --lora_path "$DREAM_LORA"
  --data_dir "$LONGBENCH_DATA"
  --config_dir "$LONGBENCH_CONFIG"
  --tasks "${TASK_ARGS[@]}"
  --max_examples "$MAX_EXAMPLES"
  --max_length "$MAX_LENGTH"
  --block_size "$BLOCK_SIZE"
  --temperature 0
  --run_name "$RUN_NAME"
  --segment_separator "$SEGMENT_SEPARATOR"
  --parallelcomp_mode
  --parallelcomp_pre_runtime_mode
  --parallelcomp_cache_compress_mode
  --parallelcomp_chunk_size "$CHUNK_SIZE"
  --parallelcomp_topk_chunks "$TOPK_CHUNKS"
  --parallelcomp_chunk_score_query_window "$CHUNK_SCORE_QUERY_WINDOW"
  --parallelcomp_chunk_score_attention_mask "$CHUNK_SCORE_ATTENTION_MASK"
  --parallelcomp_recent_token_window "$RECENT_TOKEN_WINDOW"
  --parallelcomp_min_prompt_tokens 1
  --parallelcomp_token_capacity "$TOKEN_CAPACITY"
  --parallelcomp_token_keep_min "$TOKEN_KEEP_MIN"
  --parallelcomp_tail_replay_full_mask
  --parallelcomp_fixed_query_text "Please answer the question using the long context above."
  --parallelcomp_score_mode "$SCORE_MODE"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_NEW_TOKENS" ]]; then
  cmd+=(--max_new_tokens "$MAX_NEW_TOKENS")
fi

"${cmd[@]}"
