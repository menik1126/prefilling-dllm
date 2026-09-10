#!/usr/bin/env bash
set -euo pipefail
GPU_INDEX=${1:?Specify an available GPU index}
OUTPUT_DIR=${2:?Specify a new output directory}
[[ "$GPU_INDEX" =~ ^[0-9]+$ ]] || exit 2
test ! -e "$OUTPUT_DIR" || { echo 'Output directory already exists; refusing overwrite' >&2; exit 2; }
BASE=${REFERENCE_ROOT:-/results/reference_sota_20260906}
export PYTHONPATH="$BASE/deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
unset SCORE_KV_EXPERIMENT SCORE_KV_VERIFY PREFIX_CHUNK_BIDIR DRAFT_CONTEXT_VERIFY SHARE_PREFIX_KV DRAFT_ONLINE_SCORE CAUSAL_QUERY_ONLINE
export MERGE_QUERY_PREFILL=1 INCLUDE_QUERY_SCORE=1 QUERY_CONDITIONED_KV=1 DRAFT_CONTEXT_KV=reuse
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
exec /usr/bin/python eval_fastdllm_parallelcomp_longbench.py \
 --pretrained "${MODEL_PATH:-/models/modelscope/models/Dream-org--Dream-v0-Base-7B/snapshots/master}" \
 --fastdllm_dream_dir "$BASE/fastdllm_dream" --data_dir "$BASE/data" --config_dir "$BASE/longbench_config" \
 --tasks multifieldqa_en --max_examples 0 --max_new_tokens 32 --max_length 4096 --block_length 32 \
 --temperature 0 --alg confidence_threshold --threshold 0.9 --rope_scale_factor 1.0 --dtype bfloat16 \
 --chunk_size 1024 --topk_chunks 4 --chunk_bos --score_draft_tokens 4 --score_draft_partial_rounds 1 \
 --score_batch_size 8 --score_context_mode single_chunk --cache_build_mode full_prompt_mask --token_capacity 0 \
 --token_score_layer_mode all --token_score_layers 0 --chunk_position_mode continuous \
 --query_position_mode after_selected_chunks --score_attention_mask full --score_mode shared_prefix_chunk_draft \
 --run_name retained_49381 --output_dir "$OUTPUT_DIR"
