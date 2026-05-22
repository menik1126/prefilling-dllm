#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
export PARALLELCOMP_CHUNK_SIZE=1024

CAP=128
N=5
STAMP=20260511

run_one() {
  local gpu="$1"
  local task="$2"
  local mask="$3"
  local qwin="$4"
  local label="$5"
  local tag="fourway_nokeep_tailfix_${task}_${label}_${STAMP}"
  local log="logs/${tag}.log"
  echo "[$(date '+%F %T')] START gpu=${gpu} task=${task} mask=${mask} qwin=${qwin} tag=${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PARALLELCOMP_LOCAL_ATTENTION_MASK="$mask" \
  PARALLELCOMP_QUERY_WINDOW="$qwin" \
  RUN_TAG="$tag" \
    ./run_infinitebench_parallelcomp.sh "$task" "$CAP" "$N" > "$log" 2>&1 &
}

run_batch() {
  local task="$1"
  run_one 0 "$task" query_to_chunk 0 q2c_fullq
  run_one 1 "$task" query_to_chunk 8 q2c_q8
  run_one 2 "$task" full 0 full_fullq
  run_one 3 "$task" full 8 full_q8
  wait
  echo "[$(date '+%F %T')] DONE task=${task}"
}

run_batch passkey
run_batch number_string
run_batch kv_retrieval

echo "[$(date '+%F %T')] ALL DONE"
