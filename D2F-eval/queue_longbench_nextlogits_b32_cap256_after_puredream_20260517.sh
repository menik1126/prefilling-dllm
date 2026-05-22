#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval
mkdir -p logs

WAIT_B16="settingA_nextlogits_b16_infinitebench_full_20260517_1113"
WAIT_PUREDREAM="longbench_pure_dream_chunks_ntk4k_multifieldqa_en_full_20260517_1411"
WAIT_HEADTAIL="longbench_pure_dream_headtail_ntk4k_multifieldqa_en_full_20260517_1509"
LOG="logs/longbench_nextlogits_b32_cap256_after_puredream_20260517.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

log "waiting for ${WAIT_B16}, ${WAIT_PUREDREAM}, and ${WAIT_HEADTAIL}"
while ps -eo cmd | grep -E "${WAIT_B16}|${WAIT_PUREDREAM}|${WAIT_HEADTAIL}" | grep -v grep >/dev/null 2>&1; do
  sleep 60
done

log "starting original cap256 LongBench queue"
exec bash ./queue_longbench_nextlogits_b32_cap256_full_20260517.sh
