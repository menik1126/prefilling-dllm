#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SCRIPT="${SCRIPT_DIR}/run_dream_longbench_task_fastdllm_engine_bridge.sh"
TASKS=(
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

RUN_TS_BASE="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="/home/ma-user/work/prefilling-dllm/prefilling_dllm_engine/log"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/longbench_all_fastdllm_engine_bridge_${RUN_TS_BASE}.launcher.log"

echo "START all LongBench engine bridge ${RUN_TS_BASE}" | tee -a "${SUMMARY_LOG}"
echo "TASKS=${TASKS[*]}" | tee -a "${SUMMARY_LOG}"

for task in "${TASKS[@]}"; do
  task_safe="${task//-/_}"
  task_safe="${task_safe//./_}"
  task_run_ts="${RUN_TS_BASE}_${task_safe}"
  echo "[$(date +%F %T)] START task=${task} run_ts=${task_run_ts}" | tee -a "${SUMMARY_LOG}"
  LONGBENCH_TASK="${task}" RUN_TS="${task_run_ts}" "${TASK_SCRIPT}"
  echo "[$(date +%F %T)] DONE task=${task}" | tee -a "${SUMMARY_LOG}"
done

echo "DONE all LongBench engine bridge ${RUN_TS_BASE}" | tee -a "${SUMMARY_LOG}"
