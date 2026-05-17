#!/bin/bash
# Run all D2F evaluations: LLaDA (GPU 0) and DREAM (GPU 1) in parallel
# Each GPU runs CMG → LBCG → LCB sequentially

set -u

D2F_EVAL=/home/hq/Discrete-Diffusion-Forcing/D2F-eval
VENV=/home/hq/Discrete-Diffusion-Forcing/.venv
RESULTS=$D2F_EVAL/results
LOG_DIR=/tmp/d2f_logs

LLADA_MODEL="GSAI-ML/LLaDA-8B-Instruct"
LLADA_LORA="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora"
DREAM_MODEL="$D2F_EVAL/model_weights/Dream-v0-Base-7B"
DREAM_LORA="$D2F_EVAL/model_weights/D2F_Dream_Base_7B_Lora"

TOP_PERCENT=30

mkdir -p $RESULTS $LOG_DIR

source $VENV/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

echo "=== D2F Evaluation Queue Started ==="
echo "Results dir: $RESULTS"
echo "Top percent: $TOP_PERCENT%"
echo ""

# --- GPU 0: LLaDA ---
(
  export CUDA_VISIBLE_DEVICES=0
  echo "[LLaDA] Starting CMG..."
  python $D2F_EVAL/eval_d2f_cmg.py \
    --model_type llada \
    --pretrained $LLADA_MODEL \
    --lora_path $LLADA_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/llada_cmg.log 2>&1 \
  && echo "[LLaDA] CMG done. Starting LBCG..." \
  && python $D2F_EVAL/eval_d2f_lbcg.py \
    --model_type llada \
    --pretrained $LLADA_MODEL \
    --lora_path $LLADA_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/llada_lbcg.log 2>&1 \
  && echo "[LLaDA] LBCG done. Starting LCB..." \
  && python $D2F_EVAL/eval_d2f_lcb.py \
    --model_type llada \
    --pretrained $LLADA_MODEL \
    --lora_path $LLADA_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/llada_lcb.log 2>&1 \
  && echo "[LLaDA] All tasks done." \
  || echo "[LLaDA] A task failed, check logs in $LOG_DIR"
) &
LLADA_PID=$!

# --- GPU 1: DREAM ---
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[DREAM] Starting CMG..."
  python $D2F_EVAL/eval_d2f_cmg.py \
    --model_type dream \
    --pretrained $DREAM_MODEL \
    --lora_path $DREAM_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/dream_cmg.log 2>&1 \
  && echo "[DREAM] CMG done. Starting LBCG..." \
  && python $D2F_EVAL/eval_d2f_lbcg.py \
    --model_type dream \
    --pretrained $DREAM_MODEL \
    --lora_path $DREAM_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/dream_lbcg.log 2>&1 \
  && echo "[DREAM] LBCG done. Starting LCB..." \
  && python $D2F_EVAL/eval_d2f_lcb.py \
    --model_type dream \
    --pretrained $DREAM_MODEL \
    --lora_path $DREAM_LORA \
    --top_percent $TOP_PERCENT \
    --output_dir $RESULTS \
    > $LOG_DIR/dream_lcb.log 2>&1 \
  && echo "[DREAM] All tasks done." \
  || echo "[DREAM] A task failed, check logs in $LOG_DIR"
) &
DREAM_PID=$!

echo "LLaDA queue PID: $LLADA_PID (GPU 0)"
echo "DREAM queue PID: $DREAM_PID (GPU 1)"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_DIR/llada_cmg.log"
echo "  tail -f $LOG_DIR/dream_cmg.log"
echo ""
echo "After generation, score LCB results with:"
echo "  source /home/hq/LiveCodeBench/.venv/bin/activate"
echo "  python $D2F_EVAL/eval_d2f_lcb_score.py --input_file $RESULTS/llada_lcb_release_v5_top30pct.json"
echo "  python $D2F_EVAL/eval_d2f_lcb_score.py --input_file $RESULTS/dream_lcb_release_v5_top30pct.json"

wait $LLADA_PID $DREAM_PID
echo ""
echo "=== All queues finished ==="
