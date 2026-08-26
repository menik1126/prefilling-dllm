#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
DREAM_BASE="${DREAM_BASE:-/home/ma-user/work/models/Dream-v0-Base-7B}"
DREAM_LORA="${DREAM_LORA:-/home/ma-user/work/models/Dream-v0-Base-7B-LoRA}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PROJECT_DIR DREAM_BASE DREAM_LORA CHUNK_SIZE START_INDEX LIMIT RUN_TS

LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"

if [[ -f "$LOG_DIR/gsm8k_full_results.json" ]]; then
    cp "$LOG_DIR/gsm8k_full_results.json" "$LOG_DIR/gsm8k_full_results_backup_before_chunked_${RUN_TS}.json"
    echo "Backed up existing gsm8k_full_results.json to: $LOG_DIR/gsm8k_full_results_backup_before_chunked_${RUN_TS}.json"
fi

echo "============================================"
echo "  Prefilling-dLLM Engine - Dream GSM8K Full Eval Chunked"
echo "============================================"
echo "Python           : $PYTHON"
echo "Model base       : $DREAM_BASE"
echo "LoRA path        : $DREAM_LORA"
echo "CUDA devices     : $CUDA_DEVICES"
echo "Chunk size       : $CHUNK_SIZE"
echo "Start index      : $START_INDEX"
echo "Limit            : ${LIMIT:-full}"
echo "Run timestamp    : $RUN_TS"
echo "============================================"

cd "$PROJECT_DIR"

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_DIR/gsm8k_full_eval_chunked_${RUN_TS}.log"
import json
import os
import re
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from prefilling_dllm import LLM, SamplingParams


project_dir = os.environ["PROJECT_DIR"]
model = os.environ["DREAM_BASE"]
lora_path = os.environ["DREAM_LORA"]
chunk_size = int(os.environ.get("CHUNK_SIZE", "50"))
start_index = int(os.environ.get("START_INDEX", "0"))
limit_env = os.environ.get("LIMIT", "")
limit = int(limit_env) if limit_env else None
run_ts = os.environ["RUN_TS"]
log_dir = os.path.join(project_dir, "log")
results_path = os.path.join(log_dir, f"gsm8k_full_results_chunked_{run_ts}.json")

STOP_STRINGS = ["Q:", "</s>", "<|im_end|>"]

FEW_SHOTS = '''Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

'''


def normalize_answer(value):
    if value is None:
        return None
    value = value.replace(",", "").strip()
    if value.endswith(".") and value.count(".") == 1:
        value = value[:-1]
    return value


def extract_answer(text):
    """Extract numeric answer from model output."""
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop, 1)[0]
    match = re.search(r"[Tt]he answer is\s*[\$]?\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return normalize_answer(match.group(1))
    match = re.search(r"\\boxed\{([\d,]+(?:\.\d+)?)\}", text)
    if match:
        return normalize_answer(match.group(1))
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return normalize_answer(numbers[-1])
    return None


def extract_gold_answer(answer_text):
    """Extract numeric answer from GSM8K gold answer (after ####)."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if match:
        return normalize_answer(match.group(1))
    return None


def save_results(results, total, start_time):
    correct = sum(1 for r in results if r["correct"])
    done = len(results)
    bucket_size = 100
    buckets = []
    for start in range(0, done, bucket_size):
        bucket = results[start:start + bucket_size]
        if not bucket:
            continue
        bucket_correct = sum(1 for r in bucket if r["correct"])
        buckets.append({
            "start": start,
            "end": start + len(bucket),
            "correct": bucket_correct,
            "total": len(bucket),
            "accuracy": bucket_correct / len(bucket) * 100,
        })
    payload = {
        "accuracy": correct / done * 100 if done else 0.0,
        "correct": correct,
        "completed": done,
        "total": total,
        "chunk_size": chunk_size,
        "time_seconds": time.time() - start_time,
        "contains_q_count": sum(1 for r in results if "Q:" in r["prediction"]),
        "total_tokens": sum(r["token_count"] for r in results),
        "avg_tokens": sum(r["token_count"] for r in results) / done if done else 0.0,
        "buckets_100": buckets,
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


print("Loading model...", flush=True)
llm = LLM(
    model,
    lora_path=lora_path,
    use_lora=True,
    model_name="dream",
    model_type="diffusion_lm",
    enforce_eager=True,
    data_parallel_size=1,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.60,
    max_num_batched_tokens=2048,
    max_num_seqs=5,
    max_model_len=2048,
    accept_threshold=0.95,
    complete_threshold=0.9,
    add_new_block_threshold=0.1,
    kv_cache_layout="unified",
)

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256, stop=STOP_STRINGS)

print("Loading GSM8K test set...", flush=True)
dataset = load_dataset("openai/gsm8k", "main")["test"]
questions = list(dataset["question"])
answers = list(dataset["answer"])
questions = questions[start_index:]
answers = answers[start_index:]
if limit is not None:
    questions = questions[:limit]
    answers = answers[:limit]

total = len(questions)
prompts = [tokenizer.bos_token + FEW_SHOTS + "Q: " + q + "\nA:" for q in questions]
print(f"Total examples: {total}", flush=True)
print(f"Writing incremental results to: {results_path}", flush=True)

start_time = time.time()
results = []

for start in range(0, total, chunk_size):
    end = min(start + chunk_size, total)
    chunk_t0 = time.time()
    print(f"\nRunning chunk {start}:{end}...", flush=True)
    outputs = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
    for local_idx, (output, gold_ans) in enumerate(zip(outputs, answers[start:end])):
        idx = start + local_idx
        pred = extract_answer(output["text"])
        gold = extract_gold_answer(gold_ans)
        is_correct = (pred == gold) if (pred and gold) else False
        results.append({
            "idx": start_index + idx,
            "question": questions[idx],
            "prediction": output["text"],
            "pred_answer": pred,
            "gold_answer": gold,
            "correct": is_correct,
            "token_count": len(output["token_ids"]),
            "n_diff_steps": output["n_diff_steps"],
        })
    save_results(results, total, start_time)
    done = len(results)
    correct = sum(1 for r in results if r["correct"])
    chunk_correct = sum(1 for r in results[start:end] if r["correct"])
    print(
        f"Chunk {start}:{end} accuracy: {chunk_correct}/{end - start} = {chunk_correct / (end - start) * 100:.2f}% | "
        f"running: {correct}/{done} = {correct / done * 100:.2f}% | "
        f"chunk time: {time.time() - chunk_t0:.2f}s",
        flush=True,
    )

correct = sum(1 for r in results if r["correct"])
elapsed = time.time() - start_time
print()
print("=" * 60)
print("GSM8K Full Evaluation Results (Chunked)")
print("=" * 60)
print(f"Total examples : {total}")
print(f"Correct        : {correct}")
print(f"Accuracy       : {correct / total * 100:.2f}%")
print(f"Total time     : {elapsed:.2f} seconds")
print(f"Total tokens   : {sum(r['token_count'] for r in results)}")
print(f"Avg TPS        : {sum(r['token_count'] for r in results) / elapsed:.2f} tok/s")
print(f"Contains Q:    : {sum(1 for r in results if 'Q:' in r['prediction'])}")
print("=" * 60)
print(f"Detailed results saved to: {results_path}")
PY
