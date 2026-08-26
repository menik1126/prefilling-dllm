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
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PROJECT_DIR DREAM_BASE DREAM_LORA CHUNK_SIZE START_INDEX LIMIT RUN_TS DATA_DIR MAX_MODEL_LEN

LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"

RESULTS_TAG="longbench_multifieldqa_en"

if [[ -f "$LOG_DIR/${RESULTS_TAG}_results.json" ]]; then
    cp "$LOG_DIR/${RESULTS_TAG}_results.json" \
       "$LOG_DIR/${RESULTS_TAG}_results_backup_${RUN_TS}.json"
    echo "Backed up existing results to: $LOG_DIR/${RESULTS_TAG}_results_backup_${RUN_TS}.json"
fi

echo "============================================"
echo "  Prefilling-dLLM Engine - LongBench multifieldqa_en Eval"
echo "============================================"
echo "Python           : $PYTHON"
echo "Model base       : $DREAM_BASE"
echo "LoRA path        : $DREAM_LORA"
echo "CUDA devices     : $CUDA_DEVICES"
echo "Chunk size       : $CHUNK_SIZE"
echo "Start index      : $START_INDEX"
echo "Limit            : ${LIMIT:-full}"
echo "Max model len    : $MAX_MODEL_LEN"
echo "Data dir         : $DATA_DIR"
echo "Run timestamp    : $RUN_TS"
echo "============================================"

cd "$PROJECT_DIR"

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_DIR/${RESULTS_TAG}_eval_chunked_${RUN_TS}.log"
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

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
data_dir = os.environ["DATA_DIR"]
max_model_len = int(os.environ["MAX_MODEL_LEN"])
log_dir = os.path.join(project_dir, "log")
results_path = os.path.join(log_dir, f"longbench_multifieldqa_en_results_chunked_{run_ts}.json")

TASK = "multifieldqa_en"
MAX_NEW_TOKENS = 64
STOP_STRINGS = ["</s>", "<|im_end|>"]
PROMPT_TEMPLATE = (
    "Read the following text and answer briefly.\n\n"
    "{context}\n\n"
    "Now, answer the following question based on the above text, "
    "only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)


# --------------- metrics ---------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_prediction(prediction, answers):
    return max(qa_f1_score(prediction, gt) for gt in answers)


# --------------- data loading ---------------

def load_examples(data_dir, task):
    path = Path(data_dir) / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# --------------- results saving ---------------

def save_results(results, total, start_time):
    done = len(results)
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / done if done else 0.0

    bucket_size = 50
    buckets = []
    for start in range(0, done, bucket_size):
        bucket = results[start:start + bucket_size]
        if not bucket:
            continue
        bucket_scores = [r["score"] for r in bucket]
        buckets.append({
            "start": start,
            "end": start + len(bucket),
            "avg_score": sum(bucket_scores) / len(bucket_scores) * 100,
            "n": len(bucket),
        })

    payload = {
        "task": TASK,
        "avg_f1": avg_score * 100,
        "completed": done,
        "total": total,
        "chunk_size": chunk_size,
        "time_seconds": time.time() - start_time,
        "total_tokens": sum(r["token_count"] for r in results),
        "avg_tokens": sum(r["token_count"] for r in results) / done if done else 0.0,
        "buckets_50": buckets,
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------- main ---------------

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
    max_num_batched_tokens=max_model_len,
    max_num_seqs=2,
    max_model_len=max_model_len,
    accept_threshold=0.95,
    complete_threshold=0.9,
    add_new_block_threshold=0.1,
    kv_cache_layout="unified",
)

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, stop=STOP_STRINGS)

MAX_PROMPT_TOKENS = max_model_len - MAX_NEW_TOKENS - 32  # leave margin

PROMPT_PREFIX = "Read the following text and answer briefly.\n\n"
PROMPT_SUFFIX_TEMPLATE = (
    "\n\nNow, answer the following question based on the above text, "
    "only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)


def build_prompt(ex, tokenizer, max_prompt_tokens):
    """Build prompt with context truncation if needed."""
    suffix = PROMPT_SUFFIX_TEMPLATE.format(input=ex["input"])
    full_prompt = tokenizer.bos_token + PROMPT_PREFIX + ex["context"] + suffix
    token_ids = tokenizer.encode(full_prompt)
    if len(token_ids) <= max_prompt_tokens:
        return full_prompt, len(token_ids), False

    prefix_ids = tokenizer.encode(tokenizer.bos_token + PROMPT_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    available = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if available < 64:
        available = 64
    context_ids = tokenizer.encode(ex["context"], add_special_tokens=False)
    truncated_context_ids = context_ids[:available]
    all_ids = prefix_ids + truncated_context_ids + suffix_ids
    truncated_prompt = tokenizer.decode(all_ids, skip_special_tokens=False)
    return truncated_prompt, len(all_ids), True


print(f"Loading LongBench {TASK} dataset...", flush=True)
examples = load_examples(data_dir, TASK)
examples = examples[start_index:]
if limit is not None:
    examples = examples[:limit]

total = len(examples)
prompts = []
prompt_lengths = []
n_truncated = 0
for ex in examples:
    p, p_len, was_truncated = build_prompt(ex, tokenizer, MAX_PROMPT_TOKENS)
    prompts.append(p)
    prompt_lengths.append(p_len)
    if was_truncated:
        n_truncated += 1

print(f"Total examples: {total}", flush=True)
print(f"Prompt token lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
      f"avg={sum(prompt_lengths)/len(prompt_lengths):.0f}", flush=True)
print(f"Truncated prompts: {n_truncated}/{total} (max_prompt_tokens={MAX_PROMPT_TOKENS})", flush=True)
print(f"Writing incremental results to: {results_path}", flush=True)

start_time = time.time()
results = []

for chunk_start in range(0, total, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total)
    chunk_t0 = time.time()
    print(f"\nRunning chunk {chunk_start}:{chunk_end}...", flush=True)
    outputs = llm.generate(prompts[chunk_start:chunk_end], sampling_params, use_tqdm=True)
    for local_idx, (output, ex) in enumerate(zip(outputs, examples[chunk_start:chunk_end])):
        idx = chunk_start + local_idx
        pred_text = output["text"]
        answers = ex["answers"]
        f1 = score_prediction(pred_text, answers)
        results.append({
            "idx": start_index + idx,
            "example_id": ex.get("_id", idx),
            "question": ex["input"],
            "prediction": pred_text,
            "answers": answers,
            "score": f1,
            "length": ex.get("length", 0),
            "token_count": len(output["token_ids"]),
            "n_diff_steps": output["n_diff_steps"],
        })
    save_results(results, total, start_time)
    done = len(results)
    chunk_scores = [r["score"] for r in results[chunk_start:chunk_end]]
    all_scores = [r["score"] for r in results]
    chunk_avg = sum(chunk_scores) / len(chunk_scores) * 100
    running_avg = sum(all_scores) / len(all_scores) * 100
    print(
        f"Chunk {chunk_start}:{chunk_end} F1: {chunk_avg:.2f}% | "
        f"running: {running_avg:.2f}% ({done}/{total}) | "
        f"chunk time: {time.time() - chunk_t0:.2f}s",
        flush=True,
    )

all_scores = [r["score"] for r in results]
elapsed = time.time() - start_time
print()
print("=" * 60)
print(f"LongBench {TASK} Evaluation Results (Chunked)")
print("=" * 60)
print(f"Total examples : {total}")
print(f"Avg F1 Score   : {sum(all_scores) / len(all_scores) * 100:.2f}%")
print(f"Total time     : {elapsed:.2f} seconds")
print(f"Total tokens   : {sum(r['token_count'] for r in results)}")
print(f"Avg TPS        : {sum(r['token_count'] for r in results) / elapsed:.2f} tok/s")
print("=" * 60)
print(f"Detailed results saved to: {results_path}")
PY
