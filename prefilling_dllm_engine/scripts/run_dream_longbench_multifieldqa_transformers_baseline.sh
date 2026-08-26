#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
DREAM_BASE="${DREAM_BASE:-/home/ma-user/work/models/Dream-v0-Base-7B}"
DREAM_LORA="${DREAM_LORA:-/home/ma-user/work/models/Dream-v0-Base-7B-LoRA}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PROJECT_DIR DREAM_BASE DREAM_LORA START_INDEX LIMIT RUN_TS DATA_DIR MAX_MODEL_LEN

LOG_DIR="$PROJECT_DIR/log"
mkdir -p "$LOG_DIR"

RESULTS_TAG="longbench_multifieldqa_en_transformers_baseline"

echo "============================================"
echo "  Transformers Baseline - LongBench multifieldqa_en"
echo "============================================"
echo "Python           : $PYTHON"
echo "Model base       : $DREAM_BASE"
echo "LoRA path        : $DREAM_LORA"
echo "CUDA devices     : $CUDA_DEVICES"
echo "Start index      : $START_INDEX"
echo "Limit            : ${LIMIT:-full}"
echo "Max model len    : $MAX_MODEL_LEN"
echo "Data dir         : $DATA_DIR"
echo "Run timestamp    : $RUN_TS"
echo "============================================"

cd "$PROJECT_DIR"

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_DIR/${RESULTS_TAG}_${RUN_TS}.log"
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import PeftModel

# Add prefilling_dllm_eval to path for model_cache
import sys
sys.path.insert(0, "/home/ma-user/work/prefilling-dllm/prefilling_dllm_eval")

project_dir = os.environ["PROJECT_DIR"]
model_path = os.environ["DREAM_BASE"]
lora_path = os.environ["DREAM_LORA"]
start_index = int(os.environ.get("START_INDEX", "0"))
limit_env = os.environ.get("LIMIT", "")
limit = int(limit_env) if limit_env else None
run_ts = os.environ["RUN_TS"]
data_dir = os.environ["DATA_DIR"]
max_model_len = int(os.environ["MAX_MODEL_LEN"])
log_dir = os.path.join(project_dir, "log")
results_path = os.path.join(log_dir, f"longbench_multifieldqa_en_transformers_baseline_{run_ts}.json")

TASK = "multifieldqa_en"
MAX_NEW_TOKENS = 64
STOP_STRINGS = ["</s>", "<|im_end|>"]
DIFFUSION_STEPS = 64
BLOCK_SIZE = 32
MASK_TOKEN_ID = 151666

PROMPT_PREFIX = "Read the following text and answer briefly.\n\n"
PROMPT_SUFFIX_TEMPLATE = (
    "\n\nNow, answer the following question based on the above text, "
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


# --------------- Dream diffusion generation (naive transformers) ---------------

@torch.no_grad()
def dream_generate_single(model, tokenizer, prompt_text, max_new_tokens, diffusion_steps, block_size, device):
    """Naive Dream diffusion generation: one sample at a time, no KV cache optimization."""
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_ids)

    # Truncate if too long
    max_prompt = max_model_len - max_new_tokens
    if prompt_len > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]
        prompt_len = len(prompt_ids)

    # Initialize: prompt + masked generation tokens
    x = torch.tensor([prompt_ids + [MASK_TOKEN_ID] * max_new_tokens], device=device, dtype=torch.long)
    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    total_steps = 0
    for step in range(diffusion_steps):
        # Find positions that are still masked (in generation region only)
        is_mask = (x[0] == MASK_TOKEN_ID) & (~prompt_mask)
        n_masked = is_mask.sum().item()
        if n_masked == 0:
            break

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x).logits
            # Shift logits (Dream convention)
            shifted_logits = torch.zeros_like(logits)
            shifted_logits[:, 1:, :] = logits[:, :-1, :]
            shifted_logits[:, 0, :] = logits[:, 0, :]

        total_steps += 1

        # Sample or argmax at masked positions
        probs = F.softmax(shifted_logits[0], dim=-1)
        sampled = torch.argmax(probs, dim=-1)  # greedy

        # Unmask a fraction of positions each step
        n_to_unmask = max(1, int(n_masked * (1.0 / (diffusion_steps - step))))
        # Get confidence at masked positions
        mask_indices = is_mask.nonzero(as_tuple=True)[0]
        confidences = probs[mask_indices, sampled[mask_indices]]
        # Unmask the most confident ones
        _, top_indices = confidences.topk(min(n_to_unmask, len(mask_indices)))
        unmask_positions = mask_indices[top_indices]
        x[0, unmask_positions] = sampled[unmask_positions]

    # Extract generated tokens
    gen_ids = x[0, prompt_len:].tolist()
    # Truncate at EOS
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    # Remove remaining mask tokens
    gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Apply stop strings
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop, 1)[0]

    return text, len(gen_ids), total_steps


# --------------- prompt building ---------------

def build_prompt(ex, tokenizer, max_prompt_tokens):
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


# --------------- main ---------------

print("Loading model with transformers (naive baseline)...", flush=True)
t0 = time.time()

from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig

model_config = DreamConfig.from_pretrained(model_path)
model = DreamModel.from_pretrained(
    model_path,
    config=model_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
).eval()

# Apply LoRA
model = PeftModel.from_pretrained(model, lora_path)
model = model.to(torch.bfloat16).cuda()
print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_PROMPT_TOKENS = max_model_len - MAX_NEW_TOKENS - 32

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
print(f"Truncated prompts: {n_truncated}/{total}", flush=True)
print(f"Diffusion steps: {DIFFUSION_STEPS}", flush=True)
print(f"Writing results to: {results_path}", flush=True)

device = next(model.parameters()).device
start_time = time.time()
results = []

for idx, (prompt, ex) in enumerate(zip(prompts, examples)):
    t_start = time.time()
    pred_text, token_count, n_steps = dream_generate_single(
        model, tokenizer, prompt, MAX_NEW_TOKENS, DIFFUSION_STEPS, BLOCK_SIZE, device
    )
    t_elapsed = time.time() - t_start

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
        "token_count": token_count,
        "n_diff_steps": n_steps,
        "time_seconds": t_elapsed,
    })

    # Print progress
    running_scores = [r["score"] for r in results]
    running_avg = sum(running_scores) / len(running_scores) * 100
    print(
        f"[{idx+1}/{total}] F1={f1*100:.1f}% | running={running_avg:.2f}% | "
        f"tokens={token_count} | steps={n_steps} | time={t_elapsed:.2f}s",
        flush=True,
    )

    # Save incrementally every 10 examples
    if (idx + 1) % 10 == 0 or idx == total - 1:
        done = len(results)
        all_scores = [r["score"] for r in results]
        payload = {
            "task": TASK,
            "method": "transformers_naive_diffusion",
            "diffusion_steps": DIFFUSION_STEPS,
            "avg_f1": sum(all_scores) / done * 100,
            "completed": done,
            "total": total,
            "time_seconds": time.time() - start_time,
            "total_tokens": sum(r["token_count"] for r in results),
            "avg_tokens": sum(r["token_count"] for r in results) / done,
            "avg_time_per_example": sum(r["time_seconds"] for r in results) / done,
            "results": results,
        }
        with open(results_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

elapsed = time.time() - start_time
all_scores = [r["score"] for r in results]
total_tokens = sum(r["token_count"] for r in results)
print()
print("=" * 60)
print(f"LongBench {TASK} - Transformers Baseline Results")
print("=" * 60)
print(f"Total examples     : {total}")
print(f"Avg F1 Score       : {sum(all_scores) / len(all_scores) * 100:.2f}%")
print(f"Total time         : {elapsed:.2f} seconds")
print(f"Avg time/example   : {elapsed / total:.2f} seconds")
print(f"Total tokens       : {total_tokens}")
print(f"Avg TPS            : {total_tokens / elapsed:.2f} tok/s")
print(f"Diffusion steps    : {DIFFUSION_STEPS}")
print("=" * 60)
print(f"Results saved to: {results_path}")
PY
