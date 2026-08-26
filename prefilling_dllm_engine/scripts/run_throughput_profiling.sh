#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREFILLING_DLLM_ENGINE_DIR="$REPO_DIR/prefilling_dllm_engine"
PREFILLING_DLLM_EVAL_DIR="${PREFILLING_DLLM_EVAL_DIR:-$REPO_DIR/prefilling_dllm_engine_eval}"

PYTHON="${PYTHON:-/home/ma-user/work/venvs/prefilling-dllm/bin/python}"
DREAM_BASE="${DREAM_BASE:-$PREFILLING_DLLM_EVAL_DIR/model_weights/Dream-v0-Base-7B}"
FASTDLLM_DREAM="${FASTDLLM_DREAM:-/home/ma-user/work/Fast-dLLM/v1/dream}"
DATA_DIR="${LONGBENCH_DATA_DIR:-/home/ma-user/work/ParallelComp_official/datasets/LongBench}"
CONFIG_DIR="${LONGBENCH_CONFIG_DIR:-/home/ma-user/work/ParallelComp_official/longbench_config}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20}"
WARMUP="${WARMUP:-3}"
TOKEN_CAPACITY="${TOKEN_CAPACITY:-0}"
TOKEN_SCORE_QUERY_WINDOW="${TOKEN_SCORE_QUERY_WINDOW:-8}"
TOKEN_SCORE_LAYERS="${TOKEN_SCORE_LAYERS:-0}"
TOKEN_SCORE_LAYER_MODE="${TOKEN_SCORE_LAYER_MODE:-all}"
TOKEN_SCORE_REDUCE="${TOKEN_SCORE_REDUCE:-sum}"
TOKEN_SCORE_POOLING="${TOKEN_SCORE_POOLING:-maxpool}"
TOKEN_SCORE_POOL_KERNEL="${TOKEN_SCORE_POOL_KERNEL:-7}"
TOKEN_SCORE_HEAD_REDUCE="${TOKEN_SCORE_HEAD_REDUCE:-sum}"
TOKEN_SCORE_LAYER_REDUCE="${TOKEN_SCORE_LAYER_REDUCE:-mean}"
TOKEN_SCORE_DIRECTION="${TOKEN_SCORE_DIRECTION:-query_to_chunk}"
TOKEN_SCORE_KEEP="${TOKEN_SCORE_KEEP:-high}"
TOKEN_SCORE_INCLUDE_PREFIX="${TOKEN_SCORE_INCLUDE_PREFIX:-1}"
TOKEN_SCORE_USE_GENERATED="${TOKEN_SCORE_USE_GENERATED:-0}"
TOKEN_ATTENTION_MASK="${TOKEN_ATTENTION_MASK:-causal}"
TOKEN_EVICTION_GRANULARITY="${TOKEN_EVICTION_GRANULARITY:-global}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="${HF_HOME:-/home/ma-user/work/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_DIR:$PREFILLING_DLLM_EVAL_DIR:${PYTHONPATH:-}"

LOG_DIR="$PREFILLING_DLLM_ENGINE_DIR/log"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/throughput_profiling_${RUN_TS}.log"
RESULT_FILE="$LOG_DIR/throughput_profiling_${RUN_TS}.json"

echo "============================================"
echo "  Throughput Profiling: Engine Bridge vs Transformers Baseline"
echo "============================================"
echo "Python              : $PYTHON"
echo "Model               : $DREAM_BASE"
echo "Num examples        : $NUM_EXAMPLES"
echo "Warmup examples     : $WARMUP"
echo "Run timestamp       : $RUN_TS"
echo "Log file            : $LOG_FILE"
echo "Result file         : $RESULT_FILE"
echo "============================================"

cd "$PREFILLING_DLLM_ENGINE_DIR"

export REPO_DIR PREFILLING_DLLM_ENGINE_DIR PREFILLING_DLLM_EVAL_DIR DREAM_BASE FASTDLLM_DREAM DATA_DIR CONFIG_DIR
export RUN_TS NUM_EXAMPLES WARMUP RESULT_FILE
export TOKEN_CAPACITY TOKEN_SCORE_QUERY_WINDOW TOKEN_SCORE_LAYERS TOKEN_SCORE_LAYER_MODE TOKEN_SCORE_REDUCE
export TOKEN_SCORE_POOLING TOKEN_SCORE_POOL_KERNEL TOKEN_SCORE_HEAD_REDUCE TOKEN_SCORE_LAYER_REDUCE
export TOKEN_SCORE_DIRECTION TOKEN_SCORE_KEEP TOKEN_SCORE_INCLUDE_PREFIX TOKEN_SCORE_USE_GENERATED
export TOKEN_ATTENTION_MASK TOKEN_EVICTION_GRANULARITY

"$PYTHON" <<'PY' 2>&1 | tee "$LOG_FILE"
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

repo_dir = os.environ["REPO_DIR"]
prefilling_dllm_eval_dir = os.environ["PREFILLING_DLLM_EVAL_DIR"]
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if prefilling_dllm_eval_dir not in sys.path:
    sys.path.insert(0, prefilling_dllm_eval_dir)

from prefilling_dllm import FastDLLMDreamEngine
from eval_fastdllm_parallelcomp_longbench import (
    build_token_parts,
    load_json,
    load_task_examples,
    render_prompt_parts,
    trim_stop_tokens,
)
from fastdllm_parallelcomp import load_fastdllm_parallelcomp

TASK = "multifieldqa_en"
STOP_STRINGS = ["</s>", "<|im_end|>", "\n"]
MASK_TOKEN_ID = 151666

dream_base = os.environ["DREAM_BASE"]
fastdllm_dream = os.environ["FASTDLLM_DREAM"]
data_dir = os.environ["DATA_DIR"]
config_dir = os.environ["CONFIG_DIR"]
run_ts = os.environ["RUN_TS"]
result_file = os.environ["RESULT_FILE"]
num_examples = int(os.environ.get("NUM_EXAMPLES", "20"))
warmup = int(os.environ.get("WARMUP", "3"))
token_capacity = int(os.environ.get("TOKEN_CAPACITY", "0"))
token_score_query_window = int(os.environ.get("TOKEN_SCORE_QUERY_WINDOW", "8"))
token_score_layers = int(os.environ.get("TOKEN_SCORE_LAYERS", "0"))
token_score_layer_mode = os.environ.get("TOKEN_SCORE_LAYER_MODE", "all")
token_score_reduce = os.environ.get("TOKEN_SCORE_REDUCE", "sum")
token_score_pooling = os.environ.get("TOKEN_SCORE_POOLING", "maxpool")
token_score_pool_kernel = int(os.environ.get("TOKEN_SCORE_POOL_KERNEL", "7"))
token_score_head_reduce = os.environ.get("TOKEN_SCORE_HEAD_REDUCE", "sum")
token_score_layer_reduce = os.environ.get("TOKEN_SCORE_LAYER_REDUCE", "mean")
token_score_direction = os.environ.get("TOKEN_SCORE_DIRECTION", "query_to_chunk")
token_score_keep = os.environ.get("TOKEN_SCORE_KEEP", "high")
token_score_include_prefix = os.environ.get("TOKEN_SCORE_INCLUDE_PREFIX", "1").lower() in {"1", "true", "yes", "on"}
token_score_use_generated = os.environ.get("TOKEN_SCORE_USE_GENERATED", "0").lower() in {"1", "true", "yes", "on"}
token_attention_mask = os.environ.get("TOKEN_ATTENTION_MASK", "causal")
token_eviction_granularity = os.environ.get("TOKEN_EVICTION_GRANULARITY", "global")
if token_eviction_granularity != "global":
    raise ValueError(
        "Throughput profiling can only represent global token eviction in prompt_ids. "
        "Per-head eviction needs engine KV-cache level support."
    )

prompt_templates = load_json(Path(config_dir) / "dataset2prompt_raw.json")

# ============================================================
# Load examples
# ============================================================
print("Loading LongBench examples...", flush=True)
examples = load_task_examples(TASK, data_dir, max_examples=0)
examples = examples[:num_examples + warmup]
print(f"Loaded {len(examples)} examples ({warmup} warmup + {num_examples} measured)", flush=True)

# ============================================================
# Phase 1: Engine Bridge (Ours) - Selection + Decode
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Phase 1: Engine Bridge (Ours)", flush=True)
print("=" * 60, flush=True)

print("Loading Fast-DLLM ParallelComp selector...", flush=True)
selector = load_fastdllm_parallelcomp(
    pretrained=dream_base,
    fastdllm_dream_dir=fastdllm_dream,
    max_new_tokens=32,
    max_length=4096,
    block_length=32,
    temperature=0.0,
    alg="confidence_threshold",
    threshold=0.9,
    rope_scale_factor=1.0,
    dtype="bfloat16",
    add_bos_token=True,
    chunk_size=1024,
    topk_chunks=4,
    chunk_bos=True,
    force_keep_chunk_bos=True,
    cache_build_mode="full_prompt_mask",
    score_mode="draft_self_information",
    score_draft_tokens=4,
    score_draft_partial_rounds=1,
    score_attention_mask="causal",
    score_context_mode="single_chunk",
    token_capacity=token_capacity,
    token_score_query_window=token_score_query_window,
    token_score_layers=token_score_layers,
    token_score_layer_mode=token_score_layer_mode,
    token_score_reduce=token_score_reduce,
    token_score_pooling=token_score_pooling,
    token_score_pool_kernel=token_score_pool_kernel,
    token_score_head_reduce=token_score_head_reduce,
    token_score_layer_reduce=token_score_layer_reduce,
    token_score_direction=token_score_direction,
    token_score_keep=token_score_keep,
    token_score_include_prefix=token_score_include_prefix,
    token_score_use_generated=token_score_use_generated,
    token_attention_mask=token_attention_mask,
    token_eviction_granularity=token_eviction_granularity,
    chunk_position_mode="continuous",
    query_position_mode="after_selected_chunks",
)

# Prepare compressed prompts
template = prompt_templates[TASK]
compressed_records = []
for idx, example in enumerate(examples):
    parts = render_prompt_parts(template, example, "\n")
    token_parts = build_token_parts(
        selector, parts, add_bos_token=True,
    )
    prefix_ids, context_ids, query_ids, scoring_query_ids = token_parts[:4]
    candidate_chunks, selected_indices, chunk_scores, selection_query_ids, score_token_mask = (
        selector._prepare_candidate_chunks(
            prefix_ids=prefix_ids,
            context_ids=context_ids,
            scoring_query_ids=scoring_query_ids,
        )
    )
    eviction_query_ids = selector._token_eviction_query_ids(
        scoring_query_ids=scoring_query_ids,
        selection_query_ids=selection_query_ids,
        score_token_mask=score_token_mask,
    )
    prompt_ids = list(prefix_ids)
    prompt_positions = selector._range_positions(0, len(prefix_ids))
    cache_positions = list(prompt_positions)
    kept_context_tokens = 0
    removed_context_tokens = 0
    for chunk_order, chunk_idx in enumerate(selected_indices):
        original_chunk_ids = list(candidate_chunks[chunk_idx])
        chunk_start = selector._chunk_rope_start(len(prefix_ids), chunk_order, chunk_idx)
        original_chunk_positions = selector._range_positions(chunk_start, len(original_chunk_ids))
        keep_positions = selector._keep_positions_for_chunk(prefix_ids, original_chunk_ids, eviction_query_ids)
        chunk_ids = [original_chunk_ids[pos] for pos in keep_positions]
        chunk_positions = [original_chunk_positions[pos] for pos in keep_positions]
        kept_context_tokens += len(chunk_ids)
        removed_context_tokens += max(0, len(original_chunk_ids) - len(chunk_ids))
        prompt_ids.extend(chunk_ids)
        prompt_positions.extend(chunk_positions)
        cache_positions.extend(chunk_positions)
    query_rope_start = selector._final_query_rope_start(
        len(prefix_ids), cache_positions, selected_count=len(selected_indices),
    )
    query_positions = selector._range_positions(query_rope_start, len(query_ids))
    prompt_ids.extend(query_ids)
    prompt_positions.extend(query_positions)
    compressed_records.append({
        "prompt_ids": prompt_ids,
        "prompt_positions": prompt_positions,
        "context_tokens": len(context_ids),
        "kept_context_tokens": kept_context_tokens,
        "removed_context_tokens": removed_context_tokens,
        "compressed_tokens": len(prompt_ids),
    })

del selector
torch.cuda.empty_cache()
print(f"Selection done. Compressed {len(compressed_records)} prompts.", flush=True)

# PLACEHOLDER_ENGINE_BRIDGE_DECODE

# Load engine and decode
print("Loading FastDLLMDreamEngine...", flush=True)
engine = FastDLLMDreamEngine(
    dream_base,
    max_model_len=8192,
    block_length=32,
    gpu_memory_utilization=0.60,
    threshold=0.9,
    temperature=0.0,
    max_num_batched_tokens=8192,
    max_num_seqs=1,
    kv_cache_layout="unified",
)

bridge_per_sample = []
print(f"Running decode: {warmup} warmup + {num_examples} measured...", flush=True)
for idx, compressed in enumerate(compressed_records):
    torch.cuda.synchronize()
    t0 = time.time()
    output = engine.generate_token_ids(
        compressed["prompt_ids"],
        max_new_tokens=32,
        prompt_positions=compressed["prompt_positions"],
        stop_token_ids=[engine.tokenizer.eos_token_id] if engine.tokenizer.eos_token_id is not None else None,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    prediction = trim_stop_tokens(output.text, STOP_STRINGS)
    n_tokens = len(output.token_ids)
    if idx >= warmup:
        bridge_per_sample.append({
            "idx": idx - warmup,
            "latency_s": elapsed,
            "tokens_generated": n_tokens,
            "n_diff_steps": output.n_diff_steps,
            "prompt_tokens": compressed["compressed_tokens"],
            "context_tokens": compressed["context_tokens"],
            "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0,
        })
    label = "warmup" if idx < warmup else "measured"
    print(f"  [{label}] {idx+1}/{len(compressed_records)} tokens={n_tokens} "
          f"steps={output.n_diff_steps} latency={elapsed:.3f}s "
          f"tps={n_tokens/elapsed:.1f}", flush=True)

engine.close()
del engine
torch.cuda.empty_cache()

bridge_total_tokens = sum(s["tokens_generated"] for s in bridge_per_sample)
bridge_total_time = sum(s["latency_s"] for s in bridge_per_sample)
bridge_avg_latency = bridge_total_time / len(bridge_per_sample)
bridge_tps = bridge_total_tokens / bridge_total_time

print(f"\n--- Engine Bridge Results ({num_examples} samples) ---", flush=True)
print(f"  Total tokens:     {bridge_total_tokens}", flush=True)
print(f"  Total time:       {bridge_total_time:.2f}s", flush=True)
print(f"  Avg latency:      {bridge_avg_latency:.3f}s/sample", flush=True)
print(f"  Throughput:       {bridge_tps:.2f} tokens/s", flush=True)

# PLACEHOLDER_TRANSFORMERS_BASELINE

# ============================================================
# Phase 2: Transformers Baseline (Naive Diffusion)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("Phase 2: Transformers Baseline (Naive Diffusion)", flush=True)
print("=" * 60, flush=True)

from transformers import AutoTokenizer
from peft import PeftModel

DIFFUSION_STEPS = 64
BLOCK_SIZE = 32
MAX_NEW_TOKENS_BASELINE = 32
PROMPT_PREFIX = "Read the following text and answer briefly.\n\n"
PROMPT_SUFFIX_TEMPLATE = (
    "\n\nNow, answer the following question based on the above text, "
    "only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)

sys.path.insert(0, "/home/ma-user/work/prefilling-dllm/prefilling_dllm_eval")
from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig

print("Loading model with transformers...", flush=True)
model_config = DreamConfig.from_pretrained(dream_base)
model = DreamModel.from_pretrained(
    dream_base,
    config=model_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=False,
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(dream_base, trust_remote_code=True)
print("Model loaded.", flush=True)


@torch.no_grad()
def dream_generate_naive(model, tokenizer, prompt_text, max_new_tokens, diffusion_steps, device):
    prompt_ids = tokenizer.encode(prompt_text)
    max_prompt = 8192 - max_new_tokens
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]
    prompt_len = len(prompt_ids)

    x = torch.tensor([prompt_ids + [MASK_TOKEN_ID] * max_new_tokens], device=device, dtype=torch.long)
    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    total_steps = 0
    for step in range(diffusion_steps):
        is_mask = (x[0] == MASK_TOKEN_ID) & (~prompt_mask)
        n_masked = is_mask.sum().item()
        if n_masked == 0:
            break
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x).logits
            shifted_logits = torch.zeros_like(logits)
            shifted_logits[:, 1:, :] = logits[:, :-1, :]
            shifted_logits[:, 0, :] = logits[:, 0, :]
        total_steps += 1
        probs = F.softmax(shifted_logits[0], dim=-1)
        sampled = torch.argmax(probs, dim=-1)
        n_to_unmask = max(1, int(n_masked * (1.0 / (diffusion_steps - step))))
        mask_indices = is_mask.nonzero(as_tuple=True)[0]
        confidences = probs[mask_indices, sampled[mask_indices]]
        _, top_indices = confidences.topk(min(n_to_unmask, len(mask_indices)))
        unmask_positions = mask_indices[top_indices]
        x[0, unmask_positions] = sampled[unmask_positions]

    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop, 1)[0]
    return text, len(gen_ids), total_steps


# PLACEHOLDER_BASELINE_DECODE

device = next(model.parameters()).device
baseline_per_sample = []
print(f"Running baseline decode: {warmup} warmup + {num_examples} measured...", flush=True)

for idx, example in enumerate(examples):
    suffix = PROMPT_SUFFIX_TEMPLATE.format(input=example["input"])
    full_prompt = tokenizer.bos_token + PROMPT_PREFIX + example["context"] + suffix
    prompt_ids = tokenizer.encode(full_prompt)
    max_prompt = 8192 - MAX_NEW_TOKENS_BASELINE
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

    torch.cuda.synchronize()
    t0 = time.time()
    pred_text, n_tokens, n_steps = dream_generate_naive(
        model, tokenizer, prompt_text, MAX_NEW_TOKENS_BASELINE, DIFFUSION_STEPS, device
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    if idx >= warmup:
        baseline_per_sample.append({
            "idx": idx - warmup,
            "latency_s": elapsed,
            "tokens_generated": n_tokens,
            "n_diff_steps": n_steps,
            "prompt_tokens": len(prompt_ids),
            "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0,
        })
    label = "warmup" if idx < warmup else "measured"
    print(f"  [{label}] {idx+1}/{len(examples)} tokens={n_tokens} "
          f"steps={n_steps} latency={elapsed:.3f}s "
          f"tps={n_tokens/elapsed:.1f}", flush=True)

del model
torch.cuda.empty_cache()

baseline_total_tokens = sum(s["tokens_generated"] for s in baseline_per_sample)
baseline_total_time = sum(s["latency_s"] for s in baseline_per_sample)
baseline_avg_latency = baseline_total_time / len(baseline_per_sample)
baseline_tps = baseline_total_tokens / baseline_total_time

print(f"\n--- Transformers Baseline Results ({num_examples} samples) ---", flush=True)
print(f"  Total tokens:     {baseline_total_tokens}", flush=True)
print(f"  Total time:       {baseline_total_time:.2f}s", flush=True)
print(f"  Avg latency:      {baseline_avg_latency:.3f}s/sample", flush=True)
print(f"  Throughput:       {baseline_tps:.2f} tokens/s", flush=True)

# ============================================================
# Summary
# ============================================================
speedup = baseline_avg_latency / bridge_avg_latency if bridge_avg_latency > 0 else 0
tps_ratio = bridge_tps / baseline_tps if baseline_tps > 0 else 0

print("\n" + "=" * 60, flush=True)
print("  THROUGHPUT COMPARISON SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Metric':<30} {'Engine Bridge':>15} {'Transformers':>15} {'Speedup':>10}", flush=True)
print("-" * 70, flush=True)
print(f"{'Avg latency (s/sample)':<30} {bridge_avg_latency:>15.3f} {baseline_avg_latency:>15.3f} {speedup:>9.1f}x", flush=True)
print(f"{'Throughput (tokens/s)':<30} {bridge_tps:>15.2f} {baseline_tps:>15.2f} {tps_ratio:>9.1f}x", flush=True)
print(f"{'Total tokens':<30} {bridge_total_tokens:>15} {baseline_total_tokens:>15}", flush=True)
print(f"{'Samples':<30} {num_examples:>15} {num_examples:>15}", flush=True)
print("=" * 60, flush=True)

# Save results
result = {
    "run_ts": run_ts,
    "task": TASK,
    "num_examples": num_examples,
    "warmup": warmup,
    "engine_bridge": {
        "avg_latency_s": bridge_avg_latency,
        "throughput_tps": bridge_tps,
        "total_tokens": bridge_total_tokens,
        "total_time_s": bridge_total_time,
        "token_capacity": token_capacity,
        "token_eviction_granularity": token_eviction_granularity,
        "token_score_direction": token_score_direction,
        "token_score_keep": token_score_keep,
        "per_sample": bridge_per_sample,
    },
    "transformers_baseline": {
        "avg_latency_s": baseline_avg_latency,
        "throughput_tps": baseline_tps,
        "total_tokens": baseline_total_tokens,
        "total_time_s": baseline_total_time,
        "diffusion_steps": DIFFUSION_STEPS,
        "per_sample": baseline_per_sample,
    },
    "speedup_latency": speedup,
    "speedup_tps": tps_ratio,
}

with open(result_file, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to: {result_file}", flush=True)
PY
