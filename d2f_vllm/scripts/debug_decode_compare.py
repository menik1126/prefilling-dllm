"""Compare vLLM engine vs FastDLLMDreamEngine decode step by step for one example."""
import gc
import json
import os
import sys
from pathlib import Path

import torch

repo_dir = str(Path(__file__).resolve().parent.parent.parent)
d2f_eval_dir = os.environ.get("D2F_EVAL_DIR", str(Path(repo_dir) / "D2F-eval"))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)
if d2f_eval_dir not in sys.path:
    sys.path.insert(0, d2f_eval_dir)

# Load compressed record
compressed_path = "/home/ma-user/work/Discrete-Diffusion-Forcing/d2f_vllm/log/longbench_multifieldqa_en_fastdllm_vllm_bridge_compressed_20260521_002449.json"
with open(compressed_path) as f:
    compressed = json.load(f)

# Pick example 3 (score differs: vLLM=0.452 vs engine=0.889)
record = compressed["records"][3]
prompt_ids = record["prompt_ids"]
prompt_positions = record["prompt_positions"]
print(f"Example 3: prompt_len={len(prompt_ids)}, max_pos={max(prompt_positions)}")

dream_base = os.environ.get("DREAM_BASE", str(Path(d2f_eval_dir) / "model_weights/Dream-v0-Base-7B"))
MASK_TOKEN_ID = 151666
BLOCK_LENGTH = 32
MAX_NEW_TOKENS = 32
THRESHOLD = 0.9

# --- Run FastDLLMDreamEngine ---
print("\n=== FastDLLMDreamEngine ===")
from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine

engine = FastDLLMDreamEngine(
    dream_base,
    max_model_len=8192,
    block_length=BLOCK_LENGTH,
    gpu_memory_utilization=0.30,
    max_num_seqs=1,
    threshold=THRESHOLD,
    temperature=0.0,
)

# Manual decode with logging (mirrors generate_token_ids logic)
prompt_ids_list = list(int(x) for x in prompt_ids)
pos_list = [int(x) for x in prompt_positions]
decode_len = BLOCK_LENGTH
suffix_pos_start = max(pos_list) + 1
suffix_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
full_ids = prompt_ids_list + [MASK_TOKEN_ID] * decode_len
full_positions = pos_list + suffix_positions

with torch.inference_mode():
    prefill_logits = engine._forward_prefill(full_ids, full_positions)
    prompt_len = len(prompt_ids_list)
    shifted_prefill = engine._shift_logits(prefill_logits)
    first_logits = shifted_prefill[prompt_len:prompt_len + 1, :]
    _, first_token = engine._sample_tokens(first_logits)
    last_context_logit = prefill_logits[prompt_len - 1, :].detach()

    block_ids = torch.full((decode_len,), MASK_TOKEN_ID, dtype=torch.long, device=torch.cuda.current_device())
    block_ids[0] = first_token[0]
    print(f"  After prefill: token[0]={first_token[0].item()}")

    for step in range(50):
        if not (block_ids == MASK_TOKEN_ID).any():
            break
        mask_index = block_ids == MASK_TOKEN_ID
        logits = engine._forward_replace_block(
            block_ids, prompt_len=prompt_len, slot_start=prompt_len, block_positions=suffix_positions,
        )
        shifted_logits = engine._shift_logits(logits, last_context_logit)
        confidence, sampled = engine._sample_tokens(shifted_logits[mask_index])

        candidate = torch.full_like(block_ids, MASK_TOKEN_ID)
        candidate[mask_index] = sampled
        full_confidence = torch.full_like(block_ids, -torch.inf, dtype=confidence.dtype)
        full_confidence[mask_index] = confidence
        transfer_count = int(mask_index.sum().item())
        selected_confidence, select_index = torch.topk(full_confidence, transfer_count)
        transfer_index = torch.zeros_like(block_ids, dtype=torch.bool)
        transfer_index[select_index[0]] = True
        for idx in range(1, transfer_count):
            if selected_confidence[idx] >= THRESHOLD:
                transfer_index[select_index[idx]] = True
        n_accepted = transfer_index.sum().item()
        accepted_positions = transfer_index.nonzero().squeeze(-1).tolist()
        block_ids[transfer_index] = candidate[transfer_index]
        n_mask = mask_index.sum().item()
        print(f"  Step {step+1}: {n_accepted} accepted (of {n_mask} mask), "
              f"positions={accepted_positions[:5]}{'...' if len(accepted_positions)>5 else ''}, "
              f"top_conf={selected_confidence[0].item():.4f}")

    engine_tokens = block_ids[:MAX_NEW_TOKENS].tolist()
    engine_text = engine.tokenizer.decode(engine_tokens)
    print(f"  Final tokens: {engine_tokens}")
    print(f"  Text: {repr(engine_text)}")

engine.close()
del engine
gc.collect()
torch.cuda.empty_cache()

# --- Run vLLM LLM engine ---
print("\n=== vLLM LLM engine ===")
from d2f_vllm import LLM, SamplingParams

llm = LLM(
    dream_base,
    model_name="dream",
    model_type="diffusion_lm",
    enforce_eager=True,
    data_parallel_size=1,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.30,
    max_num_batched_tokens=8192,
    max_num_seqs=1,
    max_model_len=8192,
    diffusion_block_size=BLOCK_LENGTH,
    accept_threshold=THRESHOLD,
    complete_threshold=0.0,
    add_new_block_threshold=1.0,
    kv_cache_layout="unified",
)
sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, stop=["</s>", "<|im_end|>"])

outputs = llm.generate(
    [prompt_ids],
    sp,
    use_tqdm=False,
    prompt_positions=[prompt_positions],
)
out = outputs[0]
print(f"  Token count: {len(out['token_ids'])}")
print(f"  n_diff_steps: {out['n_diff_steps']}")
print(f"  Final tokens: {out['token_ids']}")
print(f"  Text: {repr(out['text'])}")

print("\n=== Comparison ===")
if engine_tokens == out['token_ids']:
    print("MATCH: Both engines produce the same tokens")
else:
    print("MISMATCH: Different tokens!")
    for i, (e, v) in enumerate(zip(engine_tokens, out['token_ids'])):
        if e != v:
            print(f"  First diff at position {i}: engine={e} vllm={v}")
            break
