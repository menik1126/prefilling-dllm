"""Instrument vLLM LLM engine to trace decode steps vs FastDLLMDreamEngine."""
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

compressed_path = "/home/ma-user/work/Discrete-Diffusion-Forcing/d2f_vllm/log/longbench_multifieldqa_en_fastdllm_vllm_bridge_compressed_20260521_002449.json"
with open(compressed_path) as f:
    compressed = json.load(f)

record = compressed["records"][3]
prompt_ids = record["prompt_ids"]
prompt_positions = record["prompt_positions"]

dream_base = os.environ.get("DREAM_BASE", str(Path(d2f_eval_dir) / "model_weights/Dream-v0-Base-7B"))
MASK_TOKEN_ID = 151666
BLOCK_LENGTH = 32
MAX_NEW_TOKENS = 32
THRESHOLD = 0.9

# Monkey-patch the model runner to log decode inputs
_step_log = []
import d2f_vllm.engine.model_runner as mr

_orig_prepare_decode = mr.ModelRunnerForDiffusionLM.prepare_decode

def _patched_prepare_decode(self, seqs):
    result = _orig_prepare_decode(self, seqs)
    input_ids, positions = result
    _step_log.append({
        "input_ids": input_ids.cpu().tolist(),
        "positions": positions.cpu().tolist(),
        "n_mask": sum(1 for x in input_ids.cpu().tolist() if x == MASK_TOKEN_ID),
    })
    return result

mr.ModelRunnerForDiffusionLM.prepare_decode = _patched_prepare_decode

# Also patch postprocess to log accepted tokens
import d2f_vllm.engine.scheduler as sched

_orig_postprocess = sched.SchedulerForDiffusionLM.postprocess
_postprocess_log = []

def _patched_postprocess(self, seqs, sample_output):
    result = _orig_postprocess(self, seqs, sample_output)
    for seq in seqs:
        gen_start = seq.diffusion_blocks[1].global_start_id if len(seq.diffusion_blocks) > 1 else 0
        gen_end = gen_start + BLOCK_LENGTH
        block_tokens = list(seq.token_ids[gen_start:gen_end])
        _postprocess_log.append({
            "block_tokens": block_tokens,
            "n_mask": sum(1 for x in block_tokens if x == MASK_TOKEN_ID),
            "new_tokens": seq.new_tokens,
        })
    return result

sched.SchedulerForDiffusionLM.postprocess = _patched_postprocess


print("=== vLLM LLM engine (instrumented) ===")
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
print(f"  Final tokens: {out['token_ids']}")
print(f"  n_diff_steps: {out['n_diff_steps']}")

# Now print step-by-step log
print(f"\n=== Decode step log ({len(_step_log)} steps) ===")
for i, (step, post) in enumerate(zip(_step_log, _postprocess_log)):
    n_mask_before = step["n_mask"]
    n_mask_after = post["n_mask"]
    accepted = n_mask_before - n_mask_after
    # Show which positions changed
    block_tokens = post["block_tokens"]
    print(f"  Step {i+1}: masks_before={n_mask_before}, accepted={accepted}, "
          f"masks_after={n_mask_after}, new_tokens={post['new_tokens']}")
    if i < 5 or accepted > 2:
        ids = step["input_ids"]
        mask_pos = [j for j, x in enumerate(ids) if x == MASK_TOKEN_ID]
        nonmask_pos = [j for j, x in enumerate(ids) if x != MASK_TOKEN_ID]
        print(f"    Input non-mask positions: {nonmask_pos[:10]}{'...' if len(nonmask_pos)>10 else ''}")
        print(f"    Block after: {block_tokens}")

# Reference: FastDLLMDreamEngine step trace
print("\n=== Reference: FastDLLMDreamEngine expected ===")
expected = [1527, 358, 6484, 320, 6383, 517, 480, 31744, 8721, 28799, 10665, 70084, 8, 374, 264, 12005, 2673, 94856, 14346, 3671, 311, 4228, 2272, 61899, 52105, 41308, 1361, 13, 151643, 151665, 872, 25]
actual = out['token_ids']
for i, (e, a) in enumerate(zip(expected, actual)):
    if e != a:
        print(f"  First diff at position {i}: expected={e} actual={a}")
        break
else:
    print("  MATCH: all tokens identical!")
