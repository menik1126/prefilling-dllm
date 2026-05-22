"""Intercept logits at each decode step in vLLM engine and run FastDLLM acceptance on them."""
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
THRESHOLD = 0.9

# Intercept the sampler to capture raw logits and what it does with them
_sampler_log = []
import d2f_vllm.layers.sampler as sampler_mod

_orig_sampler_forward = sampler_mod.SamplerForDream.forward

def _patched_sampler_forward(self, logits, temperatures, **kwargs):
    context = sampler_mod.get_context_diffusion_lm()
    is_prefill = context.is_prefill

    result = _orig_sampler_forward(self, logits, temperatures, **kwargs)

    if not is_prefill:
        # Capture the raw logits and run FastDLLM-style acceptance for comparison
        seqs = context.seqs
        split_logits = torch.split(logits, context.seq_lens, dim=0)

        for seq, seq_logits in zip(seqs, split_logits):
            # Get the block mask positions
            for block_id, block in enumerate(seq.diffusion_blocks):
                if not block.is_active or sum(block.local_mask_tokens) == 0:
                    continue

                # Run FastDLLM shift + sample
                from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine as _E
                # Find last_context_logit (we don't have it, so use the same integer as vLLM)
                shifted = self._shift_logits(seq_logits, seq.cached_or_caching_last_token_id)

                mask_ids = block.global_mask_token_ids
                mask_logits = shifted[mask_ids, :]
                probs = F.softmax(mask_logits.float(), dim=-1)
                conf, sampled = probs.max(dim=-1)

                # FastDLLM acceptance: top-1 + all >= threshold
                n_mask = len(mask_ids)
                _, sorted_idx = torch.sort(conf, descending=True)
                fastdllm_accept = set()
                fastdllm_accept.add(sorted_idx[0].item())
                for j in range(1, n_mask):
                    if conf[sorted_idx[j]] >= THRESHOLD:
                        fastdllm_accept.add(sorted_idx[j].item())

                # vLLM acceptance: from result
                seq_id_str = str(seq.seq_id)
                vllm_accepted = set(result.accepted_ids_map.get(seq_id_str, {}).get(str(block_id), []))

                # Confidence at each mask position
                entry = {
                    "n_mask": n_mask,
                    "fastdllm_n_accept": len(fastdllm_accept),
                    "vllm_n_accept": len(vllm_accepted),
                    "fastdllm_accept": sorted(fastdllm_accept),
                    "vllm_accept": sorted(vllm_accepted),
                    "match": fastdllm_accept == vllm_accepted,
                    "top_confs": sorted(conf.tolist(), reverse=True)[:5],
                    "conf_near_threshold": [
                        (i, c) for i, c in enumerate(conf.tolist())
                        if abs(c - THRESHOLD) < 0.01
                    ],
                }
                _sampler_log.append(entry)

    return result

sampler_mod.SamplerForDream.forward = _patched_sampler_forward

print("=== vLLM LLM engine (logits intercepted) ===")
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
sp = SamplingParams(temperature=0.0, max_tokens=32, stop=["</s>", "<|im_end|>"])

outputs = llm.generate(
    [prompt_ids],
    sp,
    use_tqdm=False,
    prompt_positions=[prompt_positions],
)
out = outputs[0]
print(f"  Final tokens: {out['token_ids']}")

print(f"\n=== Sampler log ({len(_sampler_log)} decode steps) ===")
for i, entry in enumerate(_sampler_log):
    match_str = "MATCH" if entry["match"] else "MISMATCH"
    print(f"  Step {i+1}: {match_str} masks={entry['n_mask']} "
          f"fastdllm_accept={entry['fastdllm_n_accept']} vllm_accept={entry['vllm_n_accept']}")
    if not entry["match"]:
        print(f"    FastDLLM accepted: {entry['fastdllm_accept']}")
        print(f"    vLLM accepted:     {entry['vllm_accept']}")
        print(f"    Top confs: {entry['top_confs']}")
        print(f"    Near threshold: {entry['conf_near_threshold']}")
    elif entry.get("conf_near_threshold"):
        print(f"    Near threshold: {entry['conf_near_threshold']}")
