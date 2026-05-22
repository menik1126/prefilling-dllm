"""Run one model, both acceptance logics, to isolate acceptance divergence."""
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

from d2f_vllm.config import Config
from d2f_vllm.engine.model_runner import AutoModelRunner
from d2f_vllm.utils.context import set_context_diffusion_lm, reset_context_diffusion_lm
from d2f_vllm.fastdllm_engine import _StaticMaskSeq

cfg = Config(
    model=dream_base, model_name="dream", model_type="diffusion_lm",
    mask_token_id=MASK_TOKEN_ID, diffusion_block_size=BLOCK_LENGTH,
    max_model_len=8192, max_num_batched_tokens=8192, max_num_seqs=1,
    tensor_parallel_size=1, gpu_memory_utilization=0.30,
    enforce_eager=True, kv_cache_layout="unified",
)
runner = AutoModelRunner.from_config(cfg, 0, [])
model = runner.model
page_size = runner.block_size

prompt_ids_list = list(int(x) for x in prompt_ids)
pos_list = [int(x) for x in prompt_positions]
decode_len = BLOCK_LENGTH
suffix_pos_start = max(pos_list) + 1
suffix_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
full_ids = prompt_ids_list + [MASK_TOKEN_ID] * decode_len
full_positions = pos_list + suffix_positions
prompt_len = len(prompt_ids_list)

def pos_t(positions):
    return torch.tensor(list(positions), dtype=torch.long, device=torch.cuda.current_device())

def ids_t(ids):
    return torch.tensor(list(ids), dtype=torch.long, device=torch.cuda.current_device())

def full_mask(rows, cols=None):
    cols = rows if cols is None else cols
    return torch.ones((rows, cols), dtype=torch.bool, device=torch.cuda.current_device())


def fastdllm_accept(confidence, sampled, block_ids, mask_index, threshold):
    """FastDLLMDreamEngine acceptance logic."""
    candidate = torch.full_like(block_ids, MASK_TOKEN_ID)
    candidate[mask_index] = sampled
    full_confidence = torch.full_like(block_ids, -torch.inf, dtype=confidence.dtype)
    full_confidence[mask_index] = confidence
    transfer_count = int(mask_index.sum().item())
    selected_confidence, select_index = torch.topk(full_confidence, transfer_count)
    transfer_index = torch.zeros_like(block_ids, dtype=torch.bool)
    transfer_index[select_index[0]] = True
    for idx in range(1, transfer_count):
        if selected_confidence[idx] >= threshold:
            transfer_index[select_index[idx]] = True
    return transfer_index, candidate


def vllm_accept(confidence, sampled, block_ids, mask_index, threshold):
    """vLLM SamplerForDream acceptance logic (pre_block_complete path)."""
    # initial_confidence = confidence (no margin_confidence or neg_entropy)
    high_conf_indices = torch.where(confidence >= threshold)[0]  # now >= after our fix
    if len(high_conf_indices) == 0:
        _, transfer_index_in_mask = torch.topk(confidence, 1)
    else:
        transfer_index_in_mask = torch.tensor([], device=sampled.device, dtype=torch.long)
    accepted_ids = torch.unique(torch.cat([transfer_index_in_mask, high_conf_indices]))

    # Map back to block positions
    mask_positions = mask_index.nonzero().squeeze(-1)
    transfer_index = torch.zeros_like(block_ids, dtype=torch.bool)
    candidate = torch.full_like(block_ids, MASK_TOKEN_ID)
    candidate[mask_index] = sampled

    for aid in accepted_ids.tolist():
        transfer_index[mask_positions[aid]] = True

    return transfer_index, candidate


with torch.inference_mode():
    # Prefill
    seq_len = len(full_ids)
    slot_mapping_pf = torch.arange(seq_len, dtype=torch.int32, device=torch.cuda.current_device())
    seq = _StaticMaskSeq(full_mask(seq_len), BLOCK_LENGTH)
    set_context_diffusion_lm(
        True,
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
        cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
        max_seqlen_q=seq_len, max_seqlen_k=seq_len,
        slot_mapping=slot_mapping_pf,
        context_lens=torch.tensor([0], dtype=torch.int32, device=torch.cuda.current_device()),
        block_tables=None,
        seqs=[seq], seq_lens=[seq_len],
        seq_lens_ts=torch.tensor([seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
        kv_cache_layout="unified", need_kv_cache_store=True,
    )
    try:
        hidden = model(ids_t(full_ids), pos_t(full_positions))
        prefill_logits = model.compute_logits(hidden)
    finally:
        reset_context_diffusion_lm()

    from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine
    shifted_prefill = FastDLLMDreamEngine._shift_logits(prefill_logits)
    first_logits = shifted_prefill[prompt_len:prompt_len + 1, :]
    probs = F.softmax(first_logits.float(), dim=-1)
    _, first_token = probs.max(dim=-1)
    last_context_logit = prefill_logits[prompt_len - 1, :].detach()
    print(f"First token: {first_token[0].item()}")

    num_pages = math.ceil(prompt_len / page_size)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=torch.cuda.current_device()).view(1, -1)
    slot_mapping_dc = torch.arange(prompt_len, prompt_len + decode_len, dtype=torch.int32, device=torch.cuda.current_device())

    # Run both acceptance logics in parallel with SAME model
    block_a = torch.full((decode_len,), MASK_TOKEN_ID, dtype=torch.long, device=torch.cuda.current_device())
    block_a[0] = first_token[0]
    block_b = block_a.clone()

    for step in range(50):
        mask_a = block_a == MASK_TOKEN_ID
        mask_b = block_b == MASK_TOKEN_ID
        if not mask_a.any() and not mask_b.any():
            break

        # Check if blocks are still identical
        if not (block_a == block_b).all():
            for i in range(decode_len):
                if block_a[i] != block_b[i]:
                    print(f"\n*** BLOCKS DIVERGED at step {step+1}, position {i}: "
                          f"fastdllm={block_a[i].item()} vllm={block_b[i].item()}")
                    break
            break

        # Forward pass (same for both since blocks are identical)
        if not mask_a.any():
            break
        seq_dc = _StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=decode_len, max_seqlen_k=prompt_len + decode_len,
            slot_mapping=slot_mapping_dc,
            context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[seq_dc], seq_lens=[decode_len],
            seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout="unified", need_kv_cache_store=True,
        )
        try:
            h = model(block_a.clone(), pos_t(suffix_positions))
            logits = model.compute_logits(h)
        finally:
            reset_context_diffusion_lm()

        shifted = FastDLLMDreamEngine._shift_logits(logits, last_context_logit)
        mask_logits = shifted[mask_a]
        probs = F.softmax(mask_logits.float(), dim=-1)
        conf, sampled = probs.max(dim=-1)

        # Apply both acceptance logics
        transfer_a, cand_a = fastdllm_accept(conf, sampled, block_a, mask_a, THRESHOLD)
        transfer_b, cand_b = vllm_accept(conf, sampled, block_b, mask_b, THRESHOLD)

        n_a = transfer_a.sum().item()
        n_b = transfer_b.sum().item()
        pos_a = transfer_a.nonzero().squeeze(-1).tolist()
        pos_b = transfer_b.nonzero().squeeze(-1).tolist()

        if pos_a != pos_b:
            print(f"  Step {step+1}: ACCEPTANCE DIFFERS! fastdllm={pos_a} vllm={pos_b}")
            # Show confidence details
            mask_positions = mask_a.nonzero().squeeze(-1).tolist()
            print(f"    Mask positions: {mask_positions}")
            for i, (mp, c) in enumerate(zip(mask_positions, conf.tolist())):
                marker_a = "A" if mp in pos_a else " "
                marker_b = "V" if mp in pos_b else " "
                print(f"    pos={mp} conf={c:.6f} [{marker_a}{marker_b}]")
        else:
            print(f"  Step {step+1}: SAME acceptance: {n_a} tokens at {pos_a[:5]}{'...' if len(pos_a)>5 else ''}")

        block_a[transfer_a] = cand_a[transfer_a]
        block_b[transfer_b] = cand_b[transfer_b]

    print(f"\nFinal block_a: {block_a.tolist()}")
    print(f"Final block_b: {block_b.tolist()}")
    print(f"Match: {(block_a == block_b).all().item()}")

runner.exit()
