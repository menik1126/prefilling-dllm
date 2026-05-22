"""Test: does need_kv_cache_store=False cause different logits?"""
import gc
import json
import os
import sys
import math
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

from d2f_vllm.config import Config
from d2f_vllm.engine.model_runner import AutoModelRunner
from d2f_vllm.utils.context import set_context_diffusion_lm, reset_context_diffusion_lm
from d2f_vllm.fastdllm_engine import _StaticMaskSeq, FastDLLMDreamEngine
import torch.nn.functional as F

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

with torch.inference_mode():
    # Prefill (store ALL positions)
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

    shifted_prefill = FastDLLMDreamEngine._shift_logits(prefill_logits)
    first_logits = shifted_prefill[prompt_len:prompt_len+1, :]
    probs = F.softmax(first_logits.float(), dim=-1)
    _, first_token = probs.max(dim=-1)
    last_context_logit = prefill_logits[prompt_len - 1, :].detach()

    block_ids = torch.full((decode_len,), MASK_TOKEN_ID, dtype=torch.long, device=torch.cuda.current_device())
    block_ids[0] = first_token[0]
    print(f"First token: {first_token[0].item()}")

    num_pages = math.ceil(prompt_len / page_size)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=torch.cuda.current_device()).view(1, -1)

    # Test 1: need_kv_cache_store=True, proper slot_mapping (FastDLLMDreamEngine style)
    slot_mapping_store = torch.arange(prompt_len, prompt_len + decode_len, dtype=torch.int32, device=torch.cuda.current_device())
    seq_dc = _StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)
    set_context_diffusion_lm(
        False,
        cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        max_seqlen_q=decode_len, max_seqlen_k=prompt_len + decode_len,
        slot_mapping=slot_mapping_store,
        context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
        block_tables=block_tables,
        seqs=[seq_dc], seq_lens=[decode_len],
        seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        kv_cache_layout="unified", need_kv_cache_store=True,
    )
    try:
        hidden1 = model(block_ids.clone(), pos_t(suffix_positions))
        logits_store = model.compute_logits(hidden1)
    finally:
        reset_context_diffusion_lm()

    # Test 2: need_kv_cache_store=False, -1 slot_mapping (vLLM style)
    slot_mapping_nostore = torch.full((decode_len,), -1, dtype=torch.int32, device=torch.cuda.current_device())
    seq_dc2 = _StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)
    set_context_diffusion_lm(
        False,
        cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        max_seqlen_q=decode_len, max_seqlen_k=prompt_len + decode_len,
        slot_mapping=slot_mapping_nostore,
        context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
        block_tables=block_tables,
        seqs=[seq_dc2], seq_lens=[decode_len],
        seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        kv_cache_layout="unified", need_kv_cache_store=False,
    )
    try:
        hidden2 = model(block_ids.clone(), pos_t(suffix_positions))
        logits_nostore = model.compute_logits(hidden2)
    finally:
        reset_context_diffusion_lm()

    # Compare
    diff = (logits_store - logits_nostore).abs()
    print(f"\nLogits diff (store vs no-store): max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # Also test: multiple decode steps with store vs no-store
    # Run 3 decode steps with store
    block_a = block_ids.clone()
    for step in range(3):
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=decode_len, max_seqlen_k=prompt_len + decode_len,
            slot_mapping=slot_mapping_store,
            context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[_StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)],
            seq_lens=[decode_len],
            seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout="unified", need_kv_cache_store=True,
        )
        try:
            h = model(block_a, pos_t(suffix_positions))
            logits_a = model.compute_logits(h)
        finally:
            reset_context_diffusion_lm()

        shifted = FastDLLMDreamEngine._shift_logits(logits_a, last_context_logit)
        mask_idx = block_a == MASK_TOKEN_ID
        mask_logits = shifted[mask_idx]
        probs = F.softmax(mask_logits.float(), dim=-1)
        conf, sampled = probs.max(dim=-1)
        full_conf = torch.full_like(block_a, -torch.inf, dtype=conf.dtype)
        full_conf[mask_idx] = conf
        tc = mask_idx.sum().item()
        sc, si = torch.topk(full_conf, tc)
        transfer = torch.zeros_like(block_a, dtype=torch.bool)
        transfer[si[0]] = True
        for j in range(1, tc):
            if sc[j] >= 0.9:
                transfer[si[j]] = True
        cand = torch.full_like(block_a, MASK_TOKEN_ID)
        cand[mask_idx] = sampled
        block_a[transfer] = cand[transfer]

    # Run 3 decode steps with no-store
    block_b = block_ids.clone()
    for step in range(3):
        set_context_diffusion_lm(
            False,
            cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            max_seqlen_q=decode_len, max_seqlen_k=prompt_len + decode_len,
            slot_mapping=slot_mapping_nostore,
            context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
            block_tables=block_tables,
            seqs=[_StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)],
            seq_lens=[decode_len],
            seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
            kv_cache_layout="unified", need_kv_cache_store=False,
        )
        try:
            h = model(block_b, pos_t(suffix_positions))
            logits_b = model.compute_logits(h)
        finally:
            reset_context_diffusion_lm()

        shifted = FastDLLMDreamEngine._shift_logits(logits_b, last_context_logit)
        mask_idx = block_b == MASK_TOKEN_ID
        mask_logits = shifted[mask_idx]
        probs = F.softmax(mask_logits.float(), dim=-1)
        conf, sampled = probs.max(dim=-1)
        full_conf = torch.full_like(block_b, -torch.inf, dtype=conf.dtype)
        full_conf[mask_idx] = conf
        tc = mask_idx.sum().item()
        sc, si = torch.topk(full_conf, tc)
        transfer = torch.zeros_like(block_b, dtype=torch.bool)
        transfer[si[0]] = True
        for j in range(1, tc):
            if sc[j] >= 0.9:
                transfer[si[j]] = True
        cand = torch.full_like(block_b, MASK_TOKEN_ID)
        cand[mask_idx] = sampled
        block_b[transfer] = cand[transfer]

    print(f"\nAfter 3 steps, block_a == block_b: {(block_a == block_b).all().item()}")
    if not (block_a == block_b).all():
        for i in range(decode_len):
            if block_a[i] != block_b[i]:
                print(f"  First diff at pos {i}: store={block_a[i].item()} nostore={block_b[i].item()}")
                break
    print(f"  block_a: {block_a.tolist()}")
    print(f"  block_b: {block_b.tolist()}")

runner.exit()
