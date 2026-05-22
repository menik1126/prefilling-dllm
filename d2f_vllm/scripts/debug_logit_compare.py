"""Compare logits between FastDLLMDreamEngine and vLLM decode for ONE step."""
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

# --- FastDLLMDreamEngine: get prefill logits and first decode logits ---
print("=== FastDLLMDreamEngine ===")
from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine

engine = FastDLLMDreamEngine(
    dream_base, max_model_len=8192, block_length=BLOCK_LENGTH,
    gpu_memory_utilization=0.30, max_num_seqs=1, threshold=0.9, temperature=0.0,
)

prompt_ids_list = list(int(x) for x in prompt_ids)
pos_list = [int(x) for x in prompt_positions]
decode_len = BLOCK_LENGTH
suffix_pos_start = max(pos_list) + 1
suffix_positions = list(range(suffix_pos_start, suffix_pos_start + decode_len))
full_ids = prompt_ids_list + [MASK_TOKEN_ID] * decode_len
full_positions = pos_list + suffix_positions
prompt_len = len(prompt_ids_list)

with torch.inference_mode():
    # Prefill
    prefill_logits = engine._forward_prefill(full_ids, full_positions)
    shifted_prefill = engine._shift_logits(prefill_logits)
    first_logits = shifted_prefill[prompt_len:prompt_len + 1, :]
    _, first_token = engine._sample_tokens(first_logits)
    last_context_logit = prefill_logits[prompt_len - 1, :].detach()

    block_ids = torch.full((decode_len,), MASK_TOKEN_ID, dtype=torch.long, device=torch.cuda.current_device())
    block_ids[0] = first_token[0]
    print(f"  Prefill first token: {first_token[0].item()}")
    print(f"  Prefill logits at prompt[-1]: min={prefill_logits[prompt_len-1].min().item():.4f} max={prefill_logits[prompt_len-1].max().item():.4f} mean={prefill_logits[prompt_len-1].mean().item():.4f}")

    # First decode step
    engine_decode_logits = engine._forward_replace_block(
        block_ids, prompt_len=prompt_len, slot_start=prompt_len, block_positions=suffix_positions,
    )
    engine_shifted = engine._shift_logits(engine_decode_logits, last_context_logit)

    # Sample at all mask positions
    mask_index = block_ids == MASK_TOKEN_ID
    engine_mask_logits = engine_shifted[mask_index]
    engine_conf, engine_sampled = engine._sample_tokens(engine_mask_logits)

    print(f"  Decode step 1 logits shape: {engine_decode_logits.shape}")
    print(f"  Decode step 1 shifted[1] (pos 1, mask): min={engine_shifted[1].min().item():.6f} max={engine_shifted[1].max().item():.6f}")
    print(f"  Top sampled token at mask pos 0 (=gen pos 1): {engine_sampled[0].item()}, conf={engine_conf[0].item():.6f}")
    print(f"  Top sampled token at mask pos 1 (=gen pos 2): {engine_sampled[1].item()}, conf={engine_conf[1].item():.6f}")

    # Save logits for comparison
    engine_decode_logits_cpu = engine_decode_logits.cpu().clone()
    engine_shifted_cpu = engine_shifted.cpu().clone()

engine.close()
del engine
gc.collect()
torch.cuda.empty_cache()

# --- vLLM engine: manual decode using model runner ---
print("\n=== vLLM LLM engine (manual decode) ===")
from d2f_vllm.config import Config
from d2f_vllm.engine.model_runner import AutoModelRunner
from d2f_vllm.utils.context import set_context_diffusion_lm, reset_context_diffusion_lm
from d2f_vllm.fastdllm_engine import _StaticMaskSeq
import math

# Use the SAME model runner as FastDLLMDreamEngine but with vLLM-style slot mapping
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

def positions_tensor(positions):
    return torch.tensor(list(positions), dtype=torch.long, device=torch.cuda.current_device())

def ids_tensor(ids):
    return torch.tensor(list(ids), dtype=torch.long, device=torch.cuda.current_device())

def full_mask(rows, cols=None):
    cols = rows if cols is None else cols
    return torch.ones((rows, cols), dtype=torch.bool, device=torch.cuda.current_device())

with torch.inference_mode():
    # Prefill (same as FastDLLMDreamEngine)
    input_ids_t = ids_tensor(full_ids)
    seq_len = len(full_ids)
    slot_mapping_pf = torch.arange(seq_len, dtype=torch.int32, device=torch.cuda.current_device())
    seq = _StaticMaskSeq(full_mask(seq_len), BLOCK_LENGTH)
    seq_lens_ts = torch.tensor([seq_len], dtype=torch.int32, device=torch.cuda.current_device())
    set_context_diffusion_lm(
        True,
        cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
        cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=torch.cuda.current_device()),
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        slot_mapping=slot_mapping_pf,
        context_lens=torch.tensor([0], dtype=torch.int32, device=torch.cuda.current_device()),
        block_tables=None,
        seqs=[seq],
        seq_lens=[seq_len],
        seq_lens_ts=seq_lens_ts,
        kv_cache_layout="unified",
        need_kv_cache_store=True,
    )
    try:
        hidden = model(input_ids_t, positions_tensor(full_positions))
        vllm_prefill_logits = model.compute_logits(hidden)
    finally:
        reset_context_diffusion_lm()

    # Compare prefill logits
    diff = (vllm_prefill_logits - prefill_logits.to(vllm_prefill_logits.device)).abs()
    print(f"  Prefill logits diff: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

    # First decode step (same as FastDLLMDreamEngine)
    block_ids_v = torch.full((decode_len,), MASK_TOKEN_ID, dtype=torch.long, device=torch.cuda.current_device())
    block_ids_v[0] = first_token[0].item()

    num_pages = math.ceil(prompt_len / page_size)
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=torch.cuda.current_device()).view(1, -1)
    slot_mapping_dc = torch.arange(prompt_len, prompt_len + decode_len, dtype=torch.int32, device=torch.cuda.current_device())
    seq_dc = _StaticMaskSeq(full_mask(decode_len, prompt_len + decode_len), BLOCK_LENGTH)

    set_context_diffusion_lm(
        False,
        cu_seqlens_q=torch.tensor([0, decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        cu_seqlens_k=torch.tensor([0, prompt_len + decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        max_seqlen_q=decode_len,
        max_seqlen_k=prompt_len + decode_len,
        slot_mapping=slot_mapping_dc,
        context_lens=torch.tensor([prompt_len], dtype=torch.int32, device=torch.cuda.current_device()),
        block_tables=block_tables,
        seqs=[seq_dc],
        seq_lens=[decode_len],
        seq_lens_ts=torch.tensor([decode_len], dtype=torch.int32, device=torch.cuda.current_device()),
        kv_cache_layout="unified",
        need_kv_cache_store=True,
    )
    try:
        hidden = model(block_ids_v, positions_tensor(suffix_positions))
        vllm_decode_logits = model.compute_logits(hidden)
    finally:
        reset_context_diffusion_lm()

    # Compare decode logits
    vllm_decode_cpu = vllm_decode_logits.cpu()
    diff_decode = (vllm_decode_cpu - engine_decode_logits_cpu).abs()
    print(f"  Decode logits diff: max={diff_decode.max().item():.6e}, mean={diff_decode.mean().item():.6e}")

    # Compare shifted logits at mask positions
    from d2f_vllm.fastdllm_engine import FastDLLMDreamEngine as _E
    vllm_shifted = _E._shift_logits(vllm_decode_logits, last_context_logit)
    vllm_mask_logits = vllm_shifted[mask_index]
    import torch.nn.functional as F
    vllm_probs = F.softmax(vllm_mask_logits.float(), dim=-1)
    vllm_conf, vllm_sampled = vllm_probs.max(dim=-1)

    print(f"  vLLM sampled at mask pos 0: {vllm_sampled[0].item()}, conf={vllm_conf[0].item():.6f}")
    print(f"  vLLM sampled at mask pos 1: {vllm_sampled[1].item()}, conf={vllm_conf[1].item():.6f}")

runner.exit()
