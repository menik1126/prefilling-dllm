"""Benchmark: Unified vs Distinct KV cache layout for dLLM attention.

Compares FlexAttention (unified layout) against the custom Triton paged
attention kernel (distinct layout) for both prefill and decode phases.

Usage:
    python examples/bench_unified_vs_distinct.py [--seq-lengths 4096 8192 16384 32768]
"""

import argparse
import math
import time
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask
from transformers.integrations.flex_attention import (
    compile_friendly_flex_attention as flex_attention,
)

from prefilling_dllm.layers.attention.ops import (
    diffusion_lm_parallel_flash_decoding,
    load_kvcache,
    store_kvcache_distinct_layout,
    store_kvcache_unified_layout,
)
from prefilling_dllm.utils.context import ContextForDiffusionLM


NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_LENGTH = 32
PAGE_SIZE = 256
DTYPE = torch.bfloat16


def make_flex_attention():
    is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
    kernel_options = {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
    } if is_rtx_xx90(torch.cuda.get_device_name(0)) else None
    return torch.compile(
        partial(flex_attention, kernel_options=kernel_options, enable_gqa=True,
                return_lse=False, training=False),
        dynamic=True,
    )


def make_full_block_mask(seq_len: int, device: str):
    def _mask_mod(batch, head, token_q, token_kv):
        return token_q >= 0  # always True

    return create_block_mask(_mask_mod, 1, 1, seq_len, seq_len, device=device)


def make_decode_block_mask(q_len: int, kv_len: int, device: str):
    def _mask_mod(batch, head, token_q, token_kv):
        return token_q >= 0  # always True

    return create_block_mask(_mask_mod, 1, 1, q_len, kv_len, device=device)


def bench_fn(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_prefill_unified(seq_len: int, compiled_attn, warmup: int, iters: int) -> float:
    q = torch.randn(1, NUM_Q_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    block_mask = make_full_block_mask(seq_len, "cuda")

    def fn():
        compiled_attn(q, k, v, block_mask=block_mask)

    return bench_fn(fn, warmup, iters)


def bench_prefill_distinct(seq_len: int, warmup: int, iters: int) -> float:
    q = torch.randn(seq_len, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    o = torch.empty_like(q)

    num_blocks = math.ceil(seq_len / PAGE_SIZE)
    x = 8
    k_cache = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // x, PAGE_SIZE, x,
        dtype=DTYPE, device="cuda"
    )
    v_cache = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        dtype=DTYPE, device="cuda"
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, -1)
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    seq_lens_t = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device="cuda")

    def fn():
        diffusion_lm_parallel_flash_decoding(
            q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
            block_table, cu_seqlens_q, seq_lens_t,
            seq_len, seq_len, 1.0, 1.0,
            BLOCK_LENGTH, None, None,
            1.0 / math.sqrt(HEAD_DIM), mask,
            bidirectional=True,
        )

    return bench_fn(fn, warmup, iters)


def bench_decode_unified(prompt_len: int, compiled_attn, warmup: int, iters: int) -> float:
    block_len = BLOCK_LENGTH
    kv_len = prompt_len + block_len

    q = torch.randn(1, NUM_Q_HEADS, block_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    k_full = torch.randn(1, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_full = torch.randn(1, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    block_mask = make_decode_block_mask(block_len, kv_len, "cuda")

    def fn():
        compiled_attn(q, k_full, v_full, block_mask=block_mask)

    return bench_fn(fn, warmup, iters)


def bench_decode_distinct(prompt_len: int, warmup: int, iters: int) -> float:
    block_len = BLOCK_LENGTH
    total_len = prompt_len + block_len

    q = torch.randn(block_len, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(block_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(block_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    o = torch.empty_like(q)

    num_blocks = math.ceil(total_len / PAGE_SIZE)
    x = 8
    k_cache = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // x, PAGE_SIZE, x,
        dtype=DTYPE, device="cuda"
    )
    v_cache = torch.randn(
        num_blocks, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        dtype=DTYPE, device="cuda"
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(1, -1)
    cu_seqlens_q = torch.tensor([0, block_len], dtype=torch.int32, device="cuda")
    total_lens_t = torch.tensor([total_len], dtype=torch.int32, device="cuda")
    mask = torch.ones((block_len, block_len), dtype=torch.bool, device="cuda")

    def fn():
        diffusion_lm_parallel_flash_decoding(
            q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
            block_table, cu_seqlens_q, total_lens_t,
            total_len, block_len, 1.0, 1.0,
            BLOCK_LENGTH, None, None,
            1.0 / math.sqrt(HEAD_DIM), mask,
        )

    return bench_fn(fn, warmup, iters)


def main():
    parser = argparse.ArgumentParser(description="Benchmark unified vs distinct KV cache layout")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: GQA {NUM_Q_HEADS}/{NUM_KV_HEADS} heads, head_dim={HEAD_DIM}, "
          f"block_length={BLOCK_LENGTH}, dtype={DTYPE}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    compiled_attn = make_flex_attention()

    # Warmup compile
    print("Compiling FlexAttention (first call triggers torch.compile)...")
    _q = torch.randn(1, NUM_Q_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _k = torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _v = torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _bm = make_full_block_mask(128, "cuda")
    compiled_attn(_q, _k, _v, block_mask=_bm)
    torch.cuda.synchronize()
    print("Compilation done.\n")

    header = f"{'SeqLen':>8} | {'Phase':<8} | {'Unified (ms)':>13} | {'Distinct (ms)':>14} | {'Speedup':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for seq_len in args.seq_lengths:
        # Prefill
        t_unified_prefill = bench_prefill_unified(seq_len, compiled_attn, args.warmup, args.iters)
        t_distinct_prefill = bench_prefill_distinct(seq_len, args.warmup, args.iters)
        speedup_prefill = t_unified_prefill / t_distinct_prefill if t_distinct_prefill > 0 else float("inf")
        print(f"{seq_len:>8} | {'Prefill':<8} | {t_unified_prefill:>10.2f} ms | {t_distinct_prefill:>11.2f} ms | {speedup_prefill:>7.2f}x")

        # Decode
        t_unified_decode = bench_decode_unified(seq_len, compiled_attn, args.warmup, args.iters)
        t_distinct_decode = bench_decode_distinct(seq_len, args.warmup, args.iters)
        speedup_decode = t_unified_decode / t_distinct_decode if t_distinct_decode > 0 else float("inf")
        print(f"{seq_len:>8} | {'Decode':<8} | {t_unified_decode:>10.2f} ms | {t_distinct_decode:>11.2f} ms | {speedup_decode:>7.2f}x")

    print(sep)
    print("Speedup > 1.0 means distinct is faster than unified.")


if __name__ == "__main__":
    main()


def bench_decode_unified_batched(prompt_len: int, batch_size: int, compiled_attn, warmup: int, iters: int) -> float:
    block_len = BLOCK_LENGTH
    kv_len = prompt_len + block_len
    q = torch.randn(batch_size, NUM_Q_HEADS, block_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    k_full = torch.randn(batch_size, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_full = torch.randn(batch_size, NUM_KV_HEADS, kv_len, HEAD_DIM, dtype=DTYPE, device="cuda")

    def _mask_mod(batch, head, token_q, token_kv):
        return token_q >= 0
    block_mask = create_block_mask(_mask_mod, batch_size, 1, block_len, kv_len, device="cuda")

    def fn():
        compiled_attn(q, k_full, v_full, block_mask=block_mask)
    return bench_fn(fn, warmup, iters)


def bench_decode_distinct_batched(prompt_len: int, batch_size: int, warmup: int, iters: int) -> float:
    block_len = BLOCK_LENGTH
    total_len = prompt_len + block_len
    total_q_tokens = batch_size * block_len

    q = torch.randn(total_q_tokens, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(total_q_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(total_q_tokens, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    o = torch.empty_like(q)

    pages_per_seq = math.ceil(total_len / PAGE_SIZE)
    total_pages = batch_size * pages_per_seq
    x = 8
    k_cache = torch.randn(total_pages, NUM_KV_HEADS, HEAD_DIM // x, PAGE_SIZE, x, dtype=DTYPE, device="cuda")
    v_cache = torch.randn(total_pages, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, dtype=DTYPE, device="cuda")

    block_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(batch_size, pages_per_seq)
    cu_seqlens_q = torch.arange(0, total_q_tokens + 1, block_len, dtype=torch.int32, device="cuda")
    total_lens_t = torch.full((batch_size,), total_len, dtype=torch.int32, device="cuda")
    mask = torch.ones((block_len, block_len), dtype=torch.bool, device="cuda")

    def fn():
        diffusion_lm_parallel_flash_decoding(
            q, k, v, o, str(k_cache.dtype), k_cache, v_cache,
            block_table, cu_seqlens_q, total_lens_t,
            total_len, block_len, 1.0, 1.0,
            BLOCK_LENGTH, None, None,
            1.0 / math.sqrt(HEAD_DIM), mask,
        )
    return bench_fn(fn, warmup, iters)


if __name__ == "__main__" and "batched" in sys.argv:
    import sys
    warmup, iters = 5, 20
    compiled_attn = make_flex_attention()
    _q = torch.randn(1, NUM_Q_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _k = torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _v = torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM, dtype=DTYPE, device="cuda")
    _bm = create_block_mask(lambda b,h,q,k: q>=0, 1, 1, 128, 128, device="cuda")
    compiled_attn(_q, _k, _v, block_mask=_bm)
    torch.cuda.synchronize()

    print(f"{'Batch':>6} | {'PromptLen':>9} | {'Unified (ms)':>13} | {'Distinct (ms)':>14} | {'Speedup':>8}")
    print("-" * 65)
    for prompt_len in [4096, 8192]:
        for batch_size in [1, 4, 8, 16, 32]:
            t_u = bench_decode_unified_batched(prompt_len, batch_size, compiled_attn, warmup, iters)
            t_d = bench_decode_distinct_batched(prompt_len, batch_size, warmup, iters)
            sp = t_u / t_d if t_d > 0 else float("inf")
            print(f"{batch_size:>6} | {prompt_len:>9} | {t_u:>10.2f} ms | {t_d:>11.2f} ms | {sp:>7.2f}x")
