import torch
import triton
import triton.language as tl


@triton.jit
def _query_max_den_kernel(
    q_ptr,
    k_ptr,
    max_ptr,
    den_ptr,
    scale: tl.constexpr,
    prefix_len: tl.constexpr,
    chunk_len: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    mask_mode: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    kv_head_idx = head_idx // group_size
    row = prefix_len + chunk_len + query_idx

    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + (row * num_query_heads + head_idx) * head_dim + offs_d,
        mask=offs_d < head_dim,
        other=0.0,
    ).to(tl.float32)

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    for key_block_start in range(0, key_len, BLOCK_N):
        offs_n = key_block_start + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + (offs_n[:, None] * num_kv_heads + kv_head_idx) * head_dim + offs_d[None, :],
            mask=(offs_n[:, None] < key_len) & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        valid = offs_n < key_len
        if mask_mode == 1:
            valid = valid & (offs_n <= row)
        logits = tl.where(valid, logits, -float("inf"))
        block_m = tl.max(logits, axis=0)
        new_m = tl.maximum(m, block_m)
        l = l * tl.exp(m - new_m) + tl.sum(tl.exp(logits - new_m), axis=0)
        m = new_m

    out_idx = head_idx * query_len + query_idx
    tl.store(max_ptr + out_idx, m)
    tl.store(den_ptr + out_idx, l)


@triton.jit
def _query_to_chunk_scores_kernel(
    q_ptr,
    k_ptr,
    max_ptr,
    den_ptr,
    out_ptr,
    scale: tl.constexpr,
    prefix_len: tl.constexpr,
    chunk_len: tl.constexpr,
    query_len: tl.constexpr,
    key_len: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    reduce_mean: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head_idx = tl.program_id(0)
    chunk_block_idx = tl.program_id(1)
    kv_head_idx = head_idx // group_size

    offs_c = chunk_block_idx * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    key_pos = prefix_len + offs_c
    offs_d = tl.arange(0, BLOCK_D)
    k = tl.load(
        k_ptr + (key_pos[:, None] * num_kv_heads + kv_head_idx) * head_dim + offs_d[None, :],
        mask=(offs_c[:, None] < chunk_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_CHUNK,), tl.float32)
    for query_idx in range(0, query_len):
        row = prefix_len + chunk_len + query_idx
        q = tl.load(
            q_ptr + (row * num_query_heads + head_idx) * head_dim + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * scale
        max_val = tl.load(max_ptr + head_idx * query_len + query_idx)
        den_val = tl.load(den_ptr + head_idx * query_len + query_idx)
        probs = tl.exp(logits - max_val) / den_val
        probs = tl.where(offs_c < chunk_len, probs, 0.0)
        acc += probs

    if reduce_mean:
        acc = acc / query_len
    tl.store(out_ptr + head_idx * chunk_len + offs_c, acc, mask=offs_c < chunk_len)


def query_to_chunk_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    prefix_len: int,
    chunk_len: int,
    query_len: int,
    scale: float,
    reduce_mode: str = "sum",
    attention_mask: str = "causal",
    block_n: int = 64,
    block_chunk: int = 64,
) -> torch.Tensor:
    """Return per-query-head query-to-chunk attention scores.

    This is a specialized scorer for ParallelComp token eviction. It computes
    softmax denominators over all visible keys, but only writes the reduced
    probabilities for chunk tokens.
    """
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("Triton eviction scorer requires CUDA tensors.")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"Expected q/k as [seq, heads, dim], got {tuple(q.shape)} / {tuple(k.shape)}")
    if chunk_len <= 0 or query_len <= 0:
        return torch.empty(0, device=q.device)
    if int(q.shape[0]) != int(k.shape[0]):
        raise ValueError("q and k must use the same sequence length.")
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    head_dim = int(q.shape[2])
    if int(k.shape[2]) != head_dim:
        raise ValueError("q and k must use the same head dimension.")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads.")
    if head_dim > 256:
        raise ValueError(f"Unsupported head_dim for Triton eviction scorer: {head_dim}")
    if query_len > 64:
        raise ValueError(f"query_len={query_len} is too large for first Triton scorer version.")

    mask = (attention_mask or "full").lower()
    if mask in {"full", "none", "query_to_chunk"}:
        mask_mode = 0
    elif mask == "causal":
        mask_mode = 1
    else:
        raise ValueError(f"Unsupported token attention mask for Triton scorer: {attention_mask}")

    q = q.contiguous()
    k = k.contiguous()
    key_len = int(k.shape[0])
    group_size = num_query_heads // num_kv_heads
    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty((num_query_heads, int(chunk_len)), device=q.device, dtype=torch.float32)
    max_buf = torch.empty((num_query_heads, int(query_len)), device=q.device, dtype=torch.float32)
    den_buf = torch.empty((num_query_heads, int(query_len)), device=q.device, dtype=torch.float32)

    _query_max_den_kernel[(num_query_heads, int(query_len))](
        q,
        k,
        max_buf,
        den_buf,
        float(scale),
        int(prefix_len),
        int(chunk_len),
        int(query_len),
        key_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        group_size,
        mask_mode,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
    )
    grid = (num_query_heads, triton.cdiv(int(chunk_len), block_chunk))
    _query_to_chunk_scores_kernel[grid](
        q,
        k,
        max_buf,
        den_buf,
        out,
        float(scale),
        int(prefix_len),
        int(chunk_len),
        int(query_len),
        key_len,
        num_query_heads,
        num_kv_heads,
        head_dim,
        group_size,
        (reduce_mode or "sum").lower() == "mean",
        BLOCK_CHUNK=block_chunk,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out
