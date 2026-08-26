"""Minimal NCCL smoke test for the Prefilling-dLLM training environment."""

import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor([float(dist.get_rank() + 1)], device=local_rank)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    print(
        f"rank={dist.get_rank()} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} all_reduce={value.item():.1f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
