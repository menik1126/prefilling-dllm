# Prefilling-dLLM Engine

The first dLLM inference engine with native prefill–decode (PD) disaggregation.

Prefilling-dLLM computes prefix KV once, exposes transferable snapshots at an explicit PD boundary, and reuses sparse paged KV during iterative denoising.

## Core capabilities

- Native prefill and decode execution stages
- Transferable KV snapshots
- Predictive top-K chunk selection
- Optional intra-chunk token eviction
- Non-contiguous paged-KV attention
- Dream and Fast-dLLM model paths

See the [paper](https://arxiv.org/abs/2606.10537) and [source repository](https://github.com/menik1126/prefilling-dllm).
