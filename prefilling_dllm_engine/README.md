# Prefilling-dLLM Engine

The first diffusion language model inference engine with native prefill–decode (PD) disaggregation.

The engine provides a paged KV cache, explicit prefill and decode stages, transferable KV snapshots, predictive chunk selection, intra-chunk token eviction, and non-contiguous sparse attention for long-context dLLM inference.

## PD execution model

```text
Prefill worker                         Decode worker
──────────────                         ─────────────
prefill_to_pd_record()                 make_pd_record_from_kv_snapshot()
        │                                           │
snapshot_pd_record_kv() ── transfer KV snapshot ──►│
                                                    │
                                         decode_from_pd_record()
```

The main APIs are implemented by `FastDLLMDreamEngine`:

- `prefill_to_pd_record(...)` builds the prefix KV state once.
- `snapshot_pd_record_kv(...)` exports a transferable KV payload.
- `make_pd_record_from_kv_snapshot(...)` reconstructs decode-side paged KV state.
- `make_pd_record_from_remote_engine(...)` transfers state directly between engine instances.
- `decode_from_pd_record(...)` performs iterative denoising against the cached prefix.
- `release_pd_record(...)` releases owned pages explicitly.

## Installation

Python 3.12 or later is required.

```bash
uv sync
source .venv/bin/activate
uv pip install -e .
uv pip install vllm
```

The import namespace is `prefilling_dllm`:

```python
from prefilling_dllm import FastDLLMDreamEngine, LLM, SamplingParams
```

## Capabilities

- [x] Native dLLM prefill–decode disaggregation
- [x] Transferable KV snapshots
- [x] Paged KV cache allocation and reuse
- [x] Predictive top-K chunk selection
- [x] Optional per-layer, per-head token eviction
- [x] Tensor parallel and data parallel execution
- [x] Dream/Fast-dLLM integration
- [ ] Production asynchronous KV transport
- [ ] Streaming generation and online dynamic batching
- [ ] SGLang scheduler integration

## Foundation

The runtime is derived from [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) and integrates model paths from [Dream](https://github.com/HKUNLP/Dream) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM).
