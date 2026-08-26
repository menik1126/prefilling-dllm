# Engine overview

Prefilling-dLLM extends a compact vLLM-style runtime with native prefill–decode disaggregation for diffusion language models.

## Prefill stage

The engine divides a long prefix into chunks, builds paged KV representations once, predicts query relevance, and retains selected chunks and tokens. The resulting state is represented by a prefill record and can be exported as a transferable KV snapshot.

## PD boundary

A decode worker reconstructs local page ownership from the snapshot without re-running prefix computation. The boundary is exposed through:

- `prefill_to_pd_record`
- `snapshot_pd_record_kv`
- `make_pd_record_from_kv_snapshot`
- `make_pd_record_from_remote_engine`

## Decode stage

The selected prefix KV remains fixed while query and response tokens are recomputed during iterative denoising. Non-contiguous paged-KV attention lets the response attend directly to retained chunks without gathering them into a dense cache first.

## Runtime components

- Scheduler and sequence state for diffusion decoding
- Paged prompt and response KV allocation
- Prefix snapshot load/store kernels
- Predictive chunk scoring and per-head token eviction
- Tensor-parallel and data-parallel execution paths
