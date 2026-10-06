# Prefilling-dLLM on SGLang

This branch serves [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B),
a diffusion language model, with the Prefilling-dLLM long-context method on
top of [SGLang](https://github.com/sgl-project/sglang). The reference
implementation (Hugging Face runtime and the standalone engine) lives on the
[`main`](https://github.com/menik1126/prefilling-dllm/tree/main) branch of this
repository; this branch is the SGLang port.

It is a fork of SGLang. Everything outside the dLLM path behaves as upstream;
see the [upstream README](https://github.com/sgl-project/sglang#readme) and
[documentation](https://docs.sglang.io/) for SGLang itself.

## What the method does

A long document does not go through the model at every denoising step.
Instead, for one request:

1. **Draft.** Generate a short draft answer (4 slots, 1 partial denoising
   round) from `prefix + query` alone.
2. **Score chunks.** Split the document into 1024-token chunks and score each
   with one forward over `prefix + chunk + query + draft`: the score is the mean
   log-probability of the query and the confirmed draft tokens.
3. **Select.** Keep the top-4 chunks.
4. **Prefill once.** Run one bidirectional pass over
   `prefix + selected chunks + query + 32 mask tokens` and cache the prompt KV.
5. **Evict (optional).** Each KV head of each layer keeps only its top-N tokens
   of every chunk.
6. **Denoise.** Every later step recomputes only the 32 generation positions
   against the cached prompt KV (dual cache), accepting tokens whose confidence
   reaches 0.9.

## Status

| Piece | State |
| --- | --- |
| Dual-cache confidence-threshold decoding (`PrefillingDream`) | done |
| Shared draft (partial draft rounds) | done |
| Query-conditioned chunk scoring mask | done, `flashinfer` and `torch_native` |
| Scoring, drafting, and generation on one server | done |
| Full-prompt prefill with sparse RoPE positions | done |
| Per-head token eviction, bidirectional score | done |
| Prefill-decode disaggregation on stock SGLang PD | drafts and generation done; see TODO |
| Server-side chunk prefill with whole-token eviction (`--server-chunk-prefill`) | works, accuracy not aligned |

## Speed

Caching the prompt KV means each denoising step computes only the 32 generation
positions instead of the whole prompt. LongBench MultiFieldQA-en, 150 requests
sent one at a time, about 3.7K prompt tokens each after chunk selection, one
H20, BF16, CUDA graphs off. Times cover the final generation request only, not
drafting or chunk scoring.

| Attention backend | Recompute every step | Cached prompt KV | Speedup |
| --- | ---: | ---: | ---: |
| `flashinfer` | 10.84 s / request | 1.12 s / request | 9.7x |
| `torch_native` | 9.87 s / request | 2.27 s / request | 4.4x |

"Recompute every step" is the same server launched with `dual_cache: false`.

## Running it

All commands assume the repository's `python/` directory is on `PYTHONPATH`
and the Dream weights are at `$MODEL`.

### One server

A `PrefillingDream` server handles all three request kinds: chunk scoring
(answered as one ordinary forward that returns prompt log-probabilities),
drafts, and generation.

```bash
python -m sglang.launch_server --model-path $MODEL --trust-remote-code \
  --attention-backend torch_native --dtype bfloat16 \
  --disable-cuda-graph --disable-radix-cache --chunked-prefill-size -1 \
  --dllm-algorithm PrefillingDream \
  --dllm-algorithm-config benchmark/dllm/prefilling_dream_longbench.yaml \
  --mem-fraction-static 0.60 --port 30000
```

### Benchmark client

The client does the chunking, drafting, scoring requests, selection, and
prompt assembly:

```bash
python benchmark/dllm/longbench/bench_multifieldqa_chunk_selection.py \
  --base-url http://127.0.0.1:30000 --model-path $MODEL \
  --data-path multifieldqa_en.jsonl --prompt-config dataset2prompt.json \
  --output-dir out \
  --selection-mode query_logprob --score-attention-mask full \
  --position-mode continuous --query-position-mode after_selected_chunks \
  --chunk-size 1024 --top-k 4 --score-batch-size 8 --draft-tokens 4 \
  --num-examples 150
```

Add `--token-capacity 512` for per-head token eviction
(`--token-score-direction query_to_chunk` scores without the chunk-to-query
term).

### Prefill-decode disaggregation

Prefill and denoising can run on separate servers through SGLang's built-in
PD. The prefill server answers chunk-scoring requests and does the draft prompt
pass, the eviction scoring forwards, the full-prompt pass, and per-head
compaction; it then sends the prompt KV and the first generated token to the
decode server, which runs the denoising rounds.

```bash
DLLM="--model-path $MODEL --trust-remote-code --attention-backend torch_native \
  --dtype bfloat16 --disable-cuda-graph --disable-radix-cache \
  --chunked-prefill-size -1 --disable-overlap-schedule \
  --disaggregation-transfer-backend nixl --mem-fraction-static 0.55 \
  --dllm-algorithm PrefillingDream \
  --dllm-algorithm-config benchmark/dllm/prefilling_dream_longbench.yaml"

# GPU 0: prefill server
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server $DLLM \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 30008 \
  --port 30010
# GPU 1: decode server
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server $DLLM \
  --disaggregation-mode decode --port 30020

python -m sglang_router.launch_router --mini-lb --pd-disaggregation \
  --prefill http://127.0.0.1:30010 30008 --decode http://127.0.0.1:30020 \
  --port 30000
```

Run the same client with `--base-url http://127.0.0.1:30000` (the router, for
drafts and generation) and
`--score-base-url http://127.0.0.1:30010 --score-on-prefill-server` (scoring
requests go straight to the prefill server and finish there). `--mini-lb` is
required for now because the Rust router drops `sampling_params.custom_params`.

## Request parameters

The dLLM-specific inputs travel in `sampling_params.custom_params`:

| Key | Used for |
| --- | --- |
| `dream_score_attention_mask`, `dream_score_prefix_len`, `dream_score_chunk_len`, `dream_score_query_len`, `dream_score_draft_len` | scoring mask of one `prefix + chunk + query + draft` row |
| `dllm_partial_draft` | draft generation |
| `dllm_position_start`, `dllm_position_offset` | RoPE gap after a selected chunk shorter than the chunk size |
| `dllm_token_eviction` | per-head eviction: `capacity`, `prefix_len`, `chunk_lens`, `query_len` |
| `dllm_parallelcomp` | server-side chunk prefill |

## Limits

- One generation block: `max_new_tokens == block_size == 32`.
- Scoring mask and eviction need the `flashinfer` or `torch_native` attention
  backend, a disabled radix cache, and a scoring row that fits one forward
  (`--chunked-prefill-size -1` is the simple way).
- Token eviction needs `--dllm-fdfo` (the default), tensor parallel size 1,
  and KV page size 1.
- Every measured run used `--disable-cuda-graph`.
- Malformed custom parameters raise inside the scheduler instead of returning
  HTTP 400.

## TODO

Roughly in priority order:

- [ ] **Prefill-decode disaggregation (in progress).** Drafts and the final
      generation request now run on stock SGLang PD. The prefill server does
      the draft prompt pass, the eviction scoring forwards, the full-prompt
      pass, and per-head compaction, then hands the prompt KV and the first
      canvas token to the decode server, which resumes at draft suffix
      initialization or directly in dual-cache denoising; chunk-scoring requests
      finish on the prefill server. Tested layout: prefill server on one H20,
      decode server on another, NIXL transfer, `launch_router --mini-lb`.
      Predictions, selected chunks, and drafts are identical to the
      single-server pipeline, with and without eviction. Still open:
  - `dllm_parallelcomp` has no handoff;
  - the Rust `sglang_router` drops `sampling_params.custom_params`, which
    carries the sparse position offset, the draft config, and the eviction
    spans, so use `launch_router --mini-lb` until the router forwards it;
  - only tested with `--disable-overlap-schedule`; decode-side retraction,
    decode radix cache, and abort cleanup are untested;
  - no throughput measurement yet.
- [ ] **YaRN x64 (128K) RoPE** for Dream, the paper's main-table setting; only
      native RoPE has been run.
- [ ] **Multi-block generation.** Everything assumes
      `max_new_tokens == block_size == 32`.
- [ ] **Server-side orchestration.** Chunking, drafting, chunk scoring, top-k
      selection, and prompt assembly live in the benchmark client: one answer
      still takes several requests to the server.
- [ ] **Speed.** Only sequential single-request latency is measured; no
      concurrent throughput numbers, eviction scores one chunk per forward, and
      every run so far used `--disable-cuda-graph`.
- [ ] **Backend consistency.** `flashinfer` and `torch_native` do not produce
      identical outputs on identical inputs; the cause is not isolated.
- [ ] **More attention backends** for the scoring mask (Triton, FA3).
- [ ] **Token eviction with tensor parallelism** or a KV page size above 1.
- [ ] **Request validation.** Malformed custom params raise inside the scheduler
      instead of returning HTTP 400.
- [ ] **LLaDA / UltraLLaDA.**

## Acknowledgment

Built on [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0); the
Dream decoding logic follows Fast-dLLM and the Prefilling-dLLM reference
implementation.
