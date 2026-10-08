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
7. **Continue (answers longer than 32 tokens).** The canvas is denoised one
   32-token block at a time against the same prompt KV: the last position of a
   finished block predicts the first token of the next one.

## Status

| Piece | State |
| --- | --- |
| Dual-cache confidence-threshold decoding (`PrefillingDream`) | done |
| Shared draft (partial draft rounds) | done |
| Query-conditioned chunk scoring mask | done, `flashinfer` and `torch_native` |
| Scoring, drafting, and generation on one server | done |
| One call per answer (pipeline service and library outside SGLang) | done |
| Full-prompt prefill with sparse RoPE positions | done |
| Per-head token eviction, bidirectional score | done |
| Multi-block generation (`max_new_tokens` above the block size) | done |
| YaRN RoPE scaling (x64, 128K positions) | done |
| HTTP 400 for malformed dLLM request parameters | done |
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

Throughput of the same 150 requests under concurrent clients (`flashinfer`,
`--enable-deterministic-inference`, requests per second):

| Deployment | 1 client | 4 clients | 8 clients | 16 clients |
| --- | ---: | ---: | ---: | ---: |
| One server, one H20 | 0.89 | 1.54 | 1.61 | 1.62 |
| Two independent servers, two H20s | | 2.33 | 2.97 | 3.17 |

Outputs are identical across all of these runs.

Prefill-decode disaggregation needs more prefill servers than decode servers:
with about three prefill servers per decode server it matches or slightly
exceeds the same number of independent servers (measured on H100s), and it
falls behind when the two pools are the same size.

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

### Pipeline service

`prefilling_dllm/` puts the whole method behind one call: a document and a
question go in, the answer comes out. It sits outside the SGLang package and
holds no model; it tokenizes, then sends the draft, chunk-scoring, and
generation requests to the server above. Run it from the repository root (or
put the root on `PYTHONPATH`); it needs only `transformers` and `msgspec`.

```bash
python -m prefilling_dllm.server --model-path $MODEL \
  --base-url http://127.0.0.1:30000 --port 8080
```

```bash
curl -s http://127.0.0.1:8080/answer -H 'Content-Type: application/json' \
  -d '{"context": "<long document>", "question": "<question>"}'
```

The response carries `answer`, the selected chunk indices and their scores,
token counts, and the seconds spent drafting, scoring, and generating. An
optional `template` field with `{context}` and `{question}` slots replaces
the default prompt. `--token-capacity 512` turns on per-head token eviction;
`--chunk-size`, `--top-k`, `--draft-tokens`, and `--score-batch-size` default
to 1024, 4, 4, and 8.

The same pipeline as a library:

```python
from transformers import AutoTokenizer

from prefilling_dllm import PrefillingDreamPipeline, SGLangClient

pipeline = PrefillingDreamPipeline(
    tokenizer=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True),
    client=SGLangClient("http://127.0.0.1:30000", timeout=600),
)
print(pipeline.answer(context=document, question=question).answer)
```

### YaRN RoPE scaling

The reference's 128K setting scales Dream's RoPE with YaRN. Add this to the
server command (every server of a PD deployment):

```bash
--json-model-override-args '{"rope_scaling": {"rope_type": "yarn", "factor": 64.0, "original_max_position_embeddings": 131072, "max_cached_positions": 131072}}'
```

`max_cached_positions` keeps the precomputed cos/sin table at 131072 rows
instead of `original_max_position_embeddings * factor`; the table still grows
on demand.

### Benchmark client

The LongBench client runs the same steps over a dataset, with batching across
examples and the experimental modes, and scores the answers:

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
PD. The prefill server does the draft prompt pass, the eviction scoring
forwards, the full-prompt pass, and per-head compaction; it then sends the
prompt KV and the first generated token to the decode server, which runs the
denoising rounds. An answer longer than one block also takes its canvas KV
along, which the later blocks' rounds read. Either server can answer
chunk-scoring requests.

```bash
DLLM="--model-path $MODEL --trust-remote-code --attention-backend torch_native \
  --dtype bfloat16 --disable-cuda-graph --disable-radix-cache \
  --chunked-prefill-size -1 \
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
`--score-base-url http://127.0.0.1:30020 --score-on-pd-server`. Scoring requests
go straight to one PD server and finish there; sending them to the decode
server keeps the heavy prompt forwards off the prefill server. `--mini-lb` is
required for now because the Rust router drops `sampling_params.custom_params`.
The pipeline service takes the same three flags.

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

- A canvas longer than one block needs `dual_cache` and `--dllm-fdfo`. Drafts
  are 4 slots and `dllm_parallelcomp` generates one block.
- Scoring mask and eviction need the `flashinfer` or `torch_native` attention
  backend, a disabled radix cache, and a scoring row that fits one forward
  (`--chunked-prefill-size -1` is the simple way).
- Token eviction needs `--dllm-fdfo` (the default), tensor parallel size 1,
  and KV page size 1.
- Every measured run used `--disable-cuda-graph`.
- SGLang always disables the overlap scheduler for dLLM servers.
- `flashinfer` with `--enable-deterministic-inference` needs about 0.85 GB of
  workspace per 4K-token prompt prefilled in the same forward. The default 2 GB
  overflows at three concurrent long prompts; set
  `SGLANG_FLASHINFER_WORKSPACE_SIZE` (bytes) higher for concurrent load.
- `torch_native` has no deterministic mode: a few outputs change with the
  batch a request happens to share under concurrent load.

## TODO

Roughly in priority order:

- [ ] **Prefill-decode disaggregation (in progress).** Drafts and the final
      generation request now run on stock SGLang PD. The prefill server does
      the draft prompt pass, the eviction scoring forwards, the full-prompt
      pass, and per-head compaction, then hands the prompt KV (plus the
      canvas KV of a multi-block answer) and the first canvas token to the
      decode server, which resumes at draft suffix
      initialization or directly in dual-cache denoising; chunk-scoring requests
      finish on whichever PD server they are sent to. Tested layout: prefill server on one H20,
      decode server on another, NIXL transfer, `launch_router --mini-lb`.
      Predictions, selected chunks, and drafts are identical to the
      single-server pipeline, with and without eviction, for one-block and
      two-block answers. Still open:
  - `dllm_parallelcomp` has no handoff;
  - the Rust `sglang_router` drops `sampling_params.custom_params`, which
    carries the sparse position offset, the draft config, and the eviction
    spans, so use `launch_router --mini-lb` until the router forwards it;
  - decode-side retraction and the decode radix cache are untested; an abort
    storm (connections dropped at every phase) leaves both servers healthy and
    leak-free, but it did not cover draft or scoring requests;
  - throughput depends on the prefill-to-decode ratio (see Speed); the
    ratio sweep covered final generation requests only, without scoring,
    drafts, or eviction.
- [ ] **Speed.** Eviction scores one chunk per forward and roughly halves
      throughput; every run so far used `--disable-cuda-graph`.
- [ ] **Backend consistency.** `flashinfer` and `torch_native` do not produce
      identical outputs on identical inputs; the cause is not isolated.
- [ ] **More attention backends** for the scoring mask (Triton, FA3).
- [ ] **Token eviction with tensor parallelism** or a KV page size above 1.
- [ ] **LLaDA / UltraLLaDA.**

## Acknowledgment

Built on [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0); the
Dream decoding logic follows Fast-dLLM and the Prefilling-dLLM reference
implementation.
