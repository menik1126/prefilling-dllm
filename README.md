# Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models

<p align="center">
  <strong>🎉 Accepted to EMNLP 2026</strong>
</p>

<p align="center">
  <strong>The first dLLM inference engine with native prefill–decode (PD) disaggregation.</strong>
</p>

<p align="center">
  Jing Xiong · Qi Han · Shansan Gong · Yunta Hsieh · Chengyue Wu · Chaofan Tao · Chenyang Zhao · Ngai Wong
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.10537"><b>Paper</b></a> ·
  <a href="https://arxiv.org/pdf/2606.10537"><b>PDF</b></a> ·
  <a href="https://github.com/menik1126/prefilling-dllm"><b>Code</b></a>
</p>

## Overview

Diffusion large language models (dLLMs) repeatedly encode the full input prefix during iterative denoising. This redundant computation becomes increasingly expensive as the context grows.

**Prefilling-dLLM** is a training-free prefill-decode disaggregation framework for long-context dLLM inference. It computes prefix KV representations once, selects the context most relevant to the query, and reuses the resulting sparse KV cache throughout decoding.

The framework combines:

- **Chunked prefilling:** partition the long prefix into independent chunks and prefill them once.
- **Predictive chunk selection:** score chunks using the query and retain the top-*K* most relevant chunks.
- **Intra-chunk sparsity:** optionally keep only the most useful tokens within each selected chunk.
- **Disaggregated decoding:** reuse the fixed prefix KV cache while recomputing only the query and denoising response.
- **Non-contiguous KV attention:** decode directly over selected chunk KV without first gathering it into a dense contiguous cache.

## Highlights

- The first inference engine to separate dLLM prefill and decode into explicit execution stages.
- Training-free acceleration for existing diffusion language models.
- Prefill cost scales linearly with total prefix length when the chunk size is fixed.
- Decode cost is decoupled from the original full-context length.
- Evaluated on **LongBench** and **InfiniteBench** with Dream-7B, UltraLLaDA, and Fast-dLLM variants.
- Achieves **9.1–28.0× speedup** at **8K–32K** context lengths in the paper's experiments.
- Periodic chunk-level BOS tokens act as attention anchors and mitigate the lost-in-the-middle problem.

## How it works

Given a prefix of length $L_p$, Prefilling-dLLM divides it into $N$ chunks of size $C$. Each chunk is prefixed with a BOS token and processed independently:

```text
Long prefix
    │
    ├── Chunk 1 + BOS ──► KV cache 1 ──┐
    ├── Chunk 2 + BOS ──► KV cache 2 ──┼──► Predictive top-K selection
    ├── ...                            │              │
    └── Chunk N + BOS ──► KV cache N ──┘              ▼
                                               Sparse prefix KV
                                                      │
Query + masked response ──► iterative denoising ◄─────┘
```

During decoding, the selected prefix KV remains fixed. Query and response tokens are recomputed at each denoising step and attend to both the sparse prefix cache and the current response state.

The resulting complexity changes from repeatedly processing the full context,

```text
O((L_p + L_d)^2 · T)
```

to chunked prefilling followed by sparse decoding,

```text
O(N · C^2 + (L_d^2 + K · B) · T)
```

where $L_d$ is the decode length, $T$ is the number of denoising steps, and $B$ is the retained token budget per selected chunk.

## Repository layout

```text
.
├── prefilling_dllm_eval/
│   ├── fastdllm_parallelcomp.py
│   ├── eval_fastdllm_parallelcomp_longbench.py
│   ├── eval_fastdllm_parallelcomp_infinitebench.py
│   └── eval_fastdllm_parallelcomp_scbench.py
├── prefilling_dllm_engine/
│   └── prefilling_dllm/fastdllm_engine.py
├── docs/
├── requirements.txt
└── pyproject.toml
```

The main research implementation is in [`prefilling_dllm_eval/fastdllm_parallelcomp.py`](prefilling_dllm_eval/fastdllm_parallelcomp.py). It implements chunk splitting, query-conditioned scoring, top-*K* selection, optional token eviction, prefix KV construction, and block-wise Fast-dLLM decoding.

The [`prefilling_dllm_engine`](prefilling_dllm_engine/) directory contains the paged-KV inference runtime. It exposes native prefill/decode execution boundaries and transferable KV snapshot APIs, allowing prefix KV to be produced by prefill workers and consumed by decode workers. See [`prefilling_dllm_eval/README.md`](prefilling_dllm_eval/README.md) for additional experiment notes.

## Installation

```bash
git clone https://github.com/menik1126/prefilling-dllm.git
cd prefilling-dllm

python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download the required model weights and benchmark data separately. The current scripts use Dream/Fast-dLLM model code and expect their locations through command-line arguments.

## Evaluation

### LongBench

```bash
cd prefilling_dllm_eval

CUDA_VISIBLE_DEVICES=0 python eval_fastdllm_parallelcomp_longbench.py \
  --pretrained ./model_weights/Dream-v0-Base-7B \
  --fastdllm_dream_dir /path/to/Fast-dLLM/v1/dream \
  --data_dir /path/to/LongBench \
  --config_dir /path/to/longbench_config \
  --tasks multifieldqa_en \
  --max_examples 0 \
  --run_name prefilling_dllm_longbench \
  --output_dir results_longbench_prefilling_dllm \
  --max_new_tokens 32 \
  --max_length 4096 \
  --block_length 32 \
  --temperature 0 \
  --alg confidence_threshold \
  --threshold 0.9 \
  --dtype bfloat16 \
  --chunk_size 1024 \
  --topk_chunks 4 \
  --chunk_bos \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_rounds 1 \
  --score_context_mode single_chunk \
  --score_attention_mask causal \
  --score_batch_size 8 \
  --cache_build_mode full_prompt_mask \
  --token_capacity 0 \
  --chunk_position_mode continuous \
  --query_position_mode after_selected_chunks
```

### InfiniteBench

```bash
cd prefilling_dllm_eval

CUDA_VISIBLE_DEVICES=0 python eval_fastdllm_parallelcomp_infinitebench.py \
  --pretrained ./model_weights/Dream-v0-Base-7B \
  --fastdllm_dream_dir /path/to/Fast-dLLM/v1/dream \
  --data_dir /path/to/InfiniteBench/data \
  --tasks passkey number_string kv_retrieval code_debug \
  --max_examples 0 \
  --prompt_style parallelcomp_raw \
  --max_new_tokens 32 \
  --max_length 4096 \
  --block_length 32 \
  --dtype bfloat16 \
  --chunk_size 1024 \
  --topk_chunks 4 \
  --score_mode draft_self_information \
  --score_draft_tokens 4 \
  --score_draft_partial_rounds 1 \
  --score_batch_size 8 \
  --cache_build_mode full_prompt_mask \
  --chunk_position_mode continuous \
  --query_position_mode after_selected_chunks
```

Use `--topk_chunks` to control inter-chunk sparsity and `--token_capacity` to enable a fixed token budget within selected chunks. Paths and generation settings should be adjusted for the local environment and model checkpoint.

## Release status

- [x] Dream/Fast-dLLM long-context evaluation runtime
- [x] Query-conditioned chunk scoring
- [x] Inter-chunk sparsity
- [x] Optional intra-chunk token eviction
- [x] Paged KV cache prototype
- [x] Native prefill/decode disaggregation
- [x] Transferable KV snapshot APIs
- [ ] Production asynchronous prefill/decode transfer backend
- [ ] Streaming server and online dynamic batching
- [ ] SGLang integration

## Citation

If you find this work useful, please cite:

```bibtex
@article{xiong2026prefillingdllm,
  title   = {Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models},
  author  = {Xiong, Jing and Han, Qi and Gong, Shansan and Hsieh, Yunta and Wu, Chengyue and Tao, Chaofan and Zhao, Chenyang and Wong, Ngai},
  journal = {arXiv preprint arXiv:2606.10537},
  year    = {2026}
}
```

The citation will be updated when the EMNLP 2026 proceedings entry becomes available.

## Acknowledgements

This repository builds on the open-source ecosystems around [Dream](https://github.com/HKUNLP/Dream), [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), and [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

## License

See [`LICENCE`](LICENCE) for license details.
