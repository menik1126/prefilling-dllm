# Prefilling-dLLM Evaluation

This directory contains the evaluation code for Prefilling-dLLM with Dream and
Fast-dLLM long-context experiments. Historical backups, debug
scripts, queue files, local logs, and generated result folders are intentionally
excluded from the open-source tree.

## Main Entry Points

- `eval_fastdllm_parallelcomp_longbench.py`: LongBench evaluation for the
  Fast-dLLM + ParallelComp KV runtime.
- `eval_fastdllm_parallelcomp_infinitebench.py`: InfiniteBench evaluation for
  the same runtime.
- `eval_fastdllm_parallelcomp_scbench.py`: SCBench variable tracing evaluation.
- `fastdllm_parallelcomp.py`: benchmark-agnostic runtime implementation:
  chunk splitting, query-conditioned chunk scoring, optional token eviction,
  compressed/full-prompt KV construction, and Fast-dLLM block decoding.
- `eval_fastdllm_v1_longbench.py` and `eval_fastdllm_v1_infinitebench.py`:
  Fast-dLLM v1 baselines without the full ParallelComp KV path.
- `fastdllm_v1_model.py`: loader/wrapper for the upstream Fast-dLLM v1 Dream
  implementation.

General Dream and Prefilling-dLLM evaluation helpers are kept in `eval_dream.py`,
`eval_infinitebench.py`, `eval_longbench_dream.py`, `prefilling_model.py`, and
`infinitebench_tasks.py`.

## Best LongBench Configuration

The main LongBench `multifieldqa_en` configuration used by the cleaned
ParallelComp path is:

```shell
CUDA_VISIBLE_DEVICES=0 python eval_fastdllm_parallelcomp_longbench.py \
  --pretrained ./model_weights/Dream-v0-Base-7B \
  --fastdllm_dream_dir /path/to/Fast-dLLM/v1/dream \
  --data_dir /path/to/LongBench \
  --config_dir /path/to/longbench_config \
  --tasks multifieldqa_en \
  --max_examples 0 \
  --run_name fastdllm_parallelcomp_mfen_best47 \
  --output_dir results_longbench_fastdllm_parallelcomp_mfen_best47 \
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

Set `--score_batch_size 1` to reproduce the old serial chunk-scoring schedule.
Values larger than one batch independent `single_chunk` scoring forwards and
are the recommended open-source default.

## InfiniteBench

Use the same runtime for InfiniteBench:

```shell
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

## Local Artifacts

The following are intentionally ignored by git:

- model weights under `prefilling_dllm_eval/model_weights/`
- benchmark datasets and local prepared data
- `results*`, `logs`, task queues, PID files, and Python caches
- local external checkouts used during experiments

Commit source code and documentation only. Generated metrics should be reported
in papers or release notes rather than checked into the repository.
