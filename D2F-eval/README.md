# D2F Evaluation Entry Points

This directory contains both the original D2F evaluation scripts and the
ParallelComp-style Dream long-context experiments. If you just cloned this
branch and want the current experiment path, start here:

```shell
cd D2F-eval
python eval_infinitebench.py ...
```

## Main Files

- `eval_infinitebench.py`: main entry for InfiniteBench. This is the current
  script for validating long-context compression on Dream/D2F.
- `eval_dream.py`: Dream/D2F runtime. The ParallelComp-style cache compression
  logic lives here, including query-conditioned context cache rebuild and
  token eviction.
- `d2f_model.py`: thin model wrapper used by the evaluation entry points.
- `infinitebench_tasks.py`: InfiniteBench loading, prompting, answer extraction,
  and metrics.
- `eval_d2f_plcc.py`: PLCC/code-completion entry point. Keep this for PLCC, but
  do not use it as the first smoke test for the InfiniteBench branch.
- `eval_dream.sh` and `eval_llada.sh`: original benchmark shell wrappers.
- `debug_*`, `analyze_*`, `queue_*`, `*.bak_*`: debugging, queued runs, or
  historical snapshots. They are not the default scripts to run.

## mllm Environment

The paths below match the current mllm setup. Adjust them if you run elsewhere.

```shell
export REPO=/home/ma-user/work/Discrete-Diffusion-Forcing
export PYTHON=/home/ma-user/work/conda-envs/d2f_eval_parallelcomp/bin/python
export DREAM_BASE=$REPO/D2F-eval/model_weights/Dream-v0-Base-7B
export DREAM_LORA=$REPO/D2F-eval/model_weights/D2F_Dream_Base_7B_Lora
export INFINITEBENCH_DATA=/home/ma-user/work/InfiniteBench/data

cd $REPO/D2F-eval
```

## Recommended Smoke Tests

Run `passkey` first. This verifies that pre-runtime chunk selection,
query-conditioned cache rebuild, token eviction, and query replay are all wired
together.

```shell
CUDA_VISIBLE_DEVICES=0 $PYTHON eval_infinitebench.py \
  --model_type dream \
  --pretrained $DREAM_BASE \
  --lora_path $DREAM_LORA \
  --data_dir $INFINITEBENCH_DATA \
  --tasks passkey \
  --max_examples 5 \
  --max_length 131072 \
  --max_new_tokens 32 \
  --block_size 32 \
  --temperature 0 \
  --prompt_style parallelcomp_raw \
  --parallelcomp_mode \
  --parallelcomp_pre_runtime_mode \
  --parallelcomp_cache_compress_mode \
  --parallelcomp_chunk_size 1024 \
  --parallelcomp_topk_chunks 3 \
  --parallelcomp_min_prompt_tokens 1 \
  --parallelcomp_keep_first_chunk \
  --parallelcomp_token_capacity 128 \
  --parallelcomp_token_keep_min 32 \
  --parallelcomp_fixed_query_text "Please answer the question using the long context above." \
  --output_dir results_infinitebench_smoke_passkey5_cap128
```

Then run `number_string`. It is more sensitive to exact copying, so use it as a
second smoke rather than the first pipeline check.

```shell
CUDA_VISIBLE_DEVICES=0 $PYTHON eval_infinitebench.py \
  --model_type dream \
  --pretrained $DREAM_BASE \
  --lora_path $DREAM_LORA \
  --data_dir $INFINITEBENCH_DATA \
  --tasks number_string \
  --max_examples 5 \
  --max_length 131072 \
  --max_new_tokens 32 \
  --block_size 32 \
  --temperature 0 \
  --prompt_style parallelcomp_raw \
  --parallelcomp_mode \
  --parallelcomp_pre_runtime_mode \
  --parallelcomp_cache_compress_mode \
  --parallelcomp_chunk_size 1024 \
  --parallelcomp_topk_chunks 3 \
  --parallelcomp_min_prompt_tokens 1 \
  --parallelcomp_keep_first_chunk \
  --parallelcomp_token_capacity 512 \
  --parallelcomp_token_keep_min 32 \
  --parallelcomp_fixed_query_text "Please answer the question using the long context above." \
  --output_dir results_infinitebench_smoke_numberstring5_cap512
```

## Current Branch Expectations

The current main InfiniteBench path is:

1. Select long-context chunks before the Dream runtime.
2. Rebuild each kept context block cache with the real task query/tail attached.
3. Keep only context-token KV from that rebuild.
4. Apply token eviction inside the kept context blocks.
5. Replay the query/tail with full visibility against the compressed cache.

Recent 5-example smoke results on mllm:

- `passkey`, cap 128: `5/5`
- `passkey`, cap 512: `5/5`
- `passkey`, cap 1024: `5/5`
- `number_string`, cap 128: `2/5`
- `number_string`, cap 256: `1/5`
- `number_string`, cap 512: `3/5`

The outer pre-runtime compression path, without inner cache compression, reached
`20/20` on both `passkey` and `number_string` with keep-first enabled. The
remaining research issue is the inner Dream cache-compression/token-eviction
path, especially exact-copy behavior on `number_string`.

## PLCC

Use `eval_d2f_plcc.py` for code-completion experiments. It shares the Dream
runtime implementation in `eval_dream.py`, but it is a separate benchmark path.

```shell
CUDA_VISIBLE_DEVICES=0 python eval_d2f_plcc.py \
  --model_type dream \
  --pretrained /path/to/Dream-v0-Base-7B \
  --configs medium_context \
  --top_percent 30 \
  --max_length 32768 \
  --max_new_tokens 128 \
  --parallelcomp_cache_compress_mode \
  --parallelcomp_chunk_size 1024 \
  --parallelcomp_topk_chunks 3 \
  --parallelcomp_token_capacity 256 \
  --parallelcomp_token_keep_min 32 \
  --output_dir results_plcc_parallelcomp
```

## Output Files

Evaluation outputs go under `results*` directories. Local model weights, logs,
and result folders are ignored by git; commit code and docs, not generated
artifacts.
