Legacy Dream Prefill-KV Path
============================

This directory keeps the older pure-Dream prefill-KV experiment code for
result reproduction only.

The current main path for ParallelComp on Fast-DLLM v1 is:

- `../fastdllm_parallelcomp.py`
- `../eval_fastdllm_parallelcomp_longbench.py`
- `../eval_fastdllm_parallelcomp_infinitebench.py`

The legacy controller here uses a different chunk-cache semantic in its
old `sequential` mode: later chunks are built with earlier chunk KV in
`past_key_values`, so `A2` can attend to `A1`, while `A1` is not recomputed
after `A2` appears.

The default has been changed to `--chunk_cache_mode independent`, where each
selected chunk is processed independently with the query and concatenated
afterwards.  `--chunk_cache_mode joint_selected` is a diagnostic mode that
full-forwards all selected chunks plus the temporary query together, closer to a
Fast-DLLM prompt-cache construction, and then drops the temporary query KV.
