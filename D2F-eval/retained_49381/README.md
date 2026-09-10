# Retained Dream MF-en baseline (49.381 F1)

Frozen runtime snapshot validated on ci-h20, 2026-09-10. This directory is isolated from the repository's default runtime to preserve the exact evaluated implementation. Older experimental branches inside the snapshot are inactive; use the pinned launcher, not arbitrary environment switches.

```bash
# Inside the existing prefilling-dllm-exp container, with an available GPU:
bash D2F-eval/retained_49381/run_retained.sh 0 /results/NEW_RUN_DIRECTORY
```

The output directory must not already exist. Set `REFERENCE_ROOT` (contains fastdllm_dream, data, longbench_config, deps) and `MODEL_PATH` to use other installations. Dependencies/model/data are not bundled. The defaults match the validated container. This launcher does not manage GPUs or stop existing workloads; check availability first.

## Pinned algorithm

For every candidate chunk, jointly prefill prefix+chunk+query with full attention. Retain full KV and final query logits to directly generate four draft tokens, avoiding a second query prefill. Reuse pristine prefix+chunk KV for a separate causal query+draft scoring forward. Average target log-probabilities over all query+draft tokens, not a 50/50 group average. The first query token uses saved last-context logits. Because cached context already contains query information, this is not a strict causal query likelihood.

Select top four 1024-token chunks, preserving source order. Recompute prefix+selected chunks+query+32 masks using full_prompt_mask and continuous positions, then generate the final answer with confidence-threshold diffusion (threshold 0.9, temperature 0, BF16). No shared prefix and no online draft-only scoring.

## Validation

- multifieldqa_en: 150 unique examples, no empty predictions, normal exit.
- F1: **49.381386725784246** (0–100 scale).
- Merged query prefill: 443.735762 s cumulative inference vs 467.690037 s separate; 5.12% reduction in that paired run. Single-run, dual-GPU timing; not a general performance guarantee.
- Predictions identical in that comparison; one sample's selected chunks differed and 13 had score differences.
- Original public-draft baseline: 48.170469296953144; this is a single-task ablation result, not a global LongBench SOTA claim.

Source artifact: `/results/merged_query_prefill_20260910/D2F-eval`.
Result artifact: `/results/merged_query_prefill_20260910/merged/multifieldqa_en/merged_metrics.json`.
Only runtime files and aggregate metrics are retained here; dataset samples and prediction dumps are excluded.
