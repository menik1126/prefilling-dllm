import argparse
import json
import os
import types
from pathlib import Path

import torch

import eval_dream as eval_dream_module
from d2f_model import generate, load_model
from infinitebench_tasks import (
    create_prompt_parts,
    load_task_examples,
    normalize_answer_label,
    score_prediction,
)


DEFAULT_PRETRAINED = (
    "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/"
    "model_weights/Dream-v0-Base-7B"
)
DEFAULT_LORA = (
    "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval/"
    "model_weights/D2F_Dream_Base_7B_Lora"
)
DEFAULT_DATA_DIR = "/home/ma-user/work/InfiniteBench/data"


def find_subsequence_positions(values, pattern):
    if not pattern:
        return []
    width = len(pattern)
    return [
        list(range(start, start + width))
        for start in range(0, len(values) - width + 1)
        if values[start:start + width] == pattern
    ]


def coerce_answer_text(answer_label):
    if isinstance(answer_label, list):
        return str(answer_label[0])
    return str(answer_label)


def inject_required_indices(index_tensor, required_indices):
    if not required_indices:
        return index_tensor
    required = sorted({int(idx) for idx in required_indices})
    rows = []
    keep_count = int(index_tensor.shape[1])
    for row in index_tensor:
        current = [int(idx) for idx in row.tolist()]
        merged_set = set(current)
        merged_set.update(required)
        if len(merged_set) <= keep_count:
            merged = sorted(merged_set)
        else:
            required_kept = required[:keep_count]
            required_set = set(required_kept)
            remaining_slots = keep_count - len(required_kept)
            non_required = [idx for idx in current if idx not in required_set]
            merged = sorted(required_kept + non_required[:remaining_slots])
        rows.append(torch.tensor(merged, device=index_tensor.device, dtype=index_tensor.dtype))
    return torch.stack(rows, dim=0)


def install_baseline(inner, originals):
    inner._select_cache_block_token_indices_per_layer_per_head = types.MethodType(
        originals["per_head"],
        inner,
    )


def install_answer_protect(inner, originals):
    def patched(self, block_ids, query_ids, num_cache_heads_per_layer):
        keep_indices, evicted = originals["per_head"](
            self,
            block_ids,
            query_ids,
            num_cache_heads_per_layer,
        )
        answer_ids = getattr(self, "_debug_current_answer_ids", [])
        protected = []
        for span in find_subsequence_positions(block_ids, answer_ids):
            protected.extend(span)
        if protected:
            self._debug_eviction_events.append(
                {
                    "mode": "answer_protect",
                    "answer_spans": find_subsequence_positions(block_ids, answer_ids),
                    "protected_count": len(set(protected)),
                }
            )
            keep_indices = [
                inject_required_indices(layer_indices, protected)
                for layer_indices in keep_indices
            ]
        return keep_indices, evicted

    inner._select_cache_block_token_indices_per_layer_per_head = types.MethodType(
        patched,
        inner,
    )


def install_prefix_anchor(inner, originals, prefix_tokens):
    def patched(self, block_ids, query_ids, num_cache_heads_per_layer):
        keep_indices, evicted = originals["per_head"](
            self,
            block_ids,
            query_ids,
            num_cache_heads_per_layer,
        )
        protected = list(range(min(int(prefix_tokens), len(block_ids))))
        self._debug_eviction_events.append(
            {
                "mode": f"prefix_anchor_{prefix_tokens}",
                "protected_prefix": len(protected),
            }
        )
        keep_indices = [
            inject_required_indices(layer_indices, protected)
            for layer_indices in keep_indices
        ]
        return keep_indices, evicted

    inner._select_cache_block_token_indices_per_layer_per_head = types.MethodType(
        patched,
        inner,
    )


def install_shared_keep_set(inner, originals):
    def patched(self, block_ids, query_ids, num_cache_heads_per_layer):
        score_layers = self._analyze_chunk_with_attentions_per_layer_per_head(
            block_ids,
            query_ids,
            num_cache_heads_per_layer,
        )
        if not score_layers:
            return originals["per_head"](
                self,
                block_ids,
                query_ids,
                num_cache_heads_per_layer,
            )

        score_parts = []
        total_layers = max(1, len(score_layers))
        for layer_idx, head_scores in enumerate(score_layers):
            if head_scores.numel() == 0:
                continue
            for head_idx in range(head_scores.shape[0]):
                score_parts.append(
                    self._apply_layer_structural_bias(
                        token_scores=head_scores[head_idx],
                        layer_idx=layer_idx,
                        num_layers=total_layers,
                    )
                )

        if not score_parts:
            return originals["per_head"](
                self,
                block_ids,
                query_ids,
                num_cache_heads_per_layer,
            )

        shared_scores = torch.stack(score_parts, dim=0).mean(dim=0)
        keep_min = min(len(block_ids), max(1, int(self.parallelcomp_token_keep_min)))
        token_capacity = max(1, int(self.parallelcomp_token_capacity))
        keep_count = min(len(block_ids), max(keep_min, token_capacity))
        selected, evicted_high = self._select_token_indices_from_scores(
            token_scores=shared_scores,
            keep_count=keep_count,
            keep_min=keep_min,
        )
        base = torch.tensor(selected, device=self.device, dtype=torch.long)
        keep_indices = []
        for num_heads in num_cache_heads_per_layer:
            keep_indices.append(base.unsqueeze(0).expand(num_heads, -1).clone())

        answer_ids = getattr(self, "_debug_current_answer_ids", [])
        answer_spans = find_subsequence_positions(block_ids, answer_ids)
        if answer_spans:
            selected_set = set(selected)
            self._debug_eviction_events.append(
                {
                    "mode": "shared_keep_set",
                    "answer_spans": answer_spans,
                    "full_answer_kept": any(
                        all(pos in selected_set for pos in span)
                        for span in answer_spans
                    ),
                }
            )
        return keep_indices, [evicted_high for _ in num_cache_heads_per_layer]

    inner._select_cache_block_token_indices_per_layer_per_head = types.MethodType(
        patched,
        inner,
    )


def install_query_conditioned_prompt(inner, originals):
    def query_conditioned_sparse_mask(
        query_block_ids,
        cached_length,
        query_prompt_window_mask=None,
        update_kvcache=0,
        device=None,
        dtype=None,
    ):
        if query_prompt_window_mask is None or update_kvcache <= 0:
            return originals["sparse_mask"](
                query_block_ids=query_block_ids,
                cached_length=cached_length,
                query_prompt_window_mask=query_prompt_window_mask,
                update_kvcache=update_kvcache,
                device=device,
                dtype=dtype,
            )
        if dtype is None:
            dtype = torch.bfloat16
        if device is None:
            device = query_block_ids.device

        query_block_ids = query_block_ids.to(device=device, dtype=torch.long)
        query_prompt_window_mask = query_prompt_window_mask.to(device=device, dtype=torch.bool)
        q_len = query_block_ids.shape[0]
        prompt_write_len = min(int(update_kvcache), q_len)
        prompt_write_mask = query_prompt_window_mask[:prompt_write_len]
        prompt_block_ids = torch.unique(query_block_ids[:prompt_write_len][prompt_write_mask])
        if prompt_block_ids.numel() <= 1:
            return originals["sparse_mask"](
                query_block_ids=query_block_ids,
                cached_length=cached_length,
                query_prompt_window_mask=query_prompt_window_mask,
                update_kvcache=update_kvcache,
                device=device,
                dtype=dtype,
            )

        tail_block_id = int(prompt_block_ids.max().item())
        k_len = cached_length + q_len
        attention_mask = torch.full((1, 1, q_len, k_len), -torch.inf, device=device, dtype=dtype)
        if cached_length > 0:
            attention_mask[:, :, :, :cached_length] = 0

        for row_idx, q_block_id in enumerate(query_block_ids.tolist()):
            if row_idx < update_kvcache and query_prompt_window_mask[row_idx]:
                if int(q_block_id) == tail_block_id:
                    visible_current = query_prompt_window_mask
                else:
                    visible_current = (query_block_ids == q_block_id) | (query_block_ids == tail_block_id)
            else:
                visible_current = query_block_ids <= q_block_id
            attention_mask[0, 0, row_idx, cached_length:][visible_current] = 0
        return attention_mask

    def query_conditioned_positions(self, state, take):
        if state.get("is_prompt_window", False) and state.get("is_tail_prompt_window", False):
            reused_window_size = max(1, int(self.parallelcomp_chunk_size or state.get("rope_span", take)))
            return torch.arange(
                reused_window_size,
                reused_window_size + take,
                device=self.device,
                dtype=torch.long,
            )
        return originals["build_positions"](self, state, take)

    eval_dream_module.build_sparse_block_attention_mask = query_conditioned_sparse_mask
    inner._build_block_runtime_positions = types.MethodType(query_conditioned_positions, inner)
    inner.parallelcomp_tail_replay_full_mask = True
    inner._debug_eviction_events.append({"mode": "query_conditioned_prompt"})


def install_prompt_full_visible_reuse(inner, originals):
    def prompt_full_visible_sparse_mask(
        query_block_ids,
        cached_length,
        query_prompt_window_mask=None,
        update_kvcache=0,
        device=None,
        dtype=None,
    ):
        if query_prompt_window_mask is None or update_kvcache <= 0:
            return originals["sparse_mask"](
                query_block_ids=query_block_ids,
                cached_length=cached_length,
                query_prompt_window_mask=query_prompt_window_mask,
                update_kvcache=update_kvcache,
                device=device,
                dtype=dtype,
            )
        if dtype is None:
            dtype = torch.bfloat16
        if device is None:
            device = query_block_ids.device

        query_block_ids = query_block_ids.to(device=device, dtype=torch.long)
        query_prompt_window_mask = query_prompt_window_mask.to(device=device, dtype=torch.bool)
        q_len = query_block_ids.shape[0]
        prompt_write_len = min(int(update_kvcache), q_len)
        prompt_write_mask = query_prompt_window_mask[:prompt_write_len]
        if torch.unique(query_block_ids[:prompt_write_len][prompt_write_mask]).numel() <= 1:
            return originals["sparse_mask"](
                query_block_ids=query_block_ids,
                cached_length=cached_length,
                query_prompt_window_mask=query_prompt_window_mask,
                update_kvcache=update_kvcache,
                device=device,
                dtype=dtype,
            )

        k_len = cached_length + q_len
        attention_mask = torch.full((1, 1, q_len, k_len), -torch.inf, device=device, dtype=dtype)
        if cached_length > 0:
            attention_mask[:, :, :, :cached_length] = 0

        for row_idx, q_block_id in enumerate(query_block_ids.tolist()):
            if row_idx < update_kvcache and query_prompt_window_mask[row_idx]:
                visible_current = query_prompt_window_mask
            else:
                visible_current = query_block_ids <= q_block_id
            attention_mask[0, 0, row_idx, cached_length:][visible_current] = 0
        return attention_mask

    eval_dream_module.build_sparse_block_attention_mask = prompt_full_visible_sparse_mask
    inner.parallelcomp_tail_replay_full_mask = True
    inner._debug_eviction_events.append({"mode": "prompt_full_visible_reuse"})


def install_rebuild_context_with_query(inner, originals):
    def rebuild_context_cache_with_query(
        self,
        x_t,
        past_key_values,
        block_states,
        mask_id,
        cached_positions_per_layer,
        cached_block_ranges_per_layer,
        shared_cached_length,
        canonical_block_tokens,
    ):
        stable_prompt_block_ids = [
            block_id for block_id, state in sorted(block_states.items())
            if state["state"] == "in_cache" and state.get("is_prompt_window", False)
        ]
        if (
            past_key_values is not None
            and not getattr(self, "_parallelcomp_prompt_cache_compression_done", False)
            and len(stable_prompt_block_ids) > 1
        ):
            tail_prompt_block_id = stable_prompt_block_ids[-1]
            context_prompt_block_ids = stable_prompt_block_ids[:-1]
            tail_state = block_states[tail_prompt_block_id]
            tail_input_ids = x_t[:, tail_state["start_pos"]:tail_state["end_pos"]]
            tail_len = int(tail_input_ids.shape[1])
            reused_window_size = max(1, int(self.parallelcomp_chunk_size or 0))

            rebuilt_key_parts = None
            rebuilt_value_parts = None
            rebuilt_cache = None
            rebuilt_ranges = {}
            cursor = 0
            for block_id in context_prompt_block_ids:
                state = block_states[block_id]
                block_input_ids = x_t[:, state["start_pos"]:state["end_pos"]]
                block_len = int(block_input_ids.shape[1])
                joint_input_ids = torch.cat([block_input_ids, tail_input_ids], dim=1)
                joint_len = int(joint_input_ids.shape[1])
                block_positions = torch.arange(block_len, device=self.device, dtype=torch.long)
                tail_positions = torch.arange(
                    reused_window_size,
                    reused_window_size + tail_len,
                    device=self.device,
                    dtype=torch.long,
                )
                position_ids = torch.cat([block_positions, tail_positions], dim=0).unsqueeze(0)
                cache_position = torch.arange(joint_len, device=self.device, dtype=torch.long)
                outputs = self.model(
                    joint_input_ids,
                    attention_mask=self._build_full_visible_attention_mask(joint_len),
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=True,
                    update_kvcache=joint_len,
                )
                block_cache = outputs.past_key_values
                if rebuilt_cache is None:
                    rebuilt_cache = block_cache
                    rebuilt_key_parts = [[] for _ in block_cache.key_cache]
                    rebuilt_value_parts = [[] for _ in block_cache.value_cache]
                for layer_idx, (layer_k, layer_v) in enumerate(
                    zip(block_cache.key_cache, block_cache.value_cache)
                ):
                    rebuilt_key_parts[layer_idx].append(layer_k[:, :, :block_len, :])
                    rebuilt_value_parts[layer_idx].append(layer_v[:, :, :block_len, :])
                rebuilt_ranges[block_id] = (cursor, cursor + block_len)
                cursor += block_len

            if rebuilt_cache is not None:
                rebuilt_cache.key_cache = [
                    torch.cat(parts, dim=2) if parts else layer_k[:, :, :0, :]
                    for parts, layer_k in zip(rebuilt_key_parts, rebuilt_cache.key_cache)
                ]
                rebuilt_cache.value_cache = [
                    torch.cat(parts, dim=2) if parts else layer_v[:, :, :0, :]
                    for parts, layer_v in zip(rebuilt_value_parts, rebuilt_cache.value_cache)
                ]
                cached_positions_per_layer = [
                    torch.arange(cursor, device=self.device, dtype=torch.long)
                    for _ in rebuilt_cache.key_cache
                ]
                cached_block_ranges_per_layer = [
                    dict(rebuilt_ranges) for _ in rebuilt_cache.key_cache
                ]
                past_key_values = rebuilt_cache
                self._debug_eviction_events.append(
                    {
                        "mode": "rebuild_context_with_query",
                        "context_blocks": len(context_prompt_block_ids),
                        "tail_len": tail_len,
                        "rebuilt_context_tokens": cursor,
                    }
                )

        return originals["compress"](
            self,
            x_t,
            past_key_values,
            block_states,
            mask_id,
            cached_positions_per_layer,
            cached_block_ranges_per_layer,
            shared_cached_length,
            canonical_block_tokens,
        )

    inner._compress_cached_prefix_blocks = types.MethodType(
        rebuild_context_cache_with_query,
        inner,
    )
    inner.parallelcomp_tail_replay_full_mask = True
    inner._debug_eviction_events.append({"mode": "rebuild_context_with_query"})


def configure_mode(inner, originals, mode):
    inner._debug_eviction_events = []
    install_baseline(inner, originals)
    inner._compress_cached_prefix_blocks = types.MethodType(
        originals["compress"],
        inner,
    )
    inner._should_split_prompt_windows = types.MethodType(
        originals["should_split"],
        inner,
    )
    inner._build_block_runtime_positions = types.MethodType(
        originals["build_positions"],
        inner,
    )
    inner.parallelcomp_tail_replay_full_mask = originals["tail_replay_full_mask"]
    eval_dream_module.build_sparse_block_attention_mask = originals["sparse_mask"]
    if mode == "baseline":
        return
    if mode == "no_prompt_split":
        def no_prompt_split(self, prompt_length):
            self._debug_eviction_events.append(
                {"mode": "no_prompt_split", "prompt_length": int(prompt_length)}
            )
            return False

        inner._should_split_prompt_windows = types.MethodType(no_prompt_split, inner)
        return
    if mode == "skip_compress":
        def skip_compress(self, x_t, past_key_values, block_states, mask_id,
                          cached_positions_per_layer, cached_block_ranges_per_layer,
                          shared_cached_length, canonical_block_tokens):
            self._parallelcomp_prompt_cache_compression_done = True
            self._debug_eviction_events.append({"mode": "skip_compress"})
            return (
                x_t,
                past_key_values,
                cached_positions_per_layer,
                cached_block_ranges_per_layer,
                shared_cached_length,
                False,
                None,
            )

        inner._compress_cached_prefix_blocks = types.MethodType(skip_compress, inner)
        return
    if mode == "answer_protect":
        install_answer_protect(inner, originals)
        return
    if mode.startswith("prefix_anchor_"):
        prefix_tokens = int(mode.rsplit("_", 1)[1])
        install_prefix_anchor(inner, originals, prefix_tokens)
        return
    if mode == "shared_keep_set":
        install_shared_keep_set(inner, originals)
        return
    if mode == "query_conditioned_prompt":
        install_query_conditioned_prompt(inner, originals)
        return
    if mode == "prompt_full_visible_reuse":
        install_prompt_full_visible_reuse(inner, originals)
        return
    if mode == "rebuild_context_with_query":
        install_rebuild_context_with_query(inner, originals)
        return
    raise ValueError(f"Unknown ablation mode: {mode}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--lora_path", default=DEFAULT_LORA)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", default="passkey")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk_chunks", type=int, default=3)
    parser.add_argument("--token_capacity", type=int, default=128)
    parser.add_argument("--token_keep_min", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "answer_protect", "prefix_anchor_64", "shared_keep_set"],
    )
    parser.add_argument("--output_dir", default="./results_eviction_ablation")
    return parser


def main():
    args = build_arg_parser().parse_args()
    os.environ.setdefault("HF_HOME", "/home/ma-user/work/hf-cache")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    model = load_model(
        "dream",
        args.pretrained,
        args.lora_path,
        rope_scale_factor=1.0,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=0.0,
        add_bos_token=True,
        parallelcomp_mode=True,
        parallelcomp_pre_runtime_mode=True,
        parallelcomp_cache_compress_mode=True,
        parallelcomp_chunk_size=args.chunk_size,
        parallelcomp_query_tokens=0,
        parallelcomp_topk_chunks=args.topk_chunks,
        parallelcomp_min_prompt_tokens=1,
        parallelcomp_keep_first_chunk=True,
        parallelcomp_split_from_tail=False,
        parallelcomp_hidden_topk=32,
        parallelcomp_token_capacity=args.token_capacity,
        parallelcomp_token_keep_min=args.token_keep_min,
        parallelcomp_high_score_threshold=None,
        parallelcomp_select_low_score_chunks=False,
        parallelcomp_fixed_query_text="Please answer the question using the long context above.",
    )
    inner = model._inner
    originals = {
        "per_head": inner.__class__._select_cache_block_token_indices_per_layer_per_head,
        "compress": inner.__class__._compress_cached_prefix_blocks,
        "should_split": inner.__class__._should_split_prompt_windows,
        "build_positions": inner.__class__._build_block_runtime_positions,
        "tail_replay_full_mask": inner.parallelcomp_tail_replay_full_mask,
        "sparse_mask": eval_dream_module.build_sparse_block_attention_mask,
    }

    examples = load_task_examples(args.task, args.data_dir, max_examples=args.max_examples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for mode in args.modes:
        configure_mode(inner, originals, mode)
        mode_records = []
        out_file = output_dir / f"{args.task}_{mode}_n{len(examples)}.jsonl"
        print(f"\n=== mode={mode} ===", flush=True)
        with out_file.open("w", encoding="utf-8") as f:
            for idx, example in enumerate(examples):
                answer_label = normalize_answer_label(args.task, example)
                answer_text = coerce_answer_text(answer_label)
                inner._debug_current_answer_ids = inner._encode_text_fragment(answer_text)
                inner._debug_eviction_events = []

                prompt = create_prompt_parts(example, args.task, "parallelcomp_raw")
                prompt["metadata_label"] = f"{args.task}:{example.get('id', idx)}:{mode}"
                prediction = generate(model, [prompt], stop_tokens=[])[0]
                correct = score_prediction(args.task, prediction, answer_label)
                record = {
                    "mode": mode,
                    "example_id": example.get("id", idx),
                    "index": idx,
                    "answer": answer_label,
                    "prediction": prediction,
                    "correct": correct,
                    "events": inner._debug_eviction_events[:8],
                }
                mode_records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{idx + 1}/{len(examples)}] correct={int(correct)} "
                    f"answer={answer_text} pred={prediction[:80]!r} "
                    f"events={record['events'][:2]}",
                    flush=True,
                )

        accuracy = sum(int(r["correct"]) for r in mode_records) / max(1, len(mode_records))
        summary[mode] = {
            "accuracy": accuracy,
            "n": len(mode_records),
            "correct": sum(int(r["correct"]) for r in mode_records),
            "output": str(out_file),
        }
        print(f"MODE_SUMMARY {mode} {json.dumps(summary[mode], ensure_ascii=False)}", flush=True)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSUMMARY {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
