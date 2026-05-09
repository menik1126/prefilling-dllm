import argparse
import hashlib
import json

import torch

from d2f_model import load_model
from infinitebench_tasks import create_prompt, load_task_examples


def _sha1_token_list(token_ids):
    return hashlib.sha1(bytes(str(token_ids), "utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--prompt_style", default="parallelcomp_raw")
    parser.add_argument("--max_length", type=int, default=65536)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--parallelcomp_chunk_size", type=int, default=1024)
    parser.add_argument("--parallelcomp_topk_chunks", type=int, default=3)
    parser.add_argument("--parallelcomp_min_prompt_tokens", type=int, default=1)
    parser.add_argument("--parallelcomp_token_capacity", type=int, default=256)
    parser.add_argument("--parallelcomp_token_keep_min", type=int, default=32)
    parser.add_argument(
        "--parallelcomp_fixed_query_text",
        default="Please answer the question using the long context above.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
        parallelcomp_cache_compress_mode=True,
        parallelcomp_chunk_size=args.parallelcomp_chunk_size,
        parallelcomp_topk_chunks=args.parallelcomp_topk_chunks,
        parallelcomp_min_prompt_tokens=args.parallelcomp_min_prompt_tokens,
        parallelcomp_token_capacity=args.parallelcomp_token_capacity,
        parallelcomp_token_keep_min=args.parallelcomp_token_keep_min,
        parallelcomp_fixed_query_text=args.parallelcomp_fixed_query_text,
    )
    inner = model._inner
    examples = load_task_examples(args.task, args.data_dir, max_examples=max(args.indices) + 1)
    trunc_len = args.max_length - args.max_new_tokens
    query_ids = inner._build_parallelcomp_scoring_query_ids()

    samples = []
    trunc_id_lists = []

    for idx in args.indices:
        ex = examples[idx]
        prompt = create_prompt(ex, args.task, args.prompt_style)
        prompt_ids = inner.tokenizer.encode(inner.tokenizer.bos_token + prompt)
        trunc_ids = prompt_ids[-trunc_len:] if len(prompt_ids) > trunc_len else prompt_ids
        trunc_id_lists.append(trunc_ids)
        x_t = torch.tensor([trunc_ids], device=inner.device, dtype=torch.long)
        block_states = inner._init_prompt_block_states(len(trunc_ids))
        canonical = inner._init_canonical_block_tokens(x_t, block_states)
        stable_prompt_block_ids = [
            bid for bid, state in sorted(block_states.items()) if state.get("is_prompt_window", False)
        ]
        context_prompt_block_ids = stable_prompt_block_ids[:-1]
        kept_block_ids = inner._select_global_cache_blocks(
            block_states=block_states,
            stable_block_ids=context_prompt_block_ids,
            stable_prefix_len=len(trunc_ids),
            query_ids=query_ids,
            canonical_block_tokens=canonical,
        )
        kept_blocks = []
        for bid in kept_block_ids:
            ids = canonical[bid]
            kept_blocks.append(
                {
                    "block_id": bid,
                    "span": [block_states[bid]["start_pos"], block_states[bid]["end_pos"]],
                    "sha1": _sha1_token_list(ids)[:16],
                    "preview": inner.tokenizer.decode(ids[:120], skip_special_tokens=False)[:300],
                }
            )

        tail_block_id = stable_prompt_block_ids[-1]
        tail_ids = canonical[tail_block_id]
        samples.append(
            {
                "example_id": idx,
                "answer": ex["answer"],
                "orig_prompt_tokens": len(prompt_ids),
                "trunc_prompt_tokens": len(trunc_ids),
                "trunc_sha1": _sha1_token_list(trunc_ids)[:16],
                "num_prompt_blocks": len(stable_prompt_block_ids),
                "tail_block_id": tail_block_id,
                "tail_block_span": [block_states[tail_block_id]["start_pos"], block_states[tail_block_id]["end_pos"]],
                "kept_block_ids": kept_block_ids,
                "tail_last_256_tokens_text": inner.tokenizer.decode(trunc_ids[-256:], skip_special_tokens=False),
                "tail_block_preview": inner.tokenizer.decode(tail_ids, skip_special_tokens=False)[:300],
                "kept_blocks": kept_blocks,
            }
        )

    same = trunc_id_lists[0] == trunc_id_lists[1] if len(trunc_id_lists) == 2 else None
    first_diff_positions = []
    pairwise_same_kept = None
    if len(trunc_id_lists) == 2:
        limit = min(len(trunc_id_lists[0]), len(trunc_id_lists[1]))
        for i in range(limit):
            if trunc_id_lists[0][i] != trunc_id_lists[1][i]:
                first_diff_positions.append(i)
                if len(first_diff_positions) >= 20:
                    break
        pairwise_same_kept = samples[0]["kept_block_ids"] == samples[1]["kept_block_ids"]

    result = {
        "task": args.task,
        "prompt_style": args.prompt_style,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "comparison": {
            "same_truncated_token_sequence": same,
            "pairwise_same_kept_block_ids": pairwise_same_kept,
            "truncated_lengths": [len(x) for x in trunc_id_lists],
            "first_diff_positions": first_diff_positions,
        },
        "samples": samples,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
