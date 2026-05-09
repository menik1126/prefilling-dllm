import math
from pathlib import Path

from datasets import load_dataset

from d2f_model import load_model
from eval_d2f_plcc import build_repo_context, filter_top_percent, total_context_chars


TARGET = ("LAION-AI__Open-Assistant", "backend/oasst_backend/api/v1/management.py", "inproject", 113)
BASE = Path("/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval")


def main():
    model = load_model(
        "dream",
        pretrained=str(BASE / "model_weights/Dream-v0-Base-7B"),
        lora_path=str(BASE / "model_weights/D2F_Dream_Base_7B_Lora"),
        rope_scale_factor=1.0,
        max_new_tokens=128,
        max_length=8192,
        block_size=32,
        temperature=0,
        add_bos_token=True,
        parallelcomp_mode=True,
        parallelcomp_cache_compress_mode=True,
        parallelcomp_chunk_size=1024,
        parallelcomp_topk_chunks=3,
        parallelcomp_min_prompt_tokens=1,
        parallelcomp_tail_replay_full_mask=True,
        parallelcomp_fixed_query_text="Please complete the preceding code.",
    )
    inner = model._inner

    repo, cfile, _cat, line_num = TARGET
    ds = load_dataset("JetBrains-Research/lca-project-level-code-completion", "medium_context", split="test")
    subset = filter_top_percent(list(ds), 30, total_context_chars)
    item = next(
        x
        for x in subset
        if x.get("repo", "") == repo and x.get("completion_file", {}).get("filename", "") == cfile
    )

    cf = item["completion_file"]
    cf_lines = cf["content"].split("\n")
    prompt = build_repo_context(item["repo_snapshot"], completion_filepath=cfile)
    prompt += f"\n\n# path: {cfile}\n" + "\n".join(cf_lines[:line_num])

    max_prompt_chars = (8192 - 128) * 6
    if len(prompt) > max_prompt_chars:
        prompt = prompt[-max_prompt_chars:]

    input_ids = inner.tok_encode(prompt, add_special_tokens=True).to(inner.device)
    prompt_limit = 8192 - 128
    if input_ids.shape[1] > prompt_limit:
        input_ids = input_ids[:, -prompt_limit:]

    prompt_ids = input_ids[0].tolist()
    query_ids = inner._build_parallelcomp_scoring_query_ids()

    blocks = []
    chunk_size = 1024
    for start in range(0, len(prompt_ids), chunk_size):
        block_ids = prompt_ids[start:start + chunk_size]
        mean_nll = inner._score_chunk_with_self_information(block_ids, query_ids)
        neg_mean_nll = -mean_nll if math.isfinite(mean_nll) else float("-inf")
        head = inner.tok_decode(block_ids[:80], skip_special_tokens=False).replace("\n", "\\n")
        tail = inner.tok_decode(block_ids[-80:], skip_special_tokens=False).replace("\n", "\\n")
        blocks.append(
            {
                "idx": len(blocks),
                "start": start,
                "end": start + len(block_ids),
                "len": len(block_ids),
                "mean_nll": mean_nll,
                "neg_mean_nll": neg_mean_nll,
                "head": head[:260],
                "tail": tail[:260],
            }
        )

    print("PROMPT_TOKENS", len(prompt_ids), flush=True)
    print("SCORING_QUERY", repr(inner.tok_decode(query_ids, skip_special_tokens=False)), flush=True)
    print("QUERY_IDS_LEN", len(query_ids), flush=True)
    print("--- ALL BLOCKS BY POSITION ---", flush=True)
    for b in blocks:
        print(
            f"IDX={b['idx']} span=[{b['start']},{b['end']}) len={b['len']} "
            f"mean_nll={b['mean_nll']:.4f} neg_mean_nll={b['neg_mean_nll']:.4f}",
            flush=True,
        )
        print("HEAD", b["head"], flush=True)
        print("TAIL", b["tail"], flush=True)
        print("---", flush=True)

    print("--- SORT BY HIGH SCORE (current keeps) ---", flush=True)
    for b in sorted(blocks, key=lambda x: x["neg_mean_nll"], reverse=True):
        print(
            f"IDX={b['idx']} neg_mean_nll={b['neg_mean_nll']:.4f} "
            f"mean_nll={b['mean_nll']:.4f} tail={b['tail'][:180]}",
            flush=True,
        )

    print("--- SORT BY LOW SCORE (ablation keeps) ---", flush=True)
    for b in sorted(blocks, key=lambda x: x["neg_mean_nll"]):
        print(
            f"IDX={b['idx']} neg_mean_nll={b['neg_mean_nll']:.4f} "
            f"mean_nll={b['mean_nll']:.4f} tail={b['tail'][:180]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
