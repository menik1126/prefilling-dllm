import argparse

from datasets import load_dataset

from d2f_model import load_model, generate
from eval_d2f_plcc import filter_top_percent, total_context_chars
from plcc_file_retrieval import build_repo_context_bm25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["dream", "llada"], default="dream")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--config", default="medium_context")
    parser.add_argument("--top_percent", type=float, default=30.0)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--bm25_topk_files", type=int, default=8)
    parser.add_argument("--bm25_fill_window", action="store_true")
    parser.add_argument("--bm25_target_prompt_chars", type=int, default=0)
    parser.add_argument("--bm25_query_max_lines", type=int, default=128)
    parser.add_argument("--bm25_query_max_chars", type=int, default=4000)
    parser.add_argument("--parallelcomp_mode", action="store_true")
    parser.add_argument("--parallelcomp_cache_compress_mode", action="store_true")
    parser.add_argument("--parallelcomp_chunk_size", type=int, default=1024)
    parser.add_argument("--parallelcomp_topk_chunks", type=int, default=4)
    parser.add_argument("--parallelcomp_min_prompt_tokens", type=int, default=1)
    parser.add_argument("--parallelcomp_hidden_topk", type=int, default=32)
    parser.add_argument("--parallelcomp_token_capacity", type=int, default=128)
    parser.add_argument("--parallelcomp_token_keep_min", type=int, default=32)
    parser.add_argument(
        "--parallelcomp_fixed_query_text",
        default="Please complete the preceding code.",
    )
    args = parser.parse_args()

    prompt_token_limit = args.max_length - args.max_new_tokens
    max_prompt_chars = prompt_token_limit * 6
    bm25_target_prompt_chars = args.bm25_target_prompt_chars or max_prompt_chars

    print("loading dataset...", flush=True)
    ds = load_dataset(
        "JetBrains-Research/lca-project-level-code-completion",
        args.config,
        split="test",
    )
    subset = filter_top_percent(list(ds), args.top_percent, total_context_chars)
    item = subset[args.sample_index]

    snap = item.get("repo_snapshot", {})
    cf = item.get("completion_file", {})
    completion_lines = item.get("completion_lines", {})
    cf_filename = cf.get("filename", "unknown.py") if isinstance(cf, dict) else "unknown.py"
    cf_content = cf.get("content", "") if isinstance(cf, dict) else ""
    cf_line_list = cf_content.split("\n")

    repo_ctx, retrieval_meta = build_repo_context_bm25(
        snap,
        completion_filepath=cf_filename,
        completion_content=cf_content,
        completion_lines=completion_lines,
        topk_files=args.bm25_topk_files,
        fill_window=args.bm25_fill_window,
        target_prompt_chars=bm25_target_prompt_chars,
        query_max_lines=args.bm25_query_max_lines,
        query_max_chars=args.bm25_query_max_chars,
    )

    chosen = None
    for cat, line_nums in completion_lines.items():
        if not line_nums:
            continue
        ln = line_nums[0]
        prefix = "\n".join(cf_line_list[:ln])
        prompt = repo_ctx + f"\n\n# path: {cf_filename}\n{prefix}"
        if len(prompt) > max_prompt_chars:
            prompt = prompt[-max_prompt_chars:]
        gt = cf_line_list[ln] if ln < len(cf_line_list) else ""
        chosen = (cat, ln, prompt, gt)
        break

    if chosen is None:
        raise RuntimeError("No valid completion line found for the selected sample.")

    cat, ln, prompt, gt = chosen

    print(f"repo={item.get('repo', '')}", flush=True)
    print(f"completion_file={cf_filename}", flush=True)
    print(f"target_line_category={cat}", flush=True)
    print(f"target_line_num={ln}", flush=True)
    print(f"selected_files={retrieval_meta['selected_files']}", flush=True)
    print(f"num_selected_files={retrieval_meta['num_selected_files']}", flush=True)
    print(f"selected_context_chars={retrieval_meta['selected_context_chars']}", flush=True)
    print(f"prompt_chars={len(prompt)}", flush=True)

    print("loading model...", flush=True)
    model = load_model(
        args.model_type,
        args.pretrained,
        args.lora_path,
        rope_scale_factor=1.0,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=args.temperature,
        add_bos_token=True,
        parallelcomp_mode=args.parallelcomp_mode,
        parallelcomp_cache_compress_mode=args.parallelcomp_cache_compress_mode,
        parallelcomp_chunk_size=args.parallelcomp_chunk_size,
        parallelcomp_query_tokens=0,
        parallelcomp_topk_chunks=args.parallelcomp_topk_chunks,
        parallelcomp_min_prompt_tokens=args.parallelcomp_min_prompt_tokens,
        parallelcomp_keep_first_chunk=False,
        parallelcomp_split_from_tail=False,
        parallelcomp_hidden_topk=args.parallelcomp_hidden_topk,
        parallelcomp_token_capacity=args.parallelcomp_token_capacity,
        parallelcomp_token_keep_min=args.parallelcomp_token_keep_min,
        parallelcomp_high_score_threshold=None,
        parallelcomp_select_low_score_chunks=False,
        parallelcomp_fixed_query_text=args.parallelcomp_fixed_query_text,
        parallelcomp_tail_replay_full_mask=False,
    )

    print("generating...", flush=True)
    out = generate(model, [prompt], stop_tokens=["\n"])[0]
    pred_line = out.split("\n")[0]
    print(f"prediction={pred_line!r}", flush=True)
    print(f"ground_truth={gt!r}", flush=True)


if __name__ == "__main__":
    main()
