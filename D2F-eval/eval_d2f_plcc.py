"""
Evaluate LLaDA/DREAM on LCA Project-level Code Completion (top-N% longest context).
Metrics: Exact Match (EM) and Edit Similarity (ES) per line category.

Matches the official LCA baseline (lca-baselines/project_level_code_completion):
  1. Files filtered: all .py + non-hidden non-license non-py files < 16 KB.
  2. Completion file excluded from repo context; only its path marker appended.
  3. Files sorted by path distance (farthest first, closest last).
  4. Prompt format: {repo_name}METASEP{path}METASEP{content}...{cf_path}METASEP\n{prefix}
  5. Character pre-truncation (×6) then token truncation from the left.

Usage:
  python eval_d2f_plcc.py \
    --model_type llada \
    --pretrained GSAI-ML/LLaDA-8B-Instruct \
    --top_percent 30 \
    --output_dir ./results_nolora
"""
import argparse
import json
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/home/hq/Discrete-Diffusion-Forcing/D2F-eval')

CONFIGS = ['small_context', 'medium_context', 'large_context', 'huge_context']

META_SEP = 'METASEP'   # official separator between path and content


def _path_distance(path_from, path_to):
    """
    Directory-tree distance between two file paths.
    Matches official PathDistanceComposer._path_distance exactly.
    """
    parts_from = os.path.normpath(path_from).split(os.sep)
    parts_to   = os.path.normpath(path_to).split(os.sep)
    common_len = 0
    for a, b in zip(parts_from, parts_to):
        if a == b:
            common_len += 1
        else:
            break
    return (len(parts_from) - common_len - 1) + (len(parts_to) - common_len - 1)


def build_repo_context(snap, completion_filepath=None, repo_name=''):
    """
    Build the repo context string following the official LCA pipeline:
    - Filter files: all .py + non-hidden, non-license non-.py files < 16 KB
    - Exclude the completion file (it only contributes its path marker + prefix)
    - Sort farthest-first so closest files survive left-truncation
    - Format: {repo_name}METASEP{path}METASEP{content}...{cf_path}METASEP

    The caller appends "\n{prefix}" to complete the prompt.
    """
    filenames = snap.get('filename', [])
    contents  = snap.get('content', [])

    pairs = []
    for fn, ct in zip(filenames, contents):
        if fn == completion_filepath:
            continue  # completion file handled separately (path marker only)
        if fn.endswith('.py'):
            pairs.append((fn, ct))
        else:
            basename = os.path.basename(fn)
            if not basename.startswith('.') and 'license' not in fn.lower():
                if len(ct) < 16000:
                    pairs.append((fn, ct))

    if completion_filepath:
        # Farthest files first → truncated away; closest files last → survive
        pairs.sort(key=lambda x: -_path_distance(completion_filepath, x[0]))

    # Official format: path + METASEP + content (no newline separator between files)
    parts = [f"{fn}{META_SEP}{ct}" for fn, ct in pairs]

    # Completion file path marker at the very end (no content — prefix appended later)
    if completion_filepath:
        parts.append(f"{completion_filepath}{META_SEP}")

    # Prepend repo name (official adds {repo_name}METASEP header)
    header = f"{repo_name}{META_SEP}" if repo_name else ""
    return header + "".join(parts)


def total_context_chars(item):
    """Total chars across repo_snapshot + completion_file (used for top-% filtering)."""
    snap = item.get('repo_snapshot', {})
    contents = snap.get('content', []) if isinstance(snap, dict) else []
    ctx_len = sum(len(c) for c in contents if isinstance(c, str))
    cf = item.get('completion_file', {})
    file_content = cf.get('content', '') if isinstance(cf, dict) else ''
    return ctx_len + len(file_content)


def filter_top_percent(items, top_percent, key_fn):
    indexed = sorted(enumerate(items), key=lambda x: key_fn(x[1]), reverse=True)
    top_n = max(1, int(len(indexed) * top_percent / 100))
    subset = [item for _, item in indexed[:top_n]]
    threshold = key_fn(indexed[top_n - 1][1])
    print(f"  Total: {len(items)}, top {top_percent}% = {len(subset)} items")
    print(f"  Context length threshold: >= {threshold:,} chars")
    return subset


def edit_similarity(pred, gt):
    """Edit similarity following official LCA: fuzz.ratio without stripping."""
    try:
        from thefuzz import fuzz
        return fuzz.ratio(pred, gt) / 100.0
    except Exception:
        return 1.0 if pred == gt else 0.0


def main():
    parser = argparse.ArgumentParser(description='Eval LLaDA/DREAM on LCA PLCC')
    parser.add_argument('--model_type', choices=['llada', 'dream'], required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--top_percent', type=float, default=30)
    parser.add_argument('--configs', nargs='+', default=CONFIGS)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=None,
                        help='Model total context window (tokens). '
                             'Default: 2048 for dream, 4096 for llada.')
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--rope_scale_factor', type=float, default=1.0,
                        help='NTK-by-parts RoPE scale factor (1.0=off, 2.0=2x context, 4.0=4x context).')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--output_dir', default='./results_nolora')
    parser.add_argument('--apply_chat_template', action='store_true',
                        help='Wrap prompt in chat template (for Instruct models).')
    parser.add_argument('--model_tag', default=None,
                        help='Tag for output filenames (default: model_type).')
    args = parser.parse_args()

    # Determine token budget for the prompt
    if args.max_length is None:
        args.max_length = 4096 if args.model_type == 'llada' else 2048
    prompt_token_limit = args.max_length - args.max_new_tokens   # e.g. 1920 or 3968
    # Pre-truncate at character level before tokenisation (official LCA uses ×6)
    max_prompt_chars = prompt_token_limit * 6
    print(f"Prompt token limit : {prompt_token_limit}")
    print(f"Pre-truncation chars: {max_prompt_chars:,}  (={prompt_token_limit}×6)")

    from datasets import load_dataset
    from d2f_model import load_model, generate

    print(f"Loading {args.model_type} model from {args.pretrained}...")
    model = load_model(
        args.model_type, args.pretrained, args.lora_path,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=args.temperature,
        add_bos_token=True,
        rope_scale_factor=args.rope_scale_factor,
    )

    tokenizer = model.tokenizer
    chat_tag  = '_chat' if args.apply_chat_template else ''
    model_tag = args.model_tag if args.model_tag else args.model_type

    all_results = {}

    for cfg in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg}")
        print('='*60)

        # Skip if already completed
        scale_tag = f"_rope{args.rope_scale_factor:.1f}" if args.rope_scale_factor != 1.0 else ""
        out_file = os.path.join(
            args.output_dir,
            f"{model_tag}_plcc_{cfg}_top{int(args.top_percent)}pct{scale_tag}{chat_tag}_predictions.jsonl"
        )
        metrics_file = out_file.replace('_predictions.jsonl', '_metrics.json')
        if os.path.exists(metrics_file):
            print(f"  Already done, skipping. ({metrics_file})")
            with open(metrics_file) as f:
                all_results[cfg] = json.load(f)
            continue

        ds = load_dataset(
            'JetBrains-Research/lca-project-level-code-completion',
            cfg, split='test'
        )
        items = list(ds)
        subset = filter_top_percent(items, args.top_percent, total_context_chars)

        em_by_cat = {}
        es_by_cat = {}
        predictions_all = []

        for ex_idx, item in enumerate(subset):
            snap             = item.get('repo_snapshot', {})
            cf               = item.get('completion_file', {})
            completion_lines = item.get('completion_lines', {})

            cf_filename  = cf.get('filename', 'unknown.py') if isinstance(cf, dict) else 'unknown.py'
            cf_content   = cf.get('content', '')            if isinstance(cf, dict) else ''
            cf_line_list = cf_content.split('\n')
            repo_name    = item.get('repo', '')

            # Build repo context once per item:
            # - completion file excluded (only its path marker appended at end)
            # - farthest files first so closest survive left-truncation
            repo_ctx = build_repo_context(snap, completion_filepath=cf_filename, repo_name=repo_name)

            example_rec = {
                'repo':            item.get('repo', ''),
                'config':          cfg,
                'completion_file': cf_filename,
                'predictions':     {},
            }

            # Build per-line prompts
            line_prompts = []
            for cat, line_nums in completion_lines.items():
                if not line_nums:
                    continue
                for ln in line_nums:
                    prefix = '\n'.join(cf_line_list[:ln])
                    # Official format: context ends with cf_path+METASEP, then "\n"+prefix
                    prompt = repo_ctx + "\n" + prefix
                    # Character pre-truncation (official uses context_len_char, we use ×6)
                    if len(prompt) > max_prompt_chars:
                        prompt = prompt[-max_prompt_chars:]
                    if args.apply_chat_template:
                        messages = [{"role": "user", "content": prompt}]
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    gt = cf_line_list[ln] if ln < len(cf_line_list) else ''
                    line_prompts.append((cat, ln, prompt, gt))

            if not line_prompts:
                predictions_all.append(example_rec)
                continue

            # Generate (stop at newline)
            prompts = [lp[2] for lp in line_prompts]
            outputs = generate(model, prompts, stop_tokens=['\n'])

            # Evaluate
            for (cat, ln, _, gt), out in zip(line_prompts, outputs):
                # Official: strip leading/trailing newlines, then take first line
                pred_line = out.strip('\n').split('\n')[0]

                # Official: EM uses .strip() on both sides
                em_score = int(pred_line.strip() == gt.strip())
                # Official: ES uses fuzz.ratio without strip
                es_score = edit_similarity(pred_line, gt)

                em_by_cat.setdefault(cat, []).append(em_score)
                es_by_cat.setdefault(cat, []).append(es_score)

                example_rec['predictions'].setdefault(cat, []).append({
                    'line_num': ln,
                    'gt':       gt,
                    'pred':     pred_line,
                    'em':       em_score,
                    'es':       es_score,
                })

            predictions_all.append(example_rec)

            if (ex_idx + 1) % 5 == 0:
                print(f"  [{ex_idx+1}/{len(subset)}] done")

        # Save predictions
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_file, 'w') as f:
            for rec in predictions_all:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"\nPredictions saved to: {out_file}")

        # Compute and save metrics
        metrics = {}
        print(f"\n=== PLCC {cfg} Metrics ===")
        all_em, all_es = [], []
        for cat in sorted(em_by_cat.keys()):
            em_list = em_by_cat[cat]
            es_list = es_by_cat[cat]
            em_score = sum(em_list) / len(em_list) if em_list else 0.0
            es_score = sum(es_list) / len(es_list) if es_list else 0.0
            metrics[cat] = {'em': em_score, 'es': es_score, 'n': len(em_list)}
            print(f"  {cat:20s}: EM={em_score:.4f}, ES={es_score:.4f}  (n={len(em_list)})")
            all_em.extend(em_list)
            all_es.extend(es_list)

        overall_em = sum(all_em) / len(all_em) if all_em else 0.0
        overall_es = sum(all_es) / len(all_es) if all_es else 0.0
        metrics['overall'] = {'em': overall_em, 'es': overall_es, 'n': len(all_em)}
        print(f"  {'overall':20s}: EM={overall_em:.4f}, ES={overall_es:.4f}  (n={len(all_em)})")

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

        all_results[cfg] = metrics

    # Save combined metrics
    combined_file = os.path.join(
        args.output_dir,
        f"{model_tag}_plcc_all_top{int(args.top_percent)}pct{chat_tag}_metrics.json"
    )
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll configs combined metrics: {combined_file}")


if __name__ == '__main__':
    main()
