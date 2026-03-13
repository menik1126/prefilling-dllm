"""
Evaluate LLaDA/DREAM on LCA Commit Message Generation (top-N% longest diff).

Usage:
  python eval_d2f_cmg.py \
    --model_type llada \
    --pretrained /path/to/llada \
    --lora_path /path/to/lora \   # optional
    --top_percent 30 \
    --output_dir ./results_d2f

Metrics: BLEU, ChrF, ROUGE-1/2/L, BERTScore (computed inline)
"""
import argparse
import json
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/home/hq/Discrete-Diffusion-Forcing/D2F-eval')


CMG_PROMPT = (
    "Write a commit message for a given diff. "
    "Start with a heading that serves as a summary: a single sentence in imperative form, "
    "no more than 50 characters. If you have details to add, do it after a blank line.\n"
    "Diff:\n{diff}\nCommit message:\n"
)


def format_mods(mods):
    """Concatenate per-file diffs from the mods list into a unified diff string."""
    parts = []
    for mod in mods:
        if not isinstance(mod, dict):
            continue
        path = mod.get('new_path') or mod.get('old_path', '')
        diff = mod.get('diff', '')
        if path:
            parts.append(f"--- {path}\n{diff}")
        elif diff:
            parts.append(diff)
    return '\n'.join(parts)


def mods_length(item):
    """Total character length of all diffs in this item's mods list."""
    mods = item.get('mods', [])
    if not isinstance(mods, list):
        return 0
    return sum(len(m.get('diff', '')) for m in mods if isinstance(m, dict))


def filter_top_percent(items, top_percent, key_fn):
    indexed = sorted(enumerate(items), key=lambda x: key_fn(x[1]), reverse=True)
    top_n = max(1, int(len(indexed) * top_percent / 100))
    threshold = key_fn(indexed[top_n - 1][1])
    subset = [item for _, item in indexed[:top_n]]
    print(f"  Total: {len(items)}, top {top_percent}% = {len(subset)} items")
    print(f"  Diff length threshold: >= {threshold} chars")
    return subset


def compute_metrics(predictions, references):
    import evaluate
    results = {}
    bleu = evaluate.load('sacrebleu')
    chrf = evaluate.load('chrf')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    results['bleu'] = bleu.compute(predictions=predictions,
                                   references=[[r] for r in references])['score']
    results['chrf'] = chrf.compute(predictions=predictions,
                                   references=[[r] for r in references])['score']
    rouge_res = rouge.compute(predictions=predictions, references=references)
    results['rouge1'] = rouge_res['rouge1'] * 100
    results['rouge2'] = rouge_res['rouge2'] * 100
    results['rougeL'] = rouge_res['rougeL'] * 100
    bs = bertscore.compute(predictions=predictions, references=references, lang='en')
    results['bertscore_f1'] = sum(bs['f1']) / len(bs['f1'])
    return results


def main():
    parser = argparse.ArgumentParser(description='Eval LLaDA/DREAM on LCA CMG')
    parser.add_argument('--model_type', choices=['llada', 'dream'], required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--top_percent', type=float, default=30,
                        help='Use top N%% longest-diff examples')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--output_dir', default='./results_d2f')
    args = parser.parse_args()

    from datasets import load_dataset
    from d2f_model import load_model, generate

    # Load dataset
    print("Loading CMG dataset...")
    ds = load_dataset('JetBrains-Research/lca-commit-message-generation',
                      'default', split='test')
    items = list(ds)

    # Filter by total diff length across all modified files
    subset = filter_top_percent(items, args.top_percent, key_fn=mods_length)

    # Load model
    print(f"Loading {args.model_type} model from {args.pretrained}...")
    model = load_model(
        args.model_type, args.pretrained, args.lora_path,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        add_bos_token=True,
    )

    # Generate
    predictions, references = [], []
    output_records = []
    for i, item in enumerate(subset):
        diff_text = format_mods(item.get('mods', []))
        ref = item.get('message', '')
        prompt = CMG_PROMPT.format(diff=diff_text)
        pred = generate(model, [prompt], stop_tokens=['\n\n', '\n\n\n'])[0].strip()
        predictions.append(pred)
        references.append(ref)
        output_records.append({
            'repo': item.get('repo', ''),
            'hash': item.get('hash', ''),
            'reference': ref,
            'prediction': pred,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(subset)}]")

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"{args.model_type}_cmg_top{int(args.top_percent)}pct_predictions.jsonl"
    )
    with open(out_file, 'w') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\nPredictions saved to: {out_file}")

    # Compute metrics
    print("\nComputing metrics...")
    try:
        metrics = compute_metrics(predictions, references)
        print("\n=== CMG Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        metrics_file = out_file.replace('_predictions.jsonl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
    except ImportError:
        print("Warning: 'evaluate' library not installed. Skipping metrics.")
        print("Install with: uv pip install evaluate bert_score rouge_score")


if __name__ == '__main__':
    main()
