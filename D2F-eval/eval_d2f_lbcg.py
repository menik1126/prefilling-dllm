"""
Evaluate LLaDA/DREAM on LCA Library-based Code Generation (top-N% longest context).

Usage:
  python eval_d2f_lbcg.py \
    --model_type llada \
    --pretrained /path/to/llada \
    --lora_path /path/to/lora \   # optional
    --top_percent 30 \
    --output_dir ./results_d2f

Metrics: ChrF, API Recall (computed inline)
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/home/hq/Discrete-Diffusion-Forcing/D2F-eval')


LBCG_PROMPT = (
    "Generate Python code based on the following instruction. "
    "Output ONLY code. DO NOT include explanations or other textual content.\n"
    "Instruction: {instruction}"
)


def filter_top_percent(items, top_percent, key_fn):
    indexed = sorted(enumerate(items), key=lambda x: key_fn(x[1]), reverse=True)
    top_n = max(1, int(len(indexed) * top_percent / 100))
    threshold = key_fn(indexed[top_n - 1][1])
    subset = [item for _, item in indexed[:top_n]]
    print(f"  Total: {len(items)}, top {top_percent}% = {len(subset)} items")
    print(f"  Instruction length threshold: >= {threshold} chars")
    return subset


def api_recall(prediction, project_apis):
    """Fraction of expected APIs present in the generated code."""
    if not project_apis:
        return None
    hit = sum(1 for api in project_apis if api in prediction)
    return hit / len(project_apis)


def compute_chrf(predictions, references):
    try:
        import evaluate
        chrf = evaluate.load('chrf')
        return chrf.compute(predictions=predictions,
                            references=[[r] for r in references])['score']
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Eval LLaDA/DREAM on LCA LBCG')
    parser.add_argument('--model_type', choices=['llada', 'dream'], required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--top_percent', type=float, default=30,
                        help='Use top N%% longest-instruction examples')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--output_dir', default='./results_d2f')
    args = parser.parse_args()

    from datasets import load_dataset
    from d2f_model import load_model, generate

    # Load dataset
    print("Loading LBCG dataset...")
    ds = load_dataset('JetBrains-Research/lca-library-based-code-generation', split='test')
    items = list(ds)

    # Filter by instruction length
    subset = filter_top_percent(items, args.top_percent,
                                key_fn=lambda x: len(x.get('instruction', '')))

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
    predictions, references, api_recalls = [], [], []
    output_records = []
    for i, item in enumerate(subset):
        instruction = item.get('instruction', '')
        ref_code = item.get('reference', '')
        project_apis = item.get('api_calls', [])

        prompt = LBCG_PROMPT.format(instruction=instruction)
        pred = generate(model, [prompt], stop_tokens=['```\n\n', '\n\n\n\n'])[0].strip()

        recall = api_recall(pred, project_apis)
        predictions.append(pred)
        references.append(ref_code)
        if recall is not None:
            api_recalls.append(recall)

        output_records.append({
            'instruction': instruction[:200],
            'library': item.get('library', ''),
            'reference': ref_code,
            'prediction': pred,
            'api_recall': recall,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(subset)}]")

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"{args.model_type}_lbcg_top{int(args.top_percent)}pct_predictions.jsonl"
    )
    with open(out_file, 'w') as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"\nPredictions saved to: {out_file}")

    # Compute metrics
    chrf_score = compute_chrf(predictions, references)
    mean_api_recall = sum(api_recalls) / len(api_recalls) if api_recalls else None

    print("\n=== LBCG Metrics ===")
    if chrf_score is not None:
        print(f"  ChrF: {chrf_score:.4f}")
    if mean_api_recall is not None:
        print(f"  API Recall: {mean_api_recall:.4f}")

    metrics = {}
    if chrf_score is not None:
        metrics['chrf'] = chrf_score
    if mean_api_recall is not None:
        metrics['api_recall'] = mean_api_recall

    metrics_file = out_file.replace('_predictions.jsonl', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
