"""
Evaluate LLaDA/DREAM on LiveCodeBench (top-N% longest context).

Usage:
  python eval_d2f_lcb.py \
    --model_type llada \
    --pretrained /path/to/llada \
    --lora_path /path/to/lora \   # optional
    --top_percent 30 \
    --output_dir ./results_d2f

Output: ./results_d2f/{model_type}_lcb_top{N}pct.json (LCB eval format)

Compute metrics after generation:
  source /home/hq/LiveCodeBench/.venv/bin/activate
  python eval_d2f_lcb_score.py --input results_d2f/...json
"""
import argparse
import json
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, '/home/hq/Discrete-Diffusion-Forcing/D2F-eval')
sys.path.insert(0, '/home/hq/LiveCodeBench')
os.chdir('/home/hq/LiveCodeBench')


def filter_top_percent(dataset, top_percent, key_fn):
    indexed = sorted(enumerate(dataset), key=lambda x: key_fn(x[1]), reverse=True)
    top_n = max(1, int(len(indexed) * top_percent / 100))
    threshold = key_fn(indexed[top_n - 1][1])
    subset = [p for i, p in indexed[:top_n]]
    print(f"  Total: {len(dataset)}, top {top_percent}% = {len(subset)} items")
    print(f"  Context length threshold: >= {threshold} chars")
    return subset


def main():
    parser = argparse.ArgumentParser(description='Eval LLaDA/DREAM on LiveCodeBench')
    parser.add_argument('--model_type', choices=['llada', 'dream'], required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--lora_path', default=None)
    parser.add_argument('--top_percent', type=float, default=30,
                        help='Use top N%% longest-context problems')
    parser.add_argument('--n', type=int, default=1, help='Samples per problem')
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--release_version', default='release_v5')
    parser.add_argument('--output_dir', default='./results_d2f')
    args = parser.parse_args()

    from d2f_model import load_model, generate
    from lcb_runner.benchmarks import load_code_generation_dataset
    from lcb_runner.prompts.code_generation import format_prompt_generation
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle


    # Load dataset and filter
    print("Loading LiveCodeBench dataset...")
    dataset = load_code_generation_dataset(release_version=args.release_version)
    subset = filter_top_percent(dataset, args.top_percent,
                                key_fn=lambda p: len(p.question_content))

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
    results = []
    for i, problem in enumerate(subset):
        print(f"[{i+1}/{len(subset)}] {problem.question_title}")
        prompt = format_prompt_generation(problem, LMStyle.GenericBase)
        outputs = [generate(model, [prompt])[0] for _ in range(args.n)]
        results.append({
            'question_title': problem.question_title,
            'question_content': problem.question_content,
            'platform': str(problem.platform),
            'question_id': problem.question_id,
            'contest_id': problem.contest_id,
            'contest_date': problem.contest_date.isoformat(),
            'difficulty': str(problem.difficulty),
            'output_list': outputs,
            'code_list': [extract_code(o, LMStyle.GenericBase) for o in outputs],
        })

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"{args.model_type}_lcb_{args.release_version}_top{int(args.top_percent)}pct.json"
    )
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} results to: {out_file}")
    print(f"\nEvaluate with:")
    print(f"  source /home/hq/LiveCodeBench/.venv/bin/activate")
    print(f"  python eval_d2f_lcb_score.py --input_file {out_file}")


if __name__ == '__main__':
    main()
