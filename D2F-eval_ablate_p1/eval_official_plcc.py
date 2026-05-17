"""
eval_official_plcc.py

Runs PLCC evaluation using the official lca-baselines logic:
  - PathDistanceComposer  (python_standard: meta_info_sep_symbol="METASEP\n", lang_sep_symbol="", extension="")
  - LineGeneratorHF       (official StopOnNewLine, seq_max_len, max_new_tokens=100)
  - EM aggregation: per-repo mean -> global mean  (official method)

Bypasses Hydra / wandb / perplexity-composer-selection; composer is fixed to path_distance.

Usage:
  python eval_official_plcc.py \\
    --pretrained  /path/to/Qwen2.5-Coder-1.5B \\
    --seq_max_len 1024 \\
    --dataset     medium_context \\
    --top_percent 100 \\
    --output_dir  ./results_official
"""

import argparse
import json
import os
import sys

import torch

# ── make lca-baselines importable ────────────────────────────────────────────
LCA_DIR = os.path.join(os.path.dirname(__file__),
                       '../../lca-baselines/project_level_code_completion')
sys.path.insert(0, os.path.abspath(LCA_DIR))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from composers.path_distance_composer import PathDistanceComposer
from data_classes.datapoint_commit_dataset import DatapointCommitDataset
from eval.line_generators import LineGeneratorHF


# ── composer config matches python_standard.yaml ─────────────────────────────
COMPOSER_ARGS = dict(
    lang_sep_symbol      = "",
    meta_info_sep_symbol = "METASEP\n",
    extension            = "",
)


def load_hf_dataset(dataset_name: str) -> list[DatapointCommitDataset]:
    """Load HF dataset and convert to DatapointCommitDataset objects."""
    hf_data = load_dataset(
        'JetBrains-Research/lca-project-level-code-completion',
        dataset_name, split='test'
    )
    repos = list(set(dp['repo'] for dp in hf_data))
    repo_id_map = {r: i for i, r in enumerate(repos)}

    datapoints = []
    for dp in hf_data:
        filenames = dp['repo_snapshot']['filename']
        contents  = dp['repo_snapshot']['content']
        datapoints.append(DatapointCommitDataset(
            repo_id          = repo_id_map[dp['repo']],
            repo_name        = dp['repo'],
            completion_lines = dp['completion_lines'],
            context_dict     = dict(zip(filenames, contents)),
            completion_dict  = {dp['completion_file']['filename']: dp['completion_file']['content']},
        ))
    return datapoints


def apply_composer(datapoints: list[DatapointCommitDataset]) -> list[DatapointCommitDataset]:
    """Set datapoint.context using PathDistanceComposer (official)."""
    composer = PathDistanceComposer(**COMPOSER_ARGS)
    for dp in datapoints:
        dp.context    = composer.context_composer(dp)
        dp.completion = composer.completion_composer(dp)
    return datapoints


def total_context_chars(dp: DatapointCommitDataset) -> int:
    ctx = sum(len(v) for v in dp.context_dict.values())
    cmp = sum(len(v) for v in dp.completion_dict.values())
    return ctx + cmp


def filter_top_percent(datapoints, top_percent):
    indexed = sorted(enumerate(datapoints),
                     key=lambda x: total_context_chars(x[1]), reverse=True)
    top_n = max(1, int(len(indexed) * top_percent / 100))
    subset = [dp for _, dp in indexed[:top_n]]
    threshold = total_context_chars(indexed[top_n - 1][1])
    print(f"  Total: {len(datapoints)}, top {top_percent}% = {len(subset)} items")
    print(f"  Context length threshold: >= {threshold:,} chars")
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',  required=True)
    parser.add_argument('--seq_max_len', type=int,   default=1024,
                        help='Prompt token limit (official seq_max_len). Generation adds max_new_tokens=100 on top.')
    parser.add_argument('--dataset',     default='medium_context',
                        choices=['small_context', 'medium_context', 'large_context', 'huge_context'])
    parser.add_argument('--top_percent', type=float, default=100)
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['sdpa', 'eager', 'flash_attention_2'])
    parser.add_argument('--output_dir',  default='./results_official')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_tag = args.pretrained.rstrip('/').split('/')[-1].lower()
    results_path = os.path.join(
        args.output_dir,
        f"{model_tag}_official_plcc_{args.dataset}_top{int(args.top_percent)}pct_seq{args.seq_max_len}_generation.jsonl"
    )
    metrics_path = results_path.replace('_generation.jsonl', '_metrics.json')

    if os.path.exists(metrics_path):
        print(f"Already done: {metrics_path}")
        with open(metrics_path) as f:
            print(json.dumps(json.load(f), indent=2))
        return

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.pretrained} ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    ).eval()
    print("Model ready.")

    # ── load & prepare data ───────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset} ...")
    datapoints = load_hf_dataset(args.dataset)
    datapoints = apply_composer(datapoints)
    subset     = filter_top_percent(datapoints, args.top_percent)

    # ── run generation ────────────────────────────────────────────────────────
    em_by_cat = {}   # cat -> [per-repo mean, ...]
    es_by_cat = {}

    for idx, dp in enumerate(subset):
        # LineGeneratorHF creates a fresh tokenizer per datapoint — pass tokenizer_path
        generator = LineGeneratorHF(
            model         = model,
            device        = device,
            max_seq_len   = args.seq_max_len,
            results_path  = results_path,
            tokenizer_path= args.pretrained,
        )
        generator.generate_line(dp, use_zero_context=False)

        em = generator.calculate_exact_match()
        es = generator.calculate_edit_similarity()

        # aggregate within repo (official aggregate_metric logic)
        for cat, res in em.items():
            em_by_cat.setdefault(cat, []).append(res['exact_match'])
        for cat, res in es.items():
            es_by_cat.setdefault(cat, []).append(res['edit_similarity'] / 100.0)

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(subset)}] done")

    # ── metrics ───────────────────────────────────────────────────────────────
    print(f"\n=== PLCC {args.dataset}  seq_max_len={args.seq_max_len} ===")
    metrics = {}
    all_em, all_es = [], []
    for cat in sorted(em_by_cat.keys()):
        em_list = em_by_cat[cat]
        es_list = es_by_cat.get(cat, [])
        em_score = sum(em_list) / len(em_list) if em_list else 0.0
        es_score = sum(es_list) / len(es_list) if es_list else 0.0
        metrics[cat] = {'em': em_score, 'es': es_score, 'n_repos': len(em_list)}
        print(f"  {cat:20s}: EM={em_score:.4f}  ES={es_score:.4f}  (n_repos={len(em_list)})")
        all_em.extend(em_list)
        all_es.extend(es_list)

    overall_em = sum(all_em) / len(all_em) if all_em else 0.0
    overall_es = sum(all_es) / len(all_es) if all_es else 0.0
    metrics['overall'] = {'em': overall_em, 'es': overall_es, 'n_repos': len(all_em)}
    print(f"  {'overall':20s}: EM={overall_em:.4f}  ES={overall_es:.4f}  (n_repos={len(all_em)})")

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()
