"""
eval_ar_plcc.py - Evaluate any autoregressive (causal LM) on LCA PLCC.

Matches the official LCA baseline (lca-baselines/project_level_code_completion):
  1. File filtering: all .py + non-hidden non-license non-.py < 16 KB.
  2. Completion file excluded from repo context; only its path marker appended.
  3. Files sorted by path distance (farthest first, closest last).
  4. Prompt format: {repo_name}METASEP\n{path}METASEP\n{content}...{cf_path}METASEP\n\n{prefix}
     (meta_info_sep_symbol = "METASEP\n", lang_sep_symbol = "", extension = "")
  5. Character pre-truncation (x6) then token truncation from the left.
  6. StopOnNewLine: scans full vocab for all tokens containing newline,
     and requires minimum 5 generated tokens before triggering.
  7. max_new_tokens default = 100 (official).
  8. EM: pred.strip() == gt.strip(); ES: fuzz.ratio(pred, gt) (no strip).
  9. EM aggregation: per-repo average first, then average across repos (official method).

Usage:
  python eval_ar_plcc.py \\
    --pretrained model_weights/Qwen2.5-7B \\
    --top_percent 30 \\
    --max_length 16384 \\
    --output_dir ./results_ar
"""
import argparse
import json
import os

import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

CONFIGS = ['small_context', 'medium_context', 'large_context', 'huge_context']

META_SEP = 'METASEP\n'  # official: meta_info_sep_symbol = "METASEP\n" (python_standard.yaml)


# ── Shared utility functions ─────────────────────────────────────────────────

def _path_distance(path_from, path_to):
    """Directory-tree distance. Matches official PathDistanceComposer exactly."""
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
    Build repo context following official LCA pipeline:
    - Filter: all .py + non-hidden, non-license non-.py files < 16 KB
    - Exclude completion file (only path marker goes at the end)
    - Sort farthest-first so closest files survive left-truncation
    - Format: {repo_name}METASEP\n{path}METASEP\n{content}...{cf_path}METASEP\n
      (matches python_standard.yaml: meta_info_sep_symbol="METASEP\n", lang_sep_symbol="")
    Caller appends "\n{prefix}" to complete the prompt, producing a blank line
    between the completion file marker and the prefix (matches official _get_context).
    """
    filenames = snap.get('filename', [])
    contents  = snap.get('content', [])

    pairs = []
    for fn, ct in zip(filenames, contents):
        if fn == completion_filepath:
            continue  # completion file: path marker only, no content
        if fn.endswith('.py'):
            pairs.append((fn, ct))
        else:
            basename = os.path.basename(fn)
            if not basename.startswith('.') and 'license' not in fn.lower():
                if len(ct) < 16000:
                    pairs.append((fn, ct))

    if completion_filepath:
        pairs.sort(key=lambda x: -_path_distance(completion_filepath, x[0]))

    parts = [f"{fn}{META_SEP}{ct}" for fn, ct in pairs]
    if completion_filepath:
        parts.append(f"{completion_filepath}{META_SEP}")

    header = f"{repo_name}{META_SEP}" if repo_name else ""
    return header + "".join(parts)


def total_context_chars(item):
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
    """Edit similarity: fuzz.ratio without stripping (matches official)."""
    try:
        from thefuzz import fuzz
        return fuzz.ratio(pred, gt) / 100.0
    except Exception:
        return 1.0 if pred == gt else 0.0


# ── AR model wrapper ─────────────────────────────────────────────────────────

class ARModel:
    def __init__(self, pretrained, max_length, max_new_tokens, rope_scale_factor=1.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, \
            StoppingCriteria, StoppingCriteriaList

        print(f"Loading tokenizer from {pretrained} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True,
            padding_side='left',
            truncation_side='left',   # keep end of context (near completion point)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Loading model from {pretrained} ...")
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)

        # Inject NTK-by-parts (YaRN) if requested and not already set
        if rope_scale_factor > 1.0:
            orig = getattr(config, 'max_position_embeddings', max_length)
            existing = getattr(config, 'rope_scaling', None)
            if existing is None:
                config.rope_scaling = {
                    'type': 'yarn',
                    'rope_type': 'yarn',
                    'factor': float(rope_scale_factor),
                    'original_max_position_embeddings': orig,
                }
                config.max_position_embeddings = int(orig * rope_scale_factor)
                print(f"  NTK-by-parts applied: {orig} -> {config.max_position_embeddings} tokens")
            else:
                print(f"  Model already has rope_scaling ({existing.get('type','?')}), skipping.")

        # Force disable sliding window to match official baseline which used flash_attention_2
        # (flash_attention_2 previously ignored SWA, while sdpa respects it, causing discrepancies)
        if hasattr(config, 'use_sliding_window'):
            config.use_sliding_window = False

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()

        self.max_new_tokens = max_new_tokens
        self.prompt_limit   = max_length - max_new_tokens


        # Official StopOnNewLine: scan full vocab for ALL tokens containing '\n'
        vocab = self.tokenizer.get_vocab()
        nl_token_ids = set()
        for tok, tok_id in vocab.items():
            s = self.tokenizer.convert_tokens_to_string([tok])
            if '\n' in s:
                nl_token_ids.add(tok_id)
        self.nl_token_ids = nl_token_ids
        print(f"  Prompt token limit   : {self.prompt_limit}")
        print(f"  Newline token count  : {len(nl_token_ids)} tokens trigger stop")

        # Capture for use inside generate_prompts
        _nl_ids = nl_token_ids

        class StopOnNewLine(StoppingCriteria):
            """
            Official LCA stopping criterion:
            - Wait at least 5 tokens before triggering
            - Stop when the last generated token contains a newline character
            """
            def __init__(self):
                self._generated = 0

            def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
                if self._generated < 5:
                    self._generated += 1
                    return False
                if input_ids[0, -1].item() in _nl_ids:
                    self._generated = 0
                    return True
                self._generated += 1
                return False

        self._StopOnNewLine = StopOnNewLine
        self._StoppingCriteriaList = StoppingCriteriaList

    def generate_prompts(self, prompts):
        """Generate one line per prompt. Returns list of predicted strings."""
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.prompt_limit,
                add_special_tokens=True,
            )
            device = next(self.model.parameters()).device
            ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            stopping_criteria = self._StoppingCriteriaList([self._StopOnNewLine()])

            with torch.no_grad():
                out = self.model.generate(
                    ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                )

            new_tokens = out[0][ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Official: strip leading/trailing newlines, then take first line
            results.append(text.strip('\n').split('\n')[0])

        return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Eval any causal LM on LCA PLCC')
    parser.add_argument('--pretrained', required=True,
                        help='HuggingFace model id or local path')
    parser.add_argument('--model_tag', default=None,
                        help='Short name for output filenames (default: last path component).')
    parser.add_argument('--top_percent', type=float, default=30)
    parser.add_argument('--configs', nargs='+', default=CONFIGS)
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Max tokens to generate per line (official default: 100).')
    parser.add_argument('--max_length', type=int, default=16384,
                        help='Total context window tokens (prompt + generation).')
    parser.add_argument('--rope_scale_factor', type=float, default=1.0,
                        help='NTK-by-parts RoPE scale (1.0=off).')
    parser.add_argument('--output_dir', default='./results_ar')
    args = parser.parse_args()

    if args.model_tag is None:
        args.model_tag = args.pretrained.rstrip('/').split('/')[-1].lower()

    prompt_token_limit = args.max_length - args.max_new_tokens
    max_prompt_chars   = prompt_token_limit * 6

    print(f"Model tag          : {args.model_tag}")
    print(f"Prompt token limit : {prompt_token_limit}")
    print(f"Pre-truncation chars: {max_prompt_chars:,}  (={prompt_token_limit}x6)")

    model = ARModel(
        pretrained=args.pretrained,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        rope_scale_factor=args.rope_scale_factor,
    )

    from datasets import load_dataset

    all_results = {}

    for cfg in args.configs:
        print(f"\n{'='*60}")
        print(f"Config: {cfg}")
        print('='*60)

        scale_tag = f"_rope{args.rope_scale_factor:.1f}" if args.rope_scale_factor != 1.0 else ""
        out_file = os.path.join(
            args.output_dir,
            f"{args.model_tag}_plcc_{cfg}_top{int(args.top_percent)}pct{scale_tag}_predictions.jsonl"
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

        em_by_cat = {}   # cat -> list of per-repo EM means  (one scalar per repo)
        es_by_cat = {}   # cat -> list of per-repo ES means
        predictions_all = []

        for ex_idx, item in enumerate(subset):
            snap             = item.get('repo_snapshot', {})
            cf               = item.get('completion_file', {})
            completion_lines = item.get('completion_lines', {})

            cf_filename  = cf.get('filename', 'unknown.py') if isinstance(cf, dict) else 'unknown.py'
            cf_content   = cf.get('content', '')            if isinstance(cf, dict) else ''
            cf_line_list = cf_content.split('\n')
            repo_name    = item.get('repo', '')

            repo_ctx = build_repo_context(snap, completion_filepath=cf_filename,
                                          repo_name=repo_name)

            example_rec = {
                'repo':            repo_name,
                'config':          cfg,
                'completion_file': cf_filename,
                'predictions':     {},
            }

            line_prompts = []
            for cat, line_nums in completion_lines.items():
                if not line_nums:
                    continue
                for ln in line_nums:
                    prefix = '\n'.join(cf_line_list[:ln])
                    # Official prompt: repo_ctx ends with "cf_path METASEP\n",
                    # then "\n" + prefix gives the blank-line separator seen in
                    # official _get_context: "\n".join([datapoint.context, prefix])
                    prompt = repo_ctx + "\n" + prefix
                    if len(prompt) > max_prompt_chars:
                        prompt = prompt[-max_prompt_chars:]
                    gt = cf_line_list[ln] if ln < len(cf_line_list) else ''
                    line_prompts.append((cat, ln, prompt, gt))

            if not line_prompts:
                predictions_all.append(example_rec)
                continue

            prompts = [lp[2] for lp in line_prompts]
            outputs = model.generate_prompts(prompts)

            # Accumulate per-repo line-level scores, then aggregate to repo-level means.
            # This matches the official approach: aggregate_metric() per datapoint, then
            # average repo-level scalars — giving each repo equal weight.
            repo_em_by_cat = {}   # cat -> [0/1, ...] within this repo
            repo_es_by_cat = {}

            for (cat, ln, _, gt), pred_line in zip(line_prompts, outputs):
                em_score = int(pred_line.strip() == gt.strip())
                es_score = edit_similarity(pred_line, gt)

                repo_em_by_cat.setdefault(cat, []).append(em_score)
                repo_es_by_cat.setdefault(cat, []).append(es_score)

                example_rec['predictions'].setdefault(cat, []).append({
                    'line_num': ln,
                    'gt':       gt,
                    'pred':     pred_line,
                    'em':       em_score,
                    'es':       es_score,
                })

            # One mean per repo per category → appended to global lists
            for cat in repo_em_by_cat:
                em_by_cat.setdefault(cat, []).append(
                    sum(repo_em_by_cat[cat]) / len(repo_em_by_cat[cat]))
                es_by_cat.setdefault(cat, []).append(
                    sum(repo_es_by_cat[cat]) / len(repo_es_by_cat[cat]))

            predictions_all.append(example_rec)

            if (ex_idx + 1) % 5 == 0:
                print(f"  [{ex_idx+1}/{len(subset)}] done")

        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_file, 'w') as f:
            for rec in predictions_all:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"\nPredictions saved to: {out_file}")

        metrics = {}
        print(f"\n=== PLCC {cfg} Metrics ===")
        all_em, all_es = [], []
        for cat in sorted(em_by_cat.keys()):
            em_list = em_by_cat[cat]
            es_list = es_by_cat[cat]
            em_score = sum(em_list) / len(em_list) if em_list else 0.0
            es_score = sum(es_list) / len(es_list) if es_list else 0.0
            # n = number of repos (not lines) — matches official per-repo aggregation
            metrics[cat] = {'em': em_score, 'es': es_score, 'n': len(em_list)}
            print(f"  {cat:20s}: EM={em_score:.4f}, ES={es_score:.4f}  (n_repos={len(em_list)})")
            all_em.extend(em_list)
            all_es.extend(es_list)

        overall_em = sum(all_em) / len(all_em) if all_em else 0.0
        overall_es = sum(all_es) / len(all_es) if all_es else 0.0
        metrics['overall'] = {'em': overall_em, 'es': overall_es, 'n': len(all_em)}
        print(f"  {'overall':20s}: EM={overall_em:.4f}, ES={overall_es:.4f}  (n_repos={len(all_em)})")

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")

        all_results[cfg] = metrics

    combined_file = os.path.join(
        args.output_dir,
        f"{args.model_tag}_plcc_all_top{int(args.top_percent)}pct_metrics.json"
    )
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll configs combined metrics: {combined_file}")


if __name__ == '__main__':
    main()
