import argparse
import json
import os
import re
import string
from collections import Counter
from difflib import SequenceMatcher
from json import JSONDecodeError
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from prefilling_model import generate, load_model

try:
    from rouge import Rouge
except Exception:
    Rouge = None

try:
    from fuzzywuzzy import fuzz
except Exception:
    fuzz = None

try:
    import jieba
except Exception:
    jieba = None


SUPPORTED_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "trec",
    "triviaqa",
    "passage_count",
    "passage_retrieval_en",
    "qmsum",
    "samsum",
    "lcc",
    "multi_news",
    "repobench-p",
    "gov_report",
)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀"
            "｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    return white_space_fix(remove_punc(s.lower()))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    if jieba is None:
        prediction_tokens = list(normalize_zh_answer(prediction))
        ground_truth_tokens = list(normalize_zh_answer(ground_truth))
    else:
        prediction_tokens = list(jieba.cut(prediction, cut_all=False))
        ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
        prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
        ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
        prediction_tokens = [token for token in prediction_tokens if token]
        ground_truth_tokens = [token for token in ground_truth_tokens if token]
    return f1_score(prediction_tokens, ground_truth_tokens)


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_score(prediction, ground_truth, **kwargs):
    if Rouge is not None:
        try:
            scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
            return scores["rouge-l"]["f"]
        except Exception:
            return 0.0

    pred_tokens = prediction.split()
    gold_tokens = ground_truth.split()
    lcs = _lcs_len(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(gold_tokens) if gold_tokens else 0.0
    return (2 * precision * recall) / (precision + recall) if precision + recall else 0.0


def rouge_zh_score(prediction, ground_truth, **kwargs):
    if jieba is not None:
        prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
        ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    em_match_list = [class_name for class_name in all_classes if class_name in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth_id) for number in numbers) / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"段落(\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth_id) for number in numbers) / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(str(number) == str(ground_truth) for number in numbers) / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    selected_line = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            selected_line = line
            break
    if fuzz is not None:
        return fuzz.ratio(selected_line, ground_truth) / 100
    return SequenceMatcher(None, selected_line, ground_truth).ratio()


DATASET_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "qasper_new": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_en_e": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "hotpotqa_new": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "2wikimqa_new": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "trec_new": classification_score,
    "triviaqa": qa_f1_score,
    "triviaqa_new": qa_f1_score,
    "samsum": rouge_score,
    "samsum_new": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_en_new": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    "repobench-p_new": code_sim_score,
}


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_data_dir(data_dir=None):
    candidates = []
    if data_dir:
        candidates.append(Path(data_dir))
    env_value = os.environ.get("LONGBENCH_DATA_DIR") or os.environ.get("LONGBENCH_DATA")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path("/home/ma-user/work/ParallelComp_official/datasets/LongBench"),
            Path(__file__).resolve().parent.parent / "ParallelComp_official" / "datasets" / "LongBench",
            Path(__file__).resolve().parent / "LongBench",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and any((candidate / f"{task}.jsonl").exists() for task in SUPPORTED_TASKS):
            return candidate

    raise FileNotFoundError(
        "LongBench data directory not found. Pass --data_dir or set LONGBENCH_DATA_DIR."
    )


def resolve_config_dir(config_dir=None):
    candidates = []
    if config_dir:
        candidates.append(Path(config_dir))
    env_value = os.environ.get("LONGBENCH_CONFIG_DIR")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            Path("/home/ma-user/work/ParallelComp_official/longbench_config"),
            Path(__file__).resolve().parent.parent / "ParallelComp_official" / "longbench_config",
            Path(__file__).resolve().parent / "longbench_config",
        ]
    )

    for candidate in candidates:
        candidate = candidate.expanduser()
        if (candidate / "dataset2prompt_raw.json").exists() and (candidate / "dataset2maxlen.json").exists():
            return candidate

    raise FileNotFoundError(
        "LongBench config directory not found. Pass --config_dir or set LONGBENCH_CONFIG_DIR."
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_task_examples(task, data_dir, max_examples):
    task_path = Path(data_dir) / f"{task}.jsonl"
    if not task_path.exists():
        raise FileNotFoundError(f"LongBench file not found: {task_path}")
    examples = list(iter_jsonl(task_path))
    if max_examples is not None and max_examples > 0:
        examples = examples[:max_examples]
    return examples


def render_prompt(template, example):
    return template.format(
        context=example.get("context", ""),
        input=example.get("input", ""),
    )


def render_prompt_parts(template, example, segment_separator):
    context_sentinel = "__LONGBENCH_CONTEXT_SENTINEL__"
    rendered = template.format(
        context=context_sentinel,
        input=example.get("input", ""),
    )
    if context_sentinel not in rendered:
        raise ValueError("LongBench prompt template is missing a {context} slot")

    prefix, query = rendered.split(context_sentinel, 1)
    return {
        "prefix": prefix,
        "context": example.get("context", ""),
        "query": query,
        "scoring_query": query,
        "segment_separator": segment_separator,
    }


def estimate_prompt_chars(prompt):
    if isinstance(prompt, str):
        return len(prompt)
    return (
        len(prompt.get("prefix", ""))
        + len(prompt.get("context", ""))
        + len(prompt.get("query", ""))
    )


def load_existing_predictions(out_file):
    predictions = []
    completed_ids = set()
    last_good_pos = 0
    saw_decode_error = False

    if not os.path.exists(out_file):
        return predictions, completed_ids

    with open(out_file, "r", encoding="utf-8") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                last_good_pos = pos
                break
            if not line.strip():
                last_good_pos = f.tell()
                continue
            try:
                record = json.loads(line)
            except JSONDecodeError:
                saw_decode_error = True
                last_good_pos = pos
                break
            predictions.append(record)
            completed_ids.add(record.get("example_id", record.get("index")))
            last_good_pos = f.tell()

    if saw_decode_error:
        with open(out_file, "rb+") as f:
            f.truncate(last_good_pos)
        print(f"Trimmed a partial JSONL tail while resuming: {out_file}")

    return predictions, completed_ids


def append_prediction_record(out_file, record):
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def score_prediction(task, prediction, answers, all_classes):
    metric_fn = DATASET_TO_METRIC[task]
    pred = prediction
    if task in {"trec", "trec_new", "triviaqa", "triviaqa_new", "samsum", "samsum_new", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]

    best = 0.0
    for ground_truth in answers:
        best = max(best, metric_fn(pred, ground_truth, all_classes=all_classes))
    return float(best)


def summarize_metrics(records, longbench_e=False):
    scores = [float(record["score"]) for record in records]
    metrics = {
        "score": round(100 * sum(scores) / len(scores), 2) if scores else 0.0,
        "score_normalized": (sum(scores) / len(scores)) if scores else 0.0,
        "n": len(scores),
    }
    if longbench_e:
        buckets = {"0-4k": [], "4-8k": [], "8k+": []}
        for record in records:
            length = int(record.get("length") or 0)
            if length < 4000:
                buckets["0-4k"].append(float(record["score"]))
            elif length < 8000:
                buckets["4-8k"].append(float(record["score"]))
            else:
                buckets["8k+"].append(float(record["score"]))
        metrics["length_buckets"] = {
            name: round(100 * sum(values) / len(values), 2) if values else 0.0
            for name, values in buckets.items()
        }
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate Prefilling-dLLM with Dream on LongBench")
    parser.add_argument("--model_type", choices=["llada", "dream"], required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=20,
        help="Number of examples per task. Use 0 or a negative value for the full file.",
    )
    parser.add_argument("--run_name", default="dream_parallelcomp")
    parser.add_argument("--output_dir", default="./results_longbench")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument(
        "--truncate_strategy",
        choices=["left", "head_tail"],
        default="left",
        help="Prompt truncation when tokenized prompt exceeds the budget. head_tail drops the middle.",
    )
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--block_add_threshold", type=float, default=0.5)
    parser.add_argument("--skip_threshold", type=float, default=1.0)
    parser.add_argument("--decoded_token_threshold", type=float, default=0.9)
    parser.add_argument("--stop_tokens", nargs="*", default=[])
    parser.add_argument(
        "--segment_separator",
        default="\n\n",
        help="Text inserted between selected pre-runtime chunks after compression.",
    )
    parser.add_argument("--longbench_e", action="store_true")

    parser.add_argument("--parallelcomp_mode", action="store_true")
    parser.add_argument("--parallelcomp_pre_runtime_mode", action="store_true")
    parser.add_argument("--parallelcomp_cache_compress_mode", action="store_true")
    parser.add_argument("--parallelcomp_chunk_size", type=int, default=1024)
    parser.add_argument("--parallelcomp_query_tokens", type=int, default=0)
    parser.add_argument("--parallelcomp_topk_chunks", type=int, default=3)
    parser.add_argument("--parallelcomp_min_prompt_tokens", type=int, default=1)
    parser.add_argument("--parallelcomp_keep_first_chunk", action="store_true")
    parser.add_argument("--parallelcomp_split_from_tail", action="store_true")
    parser.add_argument("--parallelcomp_chunk_score_query_window", type=int, default=0)
    parser.add_argument(
        "--parallelcomp_chunk_score_attention_mask",
        choices=["causal", "full", "full_visible", "query_to_chunk", "prefix_full"],
        default="query_to_chunk",
    )
    parser.add_argument("--parallelcomp_recent_token_window", type=int, default=0)
    parser.add_argument("--parallelcomp_hidden_topk", type=int, default=32)
    parser.add_argument("--parallelcomp_token_capacity", type=int, default=128)
    parser.add_argument("--parallelcomp_token_keep_min", type=int, default=32)
    parser.add_argument("--parallelcomp_high_score_threshold", type=float, default=None)
    parser.add_argument("--parallelcomp_select_low_score_chunks", action="store_true")
    parser.add_argument(
        "--parallelcomp_fixed_query_text",
        default="Please answer the question using the long context above.",
    )
    parser.add_argument(
        "--parallelcomp_tail_replay_full_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parallelcomp_score_mode", type=str, default="self_information")
    return parser


def main():
    args = build_arg_parser().parse_args()

    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported LongBench tasks: {unknown_tasks}")

    data_dir = resolve_data_dir(args.data_dir)
    config_dir = resolve_config_dir(args.config_dir)
    prompt_templates = load_json(config_dir / "dataset2prompt_raw.json")
    dataset2maxlen = load_json(config_dir / "dataset2maxlen.json")
    max_examples = None if args.max_examples <= 0 else args.max_examples
    generation_max_new_tokens = args.max_new_tokens or max(
        int(dataset2maxlen[task]) for task in args.tasks
    )

    print(f"Data dir           : {data_dir}")
    print(f"Config dir         : {config_dir}")
    print(f"Tasks              : {', '.join(args.tasks)}")
    print(f"Run name           : {args.run_name}")
    print(f"Model max_length   : {args.max_length}")
    print(f"Generation max_new : {generation_max_new_tokens}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_type} model from {args.pretrained}...")
    model = load_model(
        args.model_type,
        args.pretrained,
        args.lora_path,
        rope_scale_factor=args.rope_scale_factor,
        max_new_tokens=generation_max_new_tokens,
        max_length=args.max_length,
        truncate_strategy=args.truncate_strategy,
        block_size=args.block_size,
        temperature=args.temperature,
        dtype=args.dtype,
        block_add_threshold=args.block_add_threshold,
        skip_threshold=args.skip_threshold,
        decoded_token_threshold=args.decoded_token_threshold,
        add_bos_token=True,
        parallelcomp_mode=args.parallelcomp_mode,
        parallelcomp_pre_runtime_mode=args.parallelcomp_pre_runtime_mode,
        parallelcomp_cache_compress_mode=args.parallelcomp_cache_compress_mode,
        parallelcomp_chunk_size=args.parallelcomp_chunk_size,
        parallelcomp_query_tokens=args.parallelcomp_query_tokens,
        parallelcomp_topk_chunks=args.parallelcomp_topk_chunks,
        parallelcomp_min_prompt_tokens=args.parallelcomp_min_prompt_tokens,
        parallelcomp_keep_first_chunk=args.parallelcomp_keep_first_chunk,
        parallelcomp_split_from_tail=args.parallelcomp_split_from_tail,
        parallelcomp_chunk_score_query_window=args.parallelcomp_chunk_score_query_window,
        parallelcomp_chunk_score_attention_mask=args.parallelcomp_chunk_score_attention_mask,
        parallelcomp_recent_token_window=args.parallelcomp_recent_token_window,
        parallelcomp_hidden_topk=args.parallelcomp_hidden_topk,
        parallelcomp_token_capacity=args.parallelcomp_token_capacity,
        parallelcomp_token_keep_min=args.parallelcomp_token_keep_min,
        parallelcomp_high_score_threshold=args.parallelcomp_high_score_threshold,
        parallelcomp_select_low_score_chunks=args.parallelcomp_select_low_score_chunks,
        parallelcomp_fixed_query_text=args.parallelcomp_fixed_query_text,
        parallelcomp_tail_replay_full_mask=args.parallelcomp_tail_replay_full_mask,
        parallelcomp_score_mode=args.parallelcomp_score_mode,
    )

    all_results = {}

    for task in args.tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task}")
        print("=" * 60)

        examples = load_task_examples(task, data_dir, max_examples=max_examples)
        print(f"Loaded examples    : {len(examples)}")
        print(f"Official max_new   : {dataset2maxlen[task]}")

        task_dir = Path(args.output_dir) / task
        task_dir.mkdir(parents=True, exist_ok=True)
        out_file = task_dir / f"{args.run_name}.json"
        metrics_file = task_dir / f"{args.run_name}_metrics.json"

        if metrics_file.exists():
            print(f"Already done, skipping. ({metrics_file})")
            with open(metrics_file, "r", encoding="utf-8") as f:
                all_results[task] = json.load(f)
            continue

        predictions, completed_ids = load_existing_predictions(out_file)
        if predictions:
            print(f"Resuming from {len(predictions)} completed examples in {out_file}")

        template = prompt_templates[task]

        for idx, example in enumerate(examples):
            example_id = example.get("_id", idx)
            if example_id in completed_ids:
                continue

            if args.parallelcomp_pre_runtime_mode and args.model_type == "dream":
                prompt = render_prompt_parts(template, example, args.segment_separator)
                prompt["metadata_label"] = f"{task}:{example_id}"
            else:
                prompt = render_prompt(template, example)

            prediction = generate(model, [prompt], stop_tokens=args.stop_tokens)[0]
            answers = example.get("answers", [])
            all_classes = example.get("all_classes")
            score = score_prediction(task, prediction, answers, all_classes)

            record = {
                "task": task,
                "example_id": example_id,
                "index": idx,
                "pred": prediction,
                "answers": answers,
                "all_classes": all_classes,
                "length": example.get("length"),
                "score": score,
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "prompt_chars": estimate_prompt_chars(prompt),
            }
            predictions.append(record)
            append_prediction_record(out_file, record)
            completed_ids.add(example_id)

            if (idx + 1) % 10 == 0:
                running = summarize_metrics(predictions, longbench_e=args.longbench_e)
                print(
                    f"  [{idx + 1}/{len(examples)}] "
                    f"running_score={running['score']:.2f}"
                )

        metrics = summarize_metrics(predictions, longbench_e=args.longbench_e)
        metrics["task"] = task
        metrics["max_examples"] = len(examples)
        metrics["official_task_max_new_tokens"] = int(dataset2maxlen[task])
        metrics["generation_max_new_tokens"] = generation_max_new_tokens
        metrics["run_name"] = args.run_name

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Predictions saved to: {out_file}")
        print(f"LongBench score      : {metrics['score']:.2f} (n={metrics['n']})")
        print(f"Metrics saved to     : {metrics_file}")

        all_results[task] = metrics

    combined_file = Path(args.output_dir) / f"{args.run_name}_all_n{args.max_examples}.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
