#!/usr/bin/env python3
import argparse
import json
import math
import os
from json import JSONDecodeError
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from infinitebench_tasks import (
    SUPPORTED_TASKS,
    TASK_TO_MAX_NEW_TOKENS,
    create_prompt_parts,
    load_task_examples,
    normalize_answer_label,
    resolve_data_dir,
    score_prediction,
)


def load_existing_predictions(out_file):
    predictions = []
    completed_ids = set()
    if not os.path.exists(out_file):
        return predictions, completed_ids

    last_good_pos = 0
    saw_decode_error = False
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
            completed_ids.add(record["example_id"])
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


def summarize_metrics(records):
    scores = [int(record["correct"]) for record in records]
    return {"accuracy": (sum(scores) / len(scores)) if scores else 0.0, "n": len(scores)}


def resolve_dtype(name):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def encode_fragment(tokenizer, text):
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def bos_ids(tokenizer, add_bos_token):
    if not add_bos_token:
        return []
    if tokenizer.bos_token_id is not None:
        return [int(tokenizer.bos_token_id)]
    if tokenizer.bos_token:
        return tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
    return []


def split_token_chunks(token_ids, chunk_size, split_from_tail=False):
    if not token_ids:
        return []
    if chunk_size <= 0 or len(token_ids) <= chunk_size:
        return [token_ids]

    chunks = []
    cursor = 0
    if split_from_tail:
        leading_remainder = len(token_ids) % chunk_size
        if leading_remainder > 0:
            chunks.append(token_ids[:leading_remainder])
            cursor = leading_remainder

    while cursor < len(token_ids):
        chunks.append(token_ids[cursor:cursor + chunk_size])
        cursor += chunk_size
    return chunks


def score_chunk_self_information(model, chunk_ids, query_ids, query_window, device):
    if not chunk_ids or not query_ids:
        return float("-inf")

    if query_window and query_window > 0:
        query_ids = query_ids[-query_window:]

    joint_ids = torch.tensor([chunk_ids + query_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones(joint_ids.shape, device=device, dtype=torch.bool)
    with torch.inference_mode():
        outputs = model(joint_ids, attention_mask=attention_mask, return_dict=True, use_cache=False)

    logits = outputs.logits
    chunk_len = len(chunk_ids)
    query_len = len(query_ids)
    if query_len <= 0 or logits.shape[1] < chunk_len + query_len - 1:
        return float("-inf")

    query_logits = logits[:, chunk_len - 1:chunk_len + query_len - 1, :]
    query_labels = joint_ids[:, chunk_len:chunk_len + query_len]
    if query_logits.shape[1] != query_labels.shape[1]:
        return float("-inf")

    log_probs = F.log_softmax(query_logits.float(), dim=-1)
    token_nll = -log_probs.gather(dim=-1, index=query_labels.unsqueeze(-1)).squeeze(-1)
    return float(-token_nll.mean().item())


def select_chunks(model, candidate_chunks, scoring_query_ids, args):
    if not candidate_chunks:
        return [], {}
    if args.score_mode == "none" or not scoring_query_ids:
        selected = list(range(len(candidate_chunks)))
        if args.topk_chunks > 0:
            selected = selected[:args.topk_chunks]
        return selected, {}

    scores = {}
    for idx, chunk_ids in enumerate(candidate_chunks):
        scores[idx] = score_chunk_self_information(
            model=model,
            chunk_ids=chunk_ids,
            query_ids=scoring_query_ids,
            query_window=args.score_query_window,
            device=args.device,
        )

    forced = [0] if args.keep_first_chunk and candidate_chunks else []
    remaining = [idx for idx in range(len(candidate_chunks)) if idx not in forced]
    ranked = sorted(remaining, key=lambda idx: scores.get(idx, float("-inf")), reverse=True)
    topk = len(ranked) if args.topk_chunks <= 0 else min(args.topk_chunks, len(ranked))
    return sorted(set(forced + ranked[:topk])), scores


def pack_chunks(candidate_chunks, selected_indices, separator_ids, available_context_tokens):
    packed_chunks = []
    packed_indices = []
    used_tokens = 0
    if available_context_tokens <= 0:
        return packed_chunks, packed_indices, used_tokens

    for idx in selected_indices:
        chunk_ids = candidate_chunks[idx]
        separator_cost = len(separator_ids) if packed_chunks else 0
        remaining = available_context_tokens - used_tokens - separator_cost
        if remaining <= 0:
            continue
        if len(chunk_ids) <= remaining:
            if separator_cost:
                used_tokens += separator_cost
            packed_chunks.append(chunk_ids)
            packed_indices.append(idx)
            used_tokens += len(chunk_ids)
            continue
        if packed_chunks:
            continue
        truncated_ids = chunk_ids[-remaining:]
        if truncated_ids:
            packed_chunks.append(truncated_ids)
            packed_indices.append(idx)
            used_tokens += len(truncated_ids)
    return packed_chunks, packed_indices, used_tokens


def build_selected_prompt_ids(tokenizer, model, example, task, args):
    parts = create_prompt_parts(example, task, args.prompt_style)
    prefix_ids = bos_ids(tokenizer, args.add_bos_token) + encode_fragment(tokenizer, parts.get("prefix", ""))
    query_ids = encode_fragment(tokenizer, parts.get("query", ""))
    scoring_query_ids = encode_fragment(tokenizer, parts.get("scoring_query", ""))
    separator_ids = encode_fragment(tokenizer, args.segment_separator)
    context_ids = encode_fragment(tokenizer, parts.get("context", ""))
    candidate_chunks = split_token_chunks(
        context_ids,
        chunk_size=args.chunk_size,
        split_from_tail=args.split_from_tail,
    )

    selected_indices, scores = select_chunks(model, candidate_chunks, scoring_query_ids, args)
    prompt_budget = max(1, args.max_length - args.max_new_tokens)
    available_context_tokens = prompt_budget - len(prefix_ids) - len(query_ids)
    packed_chunks, packed_indices, context_tokens_after_pack = pack_chunks(
        candidate_chunks=candidate_chunks,
        selected_indices=selected_indices,
        separator_ids=separator_ids,
        available_context_tokens=available_context_tokens,
    )

    prompt_ids = list(prefix_ids)
    for idx, chunk_ids in enumerate(packed_chunks):
        if idx > 0:
            prompt_ids.extend(separator_ids)
        prompt_ids.extend(chunk_ids)
    prompt_ids.extend(query_ids)

    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]

    finite_scores = [scores[idx] for idx in packed_indices if idx in scores and math.isfinite(scores[idx])]
    return prompt_ids, {
        "raw_context_tokens": len(context_ids),
        "raw_prompt_tokens": len(prefix_ids) + len(context_ids) + len(query_ids),
        "candidate_chunks": len(candidate_chunks),
        "selected_chunk_indices": packed_indices,
        "selected_chunks": len(packed_indices),
        "context_tokens_after_pack": context_tokens_after_pack,
        "score_min": min(finite_scores) if finite_scores else None,
        "score_max": max(finite_scores) if finite_scores else None,
    }


def generate_official_dream(model, tokenizer, prompt_ids, args):
    input_ids = torch.tensor([prompt_ids], device=args.device, dtype=torch.long)
    attention_mask = torch.ones(input_ids.shape, device=args.device, dtype=torch.bool)
    steps = args.steps if args.steps is not None else args.max_new_tokens
    with torch.inference_mode():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alg=args.alg,
            alg_temp=args.alg_temp,
        )
    text = tokenizer.decode(output.sequences[0, input_ids.shape[1]:].tolist(), skip_special_tokens=False)
    eos = tokenizer.eos_token
    if eos and eos in text:
        text = text.split(eos)[0]
    return text


def apply_dream_rope_scaling(model, factor):
    factor = float(factor or 1.0)
    if factor <= 1.0:
        return

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Cannot apply RoPE scaling: model has no config")

    original_max_pos = getattr(config, "max_position_embeddings", None)
    if original_max_pos is None:
        raise ValueError("Cannot apply RoPE scaling: config has no max_position_embeddings")

    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_pos,
    }

    device = next(model.parameters()).device
    for module in model.modules():
        if module.__class__.__name__ == "DreamRotaryEmbedding":
            module.__init__(config=config)
            module.to(device)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate pure Dream official diffusion_generate on InfiniteBench with text-level chunk selection."
    )
    parser.add_argument("--model_path", default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"))
    parser.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default="./results_infinitebench_pure_dream_chunks")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--prompt_style", choices=["parallelcomp_raw", "yarn-mistral", "gpt4", "slot_fill"], default="parallelcomp_raw")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--add_bos_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--topk_chunks", type=int, default=3)
    parser.add_argument("--score_mode", choices=["self_information", "none"], default="self_information")
    parser.add_argument("--score_query_window", type=int, default=0)
    parser.add_argument("--keep_first_chunk", action="store_true")
    parser.add_argument("--split_from_tail", action="store_true")
    parser.add_argument("--segment_separator", default="\n\n")
    return parser


def main():
    args = build_arg_parser().parse_args()
    unknown_tasks = sorted(set(args.tasks) - set(SUPPORTED_TASKS))
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}")

    data_dir = resolve_data_dir(args.data_dir)
    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device).eval()
    apply_dream_rope_scaling(model, args.rope_scale_factor)
    model = model.to(args.device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Model path       : {args.model_path}")
    print("Model mode       : pure Dream official diffusion_generate (no LoRA, no KV cache injection)")
    print(f"Data dir         : {data_dir}")
    print(f"Tasks            : {', '.join(args.tasks)}")
    print(f"Chunk selection  : mode={args.score_mode}, chunk_size={args.chunk_size}, topk={args.topk_chunks}")
    print(f"Max length       : {args.max_length}")
    print(f"RoPE scale factor: {args.rope_scale_factor}")
    print(f"Max new tokens   : {args.max_new_tokens}")

    all_results = {}
    for task in args.tasks:
        examples = load_task_examples(task, data_dir, max_examples=args.max_examples)
        out_file = os.path.join(
            args.output_dir,
            f"pure_dream_chunks_infinitebench_{task}_n{len(examples)}_predictions.jsonl",
        )
        metrics_file = out_file.replace("_predictions.jsonl", "_metrics.json")
        predictions, completed_ids = load_existing_predictions(out_file)
        if predictions:
            print(f"[{task}] Resuming from {len(predictions)} completed examples")

        for idx, example in enumerate(examples):
            example_id = example.get("id", idx)
            if example_id in completed_ids:
                continue

            prompt_ids, prompt_meta = build_selected_prompt_ids(tokenizer, model, example, task, args)
            print(
                f"[PureDreamChunks] label={task}:{example_id} "
                f"raw_prompt_tokens={prompt_meta['raw_prompt_tokens']} "
                f"candidate_chunks={prompt_meta['candidate_chunks']} "
                f"kept_chunks={prompt_meta['selected_chunks']} "
                f"kept_chunk_indices={prompt_meta['selected_chunk_indices']} "
                f"context_tokens_after_pack={prompt_meta['context_tokens_after_pack']}"
            )

            answer_label = normalize_answer_label(task, example)
            prediction = generate_official_dream(model, tokenizer, prompt_ids, args)
            correct = score_prediction(task, prediction, answer_label)
            record = {
                "task": task,
                "example_id": example_id,
                "index": idx,
                "correct": correct,
                "prediction": prediction,
                "answer": answer_label,
                "raw_answer": example["answer"],
                "context_chars": len(example.get("context", "")),
                "input_chars": len(example.get("input", "")),
                "prompt_tokens": len(prompt_ids),
                **prompt_meta,
            }
            if "options" in example:
                record["options"] = example["options"]
            predictions.append(record)
            append_prediction_record(out_file, record)
            completed_ids.add(example_id)

            if (idx + 1) % 10 == 0:
                running = summarize_metrics(predictions)
                print(f"  [{idx + 1}/{len(examples)}] running_acc={running['accuracy']:.4f}")

        metrics = summarize_metrics(predictions)
        metrics.update({
            "task": task,
            "max_examples": len(examples),
            "prompt_style": args.prompt_style,
            "model_mode": "pure_dream_official_chunks",
            "chunk_size": args.chunk_size,
            "topk_chunks": args.topk_chunks,
            "score_mode": args.score_mode,
            "max_length": args.max_length,
            "rope_scale_factor": args.rope_scale_factor,
            "generation_max_new_tokens": args.max_new_tokens,
            "official_task_max_new_tokens": TASK_TO_MAX_NEW_TOKENS[task],
        })
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[{task}] Accuracy: {metrics['accuracy']:.4f} (n={metrics['n']})")
        all_results[task] = metrics

    combined_file = os.path.join(args.output_dir, f"pure_dream_chunks_all_n{args.max_examples}.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Combined metrics saved to: {combined_file}")


if __name__ == "__main__":
    main()
