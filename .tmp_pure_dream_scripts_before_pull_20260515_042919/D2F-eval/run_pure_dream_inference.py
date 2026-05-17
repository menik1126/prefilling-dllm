#!/usr/bin/env python3
import argparse
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pure Dream inference via the official diffusion_generate API."
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get("DREAM_BASE", "Dream-org/Dream-v0-Base-7B"),
        help="Dream model path or HF repo. Use Base or Instruct checkpoint directly.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chat", action="store_true", help="Use chat template for instruct checkpoints.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--output_history", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def resolve_dtype(name):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


def main():
    args = parse_args()
    steps = args.steps if args.steps is not None else args.max_new_tokens
    dtype = resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device).eval()

    if args.chat:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")

    input_ids = inputs.input_ids.to(args.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    with torch.inference_mode():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=args.output_history,
            return_dict_in_generate=True,
            steps=steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alg=args.alg,
            alg_temp=args.alg_temp,
        )

    generations = [
        tokenizer.decode(g[len(p):].tolist(), skip_special_tokens=False)
        for p, g in zip(input_ids, output.sequences)
    ]
    text = generations[0]
    eos = tokenizer.eos_token
    if eos and eos in text:
        text = text.split(eos)[0]

    print(json.dumps(
        {
            "model_path": args.model_path,
            "chat": args.chat,
            "max_new_tokens": args.max_new_tokens,
            "steps": steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "alg": args.alg,
            "generation": text,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
