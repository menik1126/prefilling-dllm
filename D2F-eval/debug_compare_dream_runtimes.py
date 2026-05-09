import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

D2F_EVAL = os.path.dirname(os.path.abspath(__file__))
if D2F_EVAL not in sys.path:
    sys.path.insert(0, D2F_EVAL)

from d2f_model import load_model
from eval_d2f_plcc import build_repo_context, filter_top_percent, total_context_chars


MODEL_PATH = os.path.join(D2F_EVAL, "model_weights", "Dream-v0-Base-7B")
LORA_PATH = os.path.join(D2F_EVAL, "model_weights", "D2F_Dream_Base_7B_Lora")
CONFIG = "medium_context"
TOP_PERCENT = 30
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 128
BLOCK_SIZE = 32
TEMPERATURE = 0.0
CHAR_MULTIPLIER = 6
TRACE_STEPS = 3


class TraceComplete(Exception):
    pass


def _tensor_slice(t: Optional[torch.Tensor], k: int = 8) -> Optional[Dict[str, Any]]:
    if t is None:
        return None
    t = t.detach().cpu()
    flat = t.reshape(-1)
    head = flat[:k].tolist()
    tail = flat[-k:].tolist() if flat.numel() > k else head
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": int(flat.min().item()) if flat.numel() else None,
        "max": int(flat.max().item()) if flat.numel() else None,
        "head": head,
        "tail": tail,
    }


def _mask_summary(mask: Optional[torch.Tensor], k: int = 8) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None
    mask = mask.detach().cpu()
    visible = (mask[0, 0] == 0).sum(dim=1)
    head = visible[:k].tolist()
    tail = visible[-k:].tolist() if visible.numel() > k else head
    return {
        "shape": list(mask.shape),
        "visible_counts_head": head,
        "visible_counts_tail": tail,
    }


def _block_states_summary(block_states: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for block_id, state in sorted(block_states.items()):
        out.append(
            {
                "block_id": int(block_id),
                "start_pos": int(state["start_pos"]),
                "end_pos": int(state["end_pos"]),
                "state": state["state"],
                "mask_count": int(state["mask_count"]),
                "total_masks": int(state["total_masks"]),
                "is_prompt_window": bool(state.get("is_prompt_window", False)),
            }
        )
    return out


@dataclass
class RuntimeTracer:
    runtime_name: str
    target_filename: str
    before_line: int
    after_line: int
    max_steps: int = TRACE_STEPS
    events: List[Dict[str, Any]] = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def _capture(self, frame, phase: str) -> None:
        step = int(frame.f_locals.get("step", -1))
        key = (phase, step)
        if key in self._seen:
            return
        self._seen.add(key)

        locals_ = frame.f_locals
        input_seq = locals_.get("input_seq")
        attention_mask = locals_.get("attention_mask")
        event = {
            "runtime": self.runtime_name,
            "phase": phase,
            "step": step,
            "cache_length": int(locals_.get("cache_length", -1)),
            "process_start_pos": int(locals_.get("process_start_pos", -1)),
            "update_kvcache": int(locals_.get("update_kvcache", -1)),
            "input_length": int(input_seq.shape[1]) if input_seq is not None else None,
            "input_ids": _tensor_slice(input_seq, k=12),
            "attention_mask": _mask_summary(attention_mask),
            "block_states": _block_states_summary(locals_.get("block_states", {})),
        }

        if "input_block_ids" in locals_:
            event["input_block_ids"] = _tensor_slice(locals_["input_block_ids"], k=16)
        if "input_prompt_window_mask" in locals_:
            mask = locals_["input_prompt_window_mask"].detach().cpu()
            event["input_prompt_window_mask"] = {
                "shape": list(mask.shape),
                "sum": int(mask.sum().item()),
                "head": mask[:16].int().tolist(),
                "tail": mask[-16:].int().tolist() if mask.numel() > 16 else mask[:16].int().tolist(),
            }
        if "input_rope_positions" in locals_:
            event["input_rope_positions"] = _tensor_slice(locals_["input_rope_positions"], k=16)
        if "input_cache_positions" in locals_:
            event["input_cache_positions"] = _tensor_slice(locals_["input_cache_positions"], k=16)
        elif input_seq is not None:
            cache_start = event["cache_length"]
            inferred = torch.arange(cache_start, cache_start + input_seq.shape[1], dtype=torch.long)
            event["inferred_cache_position"] = _tensor_slice(inferred, k=16)
            event["inferred_position_ids"] = _tensor_slice(inferred, k=16)

        outputs = locals_.get("outputs")
        if outputs is not None:
            logits = outputs.logits[:, -1, :].detach().cpu()
            topk = torch.topk(logits[0], k=5)
            event["last_logits_topk"] = [
                {"token_id": int(idx), "logit": float(val)}
                for val, idx in zip(topk.values.tolist(), topk.indices.tolist())
            ]

        self.events.append(event)

        after_events = [e for e in self.events if e["phase"] == "after_forward"]
        if len(after_events) >= self.max_steps:
            raise TraceComplete

    def trace_fn(self, frame, event, arg):
        if event != "line":
            return self.trace_fn
        if frame.f_code.co_name != "_generate_block_single":
            return self.trace_fn
        if os.path.basename(frame.f_code.co_filename) != self.target_filename:
            return self.trace_fn

        if frame.f_lineno == self.before_line:
            self._capture(frame, "before_forward")
        elif frame.f_lineno == self.after_line:
            self._capture(frame, "after_forward")
        return self.trace_fn


def build_single_prompt() -> Dict[str, Any]:
    ds = load_dataset("JetBrains-Research/lca-project-level-code-completion", CONFIG, split="test")
    subset = filter_top_percent(list(ds), TOP_PERCENT, total_context_chars)
    prompt_token_limit = MAX_LENGTH - MAX_NEW_TOKENS
    max_prompt_chars = prompt_token_limit * CHAR_MULTIPLIER

    for item_idx, item in enumerate(subset):
        snap = item.get("repo_snapshot", {})
        cf = item.get("completion_file", {})
        completion_lines = item.get("completion_lines", {})
        cf_filename = cf.get("filename", "unknown.py") if isinstance(cf, dict) else "unknown.py"
        cf_content = cf.get("content", "") if isinstance(cf, dict) else ""
        cf_line_list = cf_content.split("\n")
        repo_ctx = build_repo_context(snap, completion_filepath=cf_filename)

        line_prompts = []
        for cat, line_nums in completion_lines.items():
            if not line_nums:
                continue
            for ln in line_nums:
                prefix = "\n".join(cf_line_list[:ln])
                prompt = repo_ctx + f"\n\n# path: {cf_filename}\n{prefix}"
                if len(prompt) > max_prompt_chars:
                    prompt = prompt[-max_prompt_chars:]
                gt = cf_line_list[ln] if ln < len(cf_line_list) else ""
                line_prompts.append((cat, ln, prompt, gt))

        if line_prompts:
            cat, line_no, prompt, gt = line_prompts[0]
            return {
                "item_idx": item_idx,
                "category": cat,
                "line_no": int(line_no),
                "prompt": prompt,
                "ground_truth": gt,
                "prompt_chars": len(prompt),
            }

    raise RuntimeError("Failed to build any PLCC prompt for debug comparison")


def run_trace(runtime_name: str, prompt: str) -> Dict[str, Any]:
    kwargs = dict(
        pretrained=MODEL_PATH,
        lora_path=LORA_PATH,
        rope_scale_factor=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_LENGTH,
        block_size=BLOCK_SIZE,
        temperature=TEMPERATURE,
        add_bos_token=True,
    )
    if runtime_name == "on":
        kwargs.update(
            parallelcomp_mode=True,
            parallelcomp_cache_compress_mode=True,
            parallelcomp_chunk_size=1024,
            parallelcomp_topk_chunks=4,
            parallelcomp_min_prompt_tokens=1,
            parallelcomp_fixed_query_text="Please complete the preceding code.",
        )
        tracer = RuntimeTracer(
            runtime_name="on",
            target_filename="eval_dream.py",
            before_line=2196,
            after_line=2207,
        )
    else:
        kwargs.update(
            parallelcomp_mode=False,
            parallelcomp_cache_compress_mode=False,
        )
        tracer = RuntimeTracer(
            runtime_name="off",
            target_filename="eval_dream_h20_0312.py",
            before_line=705,
            after_line=715,
        )

    model = load_model("dream", **kwargs)
    inner = model._inner

    prompt_ids = inner.tokenizer.encode(inner.tokenizer.bos_token + prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=inner.device, dtype=torch.long)
    prompt_limit = inner.max_length - inner.max_new_tokens
    if prompt_tensor.shape[1] > prompt_limit:
        prompt_tensor = prompt_tensor[:, -prompt_limit:]

    old_trace = sys.gettrace()
    sys.settrace(tracer.trace_fn)
    try:
        try:
            inner._generate_block_single(prompt_tensor)
        except TraceComplete:
            pass
    finally:
        sys.settrace(old_trace)

    result = {
        "runtime": runtime_name,
        "prompt_tokens_after_tokenizer_truncation": int(prompt_tensor.shape[1]),
        "trace_events": tracer.events,
    }

    del model
    del inner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def run_trace_subprocess(runtime_name: str, prompt: str) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--runtime",
        runtime_name,
    ]
    payload = {"prompt": prompt}
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--runtime":
        payload = json.loads(sys.stdin.read())
        result = run_trace(sys.argv[2], payload["prompt"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    sample = build_single_prompt()
    output = {
        "sample": sample,
        "results": [],
    }
    for runtime_name in ("off", "on"):
        output["results"].append(run_trace_subprocess(runtime_name, sample["prompt"]))

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
