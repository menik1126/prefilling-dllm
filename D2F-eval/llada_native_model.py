#!/usr/bin/env python3
"""Native LLaDA loading and masked-diffusion generation helpers.

This module intentionally does not import Fast-DLLM, SparseD, dKV, or any
cache-based acceleration path. It is for vanilla LLaDA baselines only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_cache.llada.configuration_llada import LLaDAConfig
from model_cache.llada.modeling_llada import LLaDAModelLM, RotaryEmbedding


def resolve_dtype(name):
    if name in (None, "auto"):
        return "auto"
    if isinstance(name, torch.dtype):
        return name
    return getattr(torch, str(name))


def get_model_input_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def set_model_device_attr(model, device):
    try:
        model.device = torch.device(device)
    except Exception:
        pass


def apply_llada_yarn_scaling(model, factor: float) -> None:
    """Apply the local LLaDA RoPE/YaRN scaling hook when factor > 1."""
    factor = float(factor or 1.0)
    if factor <= 1.0:
        return

    config = getattr(model, "config", None)
    original_max_pos = None
    if config is not None:
        original_max_pos = (
            getattr(config, "max_sequence_length", None)
            or getattr(config, "max_position_embeddings", None)
        )
        if original_max_pos is not None:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": int(original_max_pos),
            }
            if hasattr(config, "max_sequence_length"):
                config.max_sequence_length = max(
                    int(config.max_sequence_length),
                    int(int(original_max_pos) * factor),
                )

    for module in model.modules():
        if isinstance(module, RotaryEmbedding) or module.__class__.__name__ == "RotaryEmbedding":
            module.ntk_scale_factor = factor
            cache = getattr(module, "_RotaryEmbedding__cache", None)
            if isinstance(cache, dict):
                cache.pop("rope_pos_sin", None)
                cache.pop("rope_pos_cos", None)
                cache.pop("rope_scale_factor", None)


@dataclass
class LLaDANativeConfig:
    model_path: str
    device: str = "cuda"
    device_map: Optional[str] = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 32
    block_length: int = 32
    steps: Optional[int] = None
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_token_id: int = 126336
    rope_scale_factor: float = 1.0
    suffix_logits_only: bool = True

    @property
    def generation_steps(self) -> int:
        return int(self.steps if self.steps is not None else self.max_new_tokens)


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    total = mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    cols = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    return base.unsqueeze(1) + (cols < rem.unsqueeze(1)).long()


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(x0.shape, device=x0.device)
    else:
        raise ValueError(f"Unsupported remasking strategy: {remasking}")

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(confidence.dtype).min, device=x.device, dtype=confidence.dtype)
    confidence = torch.where(mask_index, confidence, neg_inf)

    transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
    for b in range(mask_index.shape[0]):
        k = int(num_transfer_tokens[b].item())
        if k <= 0:
            continue
        k = min(k, int(mask_index[b].sum().item()))
        if k <= 0:
            continue
        _, idx = torch.topk(confidence[b], k=k)
        transfer_index[b, idx] = True
    return x0, transfer_index


def distribute_steps(total_steps: int, num_blocks: int):
    if total_steps < 1:
        raise ValueError("steps must be positive")
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    base = total_steps // num_blocks
    rem = total_steps - base * num_blocks
    schedule = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    return [max(1, steps) for steps in schedule]


def forward_logits(model, x, suffix_start, suffix_logits_only):
    if suffix_logits_only:
        try:
            return model(x, logits_slice_start=suffix_start).logits, int(suffix_start)
        except TypeError:
            pass
    return model(x).logits, 0


@torch.inference_mode()
def native_llada_generate(
    model,
    prompt,
    gen_length: int,
    steps: int,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    suffix_logits_only: bool = True,
):
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape [batch, seq_len]")
    if gen_length < 1:
        raise ValueError("gen_length must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=prompt.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    suffix_start = prompt.shape[1]
    num_blocks = (gen_length + block_length - 1) // block_length
    step_schedule = distribute_steps(int(steps), num_blocks)
    nfe = 0

    for block_idx, block_steps in enumerate(step_schedule):
        block_start = suffix_start + block_idx * block_length
        block_end = min(suffix_start + gen_length, block_start + block_length)
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        for step_idx in range(block_steps):
            block_state = x[:, block_start:block_end]
            block_mask_index = block_state == mask_id
            if block_mask_index.sum() == 0:
                break

            if cfg_scale > 0.0:
                uncond_x = x.clone()
                uncond_x[:, :suffix_start] = mask_id
                model_x = torch.cat([x, uncond_x], dim=0)
                logits, logit_offset = forward_logits(model, model_x, suffix_start, suffix_logits_only)
                logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + (cfg_scale + 1.0) * (logits - uncond_logits)
            else:
                logits, logit_offset = forward_logits(model, x, suffix_start, suffix_logits_only)

            local_start = block_start - logit_offset
            local_end = block_end - logit_offset
            block_logits = logits[:, local_start:local_end, :]
            x0, transfer_index = get_transfer_index(
                block_logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=block_mask_index,
                x=block_state,
                num_transfer_tokens=num_transfer_tokens[:, step_idx],
            )
            x[:, block_start:block_end] = torch.where(transfer_index, x0, block_state)
            nfe += 1

    return x, nfe


class LLaDANativeGenerator:
    def __init__(self, config: LLaDANativeConfig):
        self.config = config
        self.dtype = resolve_dtype(config.dtype)
        self.device_map = config.device_map or None
        if self.device_map in ("", "none", "None"):
            self.device_map = None
        self.model, self.tokenizer = self._load()
        self.device = get_model_input_device(self.model)

    def _load(self):
        model_config = LLaDAConfig.from_pretrained(self.config.model_path)
        kwargs = {
            "config": model_config,
            "torch_dtype": self.dtype,
            "trust_remote_code": False,
        }
        if self.device_map:
            kwargs["device_map"] = self.device_map
            kwargs["low_cpu_mem_usage"] = True

        model = LLaDAModelLM.from_pretrained(self.config.model_path, **kwargs).eval()
        if not self.device_map:
            device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            set_model_device_attr(model, device)
        apply_llada_yarn_scaling(model, self.config.rope_scale_factor)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        return model, tokenizer

    def generate_ids(self, input_ids: Iterable[int]):
        input_ids = list(input_ids)
        prompt = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        output, nfe = native_llada_generate(
            self.model,
            prompt,
            gen_length=self.config.max_new_tokens,
            steps=self.config.generation_steps,
            block_length=self.config.block_length,
            temperature=self.config.temperature,
            cfg_scale=self.config.cfg_scale,
            remasking=self.config.remasking,
            mask_id=self.config.mask_token_id,
            suffix_logits_only=self.config.suffix_logits_only,
        )
        generated_ids = output[0, prompt.shape[1] :].tolist()
        return generated_ids, nfe

    def generate_text(self, input_ids: Iterable[int], stop_tokens=None):
        generated_ids, nfe = self.generate_ids(input_ids)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        for stop in stop_tokens or []:
            if stop:
                text = text.split(stop)[0]
        return text, generated_ids, nfe


def build_native_config(args) -> LLaDANativeConfig:
    return LLaDANativeConfig(
        model_path=args.model_path,
        device=args.device,
        device_map=getattr(args, "device_map", None),
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        mask_token_id=args.mask_token_id,
        rope_scale_factor=args.rope_scale_factor,
        suffix_logits_only=args.suffix_logits_only,
    )
