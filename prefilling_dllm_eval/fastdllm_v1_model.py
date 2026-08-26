"""Thin Fast-dLLM v1 Dream wrapper for local benchmark adapters.

This intentionally bypasses lm_eval so we can reuse our InfiniteBench and
LongBench loaders/scorers without touching the ParallelComp runtime.
"""

import os
import sys
import time
import types
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import transformers


def _ensure_default_rope_init():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return
    if "proportional" not in ROPE_INIT_FUNCTIONS:
        available = ", ".join(sorted(ROPE_INIT_FUNCTIONS))
        raise RuntimeError(f"Cannot alias default RoPE; available rope types: {available}")
    ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["proportional"]


def _ensure_generation_config_validate_compat():
    from model.generation_utils import DreamGenerationConfig

    original_validate = DreamGenerationConfig.validate
    if getattr(original_validate, "_prefilling_accepts_extra_kwargs", False):
        return

    def validate(self, is_init=False, **kwargs):
        return original_validate(self, is_init=is_init)

    validate._prefilling_accepts_extra_kwargs = True
    DreamGenerationConfig.validate = validate


def _resolve_dtype(dtype):
    if dtype in (None, "auto"):
        return dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, str(dtype), torch.bfloat16)


@dataclass
class FastDLLMv1Config:
    fastdllm_dream_dir: str
    pretrained: str
    device: str = "cuda"
    dtype: str = "auto"
    max_new_tokens: int = 32
    max_length: int = 4096
    block_length: int = 32
    diffusion_steps: Optional[int] = None
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    alg: str = "confidence_threshold"
    alg_temp: Optional[float] = 0.0
    threshold: float = 0.9
    dual_cache: bool = True
    add_bos_token: bool = True
    truncation_strategy: str = "head_tail"
    rope_scale_factor: float = 1.0
    trust_remote_code: bool = True


class FastDLLMv1Dream:
    """Direct Fast-dLLM v1 Dream generator.

    No ParallelComp operations are performed here: prompts are tokenized,
    truncated to `max_length - max_new_tokens`, then decoded with the
    official Fast-dLLM v1 block generation mixin.
    """

    def __init__(self, config: FastDLLMv1Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(config.max_new_tokens)
        self.max_length = int(config.max_length)
        self.block_length = int(config.block_length)
        if self.max_new_tokens % self.block_length != 0:
            raise ValueError(
                f"max_new_tokens ({self.max_new_tokens}) must be divisible by "
                f"block_length ({self.block_length}) for Fast-dLLM v1."
            )
        self.diffusion_steps = (
            int(config.diffusion_steps)
            if config.diffusion_steps is not None
            else max(1, self.max_new_tokens // self.block_length)
        )

        dream_dir = os.path.abspath(config.fastdllm_dream_dir)
        if not os.path.isdir(dream_dir):
            raise FileNotFoundError(f"Fast-dLLM v1 Dream dir not found: {dream_dir}")
        if dream_dir not in sys.path:
            sys.path.insert(0, dream_dir)

        _ensure_default_rope_init()
        from model.configuration_dream import DreamConfig
        from model.generation_utils import DreamGenerationConfig
        from model.generation_utils_block import DreamGenerationMixin
        from model.modeling_dream import DreamModel, DreamRotaryEmbedding
        _ensure_generation_config_validate_compat()

        model_config = DreamConfig.from_pretrained(config.pretrained)
        target_dtype = _resolve_dtype(config.dtype)
        self.model = DreamModel.from_pretrained(
            config.pretrained,
            config=model_config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate,
            self.model,
        )
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
        if target_dtype not in (None, "auto"):
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device)
        self._apply_rope_scaling(DreamRotaryEmbedding, float(config.rope_scale_factor or 1.0))

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.pretrained,
            trust_remote_code=config.trust_remote_code,
        )
        self.generated_token_num = 0
        self.total_generation_time = 0.0

    def _apply_rope_scaling(self, rotary_cls, factor: float) -> None:
        if factor <= 1.0:
            return
        original_max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if original_max_pos is None:
            raise ValueError("Cannot apply RoPE scaling: config has no max_position_embeddings")
        self.model.config.rope_scaling = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": original_max_pos,
        }
        for module in self.model.modules():
            if isinstance(module, rotary_cls):
                module.__init__(config=self.model.config)
                module.to(self.device)

    def _truncate_prompt_ids(self, prompt_ids: torch.LongTensor) -> torch.LongTensor:
        budget = self.max_length - self.max_new_tokens
        if budget <= 0:
            raise ValueError("max_length must be larger than max_new_tokens")
        if prompt_ids.shape[1] > budget:
            strategy = self.config.truncation_strategy
            if strategy == "left":
                prompt_ids = prompt_ids[:, -budget:]
            elif strategy == "head":
                prompt_ids = prompt_ids[:, :budget]
            elif strategy == "head_tail":
                head_keep = budget // 2
                tail_keep = budget - head_keep
                prompt_ids = torch.cat(
                    [prompt_ids[:, :head_keep], prompt_ids[:, -tail_keep:]],
                    dim=1,
                )
            else:
                raise ValueError(f"Unsupported truncation_strategy: {strategy}")
        return prompt_ids.to(self.device)

    def _encode_prompt(self, prompt: str) -> torch.LongTensor:
        if self.config.add_bos_token and self.tokenizer.bos_token:
            prompt = self.tokenizer.bos_token + prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return self._truncate_prompt_ids(prompt_ids)

    @torch.inference_mode()
    def generate_one_ids(self, prompt_ids: torch.LongTensor) -> str:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = self._truncate_prompt_ids(prompt_ids.to(torch.long))
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=self.device)
        else:
            attention_mask = prompt_ids.ne(pad_token_id).to(self.device)

        start = time.time()
        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            alg=self.config.alg,
            alg_temp=self.config.alg_temp,
            threshold=self.config.threshold,
            block_length=self.block_length,
            dual_cache=self.config.dual_cache,
        )
        self.total_generation_time += time.time() - start

        generated = generation_ids.sequences[0, prompt_ids.shape[1] :].tolist()
        eos = self.tokenizer.eos_token
        text = self.tokenizer.decode(generated)
        if eos:
            text = text.split(eos)[0]
        self.generated_token_num += sum(
            1 for token_id in generated if token_id != self.tokenizer.eos_token_id
        )
        return text

    @torch.inference_mode()
    def generate_one(self, prompt: str) -> str:
        prompt_ids = self._encode_prompt(prompt)
        return self.generate_one_ids(prompt_ids)

    def generate(self, prompts: Iterable[str], stop_tokens: Optional[List[str]] = None) -> List[str]:
        outputs = []
        for prompt in prompts:
            text = self.generate_one(prompt)
            for stop in stop_tokens or []:
                if stop:
                    text = text.split(stop)[0]
            outputs.append(text)
        return outputs


def default_fastdllm_dream_dir():
    return os.environ.get(
        "FASTDLLM_V1_DREAM_DIR",
        "/home/ma-user/work/Fast-dLLM/v1/dream",
    )


def load_fastdllm_v1_dream(
    pretrained,
    fastdllm_dream_dir=None,
    **kwargs,
):
    config = FastDLLMv1Config(
        fastdllm_dream_dir=fastdllm_dream_dir or default_fastdllm_dream_dir(),
        pretrained=pretrained,
        **kwargs,
    )
    return FastDLLMv1Dream(config)
