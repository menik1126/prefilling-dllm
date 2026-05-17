"""
D2F model wrappers with optional LoRA support.
Subclasses eval_llada.DreamLoRA and eval_dream.DreamLoRA without modifying them.
Dynamic dispatch ensures parent __init__ calls our overridden _create_model_and_tokenizer.
"""
import sys

D2F_EVAL = '/home/xiongjing/Discrete-Diffusion-Forcing/D2F-eval_repro_old0312_20260401'
sys.path.insert(0, D2F_EVAL)


def _resolve_dtype(dtype):
    """Resolve dtype string to torch.dtype, preserving 'auto' for from_pretrained."""
    import torch
    if dtype == 'auto' or dtype is None:
        return dtype  # pass through; from_pretrained handles 'auto', None -> float32
    return getattr(torch, dtype, torch.bfloat16)


class LLaDAModel:
    """LLaDA model with optional LoRA. Does not modify eval_llada.py."""

    def __init__(self, pretrained, lora_path=None, rope_scale_factor=1.0, **kwargs):
        from eval_llada import DreamLoRA
        import types

        self._inner = object.__new__(DreamLoRA)
        self._inner._lora_path_opt = lora_path
        self._inner._rope_scale_factor = rope_scale_factor
        self._inner._create_model_and_tokenizer = types.MethodType(
            self.__class__._llada_create_model, self._inner
        )
        DreamLoRA.__init__(
            self._inner,
            pretrained=pretrained,
            lora_path=lora_path or '',
            **kwargs
        )

    @staticmethod
    def _llada_create_model(self, pretrained, dtype, trust_remote_code):
        import torch
        from peft import PeftConfig, PeftModel
        from model_cache.llada.modeling_llada import LLaDAModelLM, LLaDAConfig
        from transformers import AutoTokenizer

        target_dtype = _resolve_dtype(dtype)
        config = LLaDAConfig.from_pretrained(pretrained)
        self.model = LLaDAModelLM.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()

        if self._lora_path_opt:
            PeftConfig.from_pretrained(self._lora_path_opt)
            self.model = PeftModel.from_pretrained(self.model, self._lora_path_opt)

        # Cast only when an explicit dtype was requested (not auto/None)
        if target_dtype is not None and target_dtype != 'auto':
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device)

        # Apply NTK-by-parts RoPE scaling (YaRN) if requested
        scale = getattr(self, '_rope_scale_factor', 1.0)
        if scale > 1.0:
            from model_cache.llada.modeling_llada import RotaryEmbedding
            for module in self.model.modules():
                if isinstance(module, RotaryEmbedding):
                    module.ntk_scale_factor = scale
                    # Invalidate cache so next forward recomputes with NTK-by-parts
                    module._RotaryEmbedding__cache.pop('rope_pos_sin', None)
                    module._RotaryEmbedding__cache.pop('rope_pos_cos', None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def generate_until(self, requests, disable_tqdm=False):
        return self._inner.generate_until(requests, disable_tqdm=disable_tqdm)

    @property
    def tokenizer(self):
        return self._inner.tokenizer


class DREAMModel:
    """DREAM model with optional LoRA. Does not modify eval_dream.py."""

    def __init__(self, pretrained, lora_path=None, rope_scale_factor=1.0, **kwargs):
        from eval_dream import DreamLoRA
        import types

        self._inner = object.__new__(DreamLoRA)
        self._inner._lora_path_opt = lora_path
        self._inner._rope_scale_factor = rope_scale_factor
        self._inner._create_model_and_tokenizer = types.MethodType(
            self.__class__._dream_create_model, self._inner
        )
        DreamLoRA.__init__(
            self._inner,
            pretrained=pretrained,
            lora_path=lora_path or '',
            **kwargs
        )

    @staticmethod
    def _dream_create_model(self, pretrained, dtype, trust_remote_code):
        import transformers
        from peft import PeftConfig, PeftModel
        from model_cache.dream.model_dream import DreamModel
        from model_cache.dream.configuration_dream import DreamConfig

        target_dtype = _resolve_dtype(dtype)
        model_config = DreamConfig.from_pretrained(pretrained)
        self.model = DreamModel.from_pretrained(
            pretrained,
            config=model_config,
            torch_dtype=target_dtype,
            trust_remote_code=False,
        ).eval()

        if self._lora_path_opt:
            PeftConfig.from_pretrained(self._lora_path_opt)
            self.model = PeftModel.from_pretrained(self.model, self._lora_path_opt)

        # Cast only when an explicit dtype was requested (not auto/None)
        if target_dtype is not None and target_dtype != 'auto':
            self.model = self.model.to(target_dtype)
        self.model = self.model.to(self.device)

        # Apply NTK-by-parts (YaRN) RoPE scaling for DREAM if requested
        scale = getattr(self, '_rope_scale_factor', 1.0)
        if scale > 1.0:
            original_max_pos = model_config.max_position_embeddings
            self.model.config.rope_scaling = {
                'rope_type': 'yarn',
                'factor': float(scale),
                'original_max_position_embeddings': original_max_pos,
            }
            from model_cache.dream.model_dream import DreamRotaryEmbedding
            for module in self.model.modules():
                if isinstance(module, DreamRotaryEmbedding):
                    module.__init__(config=self.model.config)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def generate_until(self, requests, disable_tqdm=False):
        return self._inner.generate_until(requests, disable_tqdm=disable_tqdm)

    @property
    def tokenizer(self):
        return self._inner.tokenizer


def load_model(model_type, pretrained, lora_path=None, rope_scale_factor=1.0, **kwargs):
    """Factory: load LLaDA or DREAM with optional LoRA and NTK-by-parts RoPE scaling."""
    if model_type == 'llada':
        return LLaDAModel(pretrained=pretrained, lora_path=lora_path, rope_scale_factor=rope_scale_factor, **kwargs)
    elif model_type == 'dream':
        return DREAMModel(pretrained=pretrained, lora_path=lora_path, rope_scale_factor=rope_scale_factor, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'llada' or 'dream'.")


def generate(model, prompts, stop_tokens=None):
    """Simple batch generation interface."""
    from lm_eval.api.instance import Instance
    stop_tokens = stop_tokens or ['\n\n\n']
    instances = [
        Instance(request_type='generate_until', doc={}, arguments=(p, {'until': stop_tokens}), idx=i)
        for i, p in enumerate(prompts)
    ]
    return model.generate_until(instances, disable_tqdm=True)
