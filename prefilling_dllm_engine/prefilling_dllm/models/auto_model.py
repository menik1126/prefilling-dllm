from prefilling_dllm.config import Config
from prefilling_dllm.utils.loader import load_model
from prefilling_dllm.models.dream import DreamForDiffusionLM
from prefilling_dllm.models.qwen3 import Qwen3ForCausalLM


class AutoModelLM:
    MODEL_MAPPING = {
        "qwen3": Qwen3ForCausalLM,
        "dream": DreamForDiffusionLM
    }
    @classmethod
    def from_config(cls, config: Config):
        model = cls.MODEL_MAPPING[config.model_name](config.hf_config)
        return load_model(model, config)