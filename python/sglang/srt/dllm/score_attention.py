"""Request-level predicates for Dream chunk-scoring requests."""

from typing import Any


def is_score_request(sampling_params: Any) -> bool:
    """Whether a request asks for prompt logprobs under a Dream scoring mask.

    Such a request is one ordinary forward, not a denoising generation, even
    on a server that runs a dLLM algorithm.
    """
    custom_params = sampling_params.custom_params
    return (
        isinstance(custom_params, dict)
        and "dream_score_attention_mask" in custom_params
    )


def uses_segmented_score_mask(sampling_params: Any) -> bool:
    """Whether a request asks for ``dream_score_attention_mask="full"``.

    Under that mask prefix and chunk rows attend to later rows, so the request
    must be prefilled whole in a single forward.
    """
    custom_params = sampling_params.custom_params
    if not isinstance(custom_params, dict):
        return False
    mask = custom_params.get("dream_score_attention_mask")
    return isinstance(mask, str) and mask.lower() == "full"
