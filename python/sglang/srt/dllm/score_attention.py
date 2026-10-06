"""Request-level predicate for Dream's query-conditioned scoring mask."""

from typing import Any


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
