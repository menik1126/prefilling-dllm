"""Request-level predicates for Dream chunk-scoring requests."""

from typing import Any, Optional, Tuple


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


def parse_score_attention(
    custom_params: Any, *, input_len: Optional[int]
) -> Optional[Tuple[str, Optional[Tuple[int, int, int, int]]]]:
    """Return ``('causal', None)``, ``('full', (p, c, q, d))``, or ``None``.

    ``input_len`` is the request's token count; None skips the coverage check.
    """
    if not isinstance(custom_params, dict):
        return None

    raw_mask = custom_params.get("dream_score_attention_mask")
    if raw_mask is None and custom_params.get("dream_causal_prompt_logprob") is True:
        raw_mask = "causal"
    if raw_mask is None:
        return None
    if not isinstance(raw_mask, str):
        raise ValueError(
            f"dream_score_attention_mask must be a string, got {type(raw_mask)}"
        )
    mask = raw_mask.lower()
    if mask not in {"causal", "full"}:
        raise ValueError(
            "dream_score_attention_mask must be 'causal' or 'full', "
            f"got {raw_mask!r}"
        )
    if mask == "causal":
        return ("causal", None)

    def _require_nonneg_int(name: str) -> int:
        value = custom_params.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative int for full Dream scoring"
            )
        return value

    prefix_len = _require_nonneg_int("dream_score_prefix_len")
    chunk_len = _require_nonneg_int("dream_score_chunk_len")
    query_len = _require_nonneg_int("dream_score_query_len")
    draft_len = (
        _require_nonneg_int("dream_score_draft_len")
        if "dream_score_draft_len" in custom_params
        else 0
    )
    span_len = prefix_len + chunk_len + query_len + draft_len
    if span_len <= 0:
        raise ValueError("full Dream scoring spans must cover at least one token")
    if input_len is not None and span_len != input_len:
        raise ValueError(
            "full Dream scoring spans do not cover the input: "
            f"prefix={prefix_len}, chunk={chunk_len}, query={query_len}, "
            f"draft={draft_len}, input={input_len}"
        )
    return ("full", (prefix_len, chunk_len, query_len, draft_len))
