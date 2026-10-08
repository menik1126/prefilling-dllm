"""Sparse RoPE positions for a Dream prompt whose last chunk is short."""

from typing import Any, Optional, Tuple


def parse_sparse_position_shift(
    custom_params: Any, *, prompt_len: int
) -> Optional[Tuple[int, int]]:
    """Validate ``dllm_position_start`` / ``dllm_position_offset``.

    Returns (start, offset): tokens at index ``start`` and later sit ``offset``
    positions further right. None keeps contiguous positions.
    """
    if not isinstance(custom_params, dict):
        return None
    position_start = custom_params.get("dllm_position_start")
    position_offset = custom_params.get("dllm_position_offset")
    if position_start is None and position_offset is None:
        return None
    if (
        not isinstance(position_start, int)
        or isinstance(position_start, bool)
        or not isinstance(position_offset, int)
        or isinstance(position_offset, bool)
    ):
        raise ValueError("Dream sparse position start and offset must be integers")
    if not 0 <= position_start <= prompt_len:
        raise ValueError(
            f"Dream sparse position start is outside the prompt: {position_start}"
        )
    if position_offset < 0:
        raise ValueError(
            f"Dream sparse position offset must be non-negative: {position_offset}"
        )
    return position_start, position_offset
