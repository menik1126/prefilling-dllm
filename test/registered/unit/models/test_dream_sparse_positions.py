from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.forward_batch_info import _compute_dllm_positions


def test_sparse_query_positions_preserve_fixed_chunk_slots():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            custom_params={
                "dllm_position_start": 5,
                "dllm_position_offset": 3,
            }
        ),
        origin_input_ids=list(range(7)),
        extend_range=SimpleNamespace(start=0, end=9),
    )

    assert _compute_dllm_positions(req) == [0, 1, 2, 3, 4, 8, 9, 10, 11]


def test_sparse_query_positions_default_to_contiguous():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=None),
        origin_input_ids=list(range(7)),
        extend_range=SimpleNamespace(start=2, end=6),
    )

    assert _compute_dllm_positions(req) == [2, 3, 4, 5]


def test_sparse_query_positions_reject_partial_metadata():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={"dllm_position_start": 5}),
        origin_input_ids=list(range(7)),
        extend_range=SimpleNamespace(start=0, end=7),
    )

    with pytest.raises(ValueError, match="must be integers"):
        _compute_dllm_positions(req)
