from types import SimpleNamespace

import pytest
import torch

from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.model_executor.forward_batch_info import (
    _compute_dllm_positions,
    _parallelcomp_force_causal,
)


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


def test_parallelcomp_positions_override_legacy_sparse_offsets():
    req = SimpleNamespace(
        parallelcomp_position_values=lambda: [7, 8, 20, 21],
        sampling_params=SimpleNamespace(
            custom_params={"dllm_position_start": 1, "dllm_position_offset": 99}
        ),
        origin_input_ids=list(range(4)),
        extend_range=SimpleNamespace(start=0, end=4),
    )

    assert _compute_dllm_positions(req) == [7, 8, 20, 21]


def test_parallelcomp_abort_returns_all_retained_chunks_for_page_table_filtering():
    first = torch.tensor([10, 11])
    latest = torch.tensor([20, 21])
    req = SimpleNamespace(
        dllm_parallelcomp_state={
            "stage": "chunk",
            "chunk_kv_indices": [first, latest],
        }
    )

    detached = ReqDllmMixin.take_parallelcomp_retained_kv_indices(req)

    assert len(detached) == 2
    assert torch.equal(detached[0], first)
    assert torch.equal(detached[1], latest)
    assert req.dllm_parallelcomp_state["chunk_kv_indices"] == []


def test_parallelcomp_retraction_restarts_from_common_prefix():
    req = SimpleNamespace(
        dllm_parallelcomp_state={
            "stage": "decode",
            "prefix_len": 9,
            "chunk_cursor": 3,
            "common_prefix_indices": torch.tensor([1, 2]),
            "chunk_kv_indices": [torch.tensor([3])],
            "assembled_prefix_indices": torch.tensor([1, 2, 3]),
        }
    )

    ReqDllmMixin.reset_parallelcomp_prefill_state(req)

    state = req.dllm_parallelcomp_state
    assert state["stage"] == "prefix"
    assert state["chunk_cursor"] == 0
    assert state["common_prefix_indices"] is None
    assert state["chunk_kv_indices"] == []
    assert "assembled_prefix_indices" not in state


def test_parallelcomp_causal_override_is_chunk_stage_only():
    chunk_req = SimpleNamespace(dllm_parallelcomp_state={"stage": "chunk"})
    prefix_req = SimpleNamespace(dllm_parallelcomp_state={"stage": "prefix"})
    ordinary_req = SimpleNamespace(dllm_parallelcomp_state=None)

    assert _parallelcomp_force_causal([chunk_req])
    assert not _parallelcomp_force_causal([prefix_req])
    assert not _parallelcomp_force_causal([ordinary_req])
    with pytest.raises(RuntimeError, match="cannot share a batch"):
        _parallelcomp_force_causal([chunk_req, ordinary_req])
