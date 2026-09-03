from types import SimpleNamespace

import pytest
import torch

from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
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


def test_causal_override_covers_chunk_stage_and_prompt_scoring():
    chunk_req = SimpleNamespace(dllm_parallelcomp_state={"stage": "chunk"})
    prefix_req = SimpleNamespace(dllm_parallelcomp_state={"stage": "prefix"})
    ordinary_req = SimpleNamespace(dllm_parallelcomp_state=None)
    score_req = SimpleNamespace(
        dllm_parallelcomp_state=None,
        sampling_params=SimpleNamespace(
            custom_params={"dream_causal_prompt_logprob": True}
        ),
    )

    assert _parallelcomp_force_causal([chunk_req])
    assert _parallelcomp_force_causal([score_req])
    assert not _parallelcomp_force_causal([prefix_req])
    assert not _parallelcomp_force_causal([ordinary_req])
    with pytest.raises(RuntimeError, match="cannot share a batch"):
        _parallelcomp_force_causal([chunk_req, ordinary_req])
    with pytest.raises(RuntimeError, match="cannot share a batch"):
        _parallelcomp_force_causal([score_req, ordinary_req])


def test_parallelcomp_chunk_batch_has_independent_items_and_positions():
    req = SimpleNamespace(
        dllm_parallelcomp_state={
            "stage": "chunk",
            "prefix_len": 2,
            "query_len": 2,
            "chunk_lens": [3, 4, 2],
            "chunk_batch_size": 2,
            "chunk_cursor": 1,
            "chunk_position_starts": [2, 10, 20],
            "chunk_query_position_starts": [30, 100, 200],
        },
        extend_range=SimpleNamespace(start=2, end=12),
    )
    req.parallelcomp_chunk_batch_range = lambda: (
        ReqDllmMixin.parallelcomp_chunk_batch_range(req)
    )

    assert ReqDllmMixin.parallelcomp_item_lens(req) == [6, 4]
    assert ReqDllmMixin.parallelcomp_position_values(req) == [
        10,
        11,
        12,
        13,
        100,
        101,
        20,
        21,
        200,
        201,
    ]


def test_parallelcomp_torch_mask_isolates_sibling_chunks():
    mask = TorchNativeAttnBackend._make_parallelcomp_mask(
        prefix_len=2, item_lens=[3, 2], device=torch.device("cpu")
    )

    assert mask.shape == (7, 7)
    assert mask[4].tolist() == [True, True, True, True, True, False, False]
    assert mask[5].tolist() == [True, True, False, False, False, True, False]
    assert mask[6].tolist() == [True, True, False, False, False, True, True]


def test_parallelcomp_torch_mask_is_cached_once_per_forward():
    backend = object.__new__(TorchNativeAttnBackend)
    backend.use_sliding_window_kv_pool = False
    forward_batch = SimpleNamespace(
        out_cache_loc=None,
        dllm_parallelcomp_item_lens=[[3, 2]],
        extend_prefix_lens_cpu=[2],
        input_ids=torch.arange(5),
    )

    backend.init_forward_metadata(forward_batch)

    assert len(backend.parallelcomp_masks) == 1
    assert backend.parallelcomp_masks[0][6].tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]


def test_parallelcomp_flashinfer_items_preserve_explicit_rope_positions():
    positions = torch.tensor([10, 11, 12, 20, 21])
    forward_batch = SimpleNamespace(
        dllm_parallelcomp_item_lens=[[3, 2]],
        extend_prefix_lens_cpu=[2],
        extend_seq_lens_cpu=[5],
        input_ids=torch.arange(5),
        positions=positions.clone(),
    )
    backend = object.__new__(FlashInferAttnBackend)

    params = backend._process_parallelcomp_items(forward_batch)

    assert params.prefix_len_ptr.tolist() == [2]
    assert params.token_pos_in_items_ptr.tolist() == [0, 1, 2, 0, 1]
    assert params.token_pos_in_items_len == 5
    assert params.max_item_len_ptr.tolist() == [2]
    assert torch.equal(forward_batch.positions, positions)
