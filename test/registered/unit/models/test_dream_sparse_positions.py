from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    _build_dllm_denoise_plan_key,
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


def test_denoise_plan_key_tracks_retained_request_identity():
    prefix_0 = torch.tensor([10, 11, 12])
    canvas_0 = torch.tensor([20, 21])
    prefix_1 = torch.tensor([30, 31, 32, 33])
    canvas_1 = torch.tensor([40, 41])
    req_0 = SimpleNamespace(
        rid="req-0",
        req_pool_idx=1,
        prefix_indices=prefix_0,
        dllm_kv_indices=canvas_0,
    )
    req_1 = SimpleNamespace(
        rid="req-1",
        req_pool_idx=2,
        prefix_indices=prefix_1,
        dllm_kv_indices=canvas_1,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DLLM_DENOISE,
        dllm_config=SimpleNamespace(
            flashinfer_denoise_plan_cache=True,
            flashinfer_denoise_single_paged=False,
        ),
        req_to_token_pool=SimpleNamespace(req_generation=torch.tensor([0, 7, 11])),
        reqs=[req_0, req_1],
    )

    key = _build_dllm_denoise_plan_key(batch)
    assert key == (
        ("req-0", 1, 7, prefix_0.data_ptr(), canvas_0.data_ptr()),
        ("req-1", 2, 11, prefix_1.data_ptr(), canvas_1.data_ptr()),
    )
    assert _build_dllm_denoise_plan_key(batch) == key

    # Either consumer needs the retained-layout identity. It is disabled only
    # when both the plan cache and the single-paged route are disabled.
    batch.dllm_config.flashinfer_denoise_plan_cache = False
    batch.dllm_config.flashinfer_denoise_single_paged = True
    assert _build_dllm_denoise_plan_key(batch) == key
    batch.dllm_config.flashinfer_denoise_single_paged = False
    assert _build_dllm_denoise_plan_key(batch) is None
    batch.dllm_config.flashinfer_denoise_plan_cache = True

    batch.req_to_token_pool.req_generation[1] += 1
    assert _build_dllm_denoise_plan_key(batch) != key
    batch.req_to_token_pool.req_generation[1] -= 1

    batch.reqs = [req_1, req_0]
    assert _build_dllm_denoise_plan_key(batch) == (key[1], key[0])
    batch.reqs = [req_0, req_1]

    req_0.prefix_indices = prefix_0.clone()
    assert _build_dllm_denoise_plan_key(batch) != key
    req_0.prefix_indices = prefix_0
    req_0.dllm_kv_indices = canvas_0.clone()
    assert _build_dllm_denoise_plan_key(batch) != key

    batch.forward_mode = ForwardMode.DLLM_EXTEND
    assert _build_dllm_denoise_plan_key(batch) is None
    batch.forward_mode = ForwardMode.DLLM_DENOISE
    batch.dllm_config.flashinfer_denoise_plan_cache = False
    batch.dllm_config.flashinfer_denoise_single_paged = False
    assert _build_dllm_denoise_plan_key(batch) is None


def _make_flashinfer_denoise_backend(*, single_paged: bool):
    backend = object.__new__(FlashInferAttnBackend)
    backend._model_dtype = torch.float16
    backend._dllm_denoise_single_paged_enabled = single_paged
    backend._dllm_denoise_single_paged_plain_kv = True
    backend._dllm_denoise_single_paged_layout_key = None
    backend._dllm_denoise_single_paged_offsets = None
    backend._dllm_denoise_plan_cache_enabled = True
    backend._dllm_denoise_plan_cache_key = None
    backend._dllm_denoise_plan_cache_metadata = None
    backend._dllm_denoise_plan_cache_hits = 0
    backend._dllm_denoise_plan_cache_misses = 0
    backend.dispatch_reason = None
    backend.num_wrappers = 1
    backend.use_sliding_window_kv_pool = False
    backend.enable_mis = False
    backend.is_multimodal = False
    backend.prefill_uses_dequant_workspace = False
    backend.enable_deterministic = False
    backend.use_paged = False
    backend.skip_prefill = False
    backend.page_size = 1
    backend.prefill_backend = "fa2"
    backend._dllm_denoise_plan_cache_single_rank = True
    backend._dllm_denoise_plan_cache_breakable = True
    backend.dllm_config = SimpleNamespace(
        algorithm="PrefillingDream",
        block_size=2,
        needs_full_prefill=True,
        dual_cache=True,
        first_done_first_out_mode=True,
        flashinfer_denoise_single_paged=single_paged,
        flashinfer_denoise_single_paged_max_batch_size=8,
        flashinfer_denoise_plan_cache_max_batch_size=8,
    )
    req_to_token = torch.zeros((8, 16), dtype=torch.int64)
    req_to_token[3, 5:7] = torch.tensor([20, 21])
    req_to_token[4, 7:9] = torch.tensor([30, 31])
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.prefill_wrappers_paged = [object()]
    backend.indices_updater_prefill = MagicMock()
    backend.prefill_split_tile_size = None
    backend.forward_metadata = None
    return backend


def _make_flashinfer_denoise_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.DLLM_DENOISE,
        batch_size=2,
        input_ids=torch.tensor([99, 99, 99, 99]),
        positions=torch.arange(4),
        req_pool_indices=torch.tensor([3, 4]),
        seq_lens=torch.tensor([7, 9]),
        seq_lens_cpu=torch.tensor([7, 9]),
        seq_lens_sum=16,
        extend_prefix_lens=torch.tensor([5, 7]),
        extend_prefix_lens_cpu=[5, 7],
        extend_seq_lens_cpu=[2, 2],
        out_cache_loc=torch.tensor([20, 21, 30, 31]),
        extend_num_tokens=4,
        return_logprob=False,
        spec_info=None,
        encoder_lens=None,
        dllm_parallelcomp_item_lens=None,
        dllm_force_causal=False,
        dllm_raw_last_logits_cpu=None,
        dllm_canvas_lens_cpu=None,
        dllm_disable_prefill_cuda_graph=False,
        dllm_denoise_plan_key=(
            ("req-0", 3, 7, 100, 200),
            ("req-1", 4, 11, 300, 400),
        ),
        cross_attention_custom_mask=None,
        rids=["req-0", "req-1"],
        tbo_split_seq_index=None,
        lora_ids=[None, None],
    )


def test_flashinfer_reuses_only_consecutive_stable_denoise_plans():
    backend = _make_flashinfer_denoise_backend(single_paged=True)
    # This test mutates artificial geometry without maintaining a page table;
    # row-tail validation has a dedicated test below.
    backend._validate_dllm_denoise_single_paged_layout = MagicMock()
    batch = _make_flashinfer_denoise_batch()

    valid_input_ids = batch.input_ids
    batch.input_ids = None
    assert backend._make_dllm_denoise_plan_cache_key(batch) is None
    batch.input_ids = valid_input_ids

    backend.init_forward_metadata(batch)
    first_metadata = backend.forward_metadata
    assert first_metadata.use_ragged is False
    assert first_metadata.dllm_denoise_single_paged is True
    assert (
        backend.indices_updater_prefill.update.call_args.kwargs["use_ragged"] is False
    )
    # Token/canvas contents are not part of attention planning.
    batch.input_ids = torch.tensor([1, 99, 2, 99])
    backend.init_forward_metadata(batch)

    assert backend.indices_updater_prefill.update.call_count == 1
    assert backend.forward_metadata is first_metadata
    assert backend._dllm_denoise_plan_cache_hits == 1
    assert backend._dllm_denoise_plan_cache_misses == 1

    # Per-request geometry is part of the plan even if the batch total stays
    # constant, so [7, 9] -> [8, 8] must not reuse the old plan.
    batch.seq_lens = torch.tensor([8, 8])
    batch.seq_lens_cpu = torch.tensor([8, 8])
    batch.extend_prefix_lens = torch.tensor([6, 6])
    batch.extend_prefix_lens_cpu = [6, 6]
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 2
    assert backend._dllm_denoise_plan_cache_misses == 2

    # Ordered membership matters even when the set of requests and all shapes
    # are otherwise identical.
    batch.req_pool_indices = torch.tensor([4, 3])
    batch.dllm_denoise_plan_key = tuple(reversed(batch.dllm_denoise_plan_key))
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 3
    assert backend._dllm_denoise_plan_cache_misses == 3

    batch.req_pool_indices = torch.tensor([3, 4])
    batch.seq_lens = torch.tensor([7, 9])
    batch.seq_lens_cpu = torch.tensor([7, 9])
    batch.extend_prefix_lens = torch.tensor([5, 7])
    batch.extend_prefix_lens_cpu = [5, 7]
    batch.dllm_denoise_plan_key = tuple(reversed(batch.dllm_denoise_plan_key))
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 4
    assert backend._dllm_denoise_plan_cache_misses == 4

    # Request-pool generation prevents ABA reuse of an identical slot number.
    batch.dllm_denoise_plan_key = (
        ("req-0", 3, 7, 100, 200),
        ("req-1", 4, 12, 300, 400),
    )
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 5
    assert backend._dllm_denoise_plan_cache_misses == 5

    # A membership change must rebuild even when the batch shape is unchanged.
    batch.req_pool_indices = torch.tensor([3, 5])
    batch.dllm_denoise_plan_key = (
        ("req-0", 3, 7, 100, 200),
        ("req-2", 5, 3, 500, 600),
    )
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 6
    assert backend._dllm_denoise_plan_cache_misses == 6

    # Any intervening mode mutates the shared wrappers and invalidates the
    # one-entry cache. Returning to the old denoise geometry must rebuild.
    batch.forward_mode = ForwardMode.DLLM_EXTEND
    backend.init_forward_metadata(batch)
    batch.forward_mode = ForwardMode.DLLM_DENOISE
    batch.req_pool_indices = torch.tensor([3, 4])
    batch.dllm_denoise_plan_key = (
        ("req-0", 3, 7, 100, 200),
        ("req-1", 4, 12, 300, 400),
    )
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 8
    assert backend._dllm_denoise_plan_cache_misses == 7

    # A failed miss must not leave the previously successful plan reusable.
    batch.dllm_denoise_plan_key = (
        ("req-0", 3, 7, 100, 200),
        ("req-1", 4, 13, 300, 400),
    )
    backend.indices_updater_prefill.update.side_effect = RuntimeError("plan failed")
    with pytest.raises(RuntimeError, match="plan failed"):
        backend.init_forward_metadata(batch)
    assert backend._dllm_denoise_plan_cache_key is None
    assert backend._dllm_denoise_plan_cache_metadata is None

    backend.indices_updater_prefill.update.side_effect = None
    batch.dllm_denoise_plan_key = (
        ("req-0", 3, 7, 100, 200),
        ("req-1", 4, 12, 300, 400),
    )
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 10
    assert backend._dllm_denoise_plan_cache_misses == 9


def test_flashinfer_denoise_plan_cache_isolated_by_attention_route():
    backend = _make_flashinfer_denoise_backend(single_paged=False)
    batch = _make_flashinfer_denoise_batch()

    backend.init_forward_metadata(batch)
    ragged_key = backend._dllm_denoise_plan_cache_key
    assert backend.forward_metadata.use_ragged is True
    assert backend.forward_metadata.dllm_denoise_single_paged is False

    backend._dllm_denoise_single_paged_enabled = True
    backend.dllm_config.flashinfer_denoise_single_paged = True
    backend.init_forward_metadata(batch)
    single_paged_key = backend._dllm_denoise_plan_cache_key
    assert backend.indices_updater_prefill.update.call_count == 2
    assert backend._dllm_denoise_plan_cache_hits == 0
    assert backend._dllm_denoise_plan_cache_misses == 2
    assert backend.forward_metadata.use_ragged is False
    assert backend.forward_metadata.dllm_denoise_single_paged is True
    assert single_paged_key != ragged_key
    assert single_paged_key[-1] is True
    assert ragged_key[-1] is False

    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 2
    assert backend._dllm_denoise_plan_cache_hits == 1

    backend._dllm_denoise_single_paged_enabled = False
    backend.dllm_config.flashinfer_denoise_single_paged = False
    backend.init_forward_metadata(batch)
    assert backend.indices_updater_prefill.update.call_count == 3
    assert backend._dllm_denoise_plan_cache_misses == 3
    assert backend.forward_metadata.use_ragged is True
    assert backend.forward_metadata.dllm_denoise_single_paged is False


def test_flashinfer_denoise_single_paged_requires_identity_and_plain_kv():
    backend = _make_flashinfer_denoise_backend(single_paged=True)
    batch = _make_flashinfer_denoise_batch()

    assert backend._use_dllm_denoise_single_paged(batch) is True

    scheduler_key = batch.dllm_denoise_plan_key
    batch.dllm_denoise_plan_key = None
    assert backend._use_dllm_denoise_single_paged(batch) is False
    batch.dllm_denoise_plan_key = scheduler_key

    backend._dllm_denoise_single_paged_plain_kv = False
    assert backend._use_dllm_denoise_single_paged(batch) is False
    backend.init_forward_metadata(batch)
    assert backend.forward_metadata.use_ragged is True
    assert backend.forward_metadata.dllm_denoise_single_paged is False


def test_flashinfer_skips_denoise_geometry_when_both_consumers_are_disabled():
    backend = _make_flashinfer_denoise_backend(single_paged=False)
    backend._dllm_denoise_plan_cache_enabled = False
    batch = _make_flashinfer_denoise_batch()

    with patch.object(
        backend,
        "_make_dllm_denoise_geometry_key",
        wraps=backend._make_dllm_denoise_geometry_key,
    ) as make_geometry:
        backend.init_forward_metadata(batch)

    make_geometry.assert_not_called()
    assert backend.forward_metadata.use_ragged is True


def test_flashinfer_denoise_single_paged_requires_valid_geometry_and_dtype():
    backend = _make_flashinfer_denoise_backend(single_paged=True)
    batch = _make_flashinfer_denoise_batch()

    geometry_key = backend._make_dllm_denoise_geometry_key(batch)
    assert geometry_key is not None
    assert backend._use_dllm_denoise_single_paged(batch, geometry_key=geometry_key)

    input_ids = batch.input_ids
    batch.input_ids = input_ids[:-1]
    assert backend._make_dllm_denoise_geometry_key(batch) is None
    assert backend._use_dllm_denoise_single_paged(batch) is False
    batch.input_ids = input_ids

    backend._model_dtype = torch.float32
    assert backend._use_dllm_denoise_single_paged(batch) is False
    backend._model_dtype = torch.float16
    backend.dllm_config.flashinfer_denoise_single_paged_max_batch_size = 1
    assert backend._use_dllm_denoise_single_paged(batch) is False


def test_flashinfer_denoise_single_paged_validates_page_table_canvas_once():
    backend = _make_flashinfer_denoise_backend(single_paged=True)
    batch = _make_flashinfer_denoise_batch()
    geometry_key = backend._make_dllm_denoise_geometry_key(batch)

    backend._validate_dllm_denoise_single_paged_layout(batch, geometry_key)
    assert backend._dllm_denoise_single_paged_layout_key == geometry_key

    # A stable key deliberately skips the GPU comparison on later denoise rounds.
    backend.req_to_token_pool.req_to_token[3, 5] = -1
    backend._validate_dllm_denoise_single_paged_layout(batch, geometry_key)

    backend._dllm_denoise_single_paged_layout_key = None
    with pytest.raises(RuntimeError, match="page-table canvas"):
        backend._validate_dllm_denoise_single_paged_layout(batch, geometry_key)
