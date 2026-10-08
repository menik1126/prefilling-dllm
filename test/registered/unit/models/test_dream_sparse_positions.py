from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.dllm.head_token_eviction import (
    STAGE_GENERATE,
    STAGE_SCORE,
    DllmHeadEvictionCapture,
    DllmHeadEvictionConfig,
    DllmHeadEvictionRequest,
    accumulate_head_eviction_layer,
    build_head_eviction_capture,
    compact_input_ids,
    compact_prompt_kv_per_head,
    parse_head_token_eviction,
    score_stage_input_ids,
    stack_head_eviction_keep,
)
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.dllm.score_attention import is_score_request
from sglang.srt.dllm.token_eviction import (
    DllmTokenEvictionCapture,
    DllmTokenEvictionConfig,
    DllmTokenEvictionRequest,
    accumulate_token_eviction_layer,
    select_kept_token_positions,
    select_token_eviction_keep,
    split_kept_kv_indices,
)
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    _build_dllm_denoise_plan_key,
    _compute_dllm_positions,
    _dllm_attention_override,
    _parallelcomp_force_causal,
    _require_dllm_attention_override_backend,
    _require_whole_score_rows,
    make_dream_score_full_attention_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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
        dllm_token_eviction_state=None,
        dllm_block_position_shift=0,
    )

    assert _compute_dllm_positions(req) == [0, 1, 2, 3, 4, 8, 9, 10, 11]


def test_sparse_query_positions_default_to_contiguous():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=None),
        origin_input_ids=list(range(7)),
        extend_range=SimpleNamespace(start=2, end=6),
        dllm_token_eviction_state=None,
        dllm_block_position_shift=0,
    )

    assert _compute_dllm_positions(req) == [2, 3, 4, 5]


def test_sparse_query_positions_reject_partial_metadata():
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params={"dllm_position_start": 5}),
        origin_input_ids=list(range(7)),
        extend_range=SimpleNamespace(start=0, end=7),
        dllm_token_eviction_state=None,
        dllm_block_position_shift=0,
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
            "token_eviction": None,
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


def test_score_mask_shows_the_query_but_not_the_draft_to_prefix_and_chunk():
    """Query and draft rows must stay causal, or scored tokens see themselves."""
    mask = make_dream_score_full_attention_mask(1, 2, 2, 2, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_segmented_scoring_rejects_rows_not_computed_whole():
    """A radix hit reuses KV built for another chunk, and chunked prefill
    computes a row's head before its query is in the forward."""
    _require_whole_score_rows(None, prefix_lens=[5], extend_lens=[3])
    _require_whole_score_rows([(2, 3, 2, 0)], prefix_lens=[0], extend_lens=[7])
    with pytest.raises(RuntimeError, match="disable-radix-cache"):
        _require_whole_score_rows(
            [(2, 3, 2, 0), (2, 3, 2, 0)], prefix_lens=[0, 2], extend_lens=[7, 5]
        )
    with pytest.raises(RuntimeError, match="chunked-prefill-size"):
        _require_whole_score_rows(
            [(2, 3, 2, 0), (2, 3, 2, 0)], prefix_lens=[0, 0], extend_lens=[7, 4]
        )


def test_full_score_attention_does_not_force_causal():
    full_req = SimpleNamespace(
        dllm_parallelcomp_state=None,
        origin_input_ids=list(range(7)),
        sampling_params=SimpleNamespace(
            custom_params={
                "dream_score_attention_mask": "full",
                "dream_score_prefix_len": 2,
                "dream_score_chunk_len": 3,
                "dream_score_query_len": 1,
                "dream_score_draft_len": 1,
            }
        ),
    )
    causal_req = SimpleNamespace(
        dllm_parallelcomp_state=None,
        origin_input_ids=list(range(4)),
        sampling_params=SimpleNamespace(
            custom_params={"dream_causal_prompt_logprob": True}
        ),
    )
    ordinary_req = SimpleNamespace(
        dllm_parallelcomp_state=None,
        sampling_params=SimpleNamespace(custom_params=None),
    )

    force_causal, spans = _dllm_attention_override([full_req])
    assert force_causal is False
    assert spans == [(2, 3, 1, 1)]
    with pytest.raises(RuntimeError, match="cannot share a batch"):
        _dllm_attention_override([full_req, ordinary_req])
    with pytest.raises(RuntimeError, match="cannot share a batch"):
        _dllm_attention_override([full_req, causal_req])


def test_attention_override_rejects_backends_that_ignore_it():
    supported = SimpleNamespace(
        attn_backend=SimpleNamespace(supports_dllm_attention_override=True)
    )
    unsupported = SimpleNamespace(
        attn_backend=SimpleNamespace(supports_dllm_attention_override=False)
    )

    _require_dllm_attention_override_backend(supported)
    with pytest.raises(RuntimeError, match="flashinfer and torch_native"):
        _require_dllm_attention_override_backend(unsupported)
    with pytest.raises(RuntimeError, match="flashinfer and torch_native"):
        _require_dllm_attention_override_backend(SimpleNamespace())


def test_parallelcomp_chunk_batch_has_independent_items_and_positions():
    req = SimpleNamespace(
        dllm_parallelcomp_state={
            "stage": "chunk",
            "prefix_len": 2,
            "query_len": 2,
            "token_eviction": None,
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
        dllm_score_full_spans=None,
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


def _dense_query_to_chunk_scores(
    *, q, keys, prefix_len, chunk_len, scaling, pool_kernel
):
    """Reference: full causal attention over prefix + chunk + query, per head."""
    num_q_heads, num_kv_heads = q.shape[1], keys.shape[1]
    query_len = q.shape[0] - prefix_len - chunk_len
    total = torch.zeros(chunk_len)
    for head in range(num_q_heads):
        head_keys = keys[:, head // (num_q_heads // num_kv_heads)].float()
        logits = q[:, head].float() @ head_keys.T * scaling
        logits = logits.masked_fill(
            ~torch.ones_like(logits, dtype=torch.bool).tril(), float("-inf")
        )
        attention = torch.softmax(logits, dim=-1)
        received = attention[-query_len:, prefix_len : prefix_len + chunk_len].sum(0)
        pooled = torch.nn.functional.max_pool1d(
            received[None, None],
            kernel_size=pool_kernel,
            padding=pool_kernel // 2,
            stride=1,
        )
        total += pooled[0, 0, :chunk_len]
    return total


def test_token_eviction_scores_match_dense_causal_attention_per_item():
    """Two chunks packed after a plain row must each score against only the
    shared prefix and their own chunk + query rows."""
    torch.manual_seed(0)
    num_q_heads, num_kv_heads, head_dim, scaling = 4, 2, 8, 0.35
    prefix_len, chunk_lens, query_len, plain_len = 3, [6, 4], 2, 5
    prefix_slots = torch.tensor([9, 2, 7])
    key_buffer = torch.randn(12, num_kv_heads, head_dim)
    item_lens = [chunk_len + query_len for chunk_len in chunk_lens]
    extend_len = sum(item_lens)
    q = torch.randn(plain_len + extend_len, num_q_heads, head_dim)
    k = torch.randn(plain_len + extend_len, num_kv_heads, head_dim)
    prefix_q = torch.randn(prefix_len, num_q_heads, head_dim)

    request = DllmTokenEvictionRequest(
        config=DllmTokenEvictionConfig(capacity=3, pool_kernel=3),
        prefix_kv_indices=prefix_slots,
        chunk_lens=chunk_lens,
        query_len=query_len,
    )
    capture = DllmTokenEvictionCapture(
        requests=[None, request], extend_lens=[plain_len, extend_len]
    )
    for _ in range(2):
        accumulate_token_eviction_layer(
            capture,
            q=q.reshape(-1, num_q_heads * head_dim),
            k=k.reshape(-1, num_kv_heads * head_dim),
            key_buffer=key_buffer,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scaling=scaling,
        )

    item_start = plain_len
    for chunk_order, chunk_len in enumerate(chunk_lens):
        item_end = item_start + item_lens[chunk_order]
        expected = _dense_query_to_chunk_scores(
            q=torch.cat([prefix_q, q[item_start:item_end]]),
            keys=torch.cat([key_buffer[prefix_slots], k[item_start:item_end]]),
            prefix_len=prefix_len,
            chunk_len=chunk_len,
            scaling=scaling,
            pool_kernel=3,
        )
        torch.testing.assert_close(
            request.score_sums[chunk_order], 2 * expected, rtol=1e-5, atol=1e-6
        )
        item_start = item_end

    keep = select_token_eviction_keep(capture)
    assert keep[0] is None
    assert [len(kept) for kept in keep[1]] == [3, 3]
    assert all(kept[0] == 0 and kept == sorted(kept) for kept in keep[1])


def test_token_eviction_keeps_bos_then_highest_scores_in_source_order():
    scores = torch.tensor([0.0, 5.0, 1.0, 9.0, 3.0, 7.0])

    assert select_kept_token_positions(
        scores, capacity=3, force_keep_first=True
    ).tolist() == [0, 3, 5]
    assert select_kept_token_positions(
        scores, capacity=3, force_keep_first=False
    ).tolist() == [1, 3, 5]
    assert select_kept_token_positions(
        scores, capacity=1, force_keep_first=True
    ).tolist() == [0]
    # A chunk that already fits the budget is never reordered or trimmed.
    assert select_kept_token_positions(
        scores, capacity=6, force_keep_first=True
    ).tolist() == [0, 1, 2, 3, 4, 5]


def test_token_eviction_compacts_ids_kv_and_positions_consistently():
    """After eviction the decode stage must see one id, one KV slot, and one
    RoPE position per surviving token, and retraction must undo all of it."""
    prefix_len, chunk_lens, query_len, max_new_tokens = 2, [4, 3], 3, 2
    req = SimpleNamespace(
        dllm_config=SimpleNamespace(
            needs_full_prefill=True, dual_cache=True, first_done_first_out_mode=True
        ),
        origin_input_ids=list(
            range(100, 100 + prefix_len + sum(chunk_lens) + query_len)
        ),
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens,
            custom_params={
                "dllm_parallelcomp": {
                    "prefix_len": prefix_len,
                    "chunk_lens": chunk_lens,
                    "query_len": query_len,
                    "chunk_batch_size": 2,
                    "chunk_position_starts": [2, 1026],
                    "query_position_start": 2050,
                    "token_eviction": {"capacity": 2, "score_query_ids": [7, 8]},
                }
            },
        ),
    )
    req.dllm_parallelcomp_state = ReqDllmMixin._parse_parallelcomp_state(req)
    req.parallelcomp_chunk_batch_range = lambda: (
        ReqDllmMixin.parallelcomp_chunk_batch_range(req)
    )
    state = req.dllm_parallelcomp_state
    state["stage"] = "chunk"

    # Chunk items carry the 2-token scoring query, not the 3-token real query.
    assert ReqDllmMixin.parallelcomp_item_lens(req) == [6, 5]

    kept_per_chunk = [[0, 2], [0, 1]]
    chunk_slots = [torch.tensor([40, 41, 42, 43]), torch.tensor([50, 51, 52])]
    retained = []
    for slots, kept in zip(chunk_slots, kept_per_chunk):
        kept_slots, evicted = split_kept_kv_indices(slots, kept)
        assert len(kept_slots) + len(evicted) == len(slots)
        retained.extend(kept_slots.tolist())
        state["token_eviction"].kept_positions.append(kept)
    assert retained == [40, 42, 50, 51]

    state["stage"] = "decode"
    ReqDllmMixin.compact_parallelcomp_input_ids(req)
    assert list(req.origin_input_ids) == [100, 101, 102, 104, 106, 107, 109, 110, 111]
    req.extend_range = SimpleNamespace(
        start=0, end=len(req.origin_input_ids) + max_new_tokens
    )
    assert ReqDllmMixin.parallelcomp_position_values(req) == [
        0,
        1,
        2,
        4,
        1026,
        1027,
        2050,
        2051,
        2052,
        2053,
        2054,
    ]

    ReqDllmMixin.reset_parallelcomp_prefill_state(req)
    assert list(req.origin_input_ids) == list(range(100, 112))
    assert state["token_eviction"].kept_positions == []


def test_token_eviction_decode_stage_keeps_the_full_query_as_prompt():
    """Chunk forwards append only the scoring window, but the decode stage must
    still count the whole official query as prompt, or query tokens are
    denoised as if they were generation slots."""
    chunk_lens, query_len, window = [4, 3], 3, [7, 8]
    req = SimpleNamespace(
        dllm_config=SimpleNamespace(
            needs_full_prefill=True, dual_cache=True, first_done_first_out_mode=True
        ),
        origin_input_ids=list(range(100, 100 + sum(chunk_lens) + query_len)),
        sampling_params=SimpleNamespace(
            max_new_tokens=2,
            custom_params={
                "dllm_parallelcomp": {
                    "prefix_len": 0,
                    "chunk_lens": chunk_lens,
                    "query_len": query_len,
                    "chunk_batch_size": 2,
                    "token_eviction": {"capacity": 2, "score_query_ids": window},
                }
            },
        ),
        req_pool_idx=0,
        kv=SimpleNamespace(kv_allocated_len=0),
        kv_committed_len=0,
        prefix_indices=torch.empty(0, dtype=torch.long),
        dllm_token_eviction_state=None,
        is_scoring_token_eviction=lambda: False,
    )
    req.dllm_parallelcomp_state = ReqDllmMixin._parse_parallelcomp_state(req)
    req.parallelcomp_chunk_batch_range = lambda: (
        ReqDllmMixin.parallelcomp_chunk_batch_range(req)
    )
    req.compact_parallelcomp_input_ids = lambda: (
        ReqDllmMixin.compact_parallelcomp_input_ids(req)
    )
    freed = []
    scheduler = SimpleNamespace(
        dllm_config=req.dllm_config,
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=1,
            free=lambda indices: freed.extend(indices.tolist()),
            free_group_begin=MagicMock(),
            free_group_end=MagicMock(),
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((1, 16), dtype=torch.long)
        ),
        output_streamer=SimpleNamespace(stream_output=MagicMock()),
        metrics_reporter=SimpleNamespace(report_prefill_stats=MagicMock()),
    )
    extend_len = sum(chunk_len + len(window) for chunk_len in chunk_lens)
    batch = SimpleNamespace(
        batch_size=lambda: 1,
        reqs=[req],
        extend_lens=[extend_len],
        out_cache_loc=torch.arange(20, 20 + extend_len),
        return_logprob=False,
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    result = SimpleNamespace(
        copy_done=None,
        accept_length_per_req_cpu=None,
        dllm_done_per_req_cpu=[False],
        dllm_algo_state=[None],
        dllm_token_keep_per_req=[[[0, 2], [0, 1]]],
        next_token_ids=[[]],
        can_run_cuda_graph=False,
    )

    SchedulerDllmMixin.process_batch_result_dllm(scheduler, batch, result)

    assert req.dllm_parallelcomp_state["stage"] == "decode"
    assert req.prefix_indices.tolist() == [20, 22, 26, 27]
    # Evicted chunk slots and both scoring-window copies go back to the pool.
    assert sorted(freed) == [21, 23, 24, 25, 28, 29, 30]
    assert req.dllm_algo_state["prompt_len"] == query_len
    assert req.dllm_algo_state["full_prompt_len"] == 4 + query_len
    assert list(req.origin_input_ids) == [100, 102, 104, 105, 107, 108, 109]


def test_flashinfer_single_wrapper_plans_with_the_dream_scoring_mask():
    """The scoring mask must reach the paged planner; dropping it leaves Dream's
    plain bidirectional attention, where every scored token sees itself."""
    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    updater.kv_indptr = [torch.zeros(4, dtype=torch.int32)]
    updater.qo_indptr = [torch.zeros(4, dtype=torch.int32)]
    updater._oversized_kv_indptr = None
    updater._oversized_qo_indptr = None
    updater.prefill_wrapper_ragged = None
    planned = {}
    updater.call_begin_forward = lambda *args, **kwargs: planned.update(kwargs)
    mask = torch.ones(4, dtype=torch.uint8)
    update_args = dict(
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=None,
        seq_lens_sum=2,
        prefix_lens=torch.tensor([0]),
        prefill_wrappers=[object(), object()],
        use_ragged=False,
        encoder_lens=None,
        spec_info=None,
        self_attention_custom_mask=mask,
    )

    FlashInferIndicesUpdaterPrefill.update_single_wrapper(updater, **update_args)
    assert planned["cross_attention_custom_mask"] is mask

    with pytest.raises(RuntimeError, match="sliding-window"):
        FlashInferIndicesUpdaterPrefill.update_sliding_window(updater, **update_args)
    with pytest.raises(RuntimeError, match="cross-attention"):
        FlashInferIndicesUpdaterPrefill.update_cross_attention(updater, **update_args)


def _head_eviction_state(*, capacity=2, chunk_lens=(4, 2, 5), bidirectional=True):
    # prefix [0, 1], chunks, query [90, 91, 92].
    input_ids = array("q", [0, 1])
    for chunk_order, chunk_len in enumerate(chunk_lens):
        input_ids.extend(10 * (chunk_order + 1) + offset for offset in range(chunk_len))
    input_ids.extend([90, 91, 92])
    state = parse_head_token_eviction(
        {
            "capacity": capacity,
            "prefix_len": 2,
            "chunk_lens": list(chunk_lens),
            "query_len": 3,
            "score_query_window": 2,
            "pool_kernel": 1,
            "bidirectional": bidirectional,
        },
        input_ids=input_ids,
    )
    return state, input_ids


def test_head_eviction_scores_only_oversized_chunks_then_generates():
    state, input_ids = _head_eviction_state()

    assert state.stage == STAGE_SCORE and state.chunk_cursor == 0
    assert list(score_stage_input_ids(state, input_ids)) == [
        0, 1, 10, 11, 12, 13, 91, 92,
    ]  # fmt: skip

    # The 2-token chunk already fits the capacity and is never scored.
    state.advance()
    assert state.stage == STAGE_SCORE and state.chunk_cursor == 2
    assert list(score_stage_input_ids(state, input_ids)) == [
        0, 1, 30, 31, 32, 33, 34, 91, 92,
    ]  # fmt: skip

    state.advance()
    assert state.stage == STAGE_GENERATE

    with pytest.raises(ValueError, match="do not cover the input"):
        parse_head_token_eviction(
            {"capacity": 2, "prefix_len": 2, "chunk_lens": [4], "query_len": 3},
            input_ids=input_ids,
        )


@pytest.mark.parametrize("bidirectional", [True, False])
def test_head_eviction_keeps_match_dense_unmasked_attention(bidirectional):
    torch.manual_seed(0)
    prefix_len, chunk_len, query_len = 2, 6, 2
    seq_len = prefix_len + chunk_len + query_len
    num_q_heads, num_kv_heads, head_dim, scaling = 4, 2, 8, 0.5
    config = DllmHeadEvictionConfig(
        capacity=3, pool_kernel=3, bidirectional=bidirectional
    )
    q = torch.randn(seq_len, num_q_heads * head_dim)
    k = torch.randn(seq_len, num_kv_heads * head_dim)
    capture = DllmHeadEvictionCapture(
        requests=[
            None,
            DllmHeadEvictionRequest(
                config=config,
                prefix_len=prefix_len,
                chunk_len=chunk_len,
                query_len=query_len,
            ),
        ],
        extend_lens=[5, seq_len],
    )

    accumulate_head_eviction_layer(
        capture,
        q=torch.cat([torch.randn(5, num_q_heads * head_dim), q]),
        k=torch.cat([torch.randn(5, num_kv_heads * head_dim), k]),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scaling=scaling,
    )
    keep = stack_head_eviction_keep(capture, num_layers=1)

    # Dense reference: every attention head attends the whole sequence.
    q_heads = q.view(seq_len, num_q_heads, head_dim).permute(1, 0, 2)
    k_heads = k.view(seq_len, num_kv_heads, head_dim).permute(1, 0, 2)
    k_heads = k_heads.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
    probs = torch.softmax(q_heads @ k_heads.transpose(1, 2) * scaling, dim=-1)
    chunk = slice(prefix_len, prefix_len + chunk_len)
    tail = slice(prefix_len + chunk_len, seq_len)
    scores = probs[:, tail, chunk].sum(dim=1)
    if bidirectional:
        scores = scores + probs[:, chunk, tail].sum(dim=2)
    scores = torch.nn.functional.max_pool1d(
        scores.unsqueeze(1), kernel_size=3, padding=1, stride=1
    ).squeeze(1)
    scores = scores.view(num_kv_heads, -1, chunk_len).mean(dim=1)
    expected = torch.stack(
        [
            torch.cat(
                [torch.zeros(1, dtype=torch.long), scores[head, 1:].topk(2).indices + 1]
            )
            .sort()
            .values
            for head in range(num_kv_heads)
        ]
    )

    assert keep[0] is None
    assert keep[1].shape == (1, num_kv_heads, 3)
    assert torch.equal(keep[1][0], expected)
    with pytest.raises(RuntimeError, match="scored 1 of 2 attention layers"):
        stack_head_eviction_keep(capture, num_layers=2)


def test_head_eviction_rejects_a_scoring_row_split_across_forwards():
    state, _ = _head_eviction_state()

    assert build_head_eviction_capture(states=[None, None], extend_lens=[4, 4]) is None
    with pytest.raises(RuntimeError, match="chunked-prefill-size"):
        build_head_eviction_capture(states=[state], extend_lens=[5])


def test_head_eviction_gives_each_head_its_own_tokens_and_keeps_rope_positions():
    state, input_ids = _head_eviction_state()
    num_layers, num_kv_heads, head_dim = 2, 2, 1
    # Layer 0: head 0 keeps chunk tokens {0, 3}, head 1 keeps {0, 1}.
    first_chunk_keep = torch.tensor([[[0, 3], [0, 1]], [[0, 2], [0, 3]]])
    last_chunk_keep = torch.tensor([[[0, 4], [0, 2]], [[0, 1], [0, 3]]])
    state.keep_positions = [first_chunk_keep, None, last_chunk_keep]
    state.stage = STAGE_GENERATE

    # Prompt token i lives in slot 100 + i; a slot stores 1000 * layer +
    # 100 * head + token index so every moved entry names its origin.
    prompt_len = len(input_ids)
    prompt_slots = torch.arange(100, 100 + prompt_len)
    kv_buffers = []
    for layer in range(num_layers):
        buffer = torch.zeros(200, num_kv_heads, head_dim)
        for head in range(num_kv_heads):
            buffer[100 : 100 + prompt_len, head, 0] = (
                1000 * layer + 100 * head + torch.arange(prompt_len)
            )
        kv_buffers.append((buffer, buffer.clone() + 0.5))

    retained, evicted = compact_prompt_kv_per_head(
        kv_buffers=kv_buffers, prompt_slots=prompt_slots, state=state
    )

    # prefix(2) + chunk0 -> 2 slots + whole 2-token chunk + chunk2 -> 2 slots + query(3)
    assert retained.tolist() == [100, 101, 102, 103, 106, 107, 108, 109, 113, 114, 115]
    assert sorted(evicted.tolist()) == [104, 105, 110, 111, 112]
    key_buffer, value_buffer = kv_buffers[0]
    assert key_buffer[[102, 103], 0, 0].tolist() == [2, 5]
    assert key_buffer[[102, 103], 1, 0].tolist() == [102, 103]
    assert key_buffer[[108, 109], 0, 0].tolist() == [8, 12]
    assert value_buffer[[108, 109], 1, 0].tolist() == [108.5, 110.5]
    assert kv_buffers[1][0][[102, 103], 1, 0].tolist() == [1102, 1105]
    # Prefix, the short chunk, and the query stay where they were.
    assert key_buffer[[100, 106, 113], 0, 0].tolist() == [0, 6, 13]

    req = SimpleNamespace(
        dllm_token_eviction_state=state,
        origin_input_ids=input_ids,
        sampling_params=SimpleNamespace(custom_params=None),
        is_scoring_token_eviction=lambda: False,
        dllm_block_position_shift=0,
    )
    ReqDllmMixin.compact_token_eviction_input_ids(req)
    assert list(req.origin_input_ids) == list(compact_input_ids(state, input_ids))
    assert len(req.origin_input_ids) == len(retained)
    # The generation canvas follows the compacted prompt in storage but
    # keeps the positions it had after the full 16-token prompt.
    req.extend_range = SimpleNamespace(start=len(retained), end=len(retained) + 3)
    assert _compute_dllm_positions(req) == [16, 17, 18]
    req.extend_range = SimpleNamespace(start=0, end=len(retained))
    assert _compute_dllm_positions(req) == [0, 1, 2, 3, 6, 7, 8, 9, 13, 14, 15]

    ReqDllmMixin.reset_token_eviction_state(req)
    assert list(req.origin_input_ids) == list(input_ids)
    assert state.stage == STAGE_SCORE and state.chunk_cursor == 0
    assert state.keep_positions == [None, None, None]


def test_head_eviction_leaves_the_request_ready_for_dual_cache_denoising():
    state, input_ids = _head_eviction_state()
    block_size = 3
    freed = []
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist()),
            get_kvcache=lambda: SimpleNamespace(
                start_layer=0,
                layer_num=1,
                get_kv_buffer=lambda layer_id: (
                    torch.zeros(64, 2, 1),
                    torch.zeros(64, 2, 1),
                ),
            ),
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(1, 32, dtype=torch.int32)
        ),
    )
    req = SimpleNamespace(
        dllm_token_eviction_state=state,
        origin_input_ids=input_ids,
        req_pool_idx=0,
        kv=SimpleNamespace(kv_allocated_len=8),
        kv_committed_len=8,
        prefix_indices=torch.arange(8),
        dllm_algo_state={"prompt_len": 8, "step": 0},
        dllm_phase=None,
    )
    req.is_scoring_token_eviction = lambda: ReqDllmMixin.is_scoring_token_eviction(req)
    req.compact_token_eviction_input_ids = lambda: (
        ReqDllmMixin.compact_token_eviction_input_ids(req)
    )

    # Two scoring rounds (the middle chunk is skipped) free all their KV.
    SchedulerDllmMixin._finish_token_eviction_score_round(
        scheduler,
        req,
        round_cache_loc=torch.arange(40, 48),
        keep=torch.tensor([[[0, 3], [0, 1]]]),
    )
    assert req.is_scoring_token_eviction() and state.chunk_cursor == 2
    assert (req.kv.kv_allocated_len, req.kv_committed_len) == (0, 0)
    SchedulerDllmMixin._finish_token_eviction_score_round(
        scheduler,
        req,
        round_cache_loc=torch.arange(48, 57),
        keep=torch.tensor([[[0, 4], [0, 2]]]),
    )
    assert freed == list(range(40, 57))
    assert not req.is_scoring_token_eviction()
    assert req.dllm_algo_state == {"prompt_len": len(input_ids), "step": 0}
    assert len(req.prefix_indices) == 0

    # The first full pass wrote prompt slots 0..15 and canvas slots 16..18.
    freed.clear()
    prompt_len = len(input_ids)
    req.prefix_indices = torch.arange(prompt_len)
    req.dllm_kv_indices = torch.arange(prompt_len, prompt_len + block_size)
    req.dllm_incomplete_ids = array("q", [7, 8, 9])
    req.dllm_algo_state = {"prompt_len": prompt_len, "dual_cache_ready": True}
    SchedulerDllmMixin._evict_prompt_tokens_per_head(scheduler, req)

    compact_len = prompt_len - 5
    assert sorted(freed) == [4, 5, 10, 11, 12]
    assert req.prefix_indices.tolist() == [0, 1, 2, 3, 6, 7, 8, 9, 13, 14, 15]
    row = scheduler.req_to_token_pool.req_to_token[0]
    assert row[: compact_len + block_size].tolist() == (
        req.prefix_indices.tolist() + [16, 17, 18]
    )
    # The DLLM_DENOISE fast path requires prompt, prefix, and KV row to agree.
    assert req.dllm_algo_state["prompt_len"] == compact_len
    assert req.kv.kv_allocated_len == req.kv_committed_len == compact_len + block_size
    assert len(req.origin_input_ids) == compact_len
    assert list(req.full_untruncated_fill_ids[compact_len:]) == [7, 8, 9]


def test_dream_prompt_handoff_resumes_in_dual_cache_denoising_on_the_decode_server():
    block_size, mask_id = 4, 99
    prompt_ids = array("q", [5, 6, 7])
    dllm_config = SimpleNamespace(block_size=block_size, mask_id=mask_id)

    # Prefill server: the first full pass wrote prompt slots 10..12 and canvas
    # slots 13..16; only the prompt KV and the first canvas token leave.
    freed, sent = [], []
    prefill_req = SimpleNamespace(
        prefix_indices=torch.tensor([10, 11, 12]),
        dllm_kv_indices=torch.tensor([13, 14, 15, 16]),
        dllm_incomplete_ids=array("q", [42, mask_id, mask_id, mask_id]),
        dllm_algo_state={"prompt_len": 3, "dual_cache_ready": True},
        kv=SimpleNamespace(kv_allocated_len=7),
        kv_committed_len=7,
        output_ids=array("q"),
        pending_bootstrap=False,
        time_stats=MagicMock(),
        set_extend_range=lambda start, end: sent.append(("range", start, end)),
        dllm_canvas_handoff_len=lambda: 0,
    )
    manager = SimpleNamespace(remove_req=MagicMock())
    prefill = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist())
        ),
        dllm_manager=manager,
        disagg_prefill_inflight_queue=[],
        send_kv_chunk=lambda req, last_chunk: sent.append(("send", last_chunk)),
    )
    SchedulerDllmMixin._hand_off_dllm_prompt_kv(prefill, prefill_req, first_token=42)

    assert freed == [13, 14, 15, 16]
    assert list(prefill_req.output_ids) == [42]
    assert (prefill_req.kv.kv_allocated_len, prefill_req.kv_committed_len) == (3, 3)
    assert sent == [("range", 0, 3), ("send", True)]
    assert prefill.disagg_prefill_inflight_queue == [prefill_req]
    manager.remove_req.assert_called_once_with(prefill_req)

    # Decode server: the transfer filled request-row slots 20..22 and committed
    # the handoff token as the request's only output id.
    extend_ranges = []
    decode_req = SimpleNamespace(
        origin_input_ids=prompt_ids,
        output_ids=array("q", [42]),
        req_pool_idx=0,
        sampling_params=SimpleNamespace(max_new_tokens=block_size),
        dllm_algo_state={"prompt_len": 3, "step": 0},
        dllm_partial_draft_state=None,
        kv=SimpleNamespace(kv_allocated_len=3),
        kv_committed_len=3,
        dllm_handoff_canvas_len=0,
        set_extend_range=lambda start, end: extend_ranges.append((start, end)),
    )
    row = torch.zeros(1, 16, dtype=torch.int32)
    row[0, :3] = torch.tensor([20, 21, 22])
    decode = SimpleNamespace(
        dllm_config=dllm_config,
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        tree_cache=None,
    )
    decode._enter_dllm_block = lambda req: SchedulerDllmMixin._enter_dllm_block(
        decode, req
    )
    with patch(
        "sglang.srt.dllm.mixin.scheduler.alloc_token_slots",
        return_value=torch.tensor([30, 31, 32, 33]),
    ):
        SchedulerDllmMixin.resume_dllm_after_prompt_transfer(decode, decode_req)

    assert decode_req.prefix_indices.tolist() == [20, 21, 22]
    assert decode_req.dllm_kv_indices.tolist() == [30, 31, 32, 33]
    assert row[0, :7].tolist() == [20, 21, 22, 30, 31, 32, 33]
    assert list(decode_req.dllm_incomplete_ids) == [42, mask_id, mask_id, mask_id]
    assert list(decode_req.full_untruncated_fill_ids) == [5, 6, 7, 42, 99, 99, 99]
    assert list(decode_req.output_ids) == []
    assert decode_req.dllm_algo_state["dual_cache_ready"] is True
    assert decode_req.dllm_algo_state["is_prefill"] is False
    assert decode_req.dllm_algo_state["prompt_len"] == 3
    assert (decode_req.kv.kv_allocated_len, decode_req.kv_committed_len) == (7, 7)
    assert extend_ranges == [(3, 7)]


def test_decode_server_adopts_the_evicted_prompt_layout_without_scoring():
    state, input_ids = _head_eviction_state()
    req = SimpleNamespace(
        dllm_token_eviction_state=state,
        origin_input_ids=input_ids,
        dllm_algo_state={"prompt_len": len(input_ids), "step": 0},
        dllm_phase=None,
        dllm_block_position_shift=0,
        dllm_handoff_canvas_len=0,
        dllm_canvas_handoff_len=lambda: 0,
        sampling_params=SimpleNamespace(custom_params=None),
        is_scoring_token_eviction=lambda: ReqDllmMixin.is_scoring_token_eviction(req),
    )
    req.compact_token_eviction_input_ids = lambda: (
        ReqDllmMixin.compact_token_eviction_input_ids(req)
    )
    # The prefill server announces this length before it has scored anything.
    assert ReqDllmMixin.dllm_handoff_len(req) == len(input_ids) - 5

    ReqDllmMixin.adopt_dllm_handoff_layout(req)

    assert not req.is_scoring_token_eviction()
    assert len(req.origin_input_ids) == len(input_ids) - 5
    assert req.dllm_algo_state["prompt_len"] == len(input_ids) - 5
    assert ReqDllmMixin.dllm_handoff_len(req) == len(req.origin_input_ids)
    # The canvas keeps the positions that follow the uncompacted prompt.
    req.extend_range = SimpleNamespace(start=11, end=13)
    assert _compute_dllm_positions(req) == [16, 17]


def test_transferred_draft_resumes_at_suffix_initialization():
    row = torch.zeros(1, 16, dtype=torch.int32)
    row[0, :3] = torch.tensor([20, 21, 22])
    decode = SimpleNamespace(req_to_token_pool=SimpleNamespace(req_to_token=row))
    draft_state = {
        "partial_draft": True,
        "partial_draft_stage": "prompt",
        "partial_draft_first_token": None,
        "partial_draft_rounds_done": 0,
        "partial_draft_round_limit": 1,
        "partial_draft_confirmed_mask": [False] * 4,
        "canvas_len": 4,
    }
    req = SimpleNamespace(
        origin_input_ids=array("q", [5, 6, 7]),
        output_ids=array("q", [42]),
        req_pool_idx=0,
        dllm_partial_draft_state=dict(draft_state),
        dllm_algo_state={"prompt_len": 3, "step": 0, **draft_state},
        dllm_handoff_canvas_len=0,
        init_next_round_input=MagicMock(),
    )

    SchedulerDllmMixin.resume_dllm_after_prompt_transfer(decode, req)

    assert req.prefix_indices.tolist() == [20, 21, 22]
    assert req.dllm_algo_state["partial_draft_stage"] == "suffix_init"
    assert req.dllm_algo_state["partial_draft_first_token"] == 42
    assert list(req.output_ids) == [] and req.dllm_kv_indices is None
    # The masked suffix is built behind the cached prompt on the next round.
    req.init_next_round_input.assert_called_once_with()
    assert ReqDllmMixin.has_partial_draft_prompt_cache(req)


def test_score_requests_stay_plain_prefills_on_a_dllm_server():
    score_params = SimpleNamespace(
        custom_params={"dream_score_attention_mask": "causal"}
    )
    assert is_score_request(score_params)
    assert not is_score_request(SimpleNamespace(custom_params=None))
    assert not is_score_request(
        SimpleNamespace(custom_params={"dllm_position_start": 3})
    )

    dllm_config = SimpleNamespace(needs_full_prefill=True, max_running_requests=4)

    class ScoreReq(ReqDllmMixin):
        sampling_params = score_params
        origin_input_ids = array("q", [1, 2, 3])

    score_req = ScoreReq()
    score_req.init_diffusion_llm(dllm_config)
    assert score_req.dllm_config is None
    assert score_req.dllm_request_error is None
    assert not score_req.is_dllm()
    assert score_req.dllm_phase is None and score_req.dllm_algo_state is None

    # The dLLM manager only takes generation requests; scoring requests wait
    # in the scheduler queue for an ordinary prefill batch.
    generation_req = SimpleNamespace(is_dllm=lambda: True)
    score_req.is_dllm = lambda: False
    manager = SimpleNamespace(waiting_queue=[], add_waiting_reqs=MagicMock())
    scheduler = SimpleNamespace(
        max_running_requests=4,
        dllm_config=dllm_config,
        dllm_manager=manager,
        waiting_queue=[score_req, generation_req],
    )
    SchedulerDllmMixin._fetch_waiting_reqs(scheduler)
    manager.add_waiting_reqs.assert_called_once_with([generation_req])
    assert scheduler.waiting_queue == [score_req]


def test_decode_server_slot_pool_lets_concurrent_dllm_requests_keep_their_slots():
    from sglang.srt.disaggregation.decode import DecodeReqToTokenPool

    pool = DecodeReqToTokenPool(
        size=4,
        max_context_len=8,
        device="cpu",
        enable_memory_saver=False,
        pre_alloc_size=2,
    )

    def resumed(slot, *, dllm):
        return SimpleNamespace(
            req_pool_idx=slot,
            inflight_middle_chunks=0,
            kv_committed_len=5,
            is_dllm=lambda: dllm,
        )

    # Every Dream request returns with its slot on each denoising round.
    assert pool.alloc([resumed(1, dllm=True), resumed(2, dllm=True)]) == [1, 2]
    with pytest.raises(AssertionError, match="only one chunked request"):
        pool.alloc([resumed(1, dllm=False), resumed(2, dllm=False)])
