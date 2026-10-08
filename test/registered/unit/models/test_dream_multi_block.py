import math
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.dllm.algorithm.prefilling_dream import PrefillingDream
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.layers.rotary_embedding.yarn import (
    YaRNScalingRotaryEmbedding,
    yarn_get_mscale_simple,
)
from sglang.srt.model_executor.forward_batch_info import _compute_dllm_positions
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MASK = 99
BLOCK = 2


def _dllm_config(**overrides):
    config = dict(
        algorithm="PrefillingDream",
        algorithm_config={},
        block_size=BLOCK,
        mask_id=MASK,
        needs_full_prefill=True,
        dual_cache=True,
        first_done_first_out_mode=True,
        max_running_requests=4,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def _scheduler(freed):
    row = torch.zeros(1, 32, dtype=torch.int32)
    return SimpleNamespace(
        dllm_config=_dllm_config(),
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist())
        ),
        metrics_reporter=SimpleNamespace(num_generated_tokens=0),
        tree_cache=None,
    )


def _req_after_first_pass():
    """Prompt 1 2 3 in slots 10..12; a five-token canvas in slots 20..24."""
    return SimpleNamespace(
        req_pool_idx=0,
        origin_input_ids=array("q", [1, 2, 3]),
        output_ids=array("q"),
        prefix_indices=torch.tensor([10, 11, 12]),
        dllm_kv_indices=torch.tensor([20, 21, 22, 23, 24]),
        dllm_incomplete_ids=array("q", [7, MASK, MASK, MASK, MASK]),
        full_untruncated_fill_ids=array("q", [1, 2, 3, 7, MASK, MASK, MASK, MASK]),
        dllm_algo_state={"prompt_len": 3, "canvas_len": 5, "dual_cache_ready": True},
        dllm_block_position_shift=0,
        dllm_canvas_output_len=0,
        dllm_initialized=True,
        dllm_token_eviction_state=None,
        sampling_params=SimpleNamespace(custom_params=None, max_new_tokens=5),
        kv=SimpleNamespace(kv_allocated_len=8),
        kv_committed_len=8,
        finished=lambda: False,
        update_finish_state=MagicMock(),
        is_scoring_token_eviction=lambda: False,
    )


def test_block_rounds_extend_the_row_tail_at_natural_canvas_positions():
    freed = []
    scheduler = _scheduler(freed)
    req = _req_after_first_pass()
    row = scheduler.req_to_token_pool.req_to_token

    SchedulerDllmMixin._enter_dllm_block(scheduler, req)

    # The three later mask slots move in front of the first block's two slots.
    assert req.prefix_indices.tolist() == [10, 11, 12, 22, 23, 24]
    assert req.dllm_kv_indices.tolist() == [20, 21]
    assert row[0, 3:8].tolist() == [22, 23, 24, 20, 21]
    assert list(req.full_untruncated_fill_ids) == [1, 2, 3, MASK, MASK, MASK, 7, MASK]
    assert list(req.dllm_incomplete_ids) == [7, MASK]
    assert req.dllm_algo_state["prompt_len"] == 6
    assert req.dllm_algo_state["canvas_len"] == BLOCK
    assert req.dllm_algo_state["more_blocks"] is True

    req.extend_range = SimpleNamespace(start=6, end=8)
    assert _compute_dllm_positions(req) == [3, 4]


def test_finished_block_is_committed_finalized_and_followed_by_the_next_block():
    freed = []
    scheduler = _scheduler(freed)
    req = _req_after_first_pass()
    row = scheduler.req_to_token_pool.req_to_token
    SchedulerDllmMixin._enter_dllm_block(scheduler, req)

    # An unfinished block stays in the denoising stage.
    SchedulerDllmMixin._commit_dllm_block_if_complete(scheduler, req)
    assert list(req.output_ids) == []
    assert "block_stage" not in req.dllm_algo_state

    req.dllm_incomplete_ids = array("q", [7, 8])
    SchedulerDllmMixin._commit_dllm_block_if_complete(scheduler, req)
    assert list(req.output_ids) == [7, 8]
    assert req.dllm_canvas_output_len == 2
    assert req.dllm_algo_state["block_stage"] == "finalize"
    assert scheduler.metrics_reporter.num_generated_tokens == 2
    req.update_finish_state.assert_called_once_with(new_accepted_len=2)

    # The finalize forward rewrote slots 20 and 21; the later masks go away.
    SchedulerDllmMixin._drop_later_dllm_blocks(scheduler, req)
    assert freed == [22, 23, 24]
    assert req.prefix_indices.tolist() == [10, 11, 12, 20, 21]
    assert row[0, 3:5].tolist() == [20, 21]
    assert (req.kv.kv_allocated_len, req.kv_committed_len) == (5, 5)
    assert req.dllm_block_position_shift == 0
    assert req.dllm_kv_indices is None
    assert req.dllm_initialized is False
    assert req.dllm_algo_state["prompt_len"] == 5

    # The next forward initialized the three remaining slots behind the cache.
    req.full_untruncated_fill_ids = array("q", [1, 2, 3, 7, 8, MASK, MASK, MASK])
    SchedulerDllmMixin._adopt_dllm_block_suffix(
        scheduler,
        req,
        round_tokens=array("q", [5, MASK, MASK]),
        round_cache_loc=torch.tensor([30, 31, 32]),
    )
    SchedulerDllmMixin._enter_dllm_block(scheduler, req)
    assert req.prefix_indices.tolist() == [10, 11, 12, 20, 21, 32]
    assert req.dllm_kv_indices.tolist() == [30, 31]
    assert list(req.dllm_incomplete_ids) == [5, MASK]
    assert req.dllm_block_position_shift == 1
    req.extend_range = SimpleNamespace(start=6, end=8)
    assert _compute_dllm_positions(req) == [5, 6]


def test_last_block_keeps_the_natural_row_order():
    scheduler = _scheduler([])
    req = _req_after_first_pass()
    req.dllm_kv_indices = torch.tensor([20, 21])
    req.dllm_incomplete_ids = array("q", [7, MASK])
    req.full_untruncated_fill_ids = array("q", [1, 2, 3, 7, MASK])

    SchedulerDllmMixin._enter_dllm_block(scheduler, req)

    assert req.prefix_indices.tolist() == [10, 11, 12]
    assert req.dllm_block_position_shift == 0
    assert req.dllm_algo_state["more_blocks"] is False
    assert req.dllm_algo_state["prompt_len"] == 3


def test_block_with_a_stop_token_ends_the_request_without_later_blocks():
    scheduler = _scheduler([])
    req = _req_after_first_pass()
    SchedulerDllmMixin._enter_dllm_block(scheduler, req)
    req.dllm_incomplete_ids = array("q", [7, 8])
    req.finished = lambda: True
    req.time_stats = MagicMock()

    with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
        SchedulerDllmMixin._commit_dllm_block_if_complete(scheduler, req)

    release.assert_called_once()
    assert list(req.output_ids) == [7, 8]
    assert req.dllm_algo_state is None
    assert req.dllm_block_position_shift == 0


def test_transferred_multi_block_prompt_initializes_the_whole_canvas_first():
    row = torch.zeros(1, 16, dtype=torch.int32)
    row[0, :3] = torch.tensor([20, 21, 22])
    decode = SimpleNamespace(
        dllm_config=_dllm_config(),
        req_to_token_pool=SimpleNamespace(req_to_token=row),
    )
    req = SimpleNamespace(
        origin_input_ids=array("q", [1, 2, 3]),
        output_ids=array("q", [42]),
        req_pool_idx=0,
        sampling_params=SimpleNamespace(max_new_tokens=5),
        dllm_algo_state={"prompt_len": 3, "step": 0},
        dllm_partial_draft_state=None,
        init_next_round_input=MagicMock(),
    )

    SchedulerDllmMixin.resume_dllm_after_prompt_transfer(decode, req)

    assert req.prefix_indices.tolist() == [20, 21, 22]
    assert list(req.output_ids) == []
    assert req.dllm_algo_state["block_stage"] == "suffix_init"
    assert req.dllm_algo_state["block_first_token"] == 42
    assert req.dllm_algo_state["dual_cache_ready"] is True
    assert req.dllm_kv_indices is None
    req.init_next_round_input.assert_called_once_with()


def _step(algorithm, *, input_ids, logits, state, canvas_len):
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor(input_ids),
        extend_seq_lens_cpu=[len(input_ids)],
        dllm_canvas_lens_cpu=[canvas_len],
        dllm_block_size=BLOCK,
        batch_size=1,
        return_logprob=False,
    )
    done = algorithm.step(forward_batch, logits, [state])
    return done, forward_batch.input_ids.tolist()


def test_algorithm_block_stages_carry_the_next_first_token():
    algorithm = PrefillingDream(_dllm_config())
    state = {
        "prompt_len": 6,
        "is_prefill": False,
        "dual_cache_ready": True,
        "more_blocks": True,
        "block_stage": "finalize",
    }
    logits = torch.zeros(1, 8)
    logits[0, 5] = 3.0

    done, ids = _step(
        algorithm, input_ids=[7, 8], logits=logits, state=state, canvas_len=1
    )
    assert done == [False]
    assert ids == [7, 8]
    assert state["block_first_token"] == 5
    assert state["block_stage"] == "suffix_init"
    assert state["last_block_stage"] == "finalize"

    done, ids = _step(
        algorithm,
        input_ids=[MASK, MASK, MASK],
        logits=torch.zeros(3, 8),
        state=state,
        canvas_len=3,
    )
    assert done == [False]
    assert ids == [5, MASK, MASK]
    assert state["block_stage"] is None
    assert state["last_block_stage"] == "suffix_init"


def test_algorithm_does_not_finish_on_a_block_that_has_successors():
    algorithm = PrefillingDream(_dllm_config())
    state = {
        "prompt_len": 6,
        "is_prefill": False,
        "dual_cache_ready": True,
        "more_blocks": True,
    }
    done, _ = _step(
        algorithm,
        input_ids=[7, 8],
        logits=torch.zeros(2, 8),
        state=state,
        canvas_len=BLOCK,
    )
    assert done == [False]
    state["more_blocks"] = False
    done, _ = _step(
        algorithm,
        input_ids=[7, 8],
        logits=torch.zeros(2, 8),
        state=state,
        canvas_len=BLOCK,
    )
    assert done == [True]


class _Req(ReqDllmMixin):
    def __init__(self, *, input_ids, max_new_tokens=BLOCK, custom_params=None):
        self.origin_input_ids = array("q", input_ids)
        self.output_ids = array("q")
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.dllm_initialized = False
        self.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens, custom_params=custom_params
        )


def _init_req(config=None, **kwargs):
    req = _Req(**kwargs)
    req.init_diffusion_llm(_dllm_config() if config is None else config)
    return req


@pytest.mark.parametrize("max_new_tokens", [1, BLOCK, 5])
def test_canvas_length_tracks_requests_that_are_not_one_block(max_new_tokens):
    req = _init_req(input_ids=[1, 2, 3], max_new_tokens=max_new_tokens)
    assert req.dllm_request_error is None

    req._init_fill_ids_for_dllm()

    assert list(req.full_untruncated_fill_ids) == [1, 2, 3] + [MASK] * max_new_tokens
    assert req.dllm_algo_state.get("canvas_len") == (
        None if max_new_tokens == BLOCK else max_new_tokens
    )


@pytest.mark.parametrize(
    "custom_params, message",
    [
        ({"dllm_position_start": 2}, "sparse position"),
        ({"dllm_position_start": 9, "dllm_position_offset": 1}, "outside the prompt"),
        ({"dllm_position_start": 1, "dllm_position_offset": -1}, "non-negative"),
        ({"dllm_parallelcomp": "yes"}, "must be an object"),
        ({"dllm_partial_draft": {"rounds": 3}}, "rounds=1"),
        ({"dllm_token_eviction": {"capacity": 0}}, "dllm_token_eviction"),
        (
            {"dream_score_attention_mask": "full", "dream_score_prefix_len": 1},
            "dream_score_chunk_len",
        ),
        ({"dream_score_attention_mask": "diagonal"}, "'causal' or 'full'"),
    ],
)
def test_malformed_dllm_parameters_are_reported_instead_of_raised(
    custom_params, message
):
    req = _init_req(input_ids=[1, 2, 3], custom_params=custom_params)

    assert message in req.dllm_request_error
    assert req.dllm_token_eviction_state is None
    assert req.dllm_partial_draft_state is None
    assert req.dllm_parallelcomp_state is None


def test_multi_block_generation_needs_dual_cache_and_fdfo():
    assert _init_req(input_ids=[1, 2, 3], max_new_tokens=6).dllm_request_error is None
    for overrides in ({"dual_cache": False}, {"first_done_first_out_mode": False}):
        req = _init_req(
            config=_dllm_config(**overrides), input_ids=[1, 2, 3], max_new_tokens=6
        )
        assert "requires dual_cache and --dllm-fdfo" in req.dllm_request_error


def test_token_eviction_accepts_a_multi_block_canvas():
    req = _init_req(
        input_ids=[1, 2, 3, 4, 5, 6],
        max_new_tokens=3 * BLOCK,
        custom_params={
            "dllm_token_eviction": {
                "capacity": 2,
                "prefix_len": 1,
                "chunk_lens": [3],
                "query_len": 2,
            }
        },
    )
    assert req.dllm_request_error is None
    assert req.dllm_token_eviction_state is not None


def _hf_yarn_inv_freq(*, dim, base, factor, max_position_embeddings):
    """transformers 4.46 ``_compute_yarn_parameters``, as the reference runs it."""

    def correction_dim(num_rotations):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = max(math.floor(correction_dim(32)), 0)
    high = min(math.ceil(correction_dim(1)), dim - 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    ramp = torch.clamp(
        (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
    )
    extrapolation_factor = 1 - ramp
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation_factor) + (
        1.0 / pos_freqs
    ) * extrapolation_factor
    return inv_freq, 0.1 * math.log(factor) + 1.0


def test_capped_yarn_table_matches_the_reference_frequencies():
    dim, base, factor, max_position = 128, 1000000.0, 64.0, 131072
    # Unbound methods on a stand-in: constructing the module needs the
    # platform's RoPE kernel.
    rope = SimpleNamespace(
        base=base,
        rotary_dim=dim,
        max_position_embeddings=max_position,
        scaling_factor=factor,
        beta_fast=32,
        beta_slow=1,
        truncate=True,
        extrapolation_factor=1,
        mscale=yarn_get_mscale_simple(factor),
        max_cached_positions=4096,
    )
    rope._compute_inv_freq = lambda scaling_factor: (
        YaRNScalingRotaryEmbedding._compute_inv_freq(rope, scaling_factor)
    )
    rope.cos_sin_cache = YaRNScalingRotaryEmbedding._compute_cos_sin_cache(rope)
    assert rope.cos_sin_cache.shape == (4096, dim)

    inv_freq, attention_factor = _hf_yarn_inv_freq(
        dim=dim, base=base, factor=factor, max_position_embeddings=max_position
    )
    positions = torch.tensor([0, 1, 37, 1024, 4095, 5000], dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * attention_factor

    # Position 5000 lies past the cap: the table grows with YaRN frequencies.
    YaRNScalingRotaryEmbedding._ensure_cos_sin_cache_length(rope, 5000)
    actual = rope.cos_sin_cache[positions.long()]
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_rope_factory_forwards_the_yarn_table_cap():
    with patch(
        "sglang.srt.layers.rotary_embedding.factory.YaRNScalingRotaryEmbedding"
    ) as yarn:
        get_rope(
            head_size=128,
            rotary_dim=128,
            max_position=131072,
            base=1000000.0,
            is_neox_style=True,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 64.0,
                "original_max_position_embeddings": 131072,
                "max_cached_positions": 131072,
            },
            dtype=torch.float32,
        )
    assert yarn.call_args.args[2] == 131072
    assert yarn.call_args.kwargs["max_cached_positions"] == 131072
