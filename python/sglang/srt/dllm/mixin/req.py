from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.head_token_eviction import (
    STAGE_SCORE,
    DllmHeadEvictionState,
    compact_input_ids,
    parse_head_token_eviction,
    score_stage_input_ids,
)
from sglang.srt.dllm.score_attention import is_score_request, parse_score_attention
from sglang.srt.dllm.sparse_positions import parse_sparse_position_shift
from sglang.srt.dllm.token_eviction import (
    parallelcomp_chunk_query_len,
    parse_token_eviction,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
        if dllm_config is not None and is_score_request(self.sampling_params):
            # Chunk scoring returns prompt logprobs from one plain forward.
            dllm_config = None
        self.dllm_phase: Optional[DllmReqPhase] = None
        self.dllm_incomplete_ids = array("q")
        # Physical generation KV slots retained by dual-cache rounds.
        self.dllm_kv_indices = None
        self.dllm_algo_state = (
            {"prompt_len": len(self.origin_input_ids), "step": 0}
            if dllm_config is not None and dllm_config.needs_full_prefill
            else None
        )
        self.dllm_block_offset = 0
        self.dllm_canvas_output_len = 0
        # Later-block mask tokens that sit before the block being denoised in
        # the request row; see SchedulerDllmMixin._enter_dllm_block.
        self.dllm_block_position_shift = 0
        # Mask tokens a decode server appended to origin_input_ids so that its
        # prompt-sized KV allocation and transfer also cover the canvas.
        self.dllm_handoff_canvas_len = 0
        self.dllm_config = dllm_config
        # Why the request's dLLM parameters are invalid; the scheduler answers
        # such a request with HTTP 400 instead of running it.
        self.dllm_request_error: Optional[str] = None
        try:
            self._parse_dllm_request_params()
        except ValueError as error:
            self.dllm_request_error = str(error)
            self.dllm_parallelcomp_state = None
            self.dllm_partial_draft_state = None
            self.dllm_token_eviction_state = None
        if self.dllm_partial_draft_state is not None:
            # Keep request-specific algorithm controls in the state that FDFO
            # already carries across denoising rounds.  The server-wide Dream
            # block size remains unchanged for ordinary generation requests.
            self.dllm_algo_state.update(self.dllm_partial_draft_state)

        if self.dllm_config is not None:
            if (
                self.dllm_parallelcomp_state is not None
                or self.dllm_partial_draft_state is not None
                or self.is_scoring_token_eviction()
            ):
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif self.dllm_config.needs_full_prefill:
                # Dream denoises a masked generation canvas, so it is a decode
                # request even though each round uses a full-attention prefill.
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            elif len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def _parse_dllm_request_params(self: Req) -> None:
        """Parse every dLLM custom parameter; ValueError names the bad one."""
        custom_params = self.sampling_params.custom_params
        parse_score_attention(custom_params, input_len=len(self.origin_input_ids))
        parse_sparse_position_shift(
            custom_params, prompt_len=len(self.origin_input_ids)
        )
        self.dllm_parallelcomp_state = self._parse_parallelcomp_state()
        self.dllm_partial_draft_state = self._parse_partial_draft_state()
        if (
            self.dllm_parallelcomp_state is not None
            and self.dllm_partial_draft_state is not None
        ):
            raise ValueError(
                "dllm_partial_draft cannot be combined with dllm_parallelcomp"
            )
        self.dllm_token_eviction_state = self._parse_token_eviction_state()
        self._validate_dllm_canvas()

    def _validate_dllm_canvas(self: Req) -> None:
        config = self.dllm_config
        if config is None or not config.needs_full_prefill:
            return
        max_new_tokens = self.sampling_params.max_new_tokens
        if (
            max_new_tokens is None
            or config.block_size is None
            or max_new_tokens <= config.block_size
        ):
            return
        if not (config.dual_cache and config.first_done_first_out_mode):
            raise ValueError(
                "max_new_tokens above the Dream block size requires dual_cache "
                f"and --dllm-fdfo: {max_new_tokens} > {config.block_size}"
            )
        if self.dllm_parallelcomp_state is not None:
            raise ValueError(
                "dllm_parallelcomp generates one block: max_new_tokens "
                f"{max_new_tokens} > block size {config.block_size}"
            )

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def _parse_partial_draft_state(self: Req) -> Optional[dict[str, Any]]:
        """Parse the selector-only bounded Dream draft mode.

        The first Dream forward confirms slot 0. Each requested partial round
        confirms exactly one additional highest-confidence slot, after which
        the unresolved positions are returned as mask tokens.
        """
        custom_params = getattr(self.sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        if "dllm_partial_draft" not in custom_params:
            return None

        config = custom_params["dllm_partial_draft"]
        if not isinstance(config, dict):
            raise ValueError("dllm_partial_draft must be an object")
        if not self.origin_input_ids:
            raise ValueError("dllm_partial_draft requires a non-empty prompt")
        if set(config) != {"rounds"}:
            raise ValueError("dllm_partial_draft only accepts the 'rounds' field")

        rounds = config.get("rounds")
        if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds != 1:
            raise ValueError("dllm_partial_draft currently requires rounds=1")
        if self.dllm_config is None or self.dllm_config.algorithm != "PrefillingDream":
            raise ValueError("dllm_partial_draft requires PrefillingDream")
        if not self.dllm_config.needs_full_prefill:
            raise ValueError("dllm_partial_draft requires Dream full-prefill mode")
        if not self.dllm_config.dual_cache:
            raise ValueError("dllm_partial_draft requires Dream dual_cache")
        if not self.dllm_config.first_done_first_out_mode:
            raise ValueError("dllm_partial_draft requires --dllm-fdfo")

        canvas_len = self.sampling_params.max_new_tokens
        if canvas_len != 4:
            raise ValueError("dllm_partial_draft currently requires max_new_tokens=4")
        if (
            self.dllm_config.block_size is None
            or canvas_len > self.dllm_config.block_size
        ):
            raise ValueError(
                "dllm_partial_draft canvas must fit the configured Dream block size"
            )

        return {
            "partial_draft": True,
            "partial_draft_stage": "prompt",
            "partial_draft_first_token": None,
            "canvas_len": canvas_len,
            "partial_draft_round_limit": rounds,
            "partial_draft_rounds_done": 0,
            "partial_draft_confirmed_mask": [False] * canvas_len,
        }

    def has_partial_draft_prompt_cache(self: Req) -> bool:
        """Whether a clean prompt KV prefix must survive the next stage."""
        state = self.dllm_algo_state
        return bool(
            isinstance(state, dict)
            and state.get("partial_draft", False)
            and state.get("partial_draft_stage") != "prompt"
        )

    def reset_partial_draft_state(self: Req) -> None:
        """Restart a retracted partial draft from a fresh prompt prefill."""
        state = self.dllm_algo_state
        if not isinstance(state, dict) or not state.get("partial_draft", False):
            return
        canvas_len = state["canvas_len"]
        state["partial_draft_stage"] = "prompt"
        state["partial_draft_first_token"] = None
        state["partial_draft_rounds_done"] = 0
        state["partial_draft_confirmed_mask"] = [False] * canvas_len
        for key in (
            "dual_cache_ready",
            "is_prefill",
            "last_partial_draft_stage",
            "last_round_was_dual_cache",
        ):
            state.pop(key, None)
        self.dllm_incomplete_ids = array("q")
        self.dllm_kv_indices = None

    def _parse_token_eviction_state(self: Req) -> Optional[DllmHeadEvictionState]:
        if self.dllm_config is None or not self.dllm_config.needs_full_prefill:
            return None
        custom_params = self.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        config = custom_params.get("dllm_token_eviction")
        if config is None:
            return None
        if (
            self.dllm_parallelcomp_state is not None
            or self.dllm_partial_draft_state is not None
        ):
            raise ValueError(
                "dllm_token_eviction cannot be combined with dllm_parallelcomp "
                "or dllm_partial_draft"
            )
        if not self.dllm_config.dual_cache:
            raise ValueError("dllm_token_eviction requires Dream dual_cache")
        if not self.dllm_config.first_done_first_out_mode:
            raise ValueError("dllm_token_eviction requires --dllm-fdfo")
        return parse_head_token_eviction(config, input_ids=self.origin_input_ids)

    def is_scoring_token_eviction(self: Req) -> bool:
        state = self.dllm_token_eviction_state
        return state is not None and state.stage == STAGE_SCORE

    def reset_token_eviction_state(self: Req) -> None:
        """Restart a retracted request from its first chunk-scoring forward."""
        state = self.dllm_token_eviction_state
        if state is None:
            return
        if state.full_input_ids is not None:
            self.origin_input_ids = state.full_input_ids
        state.restart()
        self.dllm_algo_state = {"prompt_len": len(self.origin_input_ids), "step": 0}
        self.dllm_incomplete_ids = array("q")
        self.dllm_kv_indices = None

    def reset_dllm_block_state(self: Req) -> None:
        """Restart a retracted multi-block request with a full pass.

        Committed blocks stay in ``output_ids``; the remaining canvas is
        prefilled again together with the prompt.
        """
        self.dllm_block_position_shift = 0
        state = self.dllm_algo_state
        if not isinstance(state, dict) or "more_blocks" not in state:
            return
        for key in (
            "more_blocks",
            "block_stage",
            "block_first_token",
            "last_block_stage",
            "canvas_len",
            "dual_cache_ready",
            "is_prefill",
            "last_round_was_dual_cache",
        ):
            state.pop(key, None)
        self.dllm_incomplete_ids = array("q")
        self.dllm_kv_indices = None

    def dllm_canvas_handoff_len(self: Req) -> int:
        """Canvas tokens whose KV a prefill server hands over with the prompt.

        Block rounds read the later blocks' mask KV of the first pass. A
        one-block canvas is recomputed whole in every round, so it stays.
        """
        config = self.dllm_config
        if (
            config is None
            or not config.needs_full_prefill
            or self.dllm_partial_draft_state is not None
            or self.dllm_parallelcomp_state is not None
        ):
            return 0
        canvas_len = self.sampling_params.max_new_tokens
        return canvas_len if canvas_len > config.block_size else 0

    def dllm_handoff_len(self: Req) -> int:
        """Tokens whose KV a prefill server hands to the decode server."""
        state = self.dllm_token_eviction_state
        prompt_len = (
            len(self.origin_input_ids) if state is None else state.compact_prompt_len
        )
        return prompt_len + self.dllm_canvas_handoff_len()

    def adopt_dllm_handoff_layout(self: Req) -> None:
        """On a decode server, take the layout the prefill server sends.

        Per-head eviction ran on the prefill server, so this side only needs
        the compacted prompt length and its RoPE position mapping.
        """
        state = self.dllm_token_eviction_state
        if state is not None and not state.compacted:
            state.skip_scoring()
            self.compact_token_eviction_input_ids()
            self.dllm_algo_state["prompt_len"] = len(self.origin_input_ids)
            self.dllm_phase = DllmReqPhase.INCOMING_DECODE
        canvas_len = self.dllm_canvas_handoff_len()
        if canvas_len and not self.dllm_handoff_canvas_len:
            # The decode server sizes its preallocation and the transfer by
            # the input length; resume_dllm_after_prompt_transfer strips these.
            self.origin_input_ids = self.origin_input_ids + array(
                "q", [self.dllm_config.mask_id] * canvas_len
            )
            self.dllm_handoff_canvas_len = canvas_len

    def compact_token_eviction_input_ids(self: Req) -> None:
        """Shrink the prompt ids to the per-head compacted KV layout."""
        state = self.dllm_token_eviction_state
        full_ids = self.origin_input_ids
        self.origin_input_ids = compact_input_ids(state, full_ids)
        state.full_input_ids = full_ids

    def _parse_parallelcomp_state(self: Req) -> Optional[dict[str, Any]]:
        if self.dllm_config is None or not self.dllm_config.needs_full_prefill:
            return None
        custom_params = getattr(self.sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        config = custom_params.get("dllm_parallelcomp")
        if config is None:
            return None
        if not isinstance(config, dict):
            raise ValueError("dllm_parallelcomp must be an object")
        if not self.dllm_config.dual_cache:
            raise ValueError("dllm_parallelcomp requires Dream dual_cache")
        if not self.dllm_config.first_done_first_out_mode:
            raise ValueError("dllm_parallelcomp requires --dllm-fdfo")

        prefix_len = config.get("prefix_len")
        query_len = config.get("query_len")
        chunk_lens = config.get("chunk_lens")
        chunk_batch_size = config.get("chunk_batch_size", 1)
        if (
            not isinstance(prefix_len, int)
            or isinstance(prefix_len, bool)
            or prefix_len < 0
            or not isinstance(query_len, int)
            or isinstance(query_len, bool)
            or query_len < 0
            or not isinstance(chunk_lens, list)
            or not chunk_lens
            or not isinstance(chunk_batch_size, int)
            or isinstance(chunk_batch_size, bool)
            or chunk_batch_size <= 0
            or any(
                not isinstance(length, int) or isinstance(length, bool) or length <= 0
                for length in chunk_lens
            )
        ):
            raise ValueError(
                "dllm_parallelcomp requires non-negative prefix_len/query_len "
                "and a non-empty list of positive chunk_lens plus a positive "
                "chunk_batch_size"
            )
        if prefix_len + sum(chunk_lens) + query_len != len(self.origin_input_ids):
            raise ValueError(
                "dllm_parallelcomp boundaries do not cover the input: "
                f"prefix={prefix_len}, chunks={sum(chunk_lens)}, "
                f"query={query_len}, input={len(self.origin_input_ids)}"
            )

        chunk_position_starts = config.get(
            "chunk_position_starts", [prefix_len] * len(chunk_lens)
        )
        chunk_query_position_starts = config.get(
            "chunk_query_position_starts",
            [
                start + length
                for start, length in zip(chunk_position_starts, chunk_lens)
            ],
        )
        query_position_start = config.get(
            "query_position_start", prefix_len + sum(chunk_lens)
        )
        if (
            not isinstance(chunk_position_starts, list)
            or len(chunk_position_starts) != len(chunk_lens)
            or not isinstance(chunk_query_position_starts, list)
            or len(chunk_query_position_starts) != len(chunk_lens)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in chunk_position_starts + chunk_query_position_starts
            )
            or not isinstance(query_position_start, int)
            or isinstance(query_position_start, bool)
            or query_position_start < 0
        ):
            raise ValueError(
                "dllm_parallelcomp position starts must be non-negative integers"
            )

        token_eviction = parse_token_eviction(
            config.get("token_eviction"),
            query_ids=self.origin_input_ids[len(self.origin_input_ids) - query_len :],
        )

        return {
            "stage": "prefix" if prefix_len else "chunk",
            "prefix_len": prefix_len,
            "query_len": query_len,
            "token_eviction": token_eviction,
            "chunk_lens": list(chunk_lens),
            "chunk_batch_size": chunk_batch_size,
            "chunk_offsets": [
                prefix_len + sum(chunk_lens[:index]) for index in range(len(chunk_lens))
            ],
            "chunk_position_starts": list(chunk_position_starts),
            "chunk_query_position_starts": list(chunk_query_position_starts),
            "query_position_start": query_position_start,
            "chunk_cursor": 0,
            "common_prefix_indices": None,
            "chunk_kv_indices": [],
        }

    def parallelcomp_chunk_batch_range(self: Req) -> range:
        state = self.dllm_parallelcomp_state
        if state is None or state["stage"] != "chunk":
            return range(0)
        start = state["chunk_cursor"]
        end = min(start + state["chunk_batch_size"], len(state["chunk_lens"]))
        return range(start, end)

    def parallelcomp_item_lens(self: Req) -> Optional[list[int]]:
        state = self.dllm_parallelcomp_state
        if state is None or state["stage"] != "chunk":
            return None
        query_len = parallelcomp_chunk_query_len(state)
        return [
            state["chunk_lens"][cursor] + query_len
            for cursor in self.parallelcomp_chunk_batch_range()
        ]

    def has_parallelcomp_prefill_cache(self: Req) -> bool:
        return self.dllm_parallelcomp_state is not None

    def take_parallelcomp_retained_kv_indices(self: Req) -> list:
        """Return retained chunk pages for page-table-aware abort cleanup."""
        state = self.dllm_parallelcomp_state
        if state is None or state["stage"] != "chunk":
            return []
        chunk_kv_indices = state["chunk_kv_indices"]
        state["chunk_kv_indices"] = []
        return chunk_kv_indices

    def reset_parallelcomp_prefill_state(self: Req) -> None:
        state = self.dllm_parallelcomp_state
        if state is None:
            return
        state["stage"] = "prefix" if state["prefix_len"] else "chunk"
        state["chunk_cursor"] = 0
        state["common_prefix_indices"] = None
        state["chunk_kv_indices"] = []
        state.pop("assembled_prefix_indices", None)
        eviction = state["token_eviction"]
        if eviction is not None:
            eviction.kept_positions = []
            if eviction.full_input_ids is not None:
                self.origin_input_ids = eviction.full_input_ids
                eviction.full_input_ids = None

    def compact_parallelcomp_input_ids(self: Req) -> None:
        """Drop evicted chunk tokens from the input ids before decoding."""
        state = self.dllm_parallelcomp_state
        eviction = state["token_eviction"]
        if eviction is None:
            return
        full_ids = self.origin_input_ids
        compact_ids = array("q", full_ids[: state["prefix_len"]])
        for chunk_offset, kept in zip(state["chunk_offsets"], eviction.kept_positions):
            compact_ids.extend(full_ids[chunk_offset + position] for position in kept)
        compact_ids.extend(full_ids[len(full_ids) - state["query_len"] :])
        eviction.full_input_ids = full_ids
        self.origin_input_ids = compact_ids

    def parallelcomp_position_values(self: Req) -> Optional[list[int]]:
        state = self.dllm_parallelcomp_state
        if state is None:
            return None
        stage = state["stage"]
        if stage == "prefix":
            values = list(range(state["prefix_len"]))
        elif stage == "chunk":
            values = list(range(state["prefix_len"]))
            for cursor in self.parallelcomp_chunk_batch_range():
                chunk_start = state["chunk_position_starts"][cursor]
                chunk_len = state["chunk_lens"][cursor]
                query_start = state["chunk_query_position_starts"][cursor]
                values.extend(range(chunk_start, chunk_start + chunk_len))
                values.extend(
                    range(
                        query_start, query_start + parallelcomp_chunk_query_len(state)
                    )
                )
        else:
            values = list(range(state["prefix_len"]))
            eviction = state["token_eviction"]
            for chunk_order, (chunk_start, chunk_len) in enumerate(
                zip(state["chunk_position_starts"], state["chunk_lens"])
            ):
                if eviction is None:
                    values.extend(range(chunk_start, chunk_start + chunk_len))
                else:
                    # Kept tokens retain the RoPE position their KV was built at.
                    values.extend(
                        chunk_start + position
                        for position in eviction.kept_positions[chunk_order]
                    )
            query_start = state["query_position_start"]
            values.extend(range(query_start, query_start + state["query_len"]))
            generation_start = query_start + state["query_len"]
            values.extend(
                range(
                    generation_start,
                    generation_start + self.sampling_params.max_new_tokens,
                )
            )
        return values[self.extend_range.start : self.extend_range.end]

    def determine_dllm_phase(self: Req):
        if (
            self.dllm_parallelcomp_state is not None
            and self.dllm_parallelcomp_state["stage"] != "decode"
        ) or self.is_scoring_token_eviction():
            if self.dllm_phase not in (
                DllmReqPhase.INCOMING_PREFILL,
                DllmReqPhase.STAGING_PREFILL,
            ):
                self.dllm_phase = DllmReqPhase.STAGING_PREFILL
            return

        if self.dllm_config.needs_full_prefill:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        if self.dllm_incomplete_ids:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE
            return

        prefix_length = len(self.prefix_indices)
        min_required_length = prefix_length + self.dllm_config.block_size

        if len(self.full_untruncated_fill_ids) < min_required_length:
            # still incoming stage
            return

        input_block = self.full_untruncated_fill_ids[prefix_length:min_required_length]
        is_prefill_phase = self.dllm_config.mask_id not in input_block

        if is_prefill_phase:
            self.dllm_phase = DllmReqPhase.STAGING_PREFILL
        else:
            self.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _init_fill_ids_for_dllm(self: Req):
        if self.dllm_config.needs_full_prefill:
            partial_draft = self.dllm_algo_state
            if isinstance(partial_draft, dict) and partial_draft.get(
                "partial_draft", False
            ):
                stage = partial_draft["partial_draft_stage"]
                if stage == "prompt":
                    if not self.origin_input_ids:
                        raise ValueError(
                            "dllm_partial_draft requires a non-empty prompt"
                        )
                    self.prefix_indices = self.prefix_indices[:0]
                    self.full_untruncated_fill_ids = array("q", self.origin_input_ids)
                    partial_draft["prompt_len"] = len(self.origin_input_ids)
                    self.dllm_block_offset = 0
                    self.dllm_canvas_output_len = 0
                    self.dllm_initialized = True
                    return
                if stage == "suffix_init":
                    prompt_len = partial_draft["prompt_len"]
                    if len(self.prefix_indices) != prompt_len:
                        raise RuntimeError(
                            "Partial draft suffix initialization lost its prompt KV: "
                            f"prefix={len(self.prefix_indices)}, prompt={prompt_len}"
                        )
                    self.full_untruncated_fill_ids = self.origin_input_ids + array(
                        "q", [self.dllm_config.mask_id] * partial_draft["canvas_len"]
                    )
                    self.dllm_block_offset = 0
                    self.dllm_canvas_output_len = 0
                    self.dllm_initialized = True
                    return

            if self.is_scoring_token_eviction():
                self.prefix_indices = self.prefix_indices[:0]
                self.full_untruncated_fill_ids = score_stage_input_ids(
                    self.dllm_token_eviction_state, self.origin_input_ids
                )
                # Mask-free, so the algorithm runs one forward and no denoising.
                self.dllm_algo_state["prompt_len"] = len(self.full_untruncated_fill_ids)
                self.dllm_initialized = True
                return

            parallelcomp = self.dllm_parallelcomp_state
            if parallelcomp is not None and parallelcomp["stage"] != "decode":
                if parallelcomp["stage"] == "prefix":
                    self.prefix_indices = self.prefix_indices[:0]
                    self.full_untruncated_fill_ids = array(
                        "q", self.origin_input_ids[: parallelcomp["prefix_len"]]
                    )
                else:
                    eviction = parallelcomp["token_eviction"]
                    chunk_query_ids = (
                        self.origin_input_ids[
                            len(self.origin_input_ids) - parallelcomp["query_len"] :
                        ]
                        if eviction is None
                        else eviction.score_query_ids
                    )
                    common_prefix_indices = parallelcomp["common_prefix_indices"]
                    self.prefix_indices = (
                        common_prefix_indices
                        if common_prefix_indices is not None
                        else self.prefix_indices[:0]
                    )
                    batch_ids = array(
                        "q", self.origin_input_ids[: parallelcomp["prefix_len"]]
                    )
                    for cursor in self.parallelcomp_chunk_batch_range():
                        chunk_start = parallelcomp["chunk_offsets"][cursor]
                        chunk_end = chunk_start + parallelcomp["chunk_lens"][cursor]
                        batch_ids.extend(self.origin_input_ids[chunk_start:chunk_end])
                        batch_ids.extend(chunk_query_ids)
                    self.full_untruncated_fill_ids = batch_ids
                # A mask-free intermediate stage makes DllmAlgorithm perform
                # exactly one model forward without entering denoising.
                self.dllm_algo_state["prompt_len"] = len(self.full_untruncated_fill_ids)
                self.dllm_initialized = True
                return

            if (
                self.dllm_algo_state is not None
                and self.dllm_algo_state.get("prompt_len", 0) == 0
                and len(self.origin_input_ids) > 0
            ):
                # Req initializes its dLLM fields before tokenization has
                # necessarily populated origin_input_ids. Capture the real
                # prompt boundary when the Dream canvas is first materialized.
                self.dllm_algo_state["prompt_len"] = len(self.origin_input_ids)

            if (
                self.dllm_initialized
                and len(self.output_ids) == self.dllm_canvas_output_len
            ):
                return

            remaining = max(
                self.sampling_params.max_new_tokens - len(self.output_ids), 0
            )
            if self.dllm_algo_state is not None:
                # The model returns logits for the last canvas_len rows only;
                # the default is one block.
                if remaining == self.dllm_config.block_size:
                    self.dllm_algo_state.pop("canvas_len", None)
                else:
                    self.dllm_algo_state["canvas_len"] = remaining
            self.dllm_block_offset = 0
            self.full_untruncated_fill_ids = (
                self.origin_input_ids
                + self.output_ids
                + array("q", [self.dllm_config.mask_id] * remaining)
            )
            self.dllm_canvas_output_len = len(self.output_ids)
            self.dllm_initialized = True
            return

        if self.dllm_incomplete_ids:
            prefix_len = len(self.prefix_indices)
            assert len(self.dllm_incomplete_ids) == self.dllm_config.block_size
            self.full_untruncated_fill_ids = (
                self.full_untruncated_fill_ids[:prefix_len] + self.dllm_incomplete_ids
            )
            # extend_range is (re)computed by the staging adder
            # (add_dllm_staging_req) before this req is scheduled, mirroring the
            # non-incomplete path which also defers it to the adder.
            return

        self.dllm_block_offset = (
            0
            if not self.dllm_initialized
            else self.dllm_block_offset + self.dllm_config.block_size
        )
        self.full_untruncated_fill_ids = (
            self.origin_input_ids
            + self.output_ids
            + array("q", [self.dllm_config.mask_id] * self.dllm_config.block_size)
        )
        self.dllm_initialized = True

    def _update_block_offset_for_dllm(self):
        prefix_len = len(self.prefix_indices)
        assert (
            prefix_len % self.dllm_config.block_size == 0
        ), f"Unexpected prefix len: {prefix_len}"
        if prefix_len > self.dllm_block_offset:
            self.dllm_block_offset = prefix_len
