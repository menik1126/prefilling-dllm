from __future__ import annotations

import enum
from array import array
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class DllmReqPhase(str, enum.Enum):
    STAGING_PREFILL = "staging_prefill"
    STAGING_DECODE = "staging_decode"
    INCOMING_PREFILL = "incoming_prefill"
    INCOMING_DECODE = "incoming_decode"


class ReqDllmMixin:
    def init_diffusion_llm(self: Req, dllm_config: DllmConfig):
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
        self.dllm_config = dllm_config
        self.dllm_parallelcomp_state = self._parse_parallelcomp_state()

        if self.dllm_config is not None:
            if self.dllm_parallelcomp_state is not None:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL
            elif self.dllm_config.needs_full_prefill:
                # Dream denoises a masked generation canvas, so it is a decode
                # request even though each round uses a full-attention prefill.
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            elif len(self.origin_input_ids) < self.dllm_config.block_size:
                self.dllm_phase = DllmReqPhase.INCOMING_DECODE
            else:
                self.dllm_phase = DllmReqPhase.INCOMING_PREFILL

    def is_dllm(self: Req) -> bool:
        return self.dllm_config is not None

    def is_dllm_prefill(self: Req) -> bool:
        return self.dllm_phase in [
            DllmReqPhase.STAGING_PREFILL,
            DllmReqPhase.INCOMING_PREFILL,
        ]

    def _parse_parallelcomp_state(self: Req) -> Optional[dict[str, Any]]:
        if self.dllm_config is None or not self.dllm_config.needs_full_prefill:
            return None
        custom_params = getattr(self.sampling_params, "custom_params", None)
        if not isinstance(custom_params, dict):
            return None
        config = custom_params.get("dllm_parallelcomp")
        if not isinstance(config, dict):
            return None
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

        return {
            "stage": "prefix" if prefix_len else "chunk",
            "prefix_len": prefix_len,
            "query_len": query_len,
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
        query_len = state["query_len"]
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
                values.extend(range(query_start, query_start + state["query_len"]))
        else:
            values = list(range(state["prefix_len"]))
            for chunk_start, chunk_len in zip(
                state["chunk_position_starts"], state["chunk_lens"]
            ):
                values.extend(range(chunk_start, chunk_start + chunk_len))
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
        ):
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
            parallelcomp = self.dllm_parallelcomp_state
            if parallelcomp is not None and parallelcomp["stage"] != "decode":
                if parallelcomp["stage"] == "prefix":
                    self.prefix_indices = self.prefix_indices[:0]
                    self.full_untruncated_fill_ids = array(
                        "q", self.origin_input_ids[: parallelcomp["prefix_len"]]
                    )
                else:
                    query_start = len(self.origin_input_ids) - parallelcomp["query_len"]
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
                        batch_ids.extend(self.origin_input_ids[query_start:])
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
