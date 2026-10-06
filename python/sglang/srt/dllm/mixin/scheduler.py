from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.head_token_eviction import compact_prompt_kv_per_head
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.token_eviction import (
    parallelcomp_chunk_query_len,
    split_kept_kv_indices,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


def _round_cache_loc(batch: ScheduleBatch, idx: int) -> torch.Tensor:
    """KV slots the forward wrote for batch row ``idx``."""
    cache_offset = sum(batch.extend_lens[:idx])
    return batch.out_cache_loc[cache_offset : cache_offset + batch.extend_lens[idx]]


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        score_batch = self._get_new_score_batch(running_batch)
        if score_batch is not None:
            return score_batch

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder, running_batch=running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        has_parallelcomp = any(
            req.dllm_parallelcomp_state is not None for req in batch.reqs
        )
        if has_parallelcomp and self.token_to_kv_pool_allocator.page_size != 1:
            raise RuntimeError(
                "dllm_parallelcomp currently requires KV cache page size 1"
            )
        has_token_eviction = any(
            req.dllm_token_eviction_state is not None for req in batch.reqs
        )
        if has_token_eviction and self.token_to_kv_pool_allocator.page_size != 1:
            raise RuntimeError("dllm_token_eviction requires KV cache page size 1")

        fdfo_mode = self.dllm_config.first_done_first_out_mode
        if fdfo_mode:
            fdfo_signal = (
                result.dllm_done_per_req_cpu
                if self.dllm_config.needs_full_prefill
                else result.accept_length_per_req_cpu
            )
            signal_name = (
                "done states"
                if self.dllm_config.needs_full_prefill
                else "accept lengths"
            )
            assert (
                fdfo_signal is not None
            ), f"FDFO dLLM result is missing {signal_name}."

        # FDFO also commits unresolved blocks so their KV can be reused.
        if fdfo_mode or result.next_token_ids or has_parallelcomp:
            algo_states = result.dllm_algo_state

            self.token_to_kv_pool_allocator.free_group_begin()
            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]
                if req.is_scoring_token_eviction():
                    self._finish_token_eviction_score_round(
                        req,
                        round_cache_loc=_round_cache_loc(batch, idx),
                        keep=result.dllm_head_keep_per_req[idx],
                    )
                    continue
                parallelcomp = req.dllm_parallelcomp_state
                if parallelcomp is not None and parallelcomp["stage"] != "decode":
                    cache_offset = sum(batch.extend_lens[:idx])
                    round_cache_loc = batch.out_cache_loc[
                        cache_offset : cache_offset + batch.extend_lens[idx]
                    ]
                    stage = parallelcomp["stage"]
                    if stage == "prefix":
                        if len(round_cache_loc) != parallelcomp["prefix_len"]:
                            raise RuntimeError(
                                "ParallelComp prefix KV span mismatch: "
                                f"kv={len(round_cache_loc)}, "
                                f"tokens={parallelcomp['prefix_len']}"
                            )
                        parallelcomp["common_prefix_indices"] = round_cache_loc.clone()
                        req.prefix_indices = round_cache_loc.clone()
                        parallelcomp["stage"] = "chunk"
                        req.dllm_phase = DllmReqPhase.STAGING_PREFILL
                    else:
                        query_len = parallelcomp["query_len"]
                        chunk_query_len = parallelcomp_chunk_query_len(parallelcomp)
                        eviction = parallelcomp["token_eviction"]
                        chunk_batch = list(req.parallelcomp_chunk_batch_range())
                        expected_kv_len = sum(
                            parallelcomp["chunk_lens"][cursor] + chunk_query_len
                            for cursor in chunk_batch
                        )
                        if len(round_cache_loc) != expected_kv_len:
                            raise RuntimeError(
                                "ParallelComp batched chunk KV span mismatch: "
                                f"kv={len(round_cache_loc)}, "
                                f"expected={expected_kv_len}, "
                                f"chunks={chunk_batch}, query={chunk_query_len}"
                            )
                        cache_cursor = 0
                        for chunk_order, cursor in enumerate(chunk_batch):
                            chunk_len = parallelcomp["chunk_lens"][cursor]
                            chunk_end = cache_cursor + chunk_len
                            query_end = chunk_end + chunk_query_len
                            chunk_kv_indices = round_cache_loc[cache_cursor:chunk_end]
                            if eviction is not None:
                                kept = result.dllm_token_keep_per_req[idx][chunk_order]
                                chunk_kv_indices, evicted = split_kept_kv_indices(
                                    chunk_kv_indices, kept
                                )
                                eviction.kept_positions.append(kept)
                                if len(evicted):
                                    self.token_to_kv_pool_allocator.free(evicted)
                            parallelcomp["chunk_kv_indices"].append(
                                chunk_kv_indices.clone()
                            )
                            if chunk_query_len:
                                self.token_to_kv_pool_allocator.free(
                                    round_cache_loc[chunk_end:query_end]
                                )
                            cache_cursor = query_end
                        parallelcomp["chunk_cursor"] = chunk_batch[-1] + 1
                        if parallelcomp["chunk_cursor"] == len(
                            parallelcomp["chunk_lens"]
                        ):
                            parts = []
                            common_prefix_indices = parallelcomp[
                                "common_prefix_indices"
                            ]
                            if common_prefix_indices is not None:
                                parts.append(common_prefix_indices)
                            parts.extend(parallelcomp["chunk_kv_indices"])
                            req.prefix_indices = (
                                torch.cat(parts)
                                if parts
                                else round_cache_loc[:0].clone()
                            )
                            # Make every retained chunk immediately visible to
                            # ordinary request cleanup. Before this write, only
                            # the most recent chunk is present in the request
                            # row and earlier chunks are intentionally detached.
                            req_row = self.req_to_token_pool.req_to_token[
                                req.req_pool_idx
                            ]
                            req_row[: len(req.prefix_indices)] = req.prefix_indices
                            req.kv.kv_allocated_len = len(req.prefix_indices)
                            req.kv_committed_len = len(req.prefix_indices)
                            parallelcomp["assembled_prefix_indices"] = (
                                req.prefix_indices.clone()
                            )
                            parallelcomp["chunk_kv_indices"] = []
                            parallelcomp["stage"] = "decode"
                            req.compact_parallelcomp_input_ids()
                            req.dllm_algo_state = {
                                # The first decode forward contains only query
                                # plus masks; cached chunks are in the page table.
                                "prompt_len": query_len,
                                "full_prompt_len": len(req.origin_input_ids),
                                "step": 0,
                            }
                            req.dllm_initialized = False
                            req.dllm_canvas_output_len = -1
                            req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        else:
                            common_prefix_indices = parallelcomp[
                                "common_prefix_indices"
                            ]
                            req.prefix_indices = (
                                common_prefix_indices
                                if common_prefix_indices is not None
                                else round_cache_loc[:0].clone()
                            )
                            req.dllm_phase = DllmReqPhase.STAGING_PREFILL
                    continue

                if self.dllm_config.needs_full_prefill and fdfo_mode:
                    round_tokens = result.next_token_ids[idx]
                    if hasattr(round_tokens, "tolist"):
                        round_tokens = round_tokens.tolist()
                    round_tokens = array("q", round_tokens)

                    algo_state = req.dllm_algo_state or {}
                    last_partial_draft_stage = algo_state.get(
                        "last_partial_draft_stage"
                    )
                    if last_partial_draft_stage == "prompt":
                        if result.dllm_done_per_req_cpu[idx]:
                            raise RuntimeError(
                                "Partial draft prompt stage completed prematurely"
                            )
                        next_state = (
                            algo_states[idx] if algo_states is not None else None
                        )
                        if (
                            not isinstance(next_state, dict)
                            or next_state.get("partial_draft_stage") != "suffix_init"
                            or not isinstance(
                                next_state.get("partial_draft_first_token"), int
                            )
                        ):
                            raise RuntimeError(
                                "Partial draft prompt stage returned invalid state"
                            )
                        cache_offset = sum(batch.extend_lens[:idx])
                        round_cache_loc = batch.out_cache_loc[
                            cache_offset : cache_offset + batch.extend_lens[idx]
                        ]
                        prompt_len = next_state["prompt_len"]
                        if (
                            len(round_tokens) != prompt_len
                            or len(round_cache_loc) != prompt_len
                        ):
                            raise RuntimeError(
                                "Partial draft prompt token/KV span mismatch: "
                                f"tokens={len(round_tokens)}, "
                                f"kv={len(round_cache_loc)}, prompt={prompt_len}"
                            )
                        req.dllm_algo_state = next_state
                        req.prefix_indices = round_cache_loc.clone()
                        req.dllm_incomplete_ids = array("q")
                        req.dllm_kv_indices = None
                        req.dllm_initialized = False
                        req.dllm_canvas_output_len = -1
                        req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        if self.disaggregation_mode == DisaggregationMode.PREFILL:
                            self._hand_off_dllm_prompt_kv(
                                req,
                                first_token=next_state["partial_draft_first_token"],
                            )
                        continue

                    if last_partial_draft_stage == "suffix_init":
                        if result.dllm_done_per_req_cpu[idx]:
                            raise RuntimeError(
                                "Partial draft suffix stage completed prematurely"
                            )
                        next_state = (
                            algo_states[idx] if algo_states is not None else None
                        )
                        if (
                            not isinstance(next_state, dict)
                            or next_state.get("partial_draft_stage") != "partial_round"
                            or not next_state.get("dual_cache_ready", False)
                        ):
                            raise RuntimeError(
                                "Partial draft suffix stage returned invalid state"
                            )
                        canvas_len = next_state["canvas_len"]
                        prompt_len = next_state["prompt_len"]
                        cache_offset = sum(batch.extend_lens[:idx])
                        round_cache_loc = batch.out_cache_loc[
                            cache_offset : cache_offset + batch.extend_lens[idx]
                        ]
                        if (
                            len(round_tokens) != canvas_len
                            or len(round_cache_loc) != canvas_len
                            or len(req.prefix_indices) != prompt_len
                        ):
                            raise RuntimeError(
                                "Partial draft suffix token/KV span mismatch: "
                                f"tokens={len(round_tokens)}, "
                                f"kv={len(round_cache_loc)}, "
                                f"prefix={len(req.prefix_indices)}, "
                                f"canvas={canvas_len}, prompt={prompt_len}"
                            )
                        req.dllm_algo_state = next_state
                        req.full_untruncated_fill_ids[
                            prompt_len : prompt_len + canvas_len
                        ] = round_tokens
                        req.dllm_incomplete_ids = array("q", round_tokens)
                        req.dllm_kv_indices = round_cache_loc.clone()
                        req.dllm_phase = DllmReqPhase.STAGING_DECODE
                        continue

                    # step() mutates the carried state in place, so the first
                    # full-canvas pass already appears dual-cache-ready here.
                    # The returned token span is the authoritative round type.
                    last_round_was_dual_cache = algo_state.get(
                        "last_round_was_dual_cache"
                    )
                    expected_canvas_len = algo_state.get(
                        "canvas_len", self.dllm_config.block_size
                    )
                    dual_cache_round = bool(
                        self.dllm_config.dual_cache
                        and (
                            last_round_was_dual_cache
                            if last_round_was_dual_cache is not None
                            else len(round_tokens) == expected_canvas_len
                        )
                    )
                    if parallelcomp is not None:
                        full_prompt_len = len(req.origin_input_ids)
                        query_len = parallelcomp["query_len"]
                        if dual_cache_round:
                            req.full_untruncated_fill_ids[
                                full_prompt_len : full_prompt_len + len(round_tokens)
                            ] = round_tokens
                        else:
                            if (
                                len(round_tokens)
                                != query_len + self.dllm_config.block_size
                            ):
                                raise RuntimeError(
                                    "ParallelComp first decode span mismatch: "
                                    f"tokens={len(round_tokens)}, "
                                    f"query={query_len}, "
                                    f"block={self.dllm_config.block_size}"
                                )
                            req.full_untruncated_fill_ids[full_prompt_len:] = (
                                round_tokens[query_len:]
                            )
                    elif dual_cache_round:
                        prompt_len = req.dllm_algo_state["prompt_len"]
                        req.full_untruncated_fill_ids[
                            prompt_len : prompt_len + len(round_tokens)
                        ] = round_tokens
                    else:
                        req.full_untruncated_fill_ids = round_tokens
                    canvas = req.full_untruncated_fill_ids

                    # A prefill server never answers: even a canvas completed
                    # by the first pass goes to the decode server.
                    hands_off = (
                        self.disaggregation_mode == DisaggregationMode.PREFILL
                        and not dual_cache_round
                    )
                    if result.dllm_done_per_req_cpu[idx] and not hands_off:
                        partial_draft = bool(
                            req.dllm_algo_state
                            and req.dllm_algo_state.get("partial_draft", False)
                        )
                        prompt_len = (
                            len(req.origin_input_ids)
                            if parallelcomp is not None
                            else req.dllm_algo_state["prompt_len"]
                        )
                        req.output_ids = array("q", canvas[prompt_len:])
                        if partial_draft:
                            confirmed_mask = req.dllm_algo_state.get(
                                "partial_draft_confirmed_mask"
                            )
                            if (
                                not isinstance(confirmed_mask, list)
                                or len(confirmed_mask) != len(req.output_ids)
                                or any(
                                    not isinstance(value, bool)
                                    for value in confirmed_mask
                                )
                            ):
                                raise RuntimeError(
                                    "Completed partial draft is missing a valid "
                                    "canvas-aligned confirmed mask"
                                )
                            if req.customized_info is None:
                                req.customized_info = {}
                            req.customized_info["dllm_confirmed_mask"] = list(
                                confirmed_mask
                            )
                        req.dllm_incomplete_ids = array("q")
                        req.dllm_kv_indices = None
                        req.dllm_algo_state = None
                        self.metrics_reporter.num_generated_tokens += len(
                            req.output_ids
                        )
                        if req.output_ids:
                            req.update_finish_state(
                                new_accepted_len=len(req.output_ids)
                            )
                        else:
                            req.finished_reason = FINISH_LENGTH(length=0)
                            req.finished_len = 0
                        if req.finished():
                            release_kv_cache(req, self.tree_cache, is_insert=False)
                            req.time_stats.set_completion_time()
                    else:
                        req.dllm_algo_state = algo_states[idx]
                        if self.dllm_config.dual_cache:
                            prompt_len = (
                                len(req.origin_input_ids)
                                if parallelcomp is not None
                                else req.dllm_algo_state["prompt_len"]
                            )
                            req.dllm_incomplete_ids = array("q", canvas[prompt_len:])
                            cache_offset = sum(batch.extend_lens[:idx])
                            round_cache_loc = batch.out_cache_loc[
                                cache_offset : cache_offset + batch.extend_lens[idx]
                            ]
                            if not dual_cache_round:
                                # The first full bidirectional pass populated
                                # the complete request row. Freeze its prompt
                                # portion; alloc_for_extend will reuse and
                                # overwrite the generation slots in place.
                                if len(round_cache_loc) != len(round_tokens):
                                    raise RuntimeError(
                                        "Dream KV span does not match returned canvas: "
                                        f"kv={len(round_cache_loc)}, "
                                        f"tokens={len(round_tokens)}"
                                    )
                                if parallelcomp is not None:
                                    query_len = parallelcomp["query_len"]
                                    assembled = parallelcomp["assembled_prefix_indices"]
                                    req.prefix_indices = torch.cat(
                                        [assembled, round_cache_loc[:query_len]]
                                    )
                                    req.dllm_kv_indices = round_cache_loc[
                                        query_len:
                                    ].clone()
                                else:
                                    req.prefix_indices = round_cache_loc[
                                        :prompt_len
                                    ].clone()
                                    req.dllm_kv_indices = round_cache_loc[
                                        prompt_len:
                                    ].clone()
                                    if req.dllm_token_eviction_state is not None:
                                        self._evict_prompt_tokens_per_head(req)
                                    if hands_off:
                                        self._hand_off_dllm_prompt_kv(
                                            req,
                                            first_token=req.dllm_incomplete_ids[0],
                                        )
                        else:
                            release_kv_cache(req, self.tree_cache, is_insert=False)
                    continue

                if not fdfo_mode:
                    next_token_ids = result.next_token_ids[idx].tolist()
                    new_tokens = len(next_token_ids)
                    if new_tokens == 0:
                        continue

                    req.full_untruncated_fill_ids[
                        req.extend_range.end - new_tokens : req.extend_range.end
                    ] = array("q", next_token_ids)
                    self.metrics_reporter.num_generated_tokens += new_tokens

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)

                    if req.finished():
                        release_kv_cache(
                            req,
                            self.tree_cache,
                            is_insert=not self.dllm_config.needs_full_prefill,
                        )
                        req.time_stats.set_completion_time()
                    continue

                block_size = self.dllm_config.block_size
                next_token_ids = result.next_token_ids[idx]
                assert len(next_token_ids) == block_size

                if result.accept_length_per_req_cpu[idx] == 0:
                    # Unresolved: keep partial state and KV for the next FDFO round.
                    req.dllm_incomplete_ids = array("q", next_token_ids)
                    req.dllm_algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm_incomplete_ids = array("q")
                req.dllm_algo_state = None

                # Mirror the resolved block into the committed fill ids so the
                # prefix cache keys on the real tokens, not the mask block, next
                # round. Index relative to extend_range.end (the truncated/
                # committed length), which can be shorter than
                # full_untruncated_fill_ids when the staging adder truncates the
                # block to the KV budget.
                req.full_untruncated_fill_ids[
                    req.extend_range.end - block_size : req.extend_range.end
                ] = array("q", next_token_ids)

                len_input = len(req.origin_input_ids)
                len_fill = req.extend_range.end
                if len_fill <= len_input:
                    continue

                if len_fill - len(next_token_ids) < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                self.metrics_reporter.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))

                if req.finished():
                    release_kv_cache(req, self.tree_cache)
                    req.time_stats.set_completion_time()

            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _finish_token_eviction_score_round(
        self: Scheduler,
        req: Req,
        *,
        round_cache_loc: torch.Tensor,
        keep: torch.Tensor,
    ) -> None:
        """Store one chunk's keep table and drop the scoring forward's KV."""
        eviction = req.dllm_token_eviction_state
        eviction.keep_positions[eviction.chunk_cursor] = keep
        self.token_to_kv_pool_allocator.free(round_cache_loc)
        req.prefix_indices = round_cache_loc[:0].clone()
        req.kv.kv_allocated_len = 0
        req.kv_committed_len = 0
        eviction.advance()
        if req.is_scoring_token_eviction():
            req.dllm_phase = DllmReqPhase.STAGING_PREFILL
            return
        req.dllm_algo_state = {"prompt_len": len(req.origin_input_ids), "step": 0}
        req.dllm_initialized = False
        req.dllm_canvas_output_len = -1
        req.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _evict_prompt_tokens_per_head(self: Scheduler, req: Req) -> None:
        """Compact the freshly prefilled prompt KV to each head's kept tokens."""
        eviction = req.dllm_token_eviction_state
        if not any(keep is not None for keep in eviction.keep_positions):
            return
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        retained, evicted = compact_prompt_kv_per_head(
            kv_buffers=[
                kv_pool.get_kv_buffer(layer_id)
                for layer_id in range(
                    kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num
                )
            ],
            prompt_slots=req.prefix_indices,
            state=eviction,
        )
        self.token_to_kv_pool_allocator.free(evicted)

        canvas_slots = req.dllm_kv_indices
        prompt_len = len(retained)
        seq_len = prompt_len + len(canvas_slots)
        req_row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        req_row[:prompt_len] = retained
        req_row[prompt_len:seq_len] = canvas_slots
        req.prefix_indices = retained
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        canvas = req.dllm_incomplete_ids
        req.compact_token_eviction_input_ids()
        req.full_untruncated_fill_ids = req.origin_input_ids + canvas
        req.dllm_algo_state["prompt_len"] = prompt_len

    def _hand_off_dllm_prompt_kv(
        self: Scheduler, req: Req, *, first_token: int
    ) -> None:
        """Send a prefilled Dream prompt to the decode server and stop here.

        The handoff is the prompt KV plus the first canvas token; the decode
        server computes canvas KV itself in every round.
        """
        prompt_len = len(req.prefix_indices)
        if req.dllm_kv_indices is not None:
            self.token_to_kv_pool_allocator.free(req.dllm_kv_indices)
        req.kv.kv_allocated_len = prompt_len
        req.kv_committed_len = prompt_len
        req.output_ids = array("q", [first_token])
        req.dllm_incomplete_ids = array("q")
        req.dllm_kv_indices = None
        req.dllm_algo_state = None
        # send_kv_chunk transfers the request row up to extend_range.end.
        req.set_extend_range(0, prompt_len)
        self.dllm_manager.remove_req(req)

        req.time_stats.set_prefill_finished_time()
        self.disagg_prefill_inflight_queue.append(req)
        if not req.pending_bootstrap:
            self.send_kv_chunk(req, last_chunk=True)
        req.time_stats.set_prefill_transfer_queue_entry_time()

    def resume_dllm_after_prompt_transfer(self: Scheduler, req: Req) -> None:
        """Rebuild, on a decode server, the state the prefill server stopped in."""
        prompt_len = len(req.origin_input_ids)
        first_token = req.output_ids.pop()
        req_row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        req.prefix_indices = req_row[:prompt_len].to(dtype=torch.int64, copy=True)

        if req.dllm_partial_draft_state is not None:
            # A draft left its prompt stage: the next forward initializes the
            # masked suffix behind the cached prompt.
            state = req.dllm_algo_state
            state["prompt_len"] = prompt_len
            state["partial_draft_stage"] = "suffix_init"
            state["partial_draft_first_token"] = first_token
            req.dllm_incomplete_ids = array("q")
            req.dllm_kv_indices = None
            req.dllm_initialized = False
            req.dllm_canvas_output_len = -1
            req.init_next_round_input()
            return

        # A generation request left its first full pass: denoising continues
        # on a canvas holding only the first token.
        canvas_len = req.sampling_params.max_new_tokens
        seq_len = prompt_len + canvas_len
        canvas = array(
            "q", [first_token] + [self.dllm_config.mask_id] * (canvas_len - 1)
        )
        canvas_slots = alloc_token_slots(self.tree_cache, canvas_len)
        req_row[prompt_len:seq_len] = canvas_slots
        req.dllm_kv_indices = canvas_slots.clone()
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        req.dllm_incomplete_ids = canvas
        req.full_untruncated_fill_ids = req.origin_input_ids + canvas
        req.dllm_canvas_output_len = len(req.output_ids)
        req.dllm_initialized = True
        req.dllm_algo_state = {
            "prompt_len": prompt_len,
            "step": 0,
            "is_prefill": False,
            "dual_cache_ready": True,
        }
        req.set_extend_range(prompt_len, seq_len)
        req.dllm_phase = DllmReqPhase.STAGING_DECODE

    def _get_new_score_batch(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Batch waiting chunk-scoring requests as one ordinary prefill."""
        score_reqs = [req for req in self.waiting_queue if not req.is_dllm()]
        if not score_reqs:
            return None

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
        )
        # Generation requests keep their request slots between rounds, so a
        # scoring batch can only take the slots that are free right now.
        free_req_slots = self.req_to_token_pool.available_size()
        for req in score_reqs:
            if len(adder.can_run_list) >= free_req_slots:
                break
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=self.truncation_align_size,
            )
            if res != AddReqResult.CONTINUE:
                break
        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        scheduled = {id(req) for req in can_run_list}
        self.waiting_queue = [
            req for req in self.waiting_queue if id(req) not in scheduled
        ]
        set_time_batch(can_run_list, "set_forward_entry_time")
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        new_batch.prepare_for_extend()
        new_batch.decoding_reqs = None

        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )
        return new_batch

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_running_reqs = (
            self.max_running_requests
            if self.dllm_config.needs_full_prefill
            else self.dllm_config.max_running_requests
        )
        max_dllm_capacity = max_running_reqs - len(self.dllm_manager.waiting_queue)
        # Scoring requests stay behind for _get_new_score_batch.
        dllm_reqs = [req for req in self.waiting_queue if req.is_dllm()]
        requests_to_add = dllm_reqs[: max(max_dllm_capacity, 0)]

        if requests_to_add:
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            added = {id(req) for req in requests_to_add}
            self.waiting_queue = [
                req for req in self.waiting_queue if id(req) not in added
            ]

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        if new_batch.can_prepare_for_denoise():
            if not getattr(self, "_dllm_denoise_fast_path_logged", False):
                logger.info(
                    "Using DLLM_DENOISE fast path (batch_size=%d, block_size=%d)",
                    len(can_run_list),
                    self.dllm_config.block_size,
                )
                self._dllm_denoise_fast_path_logged = True
            new_batch.prepare_for_denoise()
        else:
            new_batch.prepare_for_extend()
            new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            if self.dllm_config.needs_full_prefill and req.dllm_algo_state is not None:
                parallelcomp = req.dllm_parallelcomp_state
                if (
                    parallelcomp is not None and parallelcomp["stage"] != "decode"
                ) or req.is_scoring_token_eviction():
                    req.dllm_algo_state["prompt_len"] = len(
                        req.full_untruncated_fill_ids
                    )
                    res = adder.add_one_req(
                        req,
                        has_chunked_req=True,
                        truncation_align_size=self.truncation_align_size,
                    )
                    if res != AddReqResult.CONTINUE:
                        if res == AddReqResult.NO_TOKEN:
                            running_batch.batch_is_full = True
                        break
                    continue

                partial_draft_prompt = bool(
                    req.dllm_algo_state.get("partial_draft", False)
                    and req.dllm_algo_state.get("partial_draft_stage") == "prompt"
                )
                if partial_draft_prompt:
                    prompt_len = len(req.origin_input_ids)
                    if len(req.full_untruncated_fill_ids) != prompt_len:
                        raise RuntimeError(
                            "Partial draft prompt admission requires prompt-only input: "
                            f"input={len(req.full_untruncated_fill_ids)}, "
                            f"prompt={prompt_len}"
                        )
                    req.dllm_algo_state["prompt_len"] = prompt_len
                else:
                    # Req state can predate tokenization. Once Dream materializes
                    # its canvas, derive the authoritative prompt boundary from
                    # prompt + committed output + remaining-mask layout.
                    remaining = max(
                        req.sampling_params.max_new_tokens - len(req.output_ids), 0
                    )
                    prompt_len = len(req.full_untruncated_fill_ids) - remaining
                    if not 0 <= prompt_len <= len(req.full_untruncated_fill_ids):
                        raise RuntimeError(
                            "Invalid Dream prompt boundary: "
                            f"prompt_len={prompt_len}, "
                            f"canvas_len={len(req.full_untruncated_fill_ids)}, "
                            f"remaining={remaining}"
                        )
                    parallelcomp = req.dllm_parallelcomp_state
                    if (
                        parallelcomp is not None
                        and parallelcomp["stage"] == "decode"
                        and not req.dllm_algo_state.get("dual_cache_ready", False)
                    ):
                        req.dllm_algo_state["prompt_len"] = parallelcomp["query_len"]
                        req.dllm_algo_state["full_prompt_len"] = prompt_len
                    else:
                        req.dllm_algo_state["prompt_len"] = prompt_len
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def remove_req(self, req: Req) -> None:
        """Stop scheduling a request that leaves this server unfinished."""
        self.waiting_queue = [r for r in self.waiting_queue if r is not req]
        self.staging_queue = [r for r in self.staging_queue if r is not req]

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def pop_aborted_reqs(self, abort_all: bool, rid: str) -> List[Req]:
        aborted_reqs: List[Req] = []
        seen: Set[int] = set()

        for queue_name in ("waiting_queue", "staging_queue"):
            queue = getattr(self, queue_name)
            kept_queue = []
            for req in queue:
                if abort_all or req.rid.startswith(rid):
                    req_id = id(req)
                    if req_id not in seen:
                        aborted_reqs.append(req)
                        seen.add(req_id)
                else:
                    kept_queue.append(req)
            setattr(self, queue_name, kept_queue)

        return aborted_reqs

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
