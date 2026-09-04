"""Dream denoising schedule matching the Prefilling-dLLM engine sampler."""

from typing import Any, List

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PrefillingDream(DllmAlgorithm):
    """Confidence-threshold decoding used by Prefilling-dLLM for Dream.

    Prefilling-dLLM shifts Dream logits by one position, accepts the first
    generated token during prefill, then accepts every masked token whose
    greedy probability reaches the threshold. If none qualifies, it forces the
    single highest-confidence token. This implementation intentionally targets
    the one-block LongBench setting (block_size == max_new_tokens == 32).
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = float(config.algorithm_config.get("threshold", 0.9))
        self.dual_cache = config.dual_cache
        self.finish_on_final_mutation = bool(
            config.algorithm_config.get("finish_on_final_mutation", False)
        )

    def max_steps(self, block_size: int) -> int:
        # One initial prefill transfer and at most one forced token per later
        # step. The extra iteration preserves the default observer behavior and
        # retains the existing bounded retry when the model selects mask_id.
        return block_size + 1

    def init_step_state(self, forward_batch: ForwardBatch) -> List[Any]:
        return [{"is_prefill": True} for _ in range(forward_batch.batch_size)]

    def step(
        self,
        forward_batch: ForwardBatch,
        full_logits: torch.Tensor,
        states: List[Any],
    ) -> List[bool]:
        lengths = forward_batch.extend_seq_lens_cpu
        if lengths is None:
            raise RuntimeError("PrefillingDream requires CPU sequence lengths")

        input_parts = forward_batch.input_ids.split(lengths)
        logits_pruned_to_generation = full_logits.shape[0] != sum(lengths)
        if logits_pruned_to_generation:
            canvas_lens = getattr(forward_batch, "dllm_canvas_lens_cpu", None)
            block_size = getattr(forward_batch, "dllm_block_size", None)
            if canvas_lens is not None:
                if len(canvas_lens) != len(lengths):
                    raise RuntimeError(
                        "PrefillingDream canvas lengths do not match the request batch: "
                        f"{len(canvas_lens)} != {len(lengths)}"
                    )
                logits_lengths = [
                    min(length, canvas_len)
                    for length, canvas_len in zip(lengths, canvas_lens)
                ]
            elif block_size is not None:
                logits_lengths = [min(length, block_size) for length in lengths]
            else:
                raise RuntimeError(
                    "PrefillingDream received pruned logits without a dLLM block size"
                )
            if sum(logits_lengths) != full_logits.shape[0]:
                raise RuntimeError(
                    "PrefillingDream pruned-logits rows do not match generation blocks: "
                    f"{full_logits.shape[0]} != {sum(logits_lengths)}"
                )
        else:
            logits_lengths = lengths
        logits_parts = full_logits.split(logits_lengths)
        done: List[bool] = []

        for ids, logits, state in zip(input_parts, logits_parts, states):
            prompt_len = state["prompt_len"]
            partial_draft_stage = state.get("partial_draft_stage")
            if partial_draft_stage is not None:
                state["last_partial_draft_stage"] = partial_draft_stage
            dual_cache_round = bool(
                self.dual_cache and state.get("dual_cache_ready", False)
            )
            # The state is shallow-carried by FDFO, so the scheduler can use
            # this pre-step fact instead of inferring the round type from a
            # server-wide block size.
            state["last_round_was_dual_cache"] = dual_cache_round

            if partial_draft_stage == "prompt":
                if dual_cache_round or logits.shape[0] != 1:
                    raise RuntimeError(
                        "PrefillingDream partial prompt requires one raw final logit"
                    )
                prompt_probs = F.softmax(logits[0], dim=-1)
                state["partial_draft_first_token"] = int(
                    prompt_probs.argmax(dim=-1).item()
                )
                state["partial_draft_stage"] = "suffix_init"
                done.append(False)
                continue

            if partial_draft_stage == "suffix_init":
                canvas_len = state["canvas_len"]
                if dual_cache_round or len(ids) != canvas_len:
                    raise RuntimeError(
                        "PrefillingDream partial suffix initialization requires "
                        "one uncached canvas"
                    )
                if not bool(ids.eq(self.mask_id).all().item()):
                    raise RuntimeError(
                        "PrefillingDream partial suffix must start fully masked"
                    )
                first_token = state.get("partial_draft_first_token")
                if not isinstance(first_token, int):
                    raise RuntimeError(
                        "PrefillingDream partial suffix is missing its prompt token"
                    )
                confirmed_mask = state.get("partial_draft_confirmed_mask")
                if not isinstance(confirmed_mask, list) or len(confirmed_mask) != len(
                    ids
                ):
                    raise RuntimeError(
                        "PrefillingDream partial draft requires a canvas-aligned "
                        "confirmed mask"
                    )
                ids[0] = first_token
                confirmed_mask[0] = True
                state["is_prefill"] = False
                state["dual_cache_ready"] = self.dual_cache
                state["partial_draft_stage"] = "partial_round"
                done.append(False)
                continue

            generation_ids = ids if dual_cache_round else ids[prompt_len:]
            partial_draft = bool(state.get("partial_draft", False))
            confirmed_mask = None
            if partial_draft:
                confirmed_mask = state.get("partial_draft_confirmed_mask")
                if (
                    not isinstance(confirmed_mask, list)
                    or len(confirmed_mask) != len(generation_ids)
                    or any(type(value) is not bool for value in confirmed_mask)
                ):
                    raise RuntimeError(
                        "PrefillingDream partial draft requires a canvas-aligned "
                        "boolean confirmed mask"
                    )
                # Confirmation is algorithm state, not a token-id property: the
                # model is allowed to select mask_id as a real token. Excluding
                # confirmed slots by bitmap prevents such a slot from being
                # selected again in the bounded round.
                mask = torch.tensor(
                    confirmed_mask, dtype=torch.bool, device=generation_ids.device
                ).logical_not()
            else:
                mask = generation_ids.eq(self.mask_id)
            if not bool(mask.any().item()):
                done.append(True)
                continue

            # DreamModel.forward already right-shifts each request's hidden
            # states before its lm_head. This is equivalent to the reference
            # sampler's _shift_logits and must not be repeated here.
            generation_logits = (
                logits
                if logits_pruned_to_generation or dual_cache_round
                else logits[prompt_len:]
            )
            token_logits = generation_logits[mask]
            probs = F.softmax(token_logits, dim=-1)
            confidence, sampled_tokens = probs.max(dim=-1)
            mask_positions = mask.nonzero(as_tuple=False).flatten()

            # FDFO may provide a scheduler-created carried state containing only
            # prompt_len, bypassing init_step_state(). Treat that first block as
            # prefill and persist the flag for subsequent denoising iterations.
            if state.get("is_prefill", True):
                accepted = torch.zeros(1, dtype=torch.long, device=ids.device)
                state["is_prefill"] = False
                state["dual_cache_ready"] = self.dual_cache
                partial_draft_done = False
            elif partial_draft:
                # Reference partial drafting deliberately ignores the normal
                # confidence threshold: each bounded round confirms only the
                # single most-confident remaining slot.
                accepted = torch.topk(confidence, k=1).indices
                state["partial_draft_rounds_done"] += 1
                partial_draft_done = (
                    state["partial_draft_rounds_done"]
                    >= state["partial_draft_round_limit"]
                )
            else:
                accepted = torch.where(confidence >= self.threshold)[0]
                if accepted.numel() == 0:
                    accepted = torch.topk(confidence, k=1).indices
                partial_draft_done = False

            accepted_positions = mask_positions[accepted]
            accepted_tokens = sampled_tokens[accepted]
            generation_ids[accepted_positions] = accepted_tokens
            if partial_draft:
                assert confirmed_mask is not None
                for position in accepted_positions.tolist():
                    confirmed_mask[position] = True
                expected_confirmed = min(
                    len(confirmed_mask), 1 + state["partial_draft_rounds_done"]
                )
                if sum(confirmed_mask) != expected_confirmed:
                    raise RuntimeError(
                        "PrefillingDream partial draft confirmed an unexpected "
                        f"number of slots: {sum(confirmed_mask)} != "
                        f"{expected_confirmed}"
                    )
                request_done = partial_draft_done
            elif self.finish_on_final_mutation and not getattr(
                forward_batch, "return_logprob", False
            ):
                # The old path discovered completion at the start of the next
                # round, after one redundant full-canvas forward. Completion
                # here is equivalent when every remaining candidate was
                # accepted and none was rewritten to mask_id itself. Only the
                # possible final round pays the small device-to-host check.
                request_done = accepted.numel() == mask_positions.numel()
                if request_done:
                    request_done = not bool(
                        accepted_tokens.eq(self.mask_id).any().item()
                    )
            else:
                request_done = False
            done.append(request_done)

        return done


Algorithm = PrefillingDream
