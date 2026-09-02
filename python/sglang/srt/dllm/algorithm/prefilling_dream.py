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

    def max_steps(self, block_size: int) -> int:
        # One initial prefill transfer, at most one forced token per later step,
        # and one final forward to observe that no masks remain.
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
            block_size = getattr(forward_batch, "dllm_block_size", None)
            if block_size is None:
                raise RuntimeError(
                    "PrefillingDream received pruned logits without a dLLM block size"
                )
            logits_lengths = [min(length, block_size) for length in lengths]
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
            dual_cache_round = bool(
                self.dual_cache and state.get("dual_cache_ready", False)
            )
            generation_ids = ids if dual_cache_round else ids[prompt_len:]
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
            else:
                accepted = torch.where(confidence >= self.threshold)[0]
                if accepted.numel() == 0:
                    accepted = torch.topk(confidence, k=1).indices

            generation_ids[mask_positions[accepted]] = sampled_tokens[accepted]
            done.append(False)

        return done


Algorithm = PrefillingDream
