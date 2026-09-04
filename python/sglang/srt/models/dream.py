from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


class DreamModel(Qwen2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)

        if self.pp_group.world_size != 1:
            raise ValueError("DreamModel currently only supports PP=1")

        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        # Match DreamRMSNorm in the reference Hugging Face implementation:
        # normalize in FP32, cast the normalized activation back to the model
        # dtype, and only then multiply by the norm weight. Qwen2's default
        # optimized path performs the weight multiplication before that cast,
        # which changes Dream's iterative decoding decisions in FP16.
        self.model.norm.cast_x_before_out_mul = True
        self.model.norm._forward_method = self.model.norm.forward_native
        for layer in self.model.layers:
            layer.input_layernorm.cast_x_before_out_mul = True
            layer.post_attention_layernorm.cast_x_before_out_mul = True
            layer.input_layernorm._forward_method = layer.input_layernorm.forward_native
            layer.post_attention_layernorm._forward_method = (
                layer.post_attention_layernorm.forward_native
            )
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        assert not self.capture_aux_hidden_states

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not get_embedding:
            if forward_batch.forward_mode.is_dllm_full_attention():
                seq_lens = forward_batch.extend_seq_lens_cpu

                if seq_lens is None:
                    raise RuntimeError("Dream requires per-request sequence lengths")

                # Prefill BCG replays the transformer body at a padded token
                # bucket, while Dream's ragged canvas metadata still describes
                # only the real tokens.  The padding is appended after the
                # live canvas, so remove it before splitting by request.
                raw_num_tokens = sum(seq_lens)
                if raw_num_tokens > hidden_states.shape[0]:
                    raise RuntimeError(
                        "Dream sequence lengths exceed the hidden-state rows: "
                        f"{raw_num_tokens} > {hidden_states.shape[0]}"
                    )
                hidden_states = hidden_states[:raw_num_tokens]

                parts = hidden_states.split(seq_lens)
                raw_last_logits = getattr(
                    forward_batch, "dllm_raw_last_logits_cpu", None
                )
                if raw_last_logits is not None and len(raw_last_logits) != len(parts):
                    raise RuntimeError(
                        "Dream raw-logit modes do not match the request batch: "
                        f"{len(raw_last_logits)} != {len(parts)}"
                    )
                if raw_last_logits is None:
                    raw_last_logits = [False] * len(parts)
                if any(
                    use_raw_last and len(part) == 0
                    for part, use_raw_last in zip(parts, raw_last_logits)
                ):
                    raise RuntimeError(
                        "Dream raw final logits require a non-empty request"
                    )
                shifted_parts = [
                    (
                        part[-1:]
                        if use_raw_last
                        else torch.cat([part[:1], part[:-1]], dim=0)
                    )
                    for part, use_raw_last in zip(parts, raw_last_logits)
                ]

                # Dream only consumes logits for its trailing generation
                # canvas.  Projecting every prompt row through the FP32
                # vocabulary head creates a prompt-length x vocab temporary
                # (multiple GiB for long LongBench examples) that is discarded
                # immediately by the denoising algorithm.
                block_size = getattr(forward_batch, "dllm_block_size", None)
                canvas_lens = getattr(forward_batch, "dllm_canvas_lens_cpu", None)
                if canvas_lens is not None:
                    if len(canvas_lens) != len(shifted_parts):
                        raise RuntimeError(
                            "Dream canvas lengths do not match the request batch: "
                            f"{len(canvas_lens)} != {len(shifted_parts)}"
                        )
                    if any(
                        canvas_len <= 0 or canvas_len > len(part)
                        for part, canvas_len in zip(shifted_parts, canvas_lens)
                    ):
                        raise RuntimeError(
                            "Dream canvas lengths must be positive and fit their "
                            "request token spans"
                        )
                    shifted_parts = [
                        part[-canvas_len:]
                        for part, canvas_len in zip(shifted_parts, canvas_lens)
                    ]
                elif block_size is not None:
                    if block_size <= 0:
                        raise RuntimeError(
                            f"Dream dLLM block size must be positive: {block_size}"
                        )
                    shifted_parts = [part[-block_size:] for part in shifted_parts]

                hidden_states = torch.cat(shifted_parts, dim=0)
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
            )

        return self.pooler(hidden_states, forward_batch)


EntryClass = DreamModel
