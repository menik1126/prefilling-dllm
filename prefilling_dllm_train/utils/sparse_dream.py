"""Runtime patch for gradient-safe Dream sparse prefill attention."""

import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_COMPILED_FLEX_ATTENTION = None


def _get_compiled_flex_attention():
    """Compile once per training process, shared by all Dream layers."""
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


def build_uniform_sink_keep_indices(
        *,
        prompt_length,
        num_layers,
        num_kv_heads,
        token_capacity,
        sink_tokens,
        block_size,
        device,
):
    """Build block-aligned fallback indices for sparse prefill."""
    block_size = max(1, int(block_size))
    block_count = (int(prompt_length) + block_size - 1) // block_size
    keep_block_count = min(max(1, int(token_capacity) // block_size), block_count)
    sink_block_count = min(
        (max(0, int(sink_tokens)) + block_size - 1) // block_size,
        keep_block_count,
    )
    sink_blocks = torch.arange(sink_block_count, device=device, dtype=torch.long)
    remaining = keep_block_count - sink_block_count
    if remaining:
        sampled = torch.linspace(
            sink_block_count, block_count - 1, remaining, device=device
        ).round().to(torch.long)
        blocks = torch.cat([sink_blocks, sampled])
    else:
        blocks = sink_blocks
    blocks = torch.unique(blocks, sorted=True)
    if blocks.numel() < keep_block_count:
        candidates = torch.arange(block_count, device=device, dtype=torch.long)
        available = candidates[~torch.isin(candidates, blocks)]
        blocks = torch.sort(
            torch.cat([blocks, available[:keep_block_count - blocks.numel()]])
        ).values
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    keep = (blocks.unsqueeze(1) * block_size + offsets).flatten()
    keep = keep[keep < int(prompt_length)]
    return keep.view(1, 1, -1).expand(num_layers, num_kv_heads, -1).clone()


def select_blocks_by_draft_self_information(
        *,
        model,
        prompt_ids,
        mask_token_id,
        query_tokens,
        draft_tokens,
        draft_partial_rounds,
        chunk_size,
        topk_chunks,
):
    """No-grad logits scorer matching the draft-self-information selection rule."""
    prompt_length = int(prompt_ids.numel())
    query_length = min(max(1, int(query_tokens)), prompt_length - 1)
    context_ids = prompt_ids[:-query_length]
    scoring_query = prompt_ids[-query_length:]
    chunks = list(context_ids.split(max(1, int(chunk_size))))
    if not chunks:
        return torch.arange(prompt_length, device=prompt_ids.device), scoring_query

    is_rank_zero = os.environ.get("RANK", "0") == "0"
    if is_rank_zero:
        print(
            f"[sparse-selector] scoring {len(chunks)} chunks "
            f"(chunk={chunk_size}, topk={topk_chunks})",
            flush=True,
        )
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            draft = torch.full(
                (max(0, int(draft_tokens)),),
                int(mask_token_id),
                device=prompt_ids.device,
                dtype=prompt_ids.dtype,
            )
            confirmed = torch.zeros_like(draft, dtype=torch.bool)
            for _ in range(max(1, int(draft_partial_rounds))):
                open_slots = torch.nonzero(draft.eq(mask_token_id), as_tuple=False).flatten()
                if open_slots.numel() == 0:
                    break
                logits = model(torch.cat([prompt_ids, draft]).unsqueeze(0)).logits[0]
                shifted = torch.empty_like(logits)
                shifted[0] = logits[0]
                shifted[1:] = logits[:-1]
                positions = prompt_length + open_slots
                slot_logits = shifted.index_select(0, positions)
                confidence, sampled = torch.softmax(slot_logits.float(), dim=-1).max(dim=-1)
                best = torch.argmax(confidence)
                slot = open_slots[best]
                draft[slot] = sampled[best].to(draft.dtype)
                confirmed[slot] = True

            score_targets = torch.cat([scoring_query, draft])
            score_mask = torch.cat(
                [
                    torch.ones(query_length, device=prompt_ids.device, dtype=torch.bool),
                    confirmed,
                ]
            )
            scores = []
            for chunk in chunks:
                logits = model(torch.cat([chunk, score_targets]).unsqueeze(0)).logits[0]
                shifted = torch.empty_like(logits)
                shifted[0] = logits[0]
                shifted[1:] = logits[:-1]
                start = int(chunk.numel())
                target_positions = torch.arange(
                    start, start + score_targets.numel(), device=prompt_ids.device
                )
                target_logits = shifted.index_select(0, target_positions)[score_mask]
                targets = score_targets[score_mask]
                nll = -F.log_softmax(target_logits.float(), dim=-1).gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                scores.append(-nll.mean())
            scores = torch.stack(scores)
            selected = torch.topk(
                scores, k=min(max(1, int(topk_chunks)), len(chunks))
            ).indices.sort().values
            offsets = [
                sum(int(part.numel()) for part in chunks[:index])
                for index in selected.tolist()
            ]
            kept = [
                torch.arange(offset, offset + chunks[index].numel(), device=prompt_ids.device)
                for offset, index in zip(offsets, selected.tolist())
            ]
            if is_rank_zero:
                print(f"[sparse-selector] selected chunks={selected.tolist()}", flush=True)
            return torch.cat(kept), scores
    finally:
        if was_training:
            model.train()


def _sparse_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        sparse_keep_indices=None,
        sparse_prompt_len=None,
):
    """Run Dream's eviction-mask sparse prefill with FlexAttention backward."""
    keep = sparse_keep_indices
    prompt_len = sparse_prompt_len
    if keep is None:
        keep = getattr(self, "_sparse_prefill_keep_indices", None)
        prompt_len = getattr(self, "_sparse_prefill_prompt_len", None)
    if keep is None:
        return self.__class__._sparse_prefill_original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    if output_attentions or attention_mask is not None or past_key_value is not None:
        raise ValueError(
            "Sparse prefill training supports only full bidirectional attention without cache."
        )

    module = sys.modules[self.__class__.__module__]
    batch_size, query_len, _ = hidden_states.size()
    if batch_size != 1:
        raise ValueError("Sparse prefill training currently requires batch_size=1.")
    if prompt_len is None or not 0 < prompt_len <= query_len:
        raise ValueError("Sparse prompt length is outside the current sequence.")

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(
        batch_size, query_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size, query_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size, query_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = module.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    # LoRA adapters may promote only Q/K projections to fp32 under AMP while
    # V remains bf16. FlexAttention requires a single dtype for Q/K/V.
    attention_dtype = hidden_states.dtype
    query_states = query_states.to(attention_dtype)
    key_states = key_states.to(attention_dtype)
    value_states = value_states.to(attention_dtype)

    keep = keep.to(device=key_states.device, dtype=torch.long)
    if keep.ndim != 2 or keep.shape[0] != self.num_key_value_heads:
        raise ValueError(
            "Sparse keep indices must be [num_kv_heads, kept_prompt_tokens]."
        )
    if keep.numel() == 0 or keep.min() < 0 or keep.max() >= prompt_len:
        raise ValueError("Sparse keep indices are outside the prompt range.")

    # Match prefilling_dllm's eviction_mask semantics: per-head keep indices are
    # reduced to one layer-level union key mask, while response suffix keys
    # remain visible to every query.
    key_mask = torch.zeros(query_len, dtype=torch.bool, device=key_states.device)
    key_mask[torch.unique(keep)] = True
    key_mask[prompt_len:] = True

    def mask_mod(batch_idx, head_idx, query_idx, key_idx):
        return key_mask[key_idx]

    block_mask = create_block_mask(
        mask_mod,
        None,
        None,
        query_len,
        query_len,
        device=query_states.device,
    )
    attention_output = _get_compiled_flex_attention()(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,
    )
    attention_output = attention_output.transpose(1, 2).contiguous().view(
        batch_size, query_len, self.hidden_size
    )
    return self.o_proj(attention_output), None, past_key_value


def install_sparse_prefill_patch(model):
    """Enable ``sparse_keep_indices`` kwargs on a PEFT-wrapped Dream model."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_cls = base_model.__class__
    attention_cls = base_model.model.layers[0].self_attn.__class__

    if not hasattr(attention_cls, "_sparse_prefill_original_forward"):
        attention_cls._sparse_prefill_original_forward = attention_cls.forward
        attention_cls.forward = _sparse_attention_forward

    if hasattr(model_cls, "_sparse_prefill_original_forward"):
        return

    model_cls._sparse_prefill_original_forward = model_cls.forward

    def _model_forward(self, *args, **kwargs):
        keep_indices = kwargs.pop("sparse_keep_indices", None)
        prompt_len = kwargs.pop("sparse_prompt_len", None)
        if keep_indices is None:
            return self.__class__._sparse_prefill_original_forward(self, *args, **kwargs)
        if keep_indices.shape[0] != len(self.model.layers):
            raise ValueError("Sparse keep index layer count does not match the Dream model.")
        for layer_index, layer in enumerate(self.model.layers):
            layer.self_attn._sparse_prefill_keep_indices = keep_indices[layer_index]
            layer.self_attn._sparse_prefill_prompt_len = prompt_len
        try:
            return self.__class__._sparse_prefill_original_forward(self, *args, **kwargs)
        finally:
            for layer in self.model.layers:
                del layer.self_attn._sparse_prefill_keep_indices
                del layer.self_attn._sparse_prefill_prompt_len

    model_cls.forward = _model_forward
