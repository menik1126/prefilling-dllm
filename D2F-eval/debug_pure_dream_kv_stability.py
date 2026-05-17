#!/usr/bin/env python3
import json
import os

import torch

from d2f_model import load_model
from eval_dream import build_sparse_block_attention_mask


def clone_prompt_cache(past_key_values, prompt_len):
    return [
        (
            layer_k[:, :, :prompt_len, :].detach().float().cpu().clone(),
            layer_v[:, :, :prompt_len, :].detach().float().cpu().clone(),
        )
        for layer_k, layer_v in zip(past_key_values.key_cache, past_key_values.value_cache)
    ]


def max_cache_diff(left, right):
    max_k = 0.0
    max_v = 0.0
    for (lk, lv), (rk, rv) in zip(left, right):
        max_k = max(max_k, float((lk - rk).abs().max().item()))
        max_v = max(max_v, float((lv - rv).abs().max().item()))
    return {"key": max_k, "value": max_v}


def tensorize_tail(inner, text, block_size):
    ids = inner.tok_encode(text, add_special_tokens=False)[0].to(inner.device)
    if ids.numel() < block_size:
        pad = torch.full(
            (block_size - ids.numel(),),
            inner.mask_token_id,
            device=inner.device,
            dtype=torch.long,
        )
        ids = torch.cat([ids, pad], dim=0)
    return ids[:block_size].unsqueeze(0)


def make_block_states(inner, prompt_len, block_size):
    block_states = inner._init_prompt_block_states(prompt_len)
    new_block_id = max(block_states.keys()) + 1
    block_states[new_block_id] = {
        "start_pos": prompt_len,
        "end_pos": prompt_len + block_size,
        "mask_count": block_size,
        "total_masks": block_size,
        "rope_span": block_size,
        "state": "active",
        "is_complete": False,
        "is_prompt_window": False,
    }
    return block_states


def run_initial_cache_forward(inner, prompt_ids, tail_ids, block_states, prompt_len):
    x_t = torch.cat([prompt_ids, tail_ids], dim=1)
    input_seq = x_t
    update_kvcache = prompt_len
    block_ids, prompt_window_mask, rope_positions = inner._build_input_position_ids(
        block_states=block_states,
        process_start_pos=0,
        input_length=input_seq.shape[1],
        update_kvcache=update_kvcache,
    )
    cache_positions = inner._build_forward_cache_positions(
        position_ids=rope_positions,
        cached_length=0,
        update_kvcache=update_kvcache,
    )
    attention_mask = build_sparse_block_attention_mask(
        query_block_ids=block_ids,
        cached_length=0,
        query_prompt_window_mask=prompt_window_mask,
        update_kvcache=update_kvcache,
        device=inner.device,
        dtype=inner.target_dtype if inner.target_dtype not in (None, "auto") else torch.bfloat16,
    )
    return inner.model(
        input_seq,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=rope_positions.unsqueeze(0),
        cache_position=cache_positions,
        use_cache=True,
        update_kvcache=update_kvcache,
    )


def run_active_tail_forward(inner, past_key_values, tail_ids, block_states, prompt_len):
    update_kvcache = 0
    block_ids, prompt_window_mask, rope_positions = inner._build_input_position_ids(
        block_states=block_states,
        process_start_pos=prompt_len,
        input_length=tail_ids.shape[1],
        update_kvcache=update_kvcache,
    )
    cache_positions = inner._build_forward_cache_positions(
        position_ids=rope_positions,
        cached_length=prompt_len,
        update_kvcache=update_kvcache,
    )
    attention_mask = build_sparse_block_attention_mask(
        query_block_ids=block_ids,
        cached_length=prompt_len,
        query_prompt_window_mask=prompt_window_mask,
        update_kvcache=update_kvcache,
        device=inner.device,
        dtype=inner.target_dtype if inner.target_dtype not in (None, "auto") else torch.bfloat16,
    )
    return inner.model(
        tail_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=rope_positions.unsqueeze(0),
        cache_position=cache_positions,
        use_cache=True,
        update_kvcache=update_kvcache,
    )


def main():
    torch.set_grad_enabled(False)
    base_model = os.environ.get("DREAM_BASE", "/mnt/Data/xiongjing/models/Dream-v0-Base-7B")
    block_size = int(os.environ.get("DEBUG_BLOCK_SIZE", "4"))
    prompt_text = os.environ.get(
        "DEBUG_PROMPT",
        "The pass key is 71432. Remember it carefully. Question: what is the pass key?",
    )

    wrapper = load_model(
        "dream",
        base_model,
        lora_path=None,
        max_new_tokens=block_size,
        max_length=512,
        block_size=block_size,
        temperature=0.0,
        add_bos_token=True,
        parallelcomp_mode=True,
        parallelcomp_pre_runtime_mode=False,
        parallelcomp_cache_compress_mode=False,
    )
    inner = wrapper._inner
    inner.model.eval()

    prompt_ids = inner.tok_encode(prompt_text, add_special_tokens=True).to(inner.device)
    prompt_len = int(prompt_ids.shape[1])
    block_states = make_block_states(inner, prompt_len, block_size)

    tail_mask = torch.full(
        (1, block_size),
        inner.mask_token_id,
        device=inner.device,
        dtype=torch.long,
    )
    tail_words_a = tensorize_tail(inner, " alpha beta gamma delta", block_size)
    tail_words_b = tensorize_tail(inner, " red blue green yellow", block_size)

    with torch.inference_mode():
        out_mask = run_initial_cache_forward(inner, prompt_ids, tail_mask, block_states, prompt_len)
        out_words = run_initial_cache_forward(inner, prompt_ids, tail_words_a, block_states, prompt_len)
        cache_from_mask_tail = clone_prompt_cache(out_mask.past_key_values, prompt_len)
        cache_from_word_tail = clone_prompt_cache(out_words.past_key_values, prompt_len)

        before_update0 = clone_prompt_cache(out_mask.past_key_values, prompt_len)
        len_before = [int(k.shape[-2]) for k in out_mask.past_key_values.key_cache]
        _ = run_active_tail_forward(inner, out_mask.past_key_values, tail_words_b, block_states, prompt_len)
        after_update0 = clone_prompt_cache(out_mask.past_key_values, prompt_len)
        len_after = [int(k.shape[-2]) for k in out_mask.past_key_values.key_cache]

    result = {
        "model": "Dream-v0-Base-7B",
        "lora_path": None,
        "loaded_model_class": type(inner.model).__name__,
        "prompt_len": prompt_len,
        "block_size": block_size,
        "num_layers": len(out_mask.past_key_values.key_cache),
        "cache_lengths_before_update0": len_before[:3],
        "cache_lengths_after_update0": len_after[:3],
        "prompt_cache_diff_when_future_tail_changes": max_cache_diff(
            cache_from_mask_tail,
            cache_from_word_tail,
        ),
        "prompt_cache_diff_after_update0_active_tail_forward": max_cache_diff(
            before_update0,
            after_update0,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
