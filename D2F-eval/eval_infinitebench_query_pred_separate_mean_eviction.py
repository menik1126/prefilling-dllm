"""Experimental InfiniteBench runner for separated query/predicted-next-block eviction.

This monkey-patches only the local process. It scores chunk tokens by computing
mean attention over query rows and mean attention over predicted-next-block rows
separately, then adding the two token-score vectors.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from eval_dream import DreamLoRA


def _mean_attention_scores_for_range(
    layer_attn: torch.Tensor,
    row_range: Tuple[int, int],
    chunk_len: int,
) -> torch.Tensor:
    num_heads = layer_attn.shape[0]
    start, end = row_range
    start = max(0, min(int(start), int(layer_attn.shape[1])))
    end = max(0, min(int(end), int(layer_attn.shape[1])))
    if end <= start or chunk_len <= 0:
        return torch.zeros(
            (num_heads, max(0, chunk_len)),
            device=layer_attn.device,
            dtype=layer_attn.dtype,
        )
    return layer_attn[:, start:end, :chunk_len].mean(dim=1)


def _token_scores_from_separate_query_pred_attention_rows(
    self,
    attn_layers,
    chunk_len: int,
    query_range: Tuple[int, int],
    pred_range: Tuple[int, int],
    num_cache_heads_per_layer: List[int],
) -> List[torch.Tensor]:
    if attn_layers is None or len(attn_layers) == 0 or chunk_len <= 0:
        return []

    token_scores_per_layer_per_head = []
    for layer_idx, layer_attn in enumerate(attn_layers):
        layer_attn = layer_attn[0]
        num_cache_heads = (
            num_cache_heads_per_layer[layer_idx]
            if layer_idx < len(num_cache_heads_per_layer)
            else layer_attn.shape[0]
        )
        query_scores = _mean_attention_scores_for_range(layer_attn, query_range, chunk_len)
        pred_scores = _mean_attention_scores_for_range(layer_attn, pred_range, chunk_len)
        combined_scores = query_scores + pred_scores
        combined_scores = self._pool_parallelcomp_token_scores(combined_scores)

        grouped_scores = []
        for head_group in torch.tensor_split(combined_scores, num_cache_heads, dim=0):
            if head_group.shape[0] == 0:
                grouped_scores.append(
                    torch.zeros(chunk_len, device=self.device, dtype=combined_scores.dtype)
                )
            else:
                grouped_scores.append(head_group.mean(dim=0))
        token_scores_per_layer_per_head.append(torch.stack(grouped_scores, dim=0))

    return token_scores_per_layer_per_head


def _run_parallelcomp_local_block_forward_query_pred_separate_mean(
    self,
    block_input_ids: torch.Tensor,
    query_ids: List[int],
    reused_window_size: int,
) -> Optional[Dict[str, Any]]:
    block_len = int(block_input_ids.shape[1])
    query_len = len(query_ids)
    if block_len <= 0 or query_len <= 0:
        return None

    query_input_ids = torch.tensor([query_ids], device=self.device, dtype=torch.long)
    joint_ids = torch.cat([block_input_ids, query_input_ids], dim=1)
    joint_len = block_len + query_len

    block_positions = torch.arange(block_len, device=self.device, dtype=torch.long)
    query_positions = torch.arange(
        reused_window_size,
        reused_window_size + query_len,
        device=self.device,
        dtype=torch.long,
    )
    position_ids = torch.cat([block_positions, query_positions], dim=0).unsqueeze(0)
    cache_position = torch.arange(joint_len, device=self.device, dtype=torch.long)
    attention_mask = self._build_parallelcomp_scoring_attention_mask(
        joint_len,
        chunk_len=block_len,
        query_len=query_len,
    )

    with torch.inference_mode():
        outputs = self.model(
            joint_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            output_attentions=True,
            return_dict=True,
            use_cache=True,
            update_kvcache=joint_len,
        )

    block_cache = outputs.past_key_values
    num_cache_heads_per_layer = [layer_k.shape[1] for layer_k in block_cache.key_cache]
    score = self._score_parallelcomp_query_from_logits(
        logits=outputs.logits,
        joint_ids=joint_ids,
        chunk_len=block_len,
        query_len=query_len,
    )

    next_block_len = min(max(1, int(getattr(self, "block_size", 1))), joint_len)
    tail_logits = outputs.logits[:, joint_len - next_block_len:joint_len, :]
    predicted_next_ids = tail_logits.squeeze(0).argmax(dim=-1).detach()

    extended_query_ids = torch.cat(
        [query_input_ids, predicted_next_ids.unsqueeze(0)],
        dim=1,
    )
    extended_query_len = int(extended_query_ids.shape[1])
    extended_joint_ids = torch.cat([block_input_ids, extended_query_ids], dim=1)
    extended_joint_len = block_len + extended_query_len

    predicted_positions = torch.arange(
        reused_window_size + query_len,
        reused_window_size + query_len + next_block_len,
        device=self.device,
        dtype=torch.long,
    )
    extended_position_ids = torch.cat(
        [block_positions, query_positions, predicted_positions],
        dim=0,
    ).unsqueeze(0)
    extended_cache_position = torch.arange(
        extended_joint_len,
        device=self.device,
        dtype=torch.long,
    )
    extended_attention_mask = self._build_parallelcomp_scoring_attention_mask(
        extended_joint_len,
        chunk_len=block_len,
        query_len=extended_query_len,
    )

    with torch.inference_mode():
        extended_outputs = self.model(
            extended_joint_ids,
            attention_mask=extended_attention_mask,
            position_ids=extended_position_ids,
            cache_position=extended_cache_position,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
            update_kvcache=0,
        )

    query_window = self._get_parallelcomp_query_window_size(query_len)
    query_range = (block_len + query_len - query_window, block_len + query_len)
    pred_range = (block_len + query_len, block_len + query_len + next_block_len)
    token_scores_per_layer_per_head = _token_scores_from_separate_query_pred_attention_rows(
        self,
        extended_outputs.attentions,
        chunk_len=block_len,
        query_range=query_range,
        pred_range=pred_range,
        num_cache_heads_per_layer=num_cache_heads_per_layer,
    )
    per_layer_kept_indices, evicted_high_layers = (
        self._select_cache_block_token_indices_from_scores_per_layer_per_head(
            token_scores_per_layer_per_head=token_scores_per_layer_per_head,
            block_len=block_len,
            num_cache_heads_per_layer=num_cache_heads_per_layer,
        )
    )

    return {
        "score": score,
        "key_cache": [layer_k[:, :, :block_len, :] for layer_k in block_cache.key_cache],
        "value_cache": [layer_v[:, :, :block_len, :] for layer_v in block_cache.value_cache],
        "kept_indices": per_layer_kept_indices,
        "evicted_high_layers": evicted_high_layers,
        "num_cache_heads_per_layer": num_cache_heads_per_layer,
        "block_len": block_len,
    }


DreamLoRA._run_parallelcomp_local_block_forward = (
    _run_parallelcomp_local_block_forward_query_pred_separate_mean
)
print(
    "[Experiment] Enabled separated mean(query attention) + mean(predicted-next-block attention) token eviction monkey patch.",
    flush=True,
)


if __name__ == "__main__":
    import eval_infinitebench

    eval_infinitebench.main()
