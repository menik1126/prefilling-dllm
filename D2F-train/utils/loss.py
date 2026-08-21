import torch
from utils.util import forward_process_length, shift_logits,forward_process
import torch.nn.functional as F
from utils.sparse_dream import (
    build_uniform_sink_keep_indices,
    select_blocks_by_draft_self_information,
)

def compute_loss_by_config(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
        config,
        sequence_length=None,
):
    """Select different loss functions based on config file"""
    training_mode = config.get('training_mode', 'dream')
    
    if training_mode == 'llada':
        return compute_llada_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream':
        return compute_loss(
            input_ids, denoiser, question_length, mask_id, block_size,
            enable_shift, share_steps, self_align, feature_align, self_step, eos_id
        )
    elif training_mode == 'dream_full_long':
        return compute_full_diffusion_loss(
            input_ids=input_ids,
            denoiser=denoiser,
            question_length=question_length,
            sequence_length=sequence_length,
            mask_id=mask_id,
            self_align=self_align,
            enable_shift=enable_shift,
            min_mask_ratio=config.train.min_mask_ratio,
            max_mask_ratio=config.train.max_mask_ratio,
            prefill_sparse_mode=config.train.get("prefill_sparse_mode", "none"),
            sparse_token_capacity=config.train.get("sparse_token_capacity", 0),
            sparse_sink_tokens=config.train.get("sparse_sink_tokens", 0),
            sparse_block_size=config.train.get("sparse_block_size", 128),
            sparse_selection_mode=config.train.get("sparse_selection_mode", "uniform_sink"),
            sparse_score_query_tokens=config.train.get("sparse_score_query_tokens", 64),
            sparse_score_draft_tokens=config.train.get("sparse_score_draft_tokens", 4),
            sparse_score_draft_partial_rounds=config.train.get("sparse_score_draft_partial_rounds", 1),
            sparse_score_chunk_size=config.train.get("sparse_score_chunk_size", 1024),
            sparse_score_topk_chunks=config.train.get("sparse_score_topk_chunks", 3),
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def compute_full_diffusion_loss(
        input_ids,
        denoiser,
        question_length,
        sequence_length,
        mask_id,
        self_align,
        enable_shift,
        min_mask_ratio,
        max_mask_ratio,
        prefill_sparse_mode,
        sparse_token_capacity,
        sparse_sink_tokens,
        sparse_block_size,
        sparse_selection_mode,
        sparse_score_query_tokens,
        sparse_score_draft_tokens,
        sparse_score_draft_partial_rounds,
        sparse_score_chunk_size,
        sparse_score_topk_chunks,
):
    """Full-bidirectional Dream diffusion loss on assistant response tokens."""
    batch_size, sequence_width = input_ids.shape
    device = input_ids.device
    if sequence_length is None:
        sequence_length = torch.full(
            (batch_size,), sequence_width, device=device, dtype=torch.long
        )
    else:
        sequence_length = sequence_length.to(device)

    positions = torch.arange(sequence_width, device=device).unsqueeze(0)
    response_mask = (
        (positions >= question_length.unsqueeze(1))
        & (positions < sequence_length.unsqueeze(1))
    )
    if not response_mask.any():
        raise ValueError("Long-context batch contains no assistant response tokens.")

    mask_probabilities = torch.empty(batch_size, device=device).uniform_(
        min_mask_ratio, max_mask_ratio
    )
    masked_indices = response_mask & (
        torch.rand(batch_size, sequence_width, device=device)
        < mask_probabilities.unsqueeze(1)
    )
    for row in torch.nonzero(~masked_indices.any(dim=1), as_tuple=False).flatten():
        valid_positions = torch.nonzero(response_mask[row], as_tuple=False).flatten()
        masked_indices[row, valid_positions[0]] = True

    sparse_keep_indices = None
    sparse_prompt_len = None
    if (prefill_sparse_mode or "none").lower() == "eviction_mask":
        if batch_size != 1:
            raise ValueError("Sparse prefill training currently requires batch_size=1.")
        model = getattr(denoiser, "module", denoiser)
        model_config = model.config
        sparse_prompt_len = int(question_length[0].item())
        selection_mode = (sparse_selection_mode or "uniform_sink").lower()
        if selection_mode == "draft_self_information":
            selected_context, _ = select_blocks_by_draft_self_information(
                model=model,
                prompt_ids=input_ids[0, :sparse_prompt_len],
                mask_token_id=mask_id,
                query_tokens=int(sparse_score_query_tokens),
                draft_tokens=int(sparse_score_draft_tokens),
                draft_partial_rounds=int(sparse_score_draft_partial_rounds),
                chunk_size=int(sparse_score_chunk_size),
                topk_chunks=int(sparse_score_topk_chunks),
            )
            block_size = int(sparse_block_size)
            sink = torch.arange(
                min(int(sparse_sink_tokens), sparse_prompt_len),
                device=device,
                dtype=torch.long,
            )
            query_start = max(0, sparse_prompt_len - int(sparse_score_query_tokens))
            query = torch.arange(
                query_start, sparse_prompt_len, device=device, dtype=torch.long
            )
            selected = torch.cat([sink, selected_context, query])
            selected_blocks = torch.unique(selected // block_size, sorted=True)
            offsets = torch.arange(block_size, device=device, dtype=torch.long)
            keep = (selected_blocks.unsqueeze(1) * block_size + offsets).flatten()
            keep = keep[keep < sparse_prompt_len]
            sparse_keep_indices = keep.view(1, 1, -1).expand(
                int(model_config.num_hidden_layers),
                int(model_config.num_key_value_heads),
                -1,
            ).clone()
        elif selection_mode == "uniform_sink":
            sparse_keep_indices = build_uniform_sink_keep_indices(
                prompt_length=sparse_prompt_len,
                num_layers=int(model_config.num_hidden_layers),
                num_kv_heads=int(model_config.num_key_value_heads),
                token_capacity=int(sparse_token_capacity),
                sink_tokens=int(sparse_sink_tokens),
                block_size=int(sparse_block_size),
                device=device,
            )
        else:
            raise ValueError(f"Unsupported sparse_selection_mode: {sparse_selection_mode}")
    elif (prefill_sparse_mode or "none").lower() not in {"none", "off", "0", ""}:
        raise ValueError(f"Unsupported prefill_sparse_mode: {prefill_sparse_mode}")

    noisy_batch = input_ids.masked_fill(masked_indices, mask_id)
    model_kwargs = {
        "sparse_keep_indices": sparse_keep_indices,
        "sparse_prompt_len": sparse_prompt_len,
    }
    logits = denoiser(noisy_batch, **model_kwargs).logits
    if enable_shift:
        logits = shift_logits(logits)
    inverse_mask_probability = mask_probabilities.unsqueeze(1).expand_as(input_ids)

    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                teacher_logits = denoiser(noisy_batch, **model_kwargs).logits
                if enable_shift:
                    teacher_logits = shift_logits(teacher_logits)
                teacher_probabilities = torch.softmax(teacher_logits, dim=-1)
        token_loss = F.cross_entropy(
            logits[masked_indices],
            teacher_probabilities[masked_indices],
            reduction='none',
        )
    else:
        token_loss = F.cross_entropy(
            logits[masked_indices],
            input_ids[masked_indices],
            reduction='none',
        )
    return {'loss': (token_loss / inverse_mask_probability[masked_indices]).mean()}

def compute_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    logits=denoiser(noisy_batch,attention_mask=attention_mask).logits
    logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                # denoiser.eval()
                ref_logits=denoiser(noisy_batch,attention_mask=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
                # denoiser.train()
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
def compute_normal_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    logits=denoiser(noisy_batch).logits
    logits=shift_logits(logits)
    token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 
import torch
def compute_llada_loss(
        input_ids,
        denoiser,
        question_length,
        mask_id,
        block_size,
        enable_shift,
        share_steps,
        self_align,
        feature_align,
        self_step,
        eos_id,
):
    mask_id=126336
    B, L = input_ids.shape
    noisy_batch, masked_indices, p_mask = forward_process_length(input_ids, mask_id=mask_id,prompt_lengths=question_length, block_size=block_size,eos_id=eos_id)
    token_positions = torch.arange(L, device=noisy_batch.device).expand(B, L)
    prompt_mask = (token_positions < question_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # prompt_mask = prompt_mask.to(torch.int64)
    noisy_batch = noisy_batch.to(denoiser.device)
    # print(noisy_batch)
    attention_mask=build_custom_float_attention_mask(noisy_batch, question_length, block_size, device=noisy_batch.device)
    attention_mask=attention_mask.to(torch.float16)
    # print(type(denoiser),noisy_batch.shape,attention_mask.shape)
    logits=denoiser(noisy_batch,attention_bias=attention_mask).logits
    # logits=shift_logits(logits)
    if self_align:
        with torch.no_grad():
            with denoiser.disable_adapter():
                # ref_model = denoiser
            # ref_model.eval()
            # print(type(ref_model))
                ref_logits=denoiser(noisy_batch,attention_bias=torch.zeros([1,1,noisy_batch.shape[1],noisy_batch.shape[1]],dtype=torch.float16,device=denoiser.device)).logits
                # ref_logits=shift_logits(ref_logits)
                ref_logits = torch.nn.functional.softmax(ref_logits, dim=-1)
        token_loss_2 = F.cross_entropy(logits[masked_indices], ref_logits[masked_indices], reduction='none') / p_mask[masked_indices]
        # print("token_loss_2",token_loss_2.shape)
    else:
        token_loss_2= F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    losses = {
                # 'loss_1': token_loss_2.mean() * 0,
                'loss': token_loss_2.mean(),
            }

    return losses 


def build_custom_float_attention_mask(input_ids, prompt_length, block_size, device=None):
    B,seq_len= input_ids.shape
    # 初始化为全 -inf
    attn_mask = torch.full((B,1,seq_len, seq_len), float('-inf'), dtype=torch.float32, device=device)
    # 1. Prompt部分：每个token可以注意整个prompt
    for i in range(B):
        attn_mask[i,:,:,:prompt_length[i]] = 0.0  # 允许所有 token 看 prompt

        # 2. 块划分：从 prompt_length 开始划分 block
        num_blocks = (seq_len - prompt_length[i] + block_size - 1) // block_size

        for b in range(num_blocks):
            block_start = prompt_length[i] + b * block_size
            # print(block_start,block_size,seq_len)
            block_end = min(block_start + block_size, seq_len)

            # 块内全注意
            attn_mask[i,:,block_start:block_end, block_start:block_end] = 0.0

            # 块之间因果注意（只能看前面块）
            for prev_b in range(b):
                prev_start = prompt_length[i] + prev_b * block_size
                prev_end = min(prev_start + block_size, seq_len)

                # 当前块可以看前面块
                attn_mask[i,:,block_start:block_end, prev_start:prev_end] = 0.0

    return attn_mask  # [seq_len, seq_len], float, 0.0 for allowed, -inf for disallowed
if __name__ == "__main__":
    seq_len = 10
    input_ids = torch.randint(0, 100, (2, seq_len))  # 示例输入
    block_size = 4
    prompt_length = torch.tensor([2, 4])  # 示例prompt长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attn_mask = build_custom_float_attention_mask(input_ids, prompt_length, block_size, device)
    print(attn_mask)