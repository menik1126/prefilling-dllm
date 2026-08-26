import torch
import torch.nn.functional as F
import torch.distributions as dists
from peft import PeftModel, PeftConfig
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

    return attn_mask 
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0
# def generate(model,prompt,block_size,max_length,mask_id):
# def generate(model, prompt, block_size, max_length, mask_id, eos_token_id=None):
#     device = prompt.device
#     output = prompt.clone()

#     while output.shape[1] < max_length:
#         # 添加一个 block 的 mask
#         mask_block = torch.full((1, block_size), mask_id, dtype=torch.long, device=device)
#         input_ids = torch.cat([output, mask_block], dim=1)
#         attention_mask = build_custom_float_attention_mask(input_ids, torch.tensor([[prompt.shape[1]]]), block_size, device=device)
#         attention_mask = attention_mask.to(torch.bfloat16)
#         for i in range(block_size):
def generate_block(denoiser, block_size, mask_id,tokenizer,device):
    denoiser.eval()
    question = 'please give me a code about transformer model'
    # prompt = tokenizer(question)['input_ids']
    # prompt = torch.tensor(prompt).to(accelerator.device).unsqueeze(0)
    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    ).input_ids
    prompt = prompt.to(device)

    mask_id = 151666
    gen_len = (384 - prompt.shape[1])//block_size
    print(gen_len)
    temperature = 0.2
    top_p = 0.95
    with torch.inference_mode():
        for i in range(gen_len):
            if i==0:
                x_t = torch.cat([prompt, torch.tensor([[mask_id]*block_size]).to(device)], dim=1)
            else:
                x_t = torch.cat([x_t, torch.tensor([[mask_id]*block_size]).to(device)], dim=1)
            attention_mask = build_custom_float_attention_mask(x_t, torch.tensor([[prompt.shape[1]]]), block_size, device=device)
            attention_mask = attention_mask.to(torch.bfloat16)
            for n in range(block_size):
                mask_index = (x_t == mask_id)
                if mask_index.sum() == 0:
                    break
                logits =denoiser(x_t, attention_mask=attention_mask).logits
                logits = shift_logits(logits)
                mask_logits = logits[mask_index]
                confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=None, neg_entropy=True)
                number_transfer_tokens = 1
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + mask_id
                x0_[transfer_index] = x0[transfer_index].clone()
                x_t[mask_index] = x0_
            answer = tokenizer.batch_decode(x_t[:, prompt.shape[1]:], skip_special_tokens=False)[0]
            print(answer)
    answer = tokenizer.batch_decode(x_t[:, prompt.shape[1]:], skip_special_tokens=False)[0]
    print(answer)

if __name__ == "__main__":
    config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")