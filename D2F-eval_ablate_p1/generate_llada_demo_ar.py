import torch
import torch.nn.functional as F
import torch.distributions as dists
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import numpy as np
import random
import time
import os
from typing import List, Dict, Optional, Tuple, Iterator, Set
import gradio as gr
import gc

# Suppress some Hugging Face warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import necessary model classes
# Assuming these custom classes are in the correct path
from model_cache.llada.modeling_llada import LLaDAModelLM
from model_cache.llada.configuration_llada import LLaDAConfig

# --- Helper Functions (Unchanged) ---
def set_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed);
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    
def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    if dtype is None: dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size; block_end = min(prompt_length + (b + 1) * block_size, max_length)
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size; prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask
    
def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    end_pos = start_pos + input_length; total_length = cache_length + input_length
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, device=full_mask.device, dtype=full_mask.dtype)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted_mask
    
def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits
    
def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits
    
def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0: logits = logits / temperature
    if top_p is not None and top_p < 1: logits = top_p_logits(logits, top_p)
    if top_k is not None: logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except: initial_confidence, x0 = probs.max(dim=-1)
    else: initial_confidence, x0 = probs.max(dim=-1)
    confidence = initial_confidence.clone()
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
    if neg_entropy:
        epsilon = 1e-10
        confidence = torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return confidence, x0, initial_confidence


class D2FInference:
    CSS = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .model-header {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 5px;
        text-align: center;
    }
    .d2f-header {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .llama-header {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .stats-container {
        padding: 15px; 
        border: 1px solid #10B981; 
        border-radius: 8px; 
        background-color: #F0FDF4; 
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .output-textbox textarea {
        font-size: 1.5em !important;
        line-height: 1.6 !important;
        height: 70vh !important;
        overflow-y: auto !important;
    }
    """

    def __init__(self, **kwargs):
        print("Initializing D2F-LLaDA model...")
        self.device = torch.device(kwargs.get("device", "cuda:3") if torch.cuda.is_available() else "cpu")
        self.__dict__.update(kwargs)
        if self.dtype == "bfloat16" and torch.cuda.is_bf16_supported(): self.target_dtype = torch.bfloat16
        elif self.dtype == "float16": self.target_dtype = torch.float16
        else: self.target_dtype = torch.float32
        self._setup_model(self.pretrained_path, self.lora_path)
        print("D2F-LLaDA model and tokenizer setup complete.")

    def _setup_model(self, pretrained_path, lora_path):
        config = LLaDAConfig.from_pretrained(pretrained_path)
        self.model = LLaDAModelLM.from_pretrained(pretrained_path, config=config, torch_dtype=self.target_dtype).eval()
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template(self, prompt):
        chat_history = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

    def _update_block_completion_states(self, block_states, decoded_token_threshold):
        for block_id in sorted(block_states.keys()):
            decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
            if block_states[block_id]['total_masks'] > 0:
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= decoded_token_threshold:
                    if (next_block_id := block_id + 1) in block_states:
                        block_states[next_block_id]['is_complete'] = True

    @torch.inference_mode()
    def stream(
        self,
        prompt_text: str,
        max_new_tokens: int,
        block_size: int,
        block_add_threshold: float,
        decoded_token_threshold: float,
        skip_threshold: float
    ) -> Iterator[Tuple[str, str]]:
        
        start_time = time.time()
        
        input_ids = self.tokenizer(self._apply_chat_template(prompt_text), return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        
        full_attention_mask = create_full_block_attention_mask(prompt_length, self.max_length, block_size, self.device, self.target_dtype)
        x_t = input_ids
        block_states = {0: {'start_pos': 0, 'end_pos': prompt_length, 'mask_count': 0, 'total_masks': prompt_length, 'state': 'to_cache', 'is_complete': True}}
        past_key_values, current_blocks, step, eos_detected, cache_length = None, 0, 0, False, 0
        
        yield "", None

        tokens_generated = 0
        
        while True:
            step += 1
            updated_block_ids = set()

            if len(block_states) - 1 < (max_new_tokens // block_size) and not eos_detected:
                last_block_id = max(block_states.keys())
                progress_ratio = (block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks'] if block_states[last_block_id]['total_masks'] > 0 else 1.0
                if progress_ratio >= block_add_threshold:
                    new_block_id = last_block_id + 1; new_start_pos = x_t.shape[1]
                    if new_start_pos + block_size <= self.max_length:
                        x_t = torch.cat([x_t, torch.full((1, block_size), self.mask_token_id, device=self.device, dtype=torch.long)], dim=1)
                        block_states[new_block_id] = {'start_pos': new_start_pos, 'end_pos': new_start_pos + block_size, 'mask_count': block_size, 'total_masks': block_size, 'state': 'active', 'is_complete': False}
                        current_blocks += 1

            self._update_block_completion_states(block_states, decoded_token_threshold)
            if (x_t == self.mask_token_id).sum() == 0 and current_blocks == 0: break

            blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
            update_kvcache = 0
            if blocks_to_cache:
                start_pos, end_pos = block_states[min(blocks_to_cache)]['start_pos'], block_states[max(blocks_to_cache)]['end_pos']
                update_kvcache = end_pos - start_pos; input_seq, process_start_pos = x_t[:, start_pos:], start_pos
            else:
                active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active' and state['start_pos'] >= cache_length]
                if not active_blocks: break
                start_pos = min(block_states[bid]['start_pos'] for bid in active_blocks); input_seq, process_start_pos = x_t[:, start_pos:], start_pos
            
            if input_seq.shape[1] == 0: break

            attention_mask = extract_attention_mask(full_mask=full_attention_mask, 
                                                   start_pos=process_start_pos, 
                                                   input_length=input_seq.shape[1], 
                                                   cache_length=cache_length)
            
            outputs = self.model(input_seq, 
                                 attention_bias=attention_mask, 
                                 past_key_values=past_key_values, 
                                 use_cache=True, 
                                 update_kvcache=update_kvcache + cache_length)
            
            if update_kvcache > 0:
                past_key_values = outputs.past_key_values
                for bid in blocks_to_cache:
                    block_states[bid]['state'] = 'in_cache'

            blocks_to_deactivate = []
            for block_id, state in block_states.items():
                if state['state'] != 'active':
                    continue
                    
                block_mask_locs = (x_t[0, state['start_pos']:state['end_pos']] == self.mask_token_id).nonzero().squeeze(-1)
                
                if block_mask_locs.numel() == 0:
                    blocks_to_deactivate.append(block_id)
                    continue
                    
                logit_offset = state['start_pos'] - process_start_pos
                block_mask_logits = outputs.logits[:, logit_offset + block_mask_locs, :]
                _, x0, initial_confidence = sample_tokens(block_mask_logits.squeeze(0), self.temperature, self.top_p, self.top_k)
                all_indices = (initial_confidence > skip_threshold).nonzero().squeeze(-1)
                
                if state['is_complete'] and all_indices.numel() == 0 and block_mask_logits.numel() > 0:
                    all_indices = torch.tensor([torch.argmax(initial_confidence)], device=self.device)

                if all_indices.numel() > 0:
                    updated_block_ids.add(block_id)
                    positions_to_update = state['start_pos'] + block_mask_locs[all_indices]
                    x_t[0, positions_to_update] = x0[all_indices]
                    state['mask_count'] -= all_indices.numel()
                    tokens_generated += all_indices.numel()
                    
                    if self.tokenizer.eos_token_id in x0[all_indices]:
                        eos_detected = True
                        
                if state['mask_count'] == 0:
                    blocks_to_deactivate.append(block_id)
            
            for bid in blocks_to_deactivate:
                if block_states[bid]['state'] == 'active' and all(block_states.get(i, {}).get('state') != 'active' for i in range(bid)):
                    block_states[bid]['state'] = 'to_cache'
                    current_blocks -= 1
                    
            if update_kvcache > 0:
                cache_length += update_kvcache
                
            generated_ids = x_t[0, prompt_length:]
            valid_ids = generated_ids[generated_ids != self.mask_token_id]
            live_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            
            yield live_text, None
        
        total_time = time.time() - start_time
        final_generated_ids = x_t[0, prompt_length:]
        eos_positions = (final_generated_ids == self.tokenizer.eos_token_id).nonzero()
        
        if eos_positions.numel() > 0:
            final_generated_ids = final_generated_ids[:eos_positions[0, 0] + 1]

        final_text = self.tokenizer.decode(final_generated_ids, skip_special_tokens=True)
        
        tokens_incl_eos = len(final_generated_ids)
        tokens_per_second = tokens_incl_eos / total_time if total_time > 0 else 0
        
        stats = {
            "total_time": total_time,
            "tokens_generated": tokens_incl_eos,
            "tokens_per_second": tokens_per_second
        }
        
        if past_key_values is not None:
            del past_key_values
        del full_attention_mask
        torch.cuda.empty_cache()
        
        yield final_text, stats


class LlamaInference:
    def __init__(self, **kwargs):
        print("Initializing LLaMA model...")
        self.device = torch.device(kwargs.get("device", "cuda:4") if torch.cuda.is_available() else "cpu")
        self.__dict__.update(kwargs)
        self._setup_model(self.model_id)
        print("LLaMA model and tokenizer setup complete.")
    
    def _setup_model(self, model_id):
        print(f"Loading LLaMA model {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=self.device
        ).eval()
        
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "</s>"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_chat_template(self, prompt):
        chat_history = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    
    @torch.inference_mode()
    def stream(
        self, 
        prompt_text: str, 
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = None
    ) -> Iterator[Tuple[str, str]]:
        
        start_time = time.time()
        
        formatted_prompt = self._apply_chat_template(prompt_text)
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        
        yield "", None
        
        tokens_generated = 0
        current_input_ids = input_ids.clone()
        
        for i in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(current_input_ids, use_cache=True)
                
                next_token_logits = outputs.logits[:, -1, :]
                
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    if top_p is not None and top_p < 1:
                        next_token_logits = top_p_logits(next_token_logits, top_p)
                    if top_k is not None:
                        next_token_logits = top_k_logits(next_token_logits, top_k)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            tokens_generated += 1
            
            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                break
                
            generated_text = self.tokenizer.decode(
                current_input_ids[0, prompt_length:], 
                skip_special_tokens=True
            )
            
            yield generated_text, None
            
            del outputs
        
        total_time = time.time() - start_time
        tokens_per_second = tokens_generated / total_time if total_time > 0 else 0
        
        final_text = self.tokenizer.decode(current_input_ids[0, prompt_length:], skip_special_tokens=True)
        
        stats = {
            "total_time": total_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second
        }
        
        del current_input_ids
        torch.cuda.empty_cache()
        
        yield final_text, stats


# --- Comparison Helper Functions ---
def create_comparison_html(d2f_results, llama_results):
    d_tokens = d2f_results["tokens_generated"]
    d_time = d2f_results["total_time"]
    d_tokens_per_sec = d2f_results["tokens_per_second"]
    
    a_tokens = llama_results["tokens_generated"]
    a_time = llama_results["total_time"]
    a_tokens_per_sec = llama_results["tokens_per_second"]
    
    if a_tokens_per_sec > 0:
        speedup = d_tokens_per_sec / a_tokens_per_sec
    else:
        speedup = 0
    
    comparison_html = f"""
    <div class="stats-container" style="background-color: #F9FAFB; border-color: #6366F1;">
        <h3>⚡ Performance Comparison</h3>
        <table style="width:100%; text-align: left; border-collapse: collapse;">
            <tr style="background-color: #EEF2FF;">
                <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
                <th style="padding: 8px; border: 1px solid #ddd;">D2F-LLaDA-Instruct-8B</th>
                <th style="padding: 8px; border: 1px solid #ddd;">LLaMA3-Instruct-8B</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Difference</th>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">Total tokens</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{d_tokens}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{a_tokens}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">-</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">Generation time</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{d_time:.2f}s</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{a_time:.2f}s</td>
                <td style="padding: 8px; border: 1px solid #ddd;">
                    {"D2F-LLaDA is " + f"{(a_time/d_time):.1f}x faster" if d_time > 0 and d_time < a_time else "LLaMA3 is " + f"{(d_time/a_time):.1f}x faster"}
                </td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;">Tokens per second</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{d_tokens_per_sec:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{a_tokens_per_sec:.2f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">
                    {"D2F-LLaDA is " + f"{speedup:.1f}x faster" if speedup > 1 else "LLaMA3 is " + f"{(1/speedup if speedup > 0 else 0):.1f}x faster"}
                </td>
            </tr>
        </table>
    </div>
    """
    
    return comparison_html


def create_stats_html(model_name, results):
    stats_html = f"""
    <div class="stats-container">
        <h3>✓ {model_name} Generation Complete</h3>
        <ul>
            <li><b>Total time:</b> {results["total_time"]:.2f} seconds</li>
            <li><b>Tokens generated:</b> {results["tokens_generated"]}</li>
            <li><b>Tokens per second:</b> {results["tokens_per_second"]:.2f}</li>
        </ul>
    </div>
    """
    
    return stats_html


# --- Main Interface ---
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    
    torch.cuda.empty_cache()
    
    d2f_config = {
        "pretrained_path": "GSAI-ML/LLaDA-8B-Instruct",
        "lora_path": "SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora",
        "device": "cuda:0",
        "dtype": "bfloat16", 
        "max_length": 4096,
        "temperature": 0.0, 
        "top_p": None, 
        "top_k": None, 
        "mask_token_id": 126336,
        "sampling_strategy": "default",
    }
    
    llama_config = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "device": "cuda:1",
    }
    
    set_seed(42)
    
    d2f_engine = D2FInference(**d2f_config)
    llama_engine = LlamaInference(**llama_config)

    with gr.Blocks(css=D2FInference.CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 D2F-LLaDA vs LLaMA3: Speed Comparison")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Enter your question", 
                    placeholder="Example: Natalia sold clips to...", 
                    lines=5
                )
                generate_button = gr.Button("🚀 Run Speed Comparison", variant="primary")
                
                with gr.Accordion("⚙️ D2F-LLaDA Parameter Settings", open=True):
                    with gr.Row():
                        max_new_tokens_slider = gr.Slider(
                            minimum=64, maximum=2048, value=1024, step=64, 
                            label="Max Tokens to Generate"
                        )
                        block_size_slider = gr.Slider(
                            minimum=16, maximum=128, value=32, step=16, 
                            label="Block Size"
                        )
                    with gr.Row():
                        block_add_thresh_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.1, step=0.05, 
                            label="Block Add Threshold"
                        )
                        decoded_token_thresh_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05, 
                            label="Decoding Completion Threshold"
                        )
                        skip_thresh_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.9, step=0.01, 
                            label="Skip Threshold"
                        )
                
                comparison_output = gr.HTML(label="Performance Comparison", elem_id="comparison-container")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='model-header d2f-header'>✨ D2F-LLaDA-Instruct-8B (Parallel Decoding)</div>")
                d2f_output = gr.Textbox(
                    label="D2F-LLaDA Output", 
                    interactive=False, 
                    elem_classes=["output-textbox"]
                )
                d2f_status = gr.HTML(label="D2F-LLaDA Stats")
            
            with gr.Column(scale=1):
                gr.HTML("<div class='model-header llama-header'>🔄 LLaMA3-Instruct-8B (Standard)</div>")
                llama_output = gr.Textbox(
                    label="LLaMA3 Output", 
                    interactive=False, 
                    elem_classes=["output-textbox"]
                )
                llama_status = gr.HTML(label="LLaMA3 Stats")

        gr.Examples(
            examples=[
                ["Solve the equation x² - 6x + 8 = 0. First, explain what a quadratic equation is and why it can have up to two solutions. Then solve this equation using three different methods: factoring, completing the square, and the quadratic formula. For each method, explain the mathematical reasoning behind it, show all steps in detail, and discuss when this particular method is most useful. Finally, verify your solutions by substituting them back into the original equation.", 1024, 32, 0.1, 0.55, 0.9],
                ["A circular swimming pool has a diameter of 8 meters. Calculate the pool's circumference and area. First, explain the relationship between diameter, radius, circumference, and area of a circle, including the role of π in these formulas. Then perform the calculations using π ≈ 3.14159. Next, estimate how much water (in cubic meters) would be needed to fill this pool if it has a uniform depth of 1.5 meters. Finally, calculate how much it would cost to fill this pool if water costs $2.50 per cubic meter. Show all steps and include appropriate units in your answer.", 1024, 32, 0.1, 0.5, 0.9],
                ["A movie theater offers a loyalty card that costs $15 and gives a 15% discount on all tickets. If a regular movie ticket costs $10, how many tickets would you need to buy to make the loyalty card worthwhile? First, explain the concept of a break-even point. Then set up an equation to find when the total cost with the card equals the total cost without the card. Solve this equation step by step, showing all your work. Finally, interpret your answer in the context of the problem.", 1024, 32, 0.1, 0.5, 0.9],
            ],
            inputs=[
                prompt_input, max_new_tokens_slider, block_size_slider, 
                block_add_thresh_slider, decoded_token_thresh_slider, skip_thresh_slider
            ],
            label="Examples (Math Problems)"
        )
        
        def run_models_streaming(
            prompt_text,
            max_new_tokens,
            block_size,
            block_add_threshold,
            decoded_token_threshold,
            skip_threshold
        ):
            torch.cuda.empty_cache()
            
            d2f_generator = d2f_engine.stream(
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                block_add_threshold=block_add_threshold,
                decoded_token_threshold=decoded_token_threshold,
                skip_threshold=skip_threshold
            )
            
            llama_generator = llama_engine.stream(
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens
            )
            
            d2f_text = ""
            llama_text = ""
            d2f_stats = None
            llama_stats = None
            
            yield d2f_text, llama_text, "", "", ""
            
            d2f_done = False
            llama_done = False
            
            while not (d2f_done and llama_done):
                if not d2f_done:
                    try:
                        new_d2f_text, new_d2f_stats = next(d2f_generator)
                        d2f_text = new_d2f_text
                        if new_d2f_stats is not None:
                            d2f_stats = new_d2f_stats
                            d2f_done = True
                    except StopIteration:
                        d2f_done = True
                
                if not llama_done:
                    try:
                        new_llama_text, new_llama_stats = next(llama_generator)
                        llama_text = new_llama_text
                        if new_llama_stats is not None:
                            llama_stats = new_llama_stats
                            llama_done = True
                    except StopIteration:
                        llama_done = True
                
                d2f_status_html = create_stats_html("D2F-LLaDA", d2f_stats) if d2f_stats else ""
                llama_status_html = create_stats_html("LLaMA3", llama_stats) if llama_stats else ""
                
                comparison = ""
                if d2f_done and llama_done and d2f_stats and llama_stats:
                    comparison = create_comparison_html(d2f_stats, llama_stats)
                
                yield d2f_text, llama_text, d2f_status_html, llama_status_html, comparison
        
        # MODIFICATION: Removed the _js parameter from here
        generate_button.click(
            fn=run_models_streaming,
            inputs=[
                prompt_input, max_new_tokens_slider, block_size_slider,
                block_add_thresh_slider, decoded_token_thresh_slider, skip_thresh_slider
            ],
            outputs=[
                d2f_output, llama_output,
                d2f_status, llama_status,
                comparison_output
            ]
        )

        # MODIFICATION: Added a hidden HTML component with a script for auto-scrolling
        # This method is compatible with older Gradio versions.
        gr.HTML(
            """
            <script>
                function_to_run = () => {
                    const textboxes = document.querySelectorAll('.output-textbox textarea');
                    textboxes.forEach(textbox => {
                        textbox.scrollTop = textbox.scrollHeight;
                    });
                }
                // Run the function every 250ms to ensure autoscrolling
                setInterval(function_to_run, 250);
            </script>
            """,
            visible=False
        )

    demo.queue().launch(share=True)