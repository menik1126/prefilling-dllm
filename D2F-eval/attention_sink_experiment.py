"""
Attention Sink Analysis for Dream-7B (Figure 7 verification).

Memory-efficient: hooks capture Q/K after RoPE, manually compute
response→prefix attention with 4 KV heads to avoid OOM.

Setup: 8K context, YaRN x4, bf16, single A800.
"""
import sys
sys.path.insert(0, "/home/ma-user/work/Discrete-Diffusion-Forcing/D2F-eval")

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

MODEL_PATH = "/home/ma-user/work/models/Dream-v0-Base-7B"
LORA_PATH = "/home/ma-user/work/models/D2F_Dream_Base_7B_Lora"

CHUNK_SIZE = 1024
NUM_CHUNKS = 8
CONTEXT_LEN = CHUNK_SIZE * NUM_CHUNKS  # 8192
RESPONSE_LEN = 64
MASK_TOKEN_ID = 151666
NUM_LAYERS = 28
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128  # 3584 / 28
DEVICE = "cuda"
DTYPE = torch.bfloat16


def load_model():
    from model_cache.dream.model_dream import DreamModel
    from model_cache.dream.configuration_dream import DreamConfig

    config = DreamConfig.from_pretrained(MODEL_PATH)
    config.rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 2048,
        "rope_type": "yarn",
    }

    model = DreamModel.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=DTYPE,
        trust_remote_code=False,
    ).eval().to(DEVICE)

    peft_config = PeftConfig.from_pretrained(LORA_PATH)
    model = PeftModel.from_pretrained(model, LORA_PATH).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def build_vanilla_input(tokenizer):
    """Vanilla Dream: BOS + flat prefix (8K tokens) + masked response."""
    bos_id = tokenizer.bos_token_id
    text = "The study of attention mechanisms in large language models has become a central topic in modern NLP research. " * 200
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < CONTEXT_LEN - 1:
        tokens = tokens * ((CONTEXT_LEN - 1) // len(tokens) + 1)
    prefix_tokens = [bos_id] + tokens[:CONTEXT_LEN - 1]
    response_tokens = [MASK_TOKEN_ID] * RESPONSE_LEN
    input_ids = torch.tensor([prefix_tokens + response_tokens], dtype=torch.long, device=DEVICE)
    prefix_len = len(prefix_tokens)
    bos_positions = [0]
    return input_ids, prefix_len, bos_positions


def build_chunked_input(tokenizer):
    """Prefilling-dLLM: 8 chunks with BOS delimiters + BOS before masked response."""
    bos_id = tokenizer.bos_token_id
    text = "The study of attention mechanisms in large language models has become a central topic in modern NLP research. " * 200
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < CONTEXT_LEN:
        tokens = tokens * ((CONTEXT_LEN) // len(tokens) + 1)

    all_tokens = []
    bos_positions = []
    chunk_content_size = CHUNK_SIZE - 1
    for i in range(NUM_CHUNKS):
        bos_positions.append(len(all_tokens))
        all_tokens.append(bos_id)
        start = i * chunk_content_size
        end = start + chunk_content_size
        all_tokens.extend(tokens[start:end])

    prefix_len = len(all_tokens)
    # Response: BOS + masked tokens
    bos_positions.append(len(all_tokens))
    all_tokens.append(bos_id)
    all_tokens.extend([MASK_TOKEN_ID] * (RESPONSE_LEN - 1))

    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=DEVICE)
    return input_ids, prefix_len, bos_positions


class EfficientSinkCollector:
    """
    Hook into each attention layer to capture Q, K after RoPE projection.
    Manually compute response->prefix attention for 4 KV head groups,
    avoiding the full NxN matrix.
    """

    def __init__(self, prefix_len, bos_positions, response_start):
        self.prefix_len = prefix_len
        self.bos_positions = [p for p in bos_positions if p < prefix_len]
        self.response_start = response_start
        self.layer_results = {}
        self.hooks = []

    def _make_hook(self, layer_idx):
        """Hook after Q/K projections + RoPE to extract attention pattern."""
        prefix_len = self.prefix_len
        response_start = self.response_start
        bos_positions = self.bos_positions
        results = self.layer_results

        def hook_fn(module, input, output):
            # We intercept after the full attention forward.
            # Instead, let's hook into the attention module and manually compute Q*K^T
            # for just the response->prefix slice.
            pass

        return hook_fn

    def register_qk_hooks(self, model):
        """Register hooks that capture Q and K after projection + RoPE."""
        base_model = model.base_model.model if hasattr(model, 'base_model') else model
        layers = base_model.model.layers

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn

            # We'll monkey-patch the forward to capture Q, K after RoPE
            original_forward = attn.forward
            collector = self

            def make_wrapper(orig_fwd, li):
                def wrapper(*args, **kwargs):
                    # Call original forward (uses FlexAttention/SDPA, no OOM)
                    result = orig_fwd(*args, **kwargs)

                    # Now manually compute Q*K for response->prefix slice
                    hidden = args[0] if len(args) > 0 else kwargs.get('hidden_states')
                    bsz, seq_len, _ = hidden.size()

                    with torch.no_grad():
                        # Get the actual module
                        actual_attn = layers[li].self_attn
                        if hasattr(actual_attn, 'base_layer'):
                            base_attn = actual_attn.base_layer
                        else:
                            base_attn = actual_attn

                        q = base_attn.q_proj(hidden)  # [1, seq, num_heads*head_dim]
                        k = base_attn.k_proj(hidden)  # [1, seq, num_kv_heads*head_dim]

                        q = q.view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
                        k = k.view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

                        # Apply RoPE
                        position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
                        cos, sin = base_attn.rotary_emb(k, position_ids)
                        from model_cache.dream.model_dream import apply_rotary_pos_emb
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)

                        # Only compute response queries x prefix keys
                        # q_resp: [1, num_heads, resp_len, head_dim]
                        q_resp = q[:, :, collector.response_start:, :]
                        # k_prefix: [1, num_kv_heads, prefix_len, head_dim]
                        k_prefix = k[:, :, :collector.prefix_len, :]

                        # GQA: group queries by KV head
                        groups = NUM_HEADS // NUM_KV_HEADS  # 7
                        resp_len = q_resp.shape[2]
                        pref_len = k_prefix.shape[2]

                        # Compute attention scores per KV group, average across groups
                        avg_attn_profile = torch.zeros(pref_len, device=hidden.device, dtype=torch.float32)

                        for g in range(NUM_KV_HEADS):
                            # Query heads in this group
                            q_group = q_resp[:, g*groups:(g+1)*groups, :, :]  # [1, groups, resp_len, head_dim]
                            k_group = k_prefix[:, g:g+1, :, :]  # [1, 1, pref_len, head_dim]

                            # Attention score: [1, groups, resp_len, pref_len]
                            scores = torch.matmul(q_group, k_group.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
                            # Softmax over prefix positions
                            attn_probs = torch.softmax(scores.float(), dim=-1)
                            # Average: [pref_len]
                            avg_attn_profile += attn_probs.mean(dim=(0, 1, 2))

                            del scores, attn_probs, q_group, k_group

                        avg_attn_profile /= NUM_KV_HEADS

                        total = avg_attn_profile.sum().item()
                        if total > 1e-10:
                            first1 = avg_attn_profile[0].item() / total * 100
                            first5 = avg_attn_profile[:min(5, pref_len)].sum().item() / total * 100
                            bos_mass = sum(avg_attn_profile[p].item() for p in collector.bos_positions) / total * 100
                        else:
                            first1, first5, bos_mass = 0.0, 0.0, 0.0

                        collector.layer_results[li] = (first1, first5, bos_mass)

                        del q, k, q_resp, k_prefix, avg_attn_profile

                    return result
                return wrapper

            attn.forward = make_wrapper(original_forward, layer_idx)
            self.hooks.append((attn, original_forward))

    def remove(self):
        for attn, orig_fwd in self.hooks:
            attn.forward = orig_fwd
        self.hooks.clear()

    def get_results(self):
        first1 = np.array([self.layer_results.get(i, (0,0,0))[0] for i in range(NUM_LAYERS)])
        first5 = np.array([self.layer_results.get(i, (0,0,0))[1] for i in range(NUM_LAYERS)])
        all_bos = np.array([self.layer_results.get(i, (0,0,0))[2] for i in range(NUM_LAYERS)])
        return {"first1": first1, "first5": first5, "all_bos": all_bos}


def run_experiment(model, tokenizer, input_ids, prefix_len, bos_positions, label, steps=3):
    """Run denoising steps and collect attention sink measurements."""
    response_start = prefix_len
    x = input_ids.clone()

    all_step_results = []
    for step in range(steps):
        print(f"  [{label}] Denoising step {step+1}/{steps} ...", flush=True)

        collector = EfficientSinkCollector(prefix_len, bos_positions, response_start)
        collector.register_qk_hooks(model)

        with torch.no_grad():
            outputs = model(
                input_ids=x,
                output_attentions=False,
                return_dict=True,
            )

        collector.remove()
        result = collector.get_results()
        all_step_results.append(result)

        # Unmask tokens for next step
        logits = outputs.logits
        shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_positions = (x[0, response_start:] == MASK_TOKEN_ID)
        if mask_positions.any():
            mask_logits = shifted_logits[0, response_start:][mask_positions]
            probs = torch.softmax(mask_logits.float(), dim=-1)
            confidence, predicted = probs.max(dim=-1)
            n_unmask = max(1, int(mask_positions.sum().item() * 0.4))
            topk_conf, topk_idx = confidence.topk(min(n_unmask, len(confidence)))
            mask_idx = mask_positions.nonzero(as_tuple=True)[0]
            for idx in topk_idx:
                pos = mask_idx[idx].item() + response_start
                x[0, pos] = predicted[idx]

        del outputs, logits, shifted_logits
        torch.cuda.empty_cache()

    avg_result = {
        "first1": np.mean([r["first1"] for r in all_step_results], axis=0),
        "first5": np.mean([r["first5"] for r in all_step_results], axis=0),
        "all_bos": np.mean([r["all_bos"] for r in all_step_results], axis=0),
    }
    return avg_result


def plot_results(vanilla_result, ours_result, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(NUM_LAYERS)

    # Left: Per-layer
    ax = axes[0]
    width = 0.15
    ax.bar(x - 2*width, vanilla_result["first1"], width, label="Vanilla: first-1", color="#3498db", alpha=0.85)
    ax.bar(x - 1*width, vanilla_result["first5"], width, label="Vanilla: first-5", color="#2ecc71", alpha=0.85)
    ax.bar(x + 0*width, ours_result["first1"], width, label="Ours: first-1", color="#e74c3c", alpha=0.85)
    ax.bar(x + 1*width, ours_result["first5"], width, label="Ours: first-5", color="#f39c12", alpha=0.85)
    ax.bar(x + 2*width, ours_result["all_bos"], width, label="Ours: all BOS", color="#9b59b6", alpha=0.85)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Attention Ratio (%)", fontsize=11)
    ax.set_title("Per-layer attention ratio absorbed by special tokens", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(x[::2])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # Right: Summary
    ax2 = axes[1]
    categories = ["First-1 token", "First-5 tokens", "All BOS tokens"]
    vanilla_vals = [vanilla_result["first1"].mean(), vanilla_result["first5"].mean(), vanilla_result["all_bos"].mean()]
    ours_vals = [ours_result["first1"].mean(), ours_result["first5"].mean(), ours_result["all_bos"].mean()]

    x2 = np.arange(len(categories))
    w2 = 0.3
    bars1 = ax2.bar(x2 - w2/2, vanilla_vals, w2, label="Vanilla Dream", color="#3498db", alpha=0.85)
    bars2 = ax2.bar(x2 + w2/2, ours_vals, w2, label="Prefilling-dLLM", color="#e74c3c", alpha=0.85)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Avg Attention Ratio (%)", fontsize=11)
    ax2.set_title("Average attention sink ratio across layers", fontsize=11)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=9)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path + ".pdf", bbox_inches="tight")
    plt.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}.pdf and .png")


def main():
    print("Loading model...", flush=True)
    model, tokenizer = load_model()
    print(f"BOS token id: {tokenizer.bos_token_id}")
    print(f"GPU memory after loading: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Vanilla Dream
    print("\n=== Vanilla Dream (flat 8K context) ===", flush=True)
    v_input, v_prefix_len, v_bos = build_vanilla_input(tokenizer)
    print(f"Input shape: {v_input.shape}, prefix_len: {v_prefix_len}, BOS positions: {v_bos}")
    vanilla_result = run_experiment(model, tokenizer, v_input, v_prefix_len, v_bos, "Vanilla", steps=3)
    print(f"\nVanilla avg first-1: {vanilla_result['first1'].mean():.4f}%")
    print(f"Vanilla avg first-5: {vanilla_result['first5'].mean():.4f}%")
    print(f"Vanilla avg all-BOS: {vanilla_result['all_bos'].mean():.4f}%")
    del v_input; torch.cuda.empty_cache()

    # Prefilling-dLLM
    print("\n=== Prefilling-dLLM (8 chunks x 1024, BOS delimiters) ===", flush=True)
    o_input, o_prefix_len, o_bos = build_chunked_input(tokenizer)
    print(f"Input shape: {o_input.shape}, prefix_len: {o_prefix_len}, BOS positions: {o_bos}")
    ours_result = run_experiment(model, tokenizer, o_input, o_prefix_len, o_bos, "Ours", steps=3)
    print(f"\nOurs avg first-1: {ours_result['first1'].mean():.4f}%")
    print(f"Ours avg first-5: {ours_result['first5'].mean():.4f}%")
    print(f"Ours avg all-BOS: {ours_result['all_bos'].mean():.4f}%")

    # Plot
    save_path = "/home/ma-user/work/6a03da8f48b03f2bd77037da/latex/figures/attention_sink_verification"
    plot_results(vanilla_result, ours_result, save_path)

    # Per-layer details
    print("\n=== Per-layer details ===")
    print(f"{'Layer':>5} | {'V-first1':>8} {'V-first5':>8} {'V-BOS':>8} | {'O-first1':>8} {'O-first5':>8} {'O-BOS':>8}")
    print("-" * 70)
    for i in range(NUM_LAYERS):
        print(f"{i:5d} | {vanilla_result['first1'][i]:7.4f}% {vanilla_result['first5'][i]:7.4f}% {vanilla_result['all_bos'][i]:7.4f}% | "
              f"{ours_result['first1'][i]:7.4f}% {ours_result['first5'][i]:7.4f}% {ours_result['all_bos'][i]:7.4f}%")


if __name__ == "__main__":
    main()
