#!/usr/bin/env python3
"""
Baseline for TinyLlama full forward pass.
Generates reference logits and sampled token for C++ comparison.
"""

import mlx.core as mx
from pathlib import Path
import json

def forward_pass():
    model_path = Path("/Users/dex/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")

    with open(model_path / "config.json") as f:
        config = json.load(f)

    hidden_size   = config["hidden_size"]        # 2048
    n_heads       = config["num_attention_heads"] # 32
    n_kv_heads    = config["num_key_value_heads"] # 4
    n_layers      = config["num_hidden_layers"]   # 22
    vocab_size    = config["vocab_size"]          # 32000
    intermediate  = config["intermediate_size"]   # 5632
    head_dim      = hidden_size // n_heads        # 64
    n_repeats     = n_heads // n_kv_heads         # 8
    rope_theta    = config.get("rope_theta", 10000.0)
    rms_eps       = config.get("rms_norm_eps", 1e-5)
    group_size    = config["quantization"]["group_size"]  # 64
    bits          = config["quantization"]["bits"]        # 4

    print(f"Model: {n_layers} layers, hidden={hidden_size}, heads={n_heads}/{n_kv_heads}")
    print(f"       vocab={vocab_size}, intermediate={intermediate}")

    weights = mx.load(str(model_path / "model.safetensors"))

    def qmatmul(x, prefix):
        w = weights[f"{prefix}.weight"]
        s = weights[f"{prefix}.scales"]
        b = weights[f"{prefix}.biases"]
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group_size, bits=bits)

    # Fixed token IDs for reproducible baseline
    # "What is the capital of France?" (without BOS — matches our tokenizer)
    token_ids = [1, 1724, 338, 278, 7483, 310, 3444, 29973]
    seq_len = len(token_ids)
    print(f"\nInput token_ids: {token_ids} (seq_len={seq_len})")

    # Embedding
    embed_w = weights["model.embed_tokens.weight"]
    hidden = embed_w[token_ids, :]  # [seq_len, hidden_size]
    hidden = hidden.astype(mx.bfloat16)
    mx.eval(hidden)
    print(f"After embedding: {hidden.shape}")

    # Transformer blocks
    for layer_idx in range(n_layers):
        p = f"model.layers.{layer_idx}"

        # Pre-attention RMSNorm
        normed = mx.fast.rms_norm(hidden, weights[f"{p}.input_layernorm.weight"], rms_eps)

        # Attention (Q, K, V projections)
        q = qmatmul(normed, f"{p}.self_attn.q_proj")  # [seq, n_heads*head_dim]
        k = qmatmul(normed, f"{p}.self_attn.k_proj")  # [seq, n_kv_heads*head_dim]
        v = qmatmul(normed, f"{p}.self_attn.v_proj")  # [seq, n_kv_heads*head_dim]

        # Reshape for multi-head
        q = q.reshape(seq_len, n_heads,    head_dim)
        k = k.reshape(seq_len, n_kv_heads, head_dim)
        v = v.reshape(seq_len, n_kv_heads, head_dim)

        # RoPE
        q = mx.fast.rope(q, head_dim, traditional=False, base=rope_theta, scale=1.0, offset=0)
        k = mx.fast.rope(k, head_dim, traditional=False, base=rope_theta, scale=1.0, offset=0)

        # GQA: expand KV heads
        k = mx.repeat(k, n_repeats, axis=1)
        v = mx.repeat(v, n_repeats, axis=1)

        # Scaled dot-product attention (causal)
        scale = 1.0 / (head_dim ** 0.5)
        attn = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=scale, mask="causal"
        )[0]

        # Reshape back and output projection
        attn = attn.reshape(seq_len, hidden_size)
        attn_out = qmatmul(attn, f"{p}.self_attn.o_proj")

        # Attention residual
        hidden = hidden + attn_out

        # Pre-MLP RMSNorm
        normed2 = mx.fast.rms_norm(hidden, weights[f"{p}.post_attention_layernorm.weight"], rms_eps)

        # MLP (SwiGLU): silu(gate) * up, where silu(x) = x * sigmoid(x)
        gate = qmatmul(normed2, f"{p}.mlp.gate_proj")
        up   = qmatmul(normed2, f"{p}.mlp.up_proj")
        mlp_out = qmatmul(gate * mx.sigmoid(gate) * up, f"{p}.mlp.down_proj")

        # MLP residual
        hidden = hidden + mlp_out

        if layer_idx == 0 or layer_idx == n_layers - 1:
            mx.eval(hidden)
            print(f"  Layer {layer_idx}: hidden range [{hidden.min().item():.4f}, {hidden.max().item():.4f}]")

    mx.eval(hidden)
    print(f"\nAfter all layers: {hidden.shape}")

    # Final RMSNorm
    hidden = mx.fast.rms_norm(hidden, weights["model.norm.weight"], rms_eps)
    mx.eval(hidden)

    # LM head (quantized)
    logits = qmatmul(hidden, "lm_head")  # [seq_len, vocab_size]
    mx.eval(logits)
    print(f"Logits shape: {logits.shape}")

    # Last token logits
    last_logits = logits[-1, :]  # [vocab_size]
    mx.eval(last_logits)

    top5_indices = mx.argpartition(-last_logits, 5)[:5]
    mx.eval(top5_indices)
    top5 = [(int(top5_indices[i].item()), float(last_logits[top5_indices[i].item()].item()))
            for i in range(5)]
    top5.sort(key=lambda x: -x[1])
    print(f"\nTop-5 next tokens (last position):")
    for tok_id, score in top5:
        print(f"  token {tok_id}: logit={score:.4f}")

    greedy_next = int(mx.argmax(last_logits).item())
    print(f"\nGreedy next token: {greedy_next}")

    # Save baseline
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    outputs = {
        "input_hidden":  hidden.astype(mx.float32),  # after final norm [seq, hidden]
        "last_logits":   last_logits.astype(mx.float32),
    }
    mx.save_safetensors(str(output_dir / "forward_pass_baseline.safetensors"), outputs)
    print(f"\n✓ Saved to {output_dir / 'forward_pass_baseline.safetensors'}")
    print(f"  greedy_next_token = {greedy_next}")

if __name__ == "__main__":
    forward_pass()