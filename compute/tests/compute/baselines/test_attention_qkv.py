#!/usr/bin/env python3
"""
Baseline test for TinyLlama full attention layer
Generates reference outputs for C++ test comparison

Uses the correct tensor convention matching the C++ implementation:
  - Tensors are [n_heads, seq, head_dim] before RoPE and SDPA
  - SDPA handles GQA internally (no manual KV head expansion)
  - MLX fast.rope expects T=seq at shape(-2), achieved by [n_heads, seq, head_dim]
  - MLX fast.scaled_dot_product_attention expects [batch, n_heads, seq, head_dim]
"""

import mlx.core as mx
from pathlib import Path
import json

def test_full_attention_layer():
    """Test complete attention layer including RoPE, GQA, and output projection"""

    model_path = Path("/Users/dex/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")

    with open(model_path / "config.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]        # 2048
    n_heads     = config["num_attention_heads"] # 32
    n_kv_heads  = config["num_key_value_heads"] # 4
    head_dim    = hidden_size // n_heads        # 64
    rope_theta  = config.get("rope_theta", 10000.0)
    group_size  = config["quantization"]["group_size"]
    bits        = config["quantization"]["bits"]

    print(f"Config: hidden_size={hidden_size}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")

    weights = mx.load(str(model_path / "model.safetensors"))

    layer_idx = 0
    prefix = f"model.layers.{layer_idx}.self_attn."

    q_weight = weights[f"{prefix}q_proj.weight"]
    q_scales = weights[f"{prefix}q_proj.scales"]
    q_biases = weights[f"{prefix}q_proj.biases"]
    k_weight = weights[f"{prefix}k_proj.weight"]
    k_scales = weights[f"{prefix}k_proj.scales"]
    k_biases = weights[f"{prefix}k_proj.biases"]
    v_weight = weights[f"{prefix}v_proj.weight"]
    v_scales = weights[f"{prefix}v_proj.scales"]
    v_biases = weights[f"{prefix}v_proj.biases"]

    print(f"\nWeight shapes:")
    print(f"  q_weight: {q_weight.shape}, q_scales: {q_scales.shape}, q_biases: {q_biases.shape}")
    print(f"  k_weight: {k_weight.shape}, k_scales: {k_scales.shape}, k_biases: {k_biases.shape}")
    print(f"  v_weight: {v_weight.shape}, v_scales: {v_scales.shape}, v_biases: {v_biases.shape}")

    # Fixed input: [seq_len=5, hidden_size=2048]
    mx.random.seed(42)
    seq_len = 5
    input_tensor = mx.random.normal(shape=(seq_len, hidden_size)).astype(mx.bfloat16)

    print(f"\nInput shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    print(f"Input range: [{input_tensor.min().item():.6f}, {input_tensor.max().item():.6f}]")

    # Step 1: QKV projections
    queries = mx.quantized_matmul(input_tensor, q_weight, q_scales, q_biases, transpose=True, group_size=group_size, bits=bits)
    keys    = mx.quantized_matmul(input_tensor, k_weight, k_scales, k_biases, transpose=True, group_size=group_size, bits=bits)
    values  = mx.quantized_matmul(input_tensor, v_weight, v_scales, v_biases, transpose=True, group_size=group_size, bits=bits)
    mx.eval(queries, keys, values)

    print(f"\nAfter projections:")
    print(f"  queries: {queries.shape}, keys: {keys.shape}, values: {values.shape}")

    # Step 2: Reshape then swapaxes to [n_heads/n_kv_heads, seq, head_dim]
    # This is the convention where rope's T=shape(-2)=seq (correct positional encoding)
    queries_mh = queries.reshape(seq_len, n_heads, head_dim).swapaxes(0, 1)   # [32, 5, 64]
    keys_mh    = keys.reshape(seq_len, n_kv_heads, head_dim).swapaxes(0, 1)   # [4, 5, 64]
    values_mh  = values.reshape(seq_len, n_kv_heads, head_dim).swapaxes(0, 1) # [4, 5, 64]
    mx.eval(queries_mh, keys_mh, values_mh)

    print(f"\nAfter swapaxes to [heads, seq, head_dim]:")
    print(f"  queries_mh: {queries_mh.shape}")
    print(f"  keys_mh: {keys_mh.shape}")
    print(f"  values_mh: {values_mh.shape}")

    # Step 3: RoPE — with [heads, seq, head_dim], T=shape(-2)=seq=5 (correct)
    print(f"\nApplying RoPE with theta={rope_theta}")
    queries_rope = mx.fast.rope(queries_mh, head_dim, traditional=False, base=rope_theta, scale=1.0, offset=0)
    keys_rope    = mx.fast.rope(keys_mh,    head_dim, traditional=False, base=rope_theta, scale=1.0, offset=0)
    mx.eval(queries_rope, keys_rope)

    print(f"After RoPE:")
    print(f"  queries_rope: {queries_rope.shape}")
    print(f"  keys_rope: {keys_rope.shape}")

    # Step 4: SDPA — no manual GQA expansion, SDPA handles it internally.
    # Input: [n_heads, seq, head_dim] → add batch → [1, n_heads, seq, head_dim]
    # MLX SDPA: shape(-3)=n_heads, shape(-2)=seq (correct)
    scale = 1.0 / (head_dim ** 0.5)
    print(f"\nScaled dot-product attention (scale={scale:.6f})")

    attn_output_batched = mx.fast.scaled_dot_product_attention(
        queries_rope[None],  # [1, 32, 5, 64]
        keys_rope[None],     # [1,  4, 5, 64]  — SDPA tiles KV internally for GQA
        values_mh[None],     # [1,  4, 5, 64]
        scale=scale,
        mask="causal"
    )
    mx.eval(attn_output_batched)

    # Remove batch: [1, n_heads, seq, head_dim] → [n_heads, seq, head_dim]
    attn_output = attn_output_batched[0]  # [32, 5, 64]
    print(f"After attention: {attn_output.shape}")

    # Step 5: Swapaxes back and reshape: [n_heads, seq, head_dim] → [seq, hidden_size]
    attn_reshaped = attn_output.swapaxes(0, 1).reshape(seq_len, hidden_size)
    mx.eval(attn_reshaped)
    print(f"After reshape: {attn_reshaped.shape}")

    # Step 6: Output projection
    o_weight = weights[f"{prefix}o_proj.weight"]
    o_scales = weights[f"{prefix}o_proj.scales"]
    o_biases = weights[f"{prefix}o_proj.biases"]

    print(f"\nOutput projection weights:")
    print(f"  o_weight: {o_weight.shape}, o_scales: {o_scales.shape}, o_biases: {o_biases.shape}")

    final_output = mx.quantized_matmul(
        attn_reshaped, o_weight, o_scales, o_biases,
        transpose=True, group_size=group_size, bits=bits
    )
    mx.eval(final_output)

    print(f"\nFinal output:")
    print(f"  final_output: {final_output.shape}, dtype: {final_output.dtype}")
    print(f"  Output range: [{final_output.min().item():.6f}, {final_output.max().item():.6f}]")

    # Save outputs for C++ comparison
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    outputs = {
        "input":         input_tensor.astype(mx.float32),
        "queries_flat":  queries.astype(mx.float32),
        "keys_flat":     keys.astype(mx.float32),
        "values_flat":   values.astype(mx.float32),
        "queries_mh":    queries_mh.astype(mx.float32),
        "keys_mh":       keys_mh.astype(mx.float32),
        "values_mh":     values_mh.astype(mx.float32),
        "queries_rope":  queries_rope.astype(mx.float32),
        "keys_rope":     keys_rope.astype(mx.float32),
        "attn_output":   attn_output.astype(mx.float32),
        "attn_reshaped": attn_reshaped.astype(mx.float32),
        "final_output":  final_output.astype(mx.float32),
    }

    mx.save_safetensors(str(output_dir / "attention_full_baseline.safetensors"), outputs)
    print(f"\n✓ Saved baseline to {output_dir / 'attention_full_baseline.safetensors'}")
    print(f"\nSample values (first 5 elements of first position):")
    print(f"  final_output[0,:5]: {final_output[0,:5].astype(mx.float32)}")

if __name__ == "__main__":
    test_full_attention_layer()