#pragma once

#include "../core/tensor.h"
#include <optional>

namespace compute {

/**
 * Per-layer KV cache for efficient autoregressive decoding.
 * Keys/values are stored post-RoPE so they can be directly concatenated
 * with new tokens during decode without re-applying positional encoding.
 *
 * Shape when valid:
 *   keys:   [n_kv_heads, seq_so_far, head_dim]
 *   values: [n_kv_heads, seq_so_far, head_dim]
 */
struct LayerKVCache {
    std::optional<Tensor> keys;
    std::optional<Tensor> values;
    bool valid = false;
};

/**
 * Per-layer KV cache for Gemma (same shape convention as LayerKVCache).
 *
 * Gemma3 has local (sliding-window) and global layers.
 * For the first implementation we store all keys/values without
 * window truncation (correct for short sequences, conservative for long ones).
 */
struct GemmaLayerKVCache {
    std::optional<Tensor> keys;
    std::optional<Tensor> values;
    bool valid = false;
};

} // namespace compute
