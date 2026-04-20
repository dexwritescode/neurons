#pragma once

#include "language_model.h"
#include "../core/tensor.h"
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * Per-layer KV cache for Gemma (same shape convention as LlamaModel).
 *
 * Gemma3 has local (sliding-window) and global layers.
 * For the first implementation we store all keys/values without
 * window truncation (correct for short sequences, conservative for long ones).
 *
 * Shape when valid:
 *   keys:   [n_kv_heads, seq_so_far, head_dim]
 *   values: [n_kv_heads, seq_so_far, head_dim]
 */
struct GemmaLayerKVCache {
    std::optional<Tensor> keys;
    std::optional<Tensor> values;
    bool valid = false;
};

/**
 * LanguageModel implementation for Gemma/Gemma2/Gemma3 model families.
 *
 * Handles model_type: "gemma", "gemma2", "gemma3_text"
 *
 * Key differences from LlamaModel:
 *   - Embedding scale: multiply by sqrt(hidden_size) after lookup
 *   - 4 norms per block: input_layernorm, post_attention_layernorm (on attn out),
 *     pre_feedforward_layernorm, post_feedforward_layernorm (on FFN out)
 *   - Q/K norm inside each attention layer
 *   - GeGLU FFN: down(gelu(gate) * up) instead of down(silu(gate) * up)
 *   - Layer-specific RoPE theta (local vs global layers in Gemma3)
 *   - Separate lm_head (never tied to embeddings)
 */
class GemmaModel final : public LanguageModel {
public:
    // Factory — loads config, weights, and tokenizer from model_dir.
    static Result<GemmaModel> from_model_dir(
        const std::filesystem::path& model_dir,
        ComputeBackend*              backend);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids) override;
    Result<std::vector<float>> decode(int token_id) override;
    void reset_cache() override;

    const ModelConfig&        config()         const override { return config_; }
    const std::string&        model_type()     const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()      const override { return tokenizer_; }
    ComputeBackend*           backend()        const override { return backend_; }
    size_t                    num_parameters() const override;

private:
    GemmaModel(
        ModelConfig                              config,
        SimpleBpeTokenizer                       tokenizer,
        std::unordered_map<std::string, Tensor>  weights,
        ComputeBackend*                          backend);

    // ── Weight / linear helpers ───────────────────────────────────────────────

    Result<Tensor> get_weight(const std::string& name) const;

    // Dispatches to quantized_matmul or swapaxes+matmul based on presence of .scales
    Result<Tensor> linear(const Tensor& input, const std::string& weight_key);

    // ── Layer implementations ─────────────────────────────────────────────────

    Result<Tensor> embedding(const std::vector<int>& token_ids);
    Result<Tensor> rms_norm(const Tensor& input, const Tensor& weight);

    Result<Tensor> attention_layer(
        const Tensor&       input,
        int                 layer_idx,
        int                 position_offset,
        GemmaLayerKVCache*  cache);

    Result<Tensor> mlp_layer(const Tensor& input, int layer_idx);

    Result<Tensor> transformer_block(
        const Tensor&       input,
        int                 layer_idx,
        int                 position_offset,
        GemmaLayerKVCache*  cache);

    Result<std::vector<float>> forward_impl(
        const std::vector<int>&      input_ids,
        int                          position_offset,
        std::vector<GemmaLayerKVCache>* cache_vec);

    // ── State ─────────────────────────────────────────────────────────────────

    ModelConfig                              config_;
    SimpleBpeTokenizer                       tokenizer_;
    std::unordered_map<std::string, Tensor>  weights_;
    ComputeBackend*                          backend_;

    std::vector<GemmaLayerKVCache>    kv_cache_;
    size_t                            cache_position_ = 0;

    // Cached dequantized embedding table (populated on first embedding() call)
    mutable std::optional<Tensor>     dequantized_embed_tokens_;
};

} // namespace compute
