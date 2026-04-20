#pragma once

#include "language_model.h"
#include "../core/tensor.h"
#include <optional>
#include <unordered_map>

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
 * Concrete LanguageModel for all Llama-family and Mistral-family models.
 *
 * Handles model_type: "llama", "mistral", "qwen2"
 * All three families use the identical forward pass:
 *   RMSNorm → RoPE → GQA → SwiGLU
 * Differences (hidden_size, n_heads, rope_theta, …) are config-driven.
 */
class LlamaModel final : public LanguageModel {
public:
    // Factory — loads config, weights, and tokenizer from model_dir.
    static Result<LlamaModel> from_model_dir(
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

    // ── Testing / diagnostic interface ───────────────────────────────────────
    // (Not part of LanguageModel — used by unit tests only via LlamaModel& or TinyLlamaInference&)

    Result<std::vector<float>>  forward(const std::vector<int>& input_ids);
    Result<Tensor>              forward_logits(const std::vector<int>& input_ids);
    Result<Tensor>              embedding(const std::vector<int>& token_ids);
    Result<Tensor>              rms_norm(const Tensor& input, const Tensor& weight, float eps);
    Result<Tensor>              attention_layer(const Tensor& input, int layer_idx);
    Result<Tensor>              get_weight(const std::string& name) const;
    size_t                      cache_position() const { return cache_position_; }

private:
    LlamaModel(
        ModelConfig                              config,
        SimpleBpeTokenizer                       tokenizer,
        std::unordered_map<std::string, Tensor>  weights,
        ComputeBackend*                          backend);

    // ── Layer implementations ─────────────────────────────────────────────────

    // Linear projection: dispatches to quantized_matmul or matmul depending on
    // whether {weight_key}.scales exists in the weight map.  Works for both
    // quantized (mlx-community int4) and unquantized (HF fp16/bf16) models.
    Result<Tensor> linear(const Tensor& input, const std::string& weight_key);

    Result<Tensor> attention_layer(
        const Tensor& input,
        int           layer_idx,
        int           position_offset,
        LayerKVCache* cache);

    Result<Tensor> mlp_layer(const Tensor& input, int layer_idx);

    Result<Tensor> transformer_block(
        const Tensor& input,
        int           layer_idx,
        int           position_offset,
        LayerKVCache* cache);

    Result<std::vector<float>> forward_impl(
        const std::vector<int>&    input_ids,
        int                        position_offset,
        std::vector<LayerKVCache>* cache_vec);

    // ── State ─────────────────────────────────────────────────────────────────

    ModelConfig                              config_;
    SimpleBpeTokenizer                       tokenizer_;
    std::unordered_map<std::string, Tensor>  weights_;
    ComputeBackend*                          backend_;

    std::vector<LayerKVCache>         kv_cache_;
    size_t                            cache_position_ = 0;

    // Cached dequantized embedding table (populated on first use for quantized models)
    mutable std::optional<Tensor>     dequantized_embed_tokens_;
};

} // namespace compute
