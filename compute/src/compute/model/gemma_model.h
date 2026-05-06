#pragma once

#include "language_model.h"
#include "gemma_model_base.h"
#include "kv_cache.h"
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * ComputeBackend-path implementation of Gemma/Gemma2/Gemma3.
 *
 * Handles model_type: "gemma", "gemma2", "gemma3_text"
 * On Apple Silicon the factory dispatches to GemmaModelMLX instead.
 *
 * Key differences from LlamaModel:
 *   - Embedding scale: multiply by sqrt(hidden_size) after lookup
 *   - 4 norms per block: input_layernorm, post_attention_layernorm (on attn out),
 *     pre_feedforward_layernorm, post_feedforward_layernorm (on FFN out)
 *   - Q/K norm inside each attention layer (Gemma3)
 *   - GeGLU FFN: down(gelu(gate) * up) instead of down(silu(gate) * up)
 *   - Layer-specific RoPE theta (local vs global layers in Gemma3)
 */
class GemmaModel final : public GemmaModelBase, public LanguageModel {
public:
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
    size_t                    num_parameters() const override { return GemmaModelBase::num_parameters(); }

private:
    GemmaModel(
        ModelConfig                              config,
        SimpleBpeTokenizer                       tokenizer,
        std::unordered_map<std::string, Tensor>  weights,
        ComputeBackend*                          backend);

    // ── Layer implementations ─────────────────────────────────────────────────

    Result<Tensor> linear(const Tensor& input, const std::string& weight_key);
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
        const std::vector<int>&         input_ids,
        int                             position_offset,
        std::vector<GemmaLayerKVCache>* cache_vec);

    // ── State ─────────────────────────────────────────────────────────────────

    ComputeBackend*                backend_;
    std::vector<GemmaLayerKVCache> kv_cache_;
    size_t                         cache_position_ = 0;
    mutable std::optional<Tensor>  dequantized_embed_tokens_;
};

} // namespace compute
