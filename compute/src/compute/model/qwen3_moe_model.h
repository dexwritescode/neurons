#pragma once

#include "language_model.h"
#include "llama_model.h"
#include "../core/tensor.h"
#include <unordered_map>

namespace compute {

/**
 * Qwen3.5 MoE — hybrid SSM/MoE language model.
 *
 * model_type: "qwen3_5_moe"
 *
 * Architecture (40 layers):
 *   - Every layer: MoE MLP (switch_mlp batched experts + shared_expert + shared_expert_gate)
 *   - Layers 0,1,2,4,5,6,… (linear_attention): Mamba2/SSM-style linear attention
 *   - Layers 3,7,11,… (full_attention): standard GQA with q_norm/k_norm and attn_output_gate
 *
 * Weight prefix: language_model.model.layers.{i}.*
 */
class Qwen3MoeModel final : public LanguageModel {
public:
    static Result<Qwen3MoeModel> from_model_dir(
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
    Qwen3MoeModel(
        ModelConfig                             config,
        SimpleBpeTokenizer                      tokenizer,
        std::unordered_map<std::string, Tensor> weights,
        ComputeBackend*                         backend);

    // ── Helpers ───────────────────────────────────────────────────────────────

    Result<Tensor> get_weight(const std::string& name) const;

    // Linear projection: dispatches to quantized_matmul or matmul.
    // Weight names use the language_model.model.* prefix.
    Result<Tensor> linear(const Tensor& input, const std::string& weight_key);

    // MoE MLP — used by every layer regardless of attention type.
    Result<Tensor> moe_mlp(const Tensor& input, int layer_idx);

    // Full-attention transformer block (every 4th layer: 3, 7, 11, …).
    Result<Tensor> full_attention_block(
        const Tensor& input, int layer_idx, int position_offset, LayerKVCache* cache);

    // Linear-attention (SSM/Mamba2) transformer block (all other layers).
    Result<Tensor> linear_attention_block(const Tensor& input, int layer_idx);

    // Top-level forward: embedding → layer loop → norm → lm_head.
    Result<std::vector<float>> forward_impl(
        const std::vector<int>&    input_ids,
        int                        position_offset,
        std::vector<LayerKVCache>* cache_vec);

    // ── State ─────────────────────────────────────────────────────────────────

    ModelConfig                             config_;
    SimpleBpeTokenizer                      tokenizer_;
    std::unordered_map<std::string, Tensor> weights_;
    ComputeBackend*                         backend_;

    std::vector<LayerKVCache> kv_cache_;   // allocated only for full_attention layers
    size_t                    cache_position_ = 0;
};

} // namespace compute
