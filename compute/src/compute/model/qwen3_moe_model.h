#pragma once

#include "language_model.h"
#include "qwen3_moe_model_base.h"
#include "kv_cache.h"
#include <functional>
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * Qwen3.5 MoE — hybrid GatedDeltaNet SSM + GQA + MoE language model.
 *
 * model_type: "qwen3_5_moe"
 *
 * Architecture (40 layers):
 *   - Every layer: MoE MLP (switch_mlp batched experts + shared_expert + shared_expert_gate)
 *   - is_linear = (layer_idx + 1) % 4 != 0  → GatedDeltaNet SSM
 *   - is_linear = false (layers 3, 7, 11, …) → full GQA with q_norm/k_norm + output gate
 *
 * Weight prefix: language_model.model.layers.{i}.*
 */
class Qwen3MoeModel final : public Qwen3MoeModelBase, public LanguageModel {
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
    size_t num_parameters() const override { return Qwen3MoeModelBase::num_parameters(); }

private:
    // Per-SSM-layer state (GatedDeltaNet)
    struct SsmState {
        std::optional<Tensor> conv_state;  // [kernel_size-1, conv_dim]
        std::optional<Tensor> rec_state;   // [Hv, Dv, Dk]
        bool valid = false;
    };

    Qwen3MoeModel(
        ModelConfig                             config,
        SimpleBpeTokenizer                      tokenizer,
        std::unordered_map<std::string, Tensor> weights,
        ComputeBackend*                         backend);

    // ── Helpers ───────────────────────────────────────────────────────────────

    Result<Tensor> embedding(const std::vector<int>& token_ids);

    // Linear projection: dispatches to quantized_matmul or matmul.
    // Uses infer_quant_bits so gate layers (8-bit) work correctly.
    Result<Tensor> linear(const Tensor& input, const std::string& weight_key);

    // Linear projection for a single expert slice from a 3D weight bank.
    // weight_key names a weight of shape [num_experts, out, in_packed].
    Result<Tensor> expert_linear(const Tensor& input, const std::string& weight_key, int expert_idx);

    // MoE MLP — used by every layer regardless of attention type.
    Result<Tensor> moe_mlp(const Tensor& input, int layer_idx);

    // Full-attention transformer block (every 4th layer: 3, 7, 11, …).
    Result<Tensor> full_attention_block(
        const Tensor& input, int layer_idx, int position_offset, LayerKVCache* cache);

    // GatedDeltaNet SSM transformer block (all other layers).
    Result<Tensor> linear_attention_block(const Tensor& input, int layer_idx);

    // Top-level forward: embedding → layer loop → norm → lm_head.
    Result<std::vector<float>> forward_impl(
        const std::vector<int>&    input_ids,
        int                        position_offset,
        std::vector<LayerKVCache>* cache_vec);

    // ── State ─────────────────────────────────────────────────────────────────

    ComputeBackend* backend_;

    std::optional<Tensor>     dequantized_embed_tokens_;
    std::vector<LayerKVCache> kv_cache_;
    std::vector<SsmState>     ssm_cache_;
    size_t                    cache_position_ = 0;

};

} // namespace compute
