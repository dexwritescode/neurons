#pragma once

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "language_model.h"
#include <mlx/mlx.h>
#include <functional>
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * MLX-native implementation of the Qwen3 transformer MoE architecture (model_type = "qwen3_moe").
 *
 * Qwen3-30B-A3B and similar models:
 *   - All layers: standard GQA with QK-norm and RoPE (no SSM / GatedDeltaNet)
 *   - FFN: switch_mlp MoE (no shared expert); weight keys model.layers.N.mlp.switch_mlp.*
 *   - Weight prefix: "model.*" (no "language_model." outer wrapper)
 *
 * Distinct from Qwen3MoeModelMLX which implements the hybrid linear-attention +
 * full-attention architecture used by qwen3_5_moe (e.g. Qwen3.6-35B-A3B).
 */
class Qwen3TransformerMoeModelMLX final : public LanguageModel {
public:
    static Result<Qwen3TransformerMoeModelMLX> from_model_dir(
        const std::filesystem::path& model_dir,
        size_t                       context_size = 0);

    Result<std::vector<int>> generate(
        const std::vector<int>& input_ids,
        size_t max_new_tokens = 4096,
        SamplingParams params = {},
        std::function<bool(int)> on_token = nullptr) override;

    const ModelConfig&        config()          const override { return config_; }
    const std::string&        model_type()      const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()       const override { return tokenizer_; }
    size_t                    num_parameters()  const override;

private:
    struct MlxDecodeState {
        std::vector<mlx::core::array> kv_keys;  // one per layer, growing {n_kv, pos, head_dim}
        std::vector<mlx::core::array> kv_vals;
        std::function<std::vector<mlx::core::array>(
            const std::vector<mlx::core::array>&)> compiled_fn;
        bool fn_ready = false;
    };

    Qwen3TransformerMoeModelMLX(
        ModelConfig                                        config,
        SimpleBpeTokenizer                                 tokenizer,
        std::unordered_map<std::string, mlx::core::array> mlx_weights,
        mlx::core::array                                   embed_mat,
        size_t                                             context_size);

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> decode(int token_id);
    void reset_cache();
    void init_empty_decode_state();
    void build_decode_fn();
    Result<std::vector<float>> run_decode_step(int token_id);
    Result<std::vector<float>> run_prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<int>>   generate_pipelined(
        const std::vector<int>& input_ids,
        size_t max_new_tokens,
        SamplingParams params,
        std::function<bool(int)> on_token);

    ModelConfig                                        config_;
    SimpleBpeTokenizer                                 tokenizer_;
    std::unordered_map<std::string, mlx::core::array> mlx_weights_;
    mlx::core::array                                   embed_mat_;
    std::optional<MlxDecodeState>                      mlx_state_;
    size_t                                             cache_position_ = 0;
};

} // namespace compute

#endif // MLX_BACKEND_ENABLED
