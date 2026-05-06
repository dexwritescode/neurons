#pragma once

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "language_model.h"
#include <mlx/mlx.h>
#include <functional>
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * MLX-native implementation of Gemma/Gemma2/Gemma3 models.
 *
 * Owns the entire forward pass as a pure mlx::core::array graph — no
 * ComputeBackend dependency — enabling mx::compile(shapeless=true) to fuse
 * the decode step and async_eval pipelining for continuous GPU utilisation.
 *
 * Gemma-specific differences from LlamaModelMLX:
 *   - Embedding scale: h * sqrt(hidden_size)
 *   - RMSNorm: rms_norm(x, 1 + weight, eps)  — weights stored as deviations from 0
 *   - 4 norms per block: post_attention_layernorm + post_feedforward_layernorm
 *   - GeGLU FFN: gelu(gate) * up  (not SwiGLU)
 *   - Optional Q/K per-head norms (Gemma3), also using (1 + weight)
 *   - Per-layer rope_theta (local vs global in Gemma3)
 *   - LM head may be tied to embedding table
 */
class GemmaModelMLX final : public LanguageModel {
public:
    static Result<GemmaModelMLX> from_model_dir(
        const std::filesystem::path& model_dir);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<int>> generate(
        const std::vector<int>& input_ids,
        size_t max_new_tokens = 4096,
        SamplingParams params = {},
        std::function<bool(int)> on_token = nullptr) override;

    const ModelConfig&        config()     const override { return config_; }
    const std::string&        model_type() const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()  const override { return tokenizer_; }
    size_t                    num_parameters() const override;

private:
    struct MlxDecodeState {
        std::vector<mlx::core::array> kv_keys;
        std::vector<mlx::core::array> kv_vals;
        std::function<std::vector<mlx::core::array>(
            const std::vector<mlx::core::array>&)> compiled_fn;
        bool fn_ready = false;
    };

    GemmaModelMLX(
        ModelConfig                                        config,
        SimpleBpeTokenizer                                 tokenizer,
        std::unordered_map<std::string, mlx::core::array> mlx_weights,
        mlx::core::array                                   embed_mat);

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> decode(int token_id);
    void reset_cache();

    void init_empty_decode_state();
    void build_decode_fn();
    Result<std::vector<float>> run_decode_step(int token_id);
    Result<std::vector<float>> run_prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<int>>   gemma_generate_pipelined(
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
