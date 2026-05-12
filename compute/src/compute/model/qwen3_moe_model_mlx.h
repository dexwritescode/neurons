#pragma once

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "language_model.h"
#include <mlx/mlx.h>
#include <functional>
#include <optional>
#include <unordered_map>

namespace compute {

/**
 * MLX-native implementation of the Qwen3.5 MoE model.
 *
 * Owns the entire forward pass as a pure mlx::core::array graph — no ComputeBackend
 * dependency — enabling mx::compile to fuse the decode step into a single compiled
 * function (target: ≥50 tok/s on M2 Ultra for Qwen3.6-35B-A3B-4bit).
 *
 * Prefill runs all T prompt tokens in one eager pass: one Metal dispatch per SSM layer
 * (T-loop inside kernel), one SDPA per attention layer, T per-token MoE calls.
 */
class Qwen3MoeModelMLX final : public LanguageModel {
public:
    static Result<Qwen3MoeModelMLX> from_model_dir(
        const std::filesystem::path& model_dir,
        size_t                       context_size = 0);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<int>> generate(
        const std::vector<int>& input_ids,
        size_t max_new_tokens = 4096,
        SamplingParams params = {},
        std::function<bool(int)> on_token = nullptr) override;

    const ModelConfig&        config()     const override { return config_; }
    const std::string&        model_type() const override { return config_.model_type; }
    const HFTokenizer& tokenizer()  const override { return tokenizer_; }
    size_t                    num_parameters() const override;

private:
    struct MlxDecodeState {
        std::vector<mlx::core::array> kv_keys;   // one per full-attn layer, growing {nkv, pos, hd}
        std::vector<mlx::core::array> kv_vals;
        std::vector<mlx::core::array> ssm_conv;  // one per SSM layer, fixed shape
        std::vector<mlx::core::array> ssm_rec;
        std::function<std::vector<mlx::core::array>(
            const std::vector<mlx::core::array>&)> compiled_fn;
        bool fn_ready = false;
    };

    Qwen3MoeModelMLX(
        ModelConfig                                       config,
        HFTokenizer                                tokenizer,
        std::unordered_map<std::string, mlx::core::array> mlx_weights,
        mlx::core::array                                  embed_mat,
        size_t                                            context_size);

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> decode(int token_id);
    void reset_cache();

    void init_empty_decode_state();
    void build_decode_fn();
    Result<std::vector<float>> run_decode_step(int token_id);
    Result<std::vector<float>> run_prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<int>>   moe_generate_pipelined(
        const std::vector<int>& input_ids,
        size_t max_new_tokens,
        SamplingParams params,
        std::function<bool(int)> on_token);

    ModelConfig                                        config_;
    HFTokenizer                                 tokenizer_;
    std::unordered_map<std::string, mlx::core::array> mlx_weights_;
    mlx::core::array                                  embed_mat_;
    std::optional<MlxDecodeState>                     mlx_state_;
    size_t                                            cache_position_ = 0;
};

} // namespace compute

#endif // MLX_BACKEND_ENABLED
