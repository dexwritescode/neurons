#pragma once

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

#include "language_model.h"
#include "qwen3_moe_model_base.h"
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
 * Prefill runs T sequential single-token decode steps (correct, ~O(T) cost).
 * A parallel-scan prefill path is a future optimisation.
 */
class Qwen3MoeModelMLX final : public Qwen3MoeModelBase, public LanguageModel {
public:
    static Result<Qwen3MoeModelMLX> from_model_dir(
        const std::filesystem::path& model_dir);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids) override;
    Result<std::vector<float>> decode(int token_id) override;
    void reset_cache() override;

    const ModelConfig&        config()     const override { return config_; }
    const std::string&        model_type() const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()  const override { return tokenizer_; }
    ComputeBackend*           backend()    const override { return nullptr; }
    size_t                    num_parameters() const override;

private:
    struct MlxDecodeState {
        std::vector<mlx::core::array> kv_keys;   // one per full-attention layer
        std::vector<mlx::core::array> kv_vals;
        std::vector<mlx::core::array> ssm_conv;  // one per SSM layer
        std::vector<mlx::core::array> ssm_rec;
        std::function<std::vector<mlx::core::array>(
            const std::vector<mlx::core::array>&)> compiled_fn;
        bool fn_ready = false;
    };

    Qwen3MoeModelMLX(
        ModelConfig                                       config,
        SimpleBpeTokenizer                                tokenizer,
        std::unordered_map<std::string, mlx::core::array> mlx_weights,
        mlx::core::array                                  embed_mat);

    void init_empty_decode_state();
    void build_decode_fn();
    Result<std::vector<float>> run_decode_step(int token_id);

    std::unordered_map<std::string, mlx::core::array> mlx_weights_;
    mlx::core::array                                  embed_mat_;
    std::optional<MlxDecodeState>                     mlx_state_;
    size_t                                            cache_position_ = 0;
};

} // namespace compute

#endif // MLX_BACKEND_ENABLED
