#pragma once

#include "language_model.h"
#include <optional>
#include <unordered_map>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#endif

namespace compute {

/**
 * Concrete LanguageModel for all Llama-family and Mistral-family models.
 *
 * Handles model_type: "llama", "mistral", "qwen2", "qwen3"
 * All families share the same forward pass: RMSNorm → RoPE → GQA → SwiGLU.
 * Differences (hidden_size, n_heads, rope_theta, …) are config-driven.
 */
class LlamaModel final : public LanguageModel {
public:
    static Result<LlamaModel> from_model_dir(
        const std::filesystem::path& model_dir,
        ComputeBackend*              backend,
        size_t                       context_size = 0);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<int>> generate(
        const std::vector<int>& input_ids,
        size_t max_new_tokens = 4096,
        SamplingParams params = {},
        std::function<bool(int)> on_token = nullptr) override;

    const ModelConfig&        config()         const override { return config_; }
    const std::string&        model_type()     const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()      const override { return tokenizer_; }
    size_t                    num_parameters() const override;

    // ── KV-cache step interface (public for diagnostic tests) ─────────────────

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> decode(int token_id);
    void reset_cache();

    // ── Tool-use (LanguageModel overrides) ───────────────────────────────────
    bool                     supports_tool_use()          const override;
    std::string              format_tool_system_prompt(const std::string& tools_json) const override;
    std::optional<ToolCall>  detect_tool_call(const std::string& text) const override;
    std::string              format_tool_result(const std::string& tool_name,
                                                const std::string& result_json) const override;

private:
    LlamaModel(
        ModelConfig        config,
        SimpleBpeTokenizer tokenizer,
        ComputeBackend*    backend);

    // ── State ─────────────────────────────────────────────────────────────────

    enum class ToolFamily { None, Qwen25, Llama31, MistralTool };
    static ToolFamily detect_tool_family(const SimpleBpeTokenizer& tok,
                                         const ModelConfig& cfg);

    ModelConfig        config_;
    SimpleBpeTokenizer tokenizer_;
    ComputeBackend*    backend_;
    ToolFamily         tool_family_ = ToolFamily::None;

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    struct MlxDecodeState {
        std::vector<mlx::core::array> kv_keys;
        std::vector<mlx::core::array> kv_vals;
        std::function<std::vector<mlx::core::array>(std::vector<mlx::core::array>)> compiled_fn;
        bool fn_ready = false;
    };

    std::unordered_map<std::string, mlx::core::array> mlx_weights_;
    mlx::core::array                                   mlx_embed_mat_;
    std::optional<MlxDecodeState>                      mlx_state_;
    size_t                                             mlx_pos_      = 0;

    void mlx_setup(std::unordered_map<std::string, mlx::core::array> mlx_weights,
                   mlx::core::array mlx_embed_mat,
                   size_t context_size);
    void mlx_init_state();
    void mlx_build_decode_fn();
    Result<std::vector<float>> mlx_prefill_batch(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> mlx_run_step(int token_id);
    Result<std::vector<int>>   mlx_generate_pipelined(
        const std::vector<int>& input_ids,
        size_t max_new_tokens,
        SamplingParams params,
        std::function<bool(int)> on_token);
#endif
};

} // namespace compute
