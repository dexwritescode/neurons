#pragma once

#include "language_model.h"
#include "kv_cache.h"
#include <optional>
#include <unordered_map>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#endif

namespace compute {

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
        ComputeBackend*              backend,
        size_t                       context_size = 0);

    // ── LanguageModel interface ───────────────────────────────────────────────

    Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids) override;
    Result<std::vector<float>> decode(int token_id) override;
    void reset_cache() override;

    const ModelConfig&        config()         const override { return config_; }
    const std::string&        model_type()     const override { return config_.model_type; }
    const SimpleBpeTokenizer& tokenizer()      const override { return tokenizer_; }
    ComputeBackend*           backend()        const override { return backend_; }
    size_t                    num_parameters() const override;

    // ── Tool-use (LanguageModel overrides) ───────────────────────────────────
    bool                     supports_tool_use()          const override;
    std::string              format_tool_system_prompt(const std::string& tools_json) const override;
    std::optional<ToolCall>  detect_tool_call(const std::string& text) const override;
    std::string              format_tool_result(const std::string& tool_name,
                                                const std::string& result_json) const override;

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

    // Detected at construction via tokenizer vocab probe.
    enum class ToolFamily { None, Qwen25, Llama31, MistralTool };
    static ToolFamily detect_tool_family(const SimpleBpeTokenizer& tok,
                                         const ModelConfig& cfg);

    ModelConfig                              config_;
    SimpleBpeTokenizer                       tokenizer_;
    std::unordered_map<std::string, Tensor>  weights_;
    ComputeBackend*                          backend_;
    ToolFamily                               tool_family_ = ToolFamily::None;

    std::vector<LayerKVCache>         kv_cache_;
    size_t                            cache_position_ = 0;

    // Cached dequantized embedding table (populated on first use for quantized models)
    mutable std::optional<Tensor>     dequantized_embed_tokens_;

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
    size_t                                             context_size_ = 0;
    size_t                                             mlx_pos_      = 0;

    void mlx_setup(std::unordered_map<std::string, mlx::core::array> mlx_weights,
                   mlx::core::array mlx_embed_mat,
                   size_t context_size);
    void mlx_init_state();
    void mlx_build_decode_fn();
    Result<std::vector<float>> mlx_prefill_batch(const std::vector<int>& prompt_ids);
    Result<std::vector<float>> mlx_run_step(int token_id);
#endif
};

} // namespace compute
