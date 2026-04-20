#pragma once

#include "../core/compute_types.h"
#include <string>
#include <optional>
#include <vector>
#include <filesystem>

namespace compute {

/**
 * Quantization information from MLX converted models
 */
struct QuantizationConfig {
    int group_size;
    int bits;
};

/**
 * Configuration structure for transformer models
 * Parsed from config.json - NO hardcoded values
 * All parameters must be read from the actual model config file
 */
struct ModelConfig {
    // Core model architecture - REQUIRED fields
    size_t vocab_size;
    size_t hidden_size;
    size_t num_hidden_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t intermediate_size;
    size_t max_position_embeddings;

    // Normalization and activation - REQUIRED fields
    float rms_norm_eps;
    float rope_theta;
    std::string hidden_act;

    // Model behavior flags - REQUIRED fields
    bool attention_bias;
    bool tie_word_embeddings;

    // Special tokens - may be optional in config
    std::optional<int> bos_token_id;
    std::optional<std::vector<int>> eos_token_ids;  // supports Llama-3's array EOS
    std::optional<int> pad_token_id;

    // Returns true if token_id is any configured EOS token.
    bool is_eos_token(int token_id) const;
    // Returns the primary (first) EOS token ID, or 2 as fallback.
    int primary_eos_token_id() const;

    // Model metadata - REQUIRED fields
    std::string model_type;
    std::string torch_dtype;
    std::vector<std::string> architectures;

    // Quantization info - optional
    std::optional<QuantizationConfig> quantization;

    // Gemma-specific optional fields (absent in Llama/Mistral/Qwen2)
    std::optional<size_t> head_dim;               // Explicit head dim (Gemma3: 256)
    std::optional<float>  query_pre_attn_scalar;  // Q scaling before attention (Gemma3: 256)
    std::optional<size_t> sliding_window;         // Local attention window (Gemma3: 512)
    std::optional<int>    sliding_window_pattern; // Global attn period (Gemma3: 6)
    std::optional<float>  rope_local_base_freq;   // RoPE theta for local layers (Gemma3: 10000)

    // Model path info - may be optional
    std::optional<std::string> name_or_path;
    std::optional<std::string> transformers_version;

    /**
     * Parse ModelConfig from config.json file
     * @param config_path Path to config.json file
     * @return Result containing parsed config or error
     */
    static Result<ModelConfig> from_config_file(const std::filesystem::path& config_path);

    /**
     * Parse ModelConfig from JSON string
     * @param json_str JSON string content
     * @return Result containing parsed config or error
     */
    static Result<ModelConfig> from_json_string(const std::string& json_str);

    /**
     * Validate configuration for supported model types
     * Currently supports: LlamaForCausalLM (TinyLlama)
     * @return true if configuration is valid, false otherwise
     */
    bool is_valid() const;

    /**
     * Check if this is a supported model architecture (llama, mistral, qwen2, or gemma)
     */
    bool is_supported_architecture() const;

    /** True for LlamaForCausalLM models (TinyLlama, Llama-2/3.x) */
    bool is_llama_architecture() const;

    /** True for MistralForCausalLM models */
    bool is_mistral_architecture() const;

    /** True for Qwen2ForCausalLM models */
    bool is_qwen2_architecture() const;

    /** True for Gemma/Gemma2/Gemma3 models */
    bool is_gemma_architecture() const;

    // Computed helpers

    /** Returns head_dim if set, else hidden_size / num_attention_heads */
    size_t effective_head_dim() const;

    /** Returns 1/sqrt(query_pre_attn_scalar) if set, else 1/sqrt(effective_head_dim()) */
    float effective_attention_scale() const;

    /**
     * True if layer_idx uses sliding-window (local) attention.
     * A layer is global when (layer_idx % sliding_window_pattern) == (sliding_window_pattern - 1).
     * Returns false when sliding_window_pattern is not set (all layers use full causal attention).
     */
    bool is_local_layer(int layer_idx) const;

    /**
     * Get display string for this configuration
     */
    std::string to_string() const;

    /**
     * Get the effective number of key-value heads (handles GQA vs MHA)
     */
    size_t effective_num_kv_heads() const;

    /**
     * Check if model uses Grouped Query Attention
     */
    bool uses_grouped_query_attention() const;
};

} // namespace compute