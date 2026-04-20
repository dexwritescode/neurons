#pragma once

#include "../core/compute_types.h"
#include <string>
#include <unordered_map>
#include <filesystem>

namespace compute {

/**
 * Special token configuration
 */
struct SpecialToken {
    std::string content;
    bool lstrip = false;
    bool normalized = false;
    bool rstrip = false;
    bool single_word = false;
    bool special = false;
};

/**
 * TokenizerConfig holds configuration for text tokenization
 * All values parsed from tokenizer_config.json - no defaults
 */
struct TokenizerConfig {
    // Basic tokenization settings
    bool add_bos_token;
    bool add_eos_token;
    std::string tokenizer_class;
    size_t model_max_length;
    bool legacy;
    bool clean_up_tokenization_spaces;
    std::string padding_side;

    // Special tokens
    std::string bos_token;
    std::string eos_token;
    std::string unk_token;
    std::string pad_token;

    // Chat template for conversation formatting
    std::string chat_template;
    bool use_default_system_prompt;

    // Added tokens decoder mapping
    std::unordered_map<int, SpecialToken> added_tokens_decoder;

    // SentencePiece model kwargs
    std::unordered_map<std::string, std::string> sp_model_kwargs;

    /**
     * Load tokenizer configuration from tokenizer_config.json file
     * @param config_path Path to tokenizer_config.json file or directory containing it
     * @return Result with TokenizerConfig or error
     */
    static Result<TokenizerConfig> from_config_file(const std::filesystem::path& config_path);

private:
    TokenizerConfig() = default;
};

} // namespace compute