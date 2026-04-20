#include "tokenizer_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <limits>

namespace compute {

using json = nlohmann::json;

Result<TokenizerConfig> TokenizerConfig::from_config_file(const std::filesystem::path& config_path) {
    std::filesystem::path actual_config_path = config_path;

    // Handle both file path and directory path
    if (std::filesystem::is_directory(config_path)) {
        actual_config_path = config_path / "tokenizer_config.json";
    }

    if (!std::filesystem::exists(actual_config_path)) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Tokenizer config file not found: " + actual_config_path.string()});
    }

    try {
        std::ifstream config_file(actual_config_path);
        if (!config_file.is_open()) {
            return std::unexpected(Error{ErrorCode::ComputeError,
                                       "Failed to open tokenizer config file: " + actual_config_path.string()});
        }

        json j;
        config_file >> j;

        TokenizerConfig config;

        // Parse BOS/EOS flags — treat null or missing as false (Llama-3 uses null).
        // When null/missing, chat templates control the BOS via explicit token strings
        // in the prompt rather than the tokenizer auto-prepend flag.
        config.add_bos_token =
            (j.contains("add_bos_token") && j["add_bos_token"].is_boolean())
            ? j["add_bos_token"].get<bool>() : false;
        config.add_eos_token =
            (j.contains("add_eos_token") && j["add_eos_token"].is_boolean())
            ? j["add_eos_token"].get<bool>() : false;

        if (!j.contains("tokenizer_class")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: tokenizer_class"});
        }
        config.tokenizer_class = j["tokenizer_class"].get<std::string>();

        // model_max_length can be absurdly large (Mistral uses 10^30) — cap it
        if (j.contains("model_max_length") && j["model_max_length"].is_number()) {
            double raw = j["model_max_length"].get<double>();
            const double max_sane = static_cast<double>(std::numeric_limits<size_t>::max());
            config.model_max_length = static_cast<size_t>(std::min(raw, max_sane));
        }

        // Special tokens — all optional/nullable; missing or null → empty string
        auto read_string_or_null = [&](const char* key) -> std::string {
            if (!j.contains(key) || j[key].is_null()) return "";
            return j[key].get<std::string>();
        };
        config.bos_token = read_string_or_null("bos_token");
        config.eos_token = read_string_or_null("eos_token");
        config.unk_token = read_string_or_null("unk_token");
        config.pad_token = read_string_or_null("pad_token");

        // Optional fields with reasonable defaults
        config.legacy = j.value("legacy", false);
        config.clean_up_tokenization_spaces = j.value("clean_up_tokenization_spaces", false);
        config.padding_side = j.value("padding_side", "right");
        // chat_template can be a string or a list of templates — we only need string form
        if (j.contains("chat_template") && j["chat_template"].is_string()) {
            config.chat_template = j["chat_template"].get<std::string>();
        }
        config.use_default_system_prompt = j.value("use_default_system_prompt", false);

        // Parse added_tokens_decoder if present
        if (j.contains("added_tokens_decoder")) {
            const auto& added_tokens = j["added_tokens_decoder"];
            for (const auto& [key, value] : added_tokens.items()) {
                int token_id = std::stoi(key);
                SpecialToken token;

                if (value.contains("content")) {
                    token.content = value["content"].get<std::string>();
                }
                token.lstrip = value.value("lstrip", false);
                token.normalized = value.value("normalized", false);
                token.rstrip = value.value("rstrip", false);
                token.single_word = value.value("single_word", false);
                token.special = value.value("special", false);

                config.added_tokens_decoder[token_id] = std::move(token);
            }
        }

        // Parse sp_model_kwargs if present
        if (j.contains("sp_model_kwargs")) {
            const auto& sp_kwargs = j["sp_model_kwargs"];
            for (const auto& [key, value] : sp_kwargs.items()) {
                config.sp_model_kwargs[key] = value.get<std::string>();
            }
        }

        return config;

    } catch (const json::exception& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "JSON parsing error: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   "Error reading tokenizer config: " + std::string(e.what())});
    }
}

} // namespace compute