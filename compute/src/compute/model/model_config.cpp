#include "model_config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>

namespace compute {

Result<ModelConfig> ModelConfig::from_config_file(const std::filesystem::path& config_path) {
    if (!std::filesystem::exists(config_path)) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Config file does not exist: " + config_path.string()});
    }

    std::ifstream file(config_path);
    if (!file.is_open()) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Failed to open config file: " + config_path.string()});
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return from_json_string(buffer.str());
}

Result<ModelConfig> ModelConfig::from_json_string(const std::string& json_str) {
    try {
        auto json = nlohmann::json::parse(json_str);
        ModelConfig config;

        // Multimodal models (e.g. Gemma3ForConditionalGeneration) store the text
        // model config nested under "text_config".  Extract it and overlay any
        // top-level fields that text_config may be missing (eos_token_id, quantization).
        if (json.contains("text_config") && !json["text_config"].is_null()) {
            auto text_json = json["text_config"];
            if (!text_json.contains("eos_token_id") || text_json["eos_token_id"].is_null()) {
                if (json.contains("eos_token_id") && !json["eos_token_id"].is_null())
                    text_json["eos_token_id"] = json["eos_token_id"];
            }
            if (!text_json.contains("quantization") || text_json["quantization"].is_null()) {
                if (json.contains("quantization") && !json["quantization"].is_null())
                    text_json["quantization"] = json["quantization"];
            }
            json = std::move(text_json);
        }

        // Parse REQUIRED core architecture fields
        if (!json.contains("vocab_size")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: vocab_size"});
        }
        config.vocab_size = json["vocab_size"].get<size_t>();

        if (!json.contains("hidden_size")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: hidden_size"});
        }
        config.hidden_size = json["hidden_size"].get<size_t>();

        if (!json.contains("num_hidden_layers")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: num_hidden_layers"});
        }
        config.num_hidden_layers = json["num_hidden_layers"].get<size_t>();

        if (!json.contains("num_attention_heads")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: num_attention_heads"});
        }
        config.num_attention_heads = json["num_attention_heads"].get<size_t>();

        if (!json.contains("num_key_value_heads")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: num_key_value_heads"});
        }
        config.num_key_value_heads = json["num_key_value_heads"].get<size_t>();

        if (!json.contains("intermediate_size")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: intermediate_size"});
        }
        config.intermediate_size = json["intermediate_size"].get<size_t>();

        if (!json.contains("max_position_embeddings")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: max_position_embeddings"});
        }
        config.max_position_embeddings = json["max_position_embeddings"].get<size_t>();

        // Parse REQUIRED normalization and activation fields
        if (!json.contains("rms_norm_eps")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: rms_norm_eps"});
        }
        config.rms_norm_eps = json["rms_norm_eps"].get<float>();

        if (!json.contains("rope_theta")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: rope_theta"});
        }
        config.rope_theta = json["rope_theta"].get<float>();

        // hidden_act: Gemma uses "hidden_activation" instead of "hidden_act"
        if (json.contains("hidden_act") && !json["hidden_act"].is_null()) {
            config.hidden_act = json["hidden_act"].get<std::string>();
        } else if (json.contains("hidden_activation") && !json["hidden_activation"].is_null()) {
            config.hidden_act = json["hidden_activation"].get<std::string>();
        } else {
            return std::unexpected(Error{ErrorCode::InvalidInput,
                "Missing required field: hidden_act (or hidden_activation)"});
        }

        // Parse behavior flags (optional — not all architectures include these)
        config.attention_bias = json.value("attention_bias", false);

        // tie_word_embeddings: optional for Gemma (Gemma always has a separate lm_head)
        config.tie_word_embeddings = json.value("tie_word_embeddings", false);

        // Parse REQUIRED metadata
        if (!json.contains("model_type")) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Missing required field: model_type"});
        }
        config.model_type = json["model_type"].get<std::string>();

        // torch_dtype is optional — community quantized uploads sometimes omit it.
        // "bfloat16" and other values are equally valid; is_valid() only checks non-empty.
        config.torch_dtype = json.value("torch_dtype", "float16");

        // architectures is null in multimodal text_config — infer from model_type
        if (json.contains("architectures") && !json["architectures"].is_null()) {
            config.architectures = json["architectures"].get<std::vector<std::string>>();
        } else {
            // Infer architecture class from model_type
            static const std::unordered_map<std::string, std::string> kTypeToArch = {
                {"llama",        "LlamaForCausalLM"},
                {"mistral",      "MistralForCausalLM"},
                {"qwen2",        "Qwen2ForCausalLM"},
                {"gemma",        "GemmaForCausalLM"},
                {"gemma2",       "Gemma2ForCausalLM"},
                {"gemma3_text",  "Gemma3ForCausalLM"},
            };
            auto it = kTypeToArch.find(config.model_type);
            if (it != kTypeToArch.end()) {
                config.architectures = {it->second};
            } else {
                return std::unexpected(Error{ErrorCode::InvalidInput,
                    "Missing required field: architectures"});
            }
        }

        // Parse OPTIONAL special tokens
        if (json.contains("bos_token_id") && !json["bos_token_id"].is_null()) {
            config.bos_token_id = json["bos_token_id"].get<int>();
        }

        if (json.contains("eos_token_id") && !json["eos_token_id"].is_null()) {
            if (json["eos_token_id"].is_array()) {
                config.eos_token_ids = json["eos_token_id"].get<std::vector<int>>();
            } else {
                config.eos_token_ids = std::vector<int>{json["eos_token_id"].get<int>()};
            }
        }

        if (json.contains("pad_token_id") && !json["pad_token_id"].is_null()) {
            config.pad_token_id = json["pad_token_id"].get<int>();
        }

        // Parse OPTIONAL quantization info
        if (json.contains("quantization") && !json["quantization"].is_null()) {
            auto quant_json = json["quantization"];
            QuantizationConfig quant_config;

            if (!quant_json.contains("group_size")) {
                return std::unexpected(Error{ErrorCode::InvalidInput, "Quantization missing required field: group_size"});
            }
            quant_config.group_size = quant_json["group_size"].get<int>();

            if (!quant_json.contains("bits")) {
                return std::unexpected(Error{ErrorCode::InvalidInput, "Quantization missing required field: bits"});
            }
            quant_config.bits = quant_json["bits"].get<int>();

            config.quantization = quant_config;
        }

        // Parse Gemma-specific optional fields
        if (json.contains("head_dim") && !json["head_dim"].is_null()) {
            config.head_dim = json["head_dim"].get<size_t>();
        }
        if (json.contains("query_pre_attn_scalar") && !json["query_pre_attn_scalar"].is_null()) {
            config.query_pre_attn_scalar = json["query_pre_attn_scalar"].get<float>();
        }
        if (json.contains("sliding_window") && !json["sliding_window"].is_null()) {
            config.sliding_window = json["sliding_window"].get<size_t>();
        }
        if (json.contains("sliding_window_pattern") && !json["sliding_window_pattern"].is_null()) {
            config.sliding_window_pattern = json["sliding_window_pattern"].get<int>();
        }
        if (json.contains("rope_local_base_freq") && !json["rope_local_base_freq"].is_null()) {
            config.rope_local_base_freq = json["rope_local_base_freq"].get<float>();
        }

        // Parse OPTIONAL metadata
        if (json.contains("_name_or_path") && !json["_name_or_path"].is_null()) {
            config.name_or_path = json["_name_or_path"].get<std::string>();
        }

        if (json.contains("transformers_version") && !json["transformers_version"].is_null()) {
            config.transformers_version = json["transformers_version"].get<std::string>();
        }

        return config;

    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "JSON parse error: " + std::string(e.what())});
    } catch (const nlohmann::json::type_error& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "JSON type error: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Config parsing error: " + std::string(e.what())});
    }
}

bool ModelConfig::is_valid() const {
    // Check required fields are non-zero/non-empty
    if (vocab_size == 0 || hidden_size == 0 || num_hidden_layers == 0 ||
        num_attention_heads == 0 || num_key_value_heads == 0 ||
        intermediate_size == 0 || max_position_embeddings == 0) {
        return false;
    }

    if (rms_norm_eps <= 0.0f || rope_theta <= 0.0f) {
        return false;
    }

    if (hidden_act.empty() || model_type.empty() || torch_dtype.empty() || architectures.empty()) {
        return false;
    }

    // Check architecture compatibility
    if (!is_supported_architecture()) {
        return false;
    }

    // Check attention head configuration is valid
    if (num_key_value_heads > num_attention_heads) {
        return false;
    }

    // Check that attention heads are evenly divisible for GQA
    if (num_attention_heads % num_key_value_heads != 0) {
        return false;
    }

    return true;
}

bool ModelConfig::is_supported_architecture() const {
    return is_llama_architecture() || is_mistral_architecture() ||
           is_qwen2_architecture() || is_gemma_architecture();
}

bool ModelConfig::is_llama_architecture() const {
    if (model_type != "llama") return false;
    for (const auto& arch : architectures)
        if (arch == "LlamaForCausalLM") return true;
    return false;
}

bool ModelConfig::is_mistral_architecture() const {
    if (model_type != "mistral") return false;
    for (const auto& arch : architectures)
        if (arch == "MistralForCausalLM") return true;
    return false;
}

bool ModelConfig::is_qwen2_architecture() const {
    if (model_type != "qwen2" && model_type != "qwen3") return false;
    for (const auto& arch : architectures)
        if (arch == "Qwen2ForCausalLM" || arch == "Qwen3ForCausalLM") return true;
    return false;
}

bool ModelConfig::is_gemma_architecture() const {
    // model_type covers gemma (v1), gemma2, gemma3_text
    if (model_type != "gemma" && model_type != "gemma2" && model_type != "gemma3_text")
        return false;
    for (const auto& arch : architectures) {
        if (arch == "GemmaForCausalLM"  ||
            arch == "Gemma2ForCausalLM" ||
            arch == "Gemma3ForCausalLM")
            return true;
    }
    return false;
}

size_t ModelConfig::effective_head_dim() const {
    if (head_dim.has_value()) return *head_dim;
    return (num_attention_heads > 0) ? hidden_size / num_attention_heads : 0;
}

float ModelConfig::effective_attention_scale() const {
    if (query_pre_attn_scalar.has_value()) {
        return 1.0f / std::sqrt(*query_pre_attn_scalar);
    }
    const size_t hd = effective_head_dim();
    return (hd > 0) ? 1.0f / std::sqrt(static_cast<float>(hd)) : 1.0f;
}

bool ModelConfig::is_local_layer(int layer_idx) const {
    if (!sliding_window_pattern.has_value()) return false;
    const int pattern = *sliding_window_pattern;
    return (layer_idx % pattern) != (pattern - 1);
}

std::string ModelConfig::to_string() const {
    std::ostringstream oss;
    oss << "ModelConfig {\n";
    oss << "  vocab_size: " << vocab_size << "\n";
    oss << "  hidden_size: " << hidden_size << "\n";
    oss << "  num_hidden_layers: " << num_hidden_layers << "\n";
    oss << "  num_attention_heads: " << num_attention_heads << "\n";
    oss << "  num_key_value_heads: " << num_key_value_heads << "\n";
    oss << "  intermediate_size: " << intermediate_size << "\n";
    oss << "  max_position_embeddings: " << max_position_embeddings << "\n";
    oss << "  rms_norm_eps: " << rms_norm_eps << "\n";
    oss << "  rope_theta: " << rope_theta << "\n";
    oss << "  hidden_act: " << hidden_act << "\n";
    oss << "  attention_bias: " << (attention_bias ? "true" : "false") << "\n";
    oss << "  tie_word_embeddings: " << (tie_word_embeddings ? "true" : "false") << "\n";
    oss << "  model_type: " << model_type << "\n";
    oss << "  torch_dtype: " << torch_dtype << "\n";

    if (bos_token_id.has_value()) {
        oss << "  bos_token_id: " << *bos_token_id << "\n";
    }
    if (eos_token_ids.has_value()) {
        oss << "  eos_token_ids: [";
        for (size_t i = 0; i < eos_token_ids->size(); ++i) {
            if (i) oss << ", ";
            oss << (*eos_token_ids)[i];
        }
        oss << "]\n";
    }
    if (pad_token_id.has_value()) {
        oss << "  pad_token_id: " << *pad_token_id << "\n";
    }

    if (quantization.has_value()) {
        oss << "  quantization: { group_size: " << quantization->group_size
            << ", bits: " << quantization->bits << " }\n";
    }

    oss << "  uses_gqa: " << (uses_grouped_query_attention() ? "true" : "false") << "\n";
    oss << "}";

    return oss.str();
}

size_t ModelConfig::effective_num_kv_heads() const {
    return num_key_value_heads;
}

bool ModelConfig::uses_grouped_query_attention() const {
    return num_key_value_heads < num_attention_heads;
}

bool ModelConfig::is_eos_token(int token_id) const {
    if (!eos_token_ids.has_value()) return token_id == 2;
    for (int id : *eos_token_ids) {
        if (id == token_id) return true;
    }
    return false;
}

int ModelConfig::primary_eos_token_id() const {
    if (eos_token_ids.has_value() && !eos_token_ids->empty())
        return eos_token_ids->front();
    return 2;
}

} // namespace compute