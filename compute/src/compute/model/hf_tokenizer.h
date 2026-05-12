#pragma once

#include "../core/compute_types.h"
#include "tokenizer_config.h"
#include <filesystem>
#include <string>
#include <vector>

// Forward-declare the opaque Rust handle so this header needs no C include.
struct TokenizerHandle;

namespace compute {

// C++ wrapper around the HuggingFace tokenizers Rust crate (0.32.1).
// Drop-in replacement for SimpleBpeTokenizer (now deleted) with the same public API.
// Non-copyable; movable.
class HFTokenizer {
public:
    ~HFTokenizer();

    HFTokenizer(const HFTokenizer&)            = delete;
    HFTokenizer& operator=(const HFTokenizer&) = delete;
    HFTokenizer(HFTokenizer&&) noexcept;
    HFTokenizer& operator=(HFTokenizer&&) noexcept;

    // Load from a model directory containing tokenizer.json and tokenizer_config.json.
    static Result<HFTokenizer> from_model_dir(const std::filesystem::path& model_dir);

    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string      decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    const TokenizerConfig& config()        const { return config_; }
    size_t                 vocab_size()    const { return vocab_size_; }
    int                    bos_token_id() const { return bos_token_id_; }
    int                    eos_token_id() const { return eos_token_id_; }
    int                    unk_token_id() const { return unk_token_id_; }
    int                    pad_token_id() const { return pad_token_id_; }

    std::string get_token_string(int token_id) const;
    int         find_token_id(const std::string& token) const;

private:
    explicit HFTokenizer(TokenizerConfig config);

    TokenizerHandle* handle_       = nullptr;
    TokenizerConfig  config_;
    size_t           vocab_size_   = 0;
    int              bos_token_id_ = -1;
    int              eos_token_id_ = -1;
    int              unk_token_id_ = -1;
    int              pad_token_id_ = -1;
};

} // namespace compute
