#include "hf_tokenizer.h"
#include "tokenizers_c.h"

#include <algorithm>
#include <vector>

namespace compute {

HFTokenizer::~HFTokenizer() {
    tokenizer_free(handle_);
}

HFTokenizer::HFTokenizer(HFTokenizer&& o) noexcept
    : handle_(o.handle_)
    , config_(std::move(o.config_))
    , vocab_size_(o.vocab_size_)
    , bos_token_id_(o.bos_token_id_)
    , eos_token_id_(o.eos_token_id_)
    , unk_token_id_(o.unk_token_id_)
    , pad_token_id_(o.pad_token_id_)
{
    o.handle_ = nullptr;
}

HFTokenizer& HFTokenizer::operator=(HFTokenizer&& o) noexcept {
    if (this != &o) {
        tokenizer_free(handle_);
        handle_       = o.handle_;
        config_       = std::move(o.config_);
        vocab_size_   = o.vocab_size_;
        bos_token_id_ = o.bos_token_id_;
        eos_token_id_ = o.eos_token_id_;
        unk_token_id_ = o.unk_token_id_;
        pad_token_id_ = o.pad_token_id_;
        o.handle_     = nullptr;
    }
    return *this;
}

HFTokenizer::HFTokenizer(TokenizerConfig config) : config_(std::move(config)) {}

Result<HFTokenizer> HFTokenizer::from_model_dir(const std::filesystem::path& model_dir) {
    auto cfg_result = TokenizerConfig::from_config_file(model_dir);
    if (!cfg_result.has_value())
        return std::unexpected(cfg_result.error());

    const auto tok_json = model_dir / "tokenizer.json";
    TokenizerHandle* handle = tokenizer_from_file(tok_json.c_str());
    if (!handle)
        return std::unexpected(Error{ErrorCode::InvalidModel, "failed to load tokenizer.json from " + model_dir.string()});

    HFTokenizer tok(std::move(*cfg_result));
    tok.handle_     = handle;
    tok.vocab_size_ = static_cast<size_t>(tokenizer_vocab_size(handle));

    auto lookup = [&](const std::string& s) -> int {
        return s.empty() ? -1 : tokenizer_token_to_id(handle, s.c_str());
    };
    tok.bos_token_id_ = lookup(tok.config_.bos_token);
    tok.eos_token_id_ = lookup(tok.config_.eos_token);
    tok.unk_token_id_ = lookup(tok.config_.unk_token);
    tok.pad_token_id_ = lookup(tok.config_.pad_token);

    return tok;
}

std::vector<int> HFTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    constexpr int32_t kInitialCap = 8192;
    std::vector<int32_t> buf(kInitialCap);
    int32_t count = tokenizer_encode(handle_, text.c_str(), add_special_tokens,
                                     buf.data(), kInitialCap);
    if (count < 0) return {};
    if (count > kInitialCap) {
        buf.resize(count);
        tokenizer_encode(handle_, text.c_str(), add_special_tokens, buf.data(), count);
    }
    return std::vector<int>(buf.begin(), buf.begin() + count);
}

std::string HFTokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::vector<int32_t> i32(ids.begin(), ids.end());
    char* s = tokenizer_decode(handle_, i32.data(), static_cast<int32_t>(i32.size()),
                               skip_special_tokens);
    if (!s) return {};
    std::string result(s);
    tokenizer_free_string(s);
    return result;
}

std::string HFTokenizer::get_token_string(int token_id) const {
    char* s = tokenizer_id_to_token(handle_, token_id);
    if (!s) return {};
    std::string result(s);
    tokenizer_free_string(s);
    return result;
}

int HFTokenizer::find_token_id(const std::string& token) const {
    return tokenizer_token_to_id(handle_, token.c_str());
}

} // namespace compute
