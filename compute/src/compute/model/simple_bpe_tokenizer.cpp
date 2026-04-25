#include "simple_bpe_tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>

namespace compute {

SimpleBpeTokenizer::SimpleBpeTokenizer(TokenizerConfig config)
    : config_(std::move(config)) {
}

Result<SimpleBpeTokenizer> SimpleBpeTokenizer::from_model_dir(const std::filesystem::path& model_dir) {
    // 1. Load tokenizer config (reuse existing implementation)
    auto config_result = TokenizerConfig::from_config_file(model_dir);
    if (!config_result) {
        return std::unexpected(config_result.error());
    }

    // 2. Create tokenizer with config
    SimpleBpeTokenizer tokenizer(std::move(*config_result));

    // 3. Load vocabulary and merges from tokenizer.json
    auto tokenizer_json_path = model_dir / "tokenizer.json";
    auto vocab_result = tokenizer.load_vocabulary_from_json(tokenizer_json_path);
    if (!vocab_result) {
        return std::unexpected(vocab_result.error());
    }

    // 4. Setup special token IDs
    tokenizer.setup_special_token_ids();

    return tokenizer;
}

Result<void> SimpleBpeTokenizer::load_vocabulary_from_json(const std::filesystem::path& tokenizer_json_path) {
    if (!std::filesystem::exists(tokenizer_json_path)) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "tokenizer.json not found: " + tokenizer_json_path.string()});
    }

    try {
        // Parse JSON file
        std::ifstream file(tokenizer_json_path);
        if (!file.is_open()) {
            return std::unexpected(Error{ErrorCode::InvalidInput,
                                       "Failed to open tokenizer.json: " + tokenizer_json_path.string()});
        }

        nlohmann::json j;
        file >> j;

        // Verify JSON structure
        if (!j.contains("model") || !j["model"].contains("vocab") || !j["model"].contains("merges")) {
            return std::unexpected(Error{ErrorCode::InvalidInput,
                                       "Invalid tokenizer.json format: missing model/vocab/merges"});
        }

        // Load vocabulary: token -> id mapping
        const auto& vocab_obj = j["model"]["vocab"];
        for (const auto& [token, id] : vocab_obj.items()) {
            int token_id = id.get<int>();
            vocab_[token] = token_id;
            vocab_reverse_[token_id] = token;
        }

        // Load merges: ordered array where index = priority rank
        const auto& merges_array = j["model"]["merges"];
        merges_.clear();
        merge_ranks_.clear();

        for (size_t i = 0; i < merges_array.size(); ++i) {
            std::string left, right;
            const auto& m = merges_array[i];

            if (m.is_array() && m.size() == 2) {
                // Llama-3 / newer HuggingFace format: ["token1", "token2"]
                left  = m[0].get<std::string>();
                right = m[1].get<std::string>();
            } else if (m.is_string()) {
                // Classic HuggingFace format: "token1 token2"
                std::string merge_str = m.get<std::string>();
                auto space_pos = merge_str.find(' ');
                if (space_pos == std::string::npos) {
                    return std::unexpected(Error{ErrorCode::InvalidInput,
                                               "Invalid merge format: " + merge_str});
                }
                left  = merge_str.substr(0, space_pos);
                right = merge_str.substr(space_pos + 1);
            } else {
                return std::unexpected(Error{ErrorCode::InvalidInput,
                    "Unrecognised merge format at index " + std::to_string(i)});
            }

            merges_.emplace_back(left, right);
            merge_ranks_[{left, right}] = static_cast<int>(i); // Index = priority rank
        }

        // Load normalizer configuration and pre-tokenizer
        load_normalizer_config(j);

        // Load added_tokens (special tokens like [INST], [/INST], <s>, </s>)
        // These must be matched as whole strings before BPE runs.
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto& tok : j["added_tokens"]) {
                if (!tok.contains("content") || !tok.contains("id")) continue;
                std::string content = tok["content"].get<std::string>();
                int id = tok["id"].get<int>();
                if (!content.empty()) {
                    added_tokens_sorted_.emplace_back(content, id);
                    // Ensure the token is in the vocab lookup tables
                    vocab_[content]        = id;
                    vocab_reverse_[id]     = content;
                }
            }
            // Sort longest-first so we match [/INST] before [ in ambiguous cases
            std::sort(added_tokens_sorted_.begin(), added_tokens_sorted_.end(),
                [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        }

        // Detect pre-tokenizer type
        if (j.contains("pre_tokenizer") && !j["pre_tokenizer"].is_null()) {
            const auto& pt = j["pre_tokenizer"];
            if (pt.contains("type")) {
                const std::string pt_type = pt["type"].get<std::string>();
                if (pt_type == "Metaspace") {
                    use_metaspace_ = true;
                } else if (pt_type == "ByteLevel") {
                    use_byte_level_ = true;
                } else if (pt_type == "Sequence" && pt.contains("pretokenizers")) {
                    // Some models wrap ByteLevel in a Sequence pre-tokenizer
                    for (const auto& sub : pt["pretokenizers"]) {
                        if (sub.contains("type") && sub["type"] == "ByteLevel") {
                            use_byte_level_ = true;
                        }
                    }
                } else if (pt_type == "Split") {
                    // Gemma uses Split(" ", MergedWithPrevious).  After the normalizer
                    // has replaced every " " with "▁" there are no spaces left for the
                    // splitter to act on, so the whole string is one pre-token.  For
                    // our BPE, that is equivalent to running Metaspace-style: replace
                    // spaces with ▁ and segment at ▁ boundaries.
                    // We only activate this path when the normalizer is already doing
                    // the " " → "▁" replacement (replace_pattern_ / replace_content_
                    // are set by load_normalizer_config above).
                    if (replace_pattern_ == " " && replace_content_ == "▁") {
                        use_metaspace_ = true;
                    }
                }
            }
        }

        return {};

    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                               "JSON parsing error: " + std::string(e.what())});
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                               "Error loading tokenizer vocabulary: " + std::string(e.what())});
    }
}

void SimpleBpeTokenizer::setup_special_token_ids() {
    // Find special token IDs by looking up token strings in vocabulary
    bos_token_id_ = find_token_id(config_.bos_token);
    eos_token_id_ = find_token_id(config_.eos_token);
    unk_token_id_ = find_token_id(config_.unk_token);
    pad_token_id_ = find_token_id(config_.pad_token);

}

bool SimpleBpeTokenizer::is_special_token(int token_id) const {
    return token_id == bos_token_id_ || token_id == eos_token_id_ ||
           token_id == unk_token_id_ || token_id == pad_token_id_;
}

std::vector<std::string> SimpleBpeTokenizer::pre_tokenize(const std::string& text) const {
    if (text.empty()) {
        return {};
    }

    // Metaspace pre-tokenizer (Mistral / SentencePiece-with-space-replacement):
    // HuggingFace tokenizers uses SplitDelimiterBehavior::MergedWithNext — each ▁
    // marks the START of a new word segment. BPE must run on each word independently
    // so that ▁ can bind to its following characters (e.g. ▁+W+h+a+t → ▁What).
    // If we pass the whole string to BPE at once, W+h+a+t merges before ▁ can bind,
    // leaving ▁ as a spurious standalone token (the F.9 Mistral inference quality bug).
    if (use_metaspace_) {
        // UTF-8 encoding of ▁ (U+2581): E2 96 81
        static const unsigned char mark[3] = {0xE2, 0x96, 0x81};

        std::vector<std::string> segments;
        std::string current;

        size_t i = 0;
        while (i < text.size()) {
            // Detect the 3-byte ▁ sequence
            if (i + 2 < text.size() &&
                (unsigned char)text[i]   == mark[0] &&
                (unsigned char)text[i+1] == mark[1] &&
                (unsigned char)text[i+2] == mark[2])
            {
                // ▁ starts a new word segment — flush the current one first
                if (!current.empty()) {
                    segments.push_back(std::move(current));
                    current.clear();
                }
                // Begin new segment with the ▁ prefix
                current.push_back(text[i]);
                current.push_back(text[i+1]);
                current.push_back(text[i+2]);
                i += 3;
            } else {
                current.push_back(text[i++]);
            }
        }
        if (!current.empty()) {
            segments.push_back(std::move(current));
        }
        return segments;
    }

    // Normalizer path (TinyLlama / DEFAULT): BPE runs on the whole normalised text
    return {text};
}

std::vector<int> SimpleBpeTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int> result;

    // Add BOS token if configured
    if (add_special_tokens && config_.add_bos_token && bos_token_id_ != -1) {
        result.push_back(bos_token_id_);
    }

    // Step 1: Split text on added special tokens (e.g. [INST], [/INST]).
    // These must be emitted as their exact token IDs without going through BPE.
    // We split left-to-right, matching the longest added token at each position.
    struct Chunk { std::string text; bool is_special; int special_id = -1; };
    std::vector<Chunk> chunks;

    if (added_tokens_sorted_.empty()) {
        chunks.push_back({text, false});
    } else {
        size_t pos = 0;
        while (pos < text.size()) {
            bool found = false;
            for (const auto& [tok_str, tok_id] : added_tokens_sorted_) {
                if (text.compare(pos, tok_str.size(), tok_str) == 0) {
                    // Flush any accumulated plain text first
                    if (!chunks.empty() && !chunks.back().is_special) {
                        // already accumulated in back
                    }
                    chunks.push_back({"", false});       // placeholder — will be filled by plain accumulation
                    // Actually: emit the special token chunk
                    chunks.back() = {tok_str, true, tok_id};
                    pos += tok_str.size();
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Accumulate into a plain-text chunk
                if (chunks.empty() || chunks.back().is_special) {
                    chunks.push_back({"", false});
                }
                chunks.back().text += text[pos++];
            }
        }
    }

    // Step 2: For each chunk, either emit the special token ID or run BPE.
    for (const auto& chunk : chunks) {
        if (chunk.is_special) {
            result.push_back(chunk.special_id);
            continue;
        }
        if (chunk.text.empty()) continue;

        // Do NOT strip leading spaces — HuggingFace's SentencePiece tokenizer
        // keeps them.  A leading space on any chunk (whether it follows a special
        // token or is the very first chunk) is part of the input and must survive
        // normalization so that ` what` → `▁▁what` → [▁(29473), ▁what(1535)].
        // Stripping it here would drop the standalone ▁ token and diverge from HF.
        std::string chunk_text = chunk.text;

        if (use_byte_level_) {
            // ByteLevel path (Llama-3, GPT-2):
            // 1. Split raw text on GPT-2 word boundaries
            // 2. Byte-encode each segment
            // 3. BPE each encoded segment
            auto raw_segs = byte_level_split_raw(chunk_text);
            for (const auto& raw_seg : raw_segs) {
                if (raw_seg.empty()) continue;
                std::string encoded = apply_byte_level_encoding(raw_seg);
                auto seg_tokens = encode_word(encoded);
                result.insert(result.end(), seg_tokens.begin(), seg_tokens.end());
            }
            continue;
        }

        std::string prepared;
        if (use_metaspace_) {
            // Replace spaces with ▁.
            prepared = chunk_text;
            const std::string space_mark = "\xE2\x96\x81";
            std::string::size_type p = 0;
            while ((p = prepared.find(' ', p)) != std::string::npos) {
                prepared.replace(p, 1, space_mark);
                p += space_mark.size();
            }
            // Only prepend ▁ when the normalizer includes a Prepend step (e.g. Mistral).
            // Gemma's normalizer is a plain Replace with no Prepend, so its first word
            // has no ▁ prefix — matching HuggingFace behaviour.
            if (!prepend_text_.empty() &&
                prepared.compare(0, space_mark.size(), space_mark) != 0) {
                prepared = space_mark + prepared;
            }
        } else {
            prepared = apply_normalizer(chunk_text);
        }

        // Pre-tokenize and BPE
        auto segments = pre_tokenize(prepared);
        for (const auto& segment : segments) {
            if (segment.empty()) continue;
            auto segment_tokens = encode_word(segment);
            result.insert(result.end(), segment_tokens.begin(), segment_tokens.end());
        }
    }

    // Add EOS token if configured
    if (add_special_tokens && config_.add_eos_token && eos_token_id_ != -1) {
        result.push_back(eos_token_id_);
    }

    return result;
}

std::vector<int> SimpleBpeTokenizer::encode_word(const std::string& word) const {
    if (word.empty()) {
        return {};
    }

    // Try direct vocab lookup first (optimization for common words)
    auto direct_it = vocab_.find(word);
    if (direct_it != vocab_.end()) {
        return {direct_it->second};
    }

    // Apply full BPE algorithm
    // 1. Initialize symbols (character-level, UTF-8 aware)
    std::vector<Symbol> symbols;

    // Convert word to UTF-8 characters and create symbol linked list
    // Handle UTF-8 multi-byte characters properly
    size_t i = 0;
    int symbol_idx = 0;
    while (i < word.length()) {
        // Determine UTF-8 character length
        unsigned char c = static_cast<unsigned char>(word[i]);
        size_t char_len = 1;

        if ((c & 0x80) == 0) {
            // ASCII (0xxxxxxx)
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte UTF-8 (110xxxxx)
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte UTF-8 (1110xxxx)
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte UTF-8 (11110xxx)
            char_len = 4;
        }

        // Extract the complete UTF-8 character
        std::string utf8_char = word.substr(i, std::min(char_len, word.length() - i));
        symbols.emplace_back(utf8_char);

        // Set up linked list pointers
        if (symbol_idx > 0) {
            symbols[symbol_idx].prev = symbol_idx - 1;
            symbols[symbol_idx - 1].next = symbol_idx;
        }

        i += char_len;
        symbol_idx++;
    }

    if (symbols.empty()) {
        return {};
    }

    // 2. Build initial merge priority queue
    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>> merge_queue;

    for (int i = 0; i < static_cast<int>(symbols.size()) - 1; ++i) {
        if (symbols[i].next == i + 1) {  // Ensure they're adjacent
            add_merge_if_exists(symbols, merge_queue, i);
        }
    }

    // 3. Apply BPE merges iteratively
    while (!merge_queue.empty()) {
        Merge top_merge = merge_queue.top();
        merge_queue.pop();

        // Validate merge is still valid (symbols might have changed)
        if (!is_merge_valid(symbols, top_merge)) {
            continue;
        }

        // Apply the merge
        apply_merge(symbols, top_merge);

        // Add new potential merges around the merged position
        add_potential_merges_around(symbols, merge_queue, top_merge.left);
    }

    // 4. Convert final symbols to token IDs
    return symbols_to_token_ids(symbols);
}

std::string SimpleBpeTokenizer::decode(const std::vector<int>& token_ids, bool skip_special_tokens) const {
    // ── ByteLevel decode ──────────────────────────────────────────────────────
    // Each token string is a sequence of byte-mapped Unicode codepoints.
    // Map each codepoint back to its original byte, concatenate, return as UTF-8.
    if (use_byte_level_) {
        const auto& u2b = unicode_to_byte_map();
        std::string raw_bytes;
        for (int token_id : token_ids) {
            if (skip_special_tokens && is_special_token(token_id)) continue;
            auto it = vocab_reverse_.find(token_id);
            if (it == vocab_reverse_.end()) {
                raw_bytes += config_.unk_token;
                continue;
            }
            const std::string& tok = it->second;
            // Iterate over UTF-8 codepoints and reverse-map each one
            size_t j = 0;
            while (j < tok.size()) {
                unsigned char c = (unsigned char)tok[j];
                std::string cp;
                if (c < 0x80) {
                    cp = std::string(1, tok[j++]);
                } else if ((c & 0xE0) == 0xC0 && j + 1 < tok.size()) {
                    cp = tok.substr(j, 2); j += 2;
                } else if ((c & 0xF0) == 0xE0 && j + 2 < tok.size()) {
                    cp = tok.substr(j, 3); j += 3;
                } else {
                    cp = std::string(1, tok[j++]); // fallback: treat as single byte
                }
                auto bit = u2b.find(cp);
                if (bit != u2b.end()) {
                    raw_bytes += (char)bit->second;
                } else {
                    raw_bytes += cp; // unmapped codepoint — pass through
                }
            }
        }
        return raw_bytes;
    }

    std::ostringstream result;
    bool is_first_token = true;

    for (int token_id : token_ids) {
        // Skip special tokens if requested
        if (skip_special_tokens && is_special_token(token_id)) {
            continue;
        }

        // Look up token string
        auto it = vocab_reverse_.find(token_id);
        if (it != vocab_reverse_.end()) {
            std::string token = it->second;

            // Handle BPE whitespace encoding with ▁ (U+2581)
            // The ▁ character is 3 bytes in UTF-8: E2 96 81
            const char* space_char = "\xE2\x96\x81";  // ▁ character
            const size_t space_char_len = 3;

            // Count how many ▁ characters are in this token
            size_t space_count = 0;
            for (size_t i = 0; i + space_char_len <= token.length(); i += space_char_len) {
                if (token.substr(i, space_char_len) == space_char) {
                    space_count++;
                } else {
                    break;  // Stop at first non-▁ character
                }
            }

            if (space_count > 0 && space_count * space_char_len == token.length()) {
                // Token is ONLY ▁ characters (e.g., "▁" or "▁▁")
                if (is_first_token) {
                    // First ▁ was prepended, skip it. Remaining ▁ become spaces
                    for (size_t i = 1; i < space_count; ++i) {
                        result << " ";
                    }
                    is_first_token = false;
                } else {
                    // All ▁ characters represent spaces
                    for (size_t i = 0; i < space_count; ++i) {
                        result << " ";
                    }
                }
            } else if (space_count > 0) {
                // Token starts with ▁ but has other content (e.g., "▁Hello")
                if (is_first_token) {
                    // First ▁ was prepended, remove it
                    result << token.substr(space_char_len);
                    is_first_token = false;
                } else {
                    // ▁ represents space, replace with space
                    result << " " << token.substr(space_char_len);
                }
            } else if (token.size() == 6 && token[0] == '<' && token[1] == '0' &&
                       token[2] == 'x' && token[5] == '>') {
                // Byte fallback token like <0x0A> → decode to actual byte
                const char hex[3] = {token[3], token[4], '\0'};
                char byte_val = static_cast<char>(std::strtol(hex, nullptr, 16));
                result << byte_val;
                is_first_token = false;
            } else {
                // Token doesn't contain ▁ - just append it as-is
                result << token;
                is_first_token = false;
            }
        } else {
            // Unknown token ID - use UNK token string
            result << config_.unk_token;
            is_first_token = false;
        }
    }

    return result.str();
}

std::string SimpleBpeTokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt) const {

    std::ostringstream result;

    // ChatML format: used by Qwen2, Qwen3, and any model whose chat_template
    // references <|im_start|>/<|im_end|> control tokens.
    if (config_.chat_template.find("<|im_start|>") != std::string::npos) {
        for (const auto& [role, content] : messages) {
            result << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
        }
        if (add_generation_prompt) {
            result << "<|im_start|>assistant\n";
        }
        return result.str();
    }

    // Fallback: TinyLlama / Zephyr-style tags
    for (const auto& [role, content] : messages) {
        if (role == "user") {
            result << "<|user|>\n" << content << "\n";
        } else if (role == "assistant") {
            result << "<|assistant|>\n" << content << "\n";
        } else if (role == "system") {
            result << "<|system|>\n" << content << "\n";
        }
    }
    if (add_generation_prompt) {
        result << "<|assistant|>\n";
    }
    return result.str();
}

// BPE Algorithm helper methods
void SimpleBpeTokenizer::apply_merge(std::vector<Symbol>& symbols, const Merge& merge) const {
    // Merge left and right symbols into left position
    int left_idx = merge.left;
    int right_idx = merge.right;

    // Create merged text
    symbols[left_idx].text += symbols[right_idx].text;
    symbols[left_idx].len = symbols[left_idx].text.length();

    // Update linked list: left now points to what right was pointing to
    symbols[left_idx].next = symbols[right_idx].next;

    // Update next symbol's prev pointer if it exists
    if (symbols[right_idx].next != -1) {
        symbols[symbols[right_idx].next].prev = left_idx;
    }

    // Mark right symbol as deleted by clearing its text and breaking links
    symbols[right_idx].text.clear();
    symbols[right_idx].prev = -1;
    symbols[right_idx].next = -1;
}

bool SimpleBpeTokenizer::is_merge_valid(const std::vector<Symbol>& symbols, const Merge& merge) const {
    int left_idx = merge.left;
    int right_idx = merge.right;

    // Check that indices are valid
    if (left_idx < 0 || left_idx >= static_cast<int>(symbols.size()) ||
        right_idx < 0 || right_idx >= static_cast<int>(symbols.size())) {
        return false;
    }

    // Check that symbols haven't been deleted
    if (symbols[left_idx].text.empty() || symbols[right_idx].text.empty()) {
        return false;
    }

    // Check that they are still adjacent
    if (symbols[left_idx].next != right_idx) {
        return false;
    }

    // CRITICAL: verify symbol texts still match the merge that was queued.
    // A higher-priority merge may have consumed one of these symbols and
    // replaced its text (e.g. a queued (▁,▁) merge becomes invalid after
    // (▁,w)→▁w rewrote symbol[1], then (▁w,hat)→▁what rewrote it again).
    // Without this check we blindly concatenate stale text and produce
    // tokens not in the vocab, causing byte-fallback garbage.
    if (symbols[left_idx].text != merge.left_text ||
        symbols[right_idx].text != merge.right_text) {
        return false;
    }

    return true;
}

void SimpleBpeTokenizer::add_merge_if_exists(const std::vector<Symbol>& symbols,
                                            std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                                            int left_pos) const {
    int right_pos = symbols[left_pos].next;
    if (right_pos == -1) {
        return; // No right neighbor
    }

    // Check if symbols are still valid (not deleted)
    if (symbols[left_pos].text.empty() || symbols[right_pos].text.empty()) {
        return;
    }

    const std::string& left_text = symbols[left_pos].text;
    const std::string& right_text = symbols[right_pos].text;

    // Look up merge in merge rules
    auto merge_it = merge_ranks_.find({left_text, right_text});
    if (merge_it != merge_ranks_.end()) {
        // Found a merge rule - create merged text and look up token ID
        std::string merged_text = left_text + right_text;
        auto token_it = vocab_.find(merged_text);

        if (token_it != vocab_.end()) {
            Merge merge;
            merge.left = left_pos;
            merge.right = right_pos;
            merge.rank = merge_it->second;
            merge.new_token_id = token_it->second;
            merge.left_text = left_text;
            merge.right_text = right_text;

            merge_queue.push(merge);
        }
    }
}

void SimpleBpeTokenizer::add_potential_merges_around(const std::vector<Symbol>& symbols,
                                                    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                                                    int pos) const {
    // Add merge with left neighbor
    if (symbols[pos].prev != -1) {
        add_merge_if_exists(symbols, merge_queue, symbols[pos].prev);
    }

    // Add merge with right neighbor
    add_merge_if_exists(symbols, merge_queue, pos);
}

void SimpleBpeTokenizer::add_potential_merges(const std::vector<Symbol>& symbols,
                                             std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                                             int pos) const {
    // This method is used for adding initial merges - same as around for now
    add_potential_merges_around(symbols, merge_queue, pos);
}

std::vector<int> SimpleBpeTokenizer::symbols_to_token_ids(const std::vector<Symbol>& symbols) const {
    std::vector<int> token_ids;

    for (const auto& symbol : symbols) {
        // Skip deleted symbols (empty text)
        if (symbol.text.empty()) {
            continue;
        }

        // Look up token in vocabulary
        auto it = vocab_.find(symbol.text);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Byte fallback: encode each byte as <0xHH> (sentencepiece convention)
            bool all_bytes_found = true;
            std::vector<int> byte_ids;
            for (unsigned char c : symbol.text) {
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", c);
                auto bit = vocab_.find(std::string(hex));
                if (bit != vocab_.end()) {
                    byte_ids.push_back(bit->second);
                } else {
                    all_bytes_found = false;
                    break;
                }
            }
            if (all_bytes_found && !byte_ids.empty()) {
                for (int id : byte_ids) token_ids.push_back(id);
            } else if (unk_token_id_ != -1) {
                token_ids.push_back(unk_token_id_);
            }
        }
    }

    return token_ids;
}

// Normalizer implementation
void SimpleBpeTokenizer::load_normalizer_config(const nlohmann::json& j) {
    // Load normalizer configuration from tokenizer.json
    // TinyLlama / Mistral use: Sequence[Prepend("▁"), Replace(" ", "▁")]
    // Gemma uses:              Replace(" ", "▁")  (top-level, no Prepend)

    if (!j.contains("normalizer") || j["normalizer"].is_null()) {
        return;
    }

    const auto& normalizer = j["normalizer"];
    if (!normalizer.contains("type")) return;

    const std::string norm_type = normalizer["type"].get<std::string>();

    // Helper lambda to parse a single Replace step
    auto parse_replace = [&](const nlohmann::json& norm) {
        if (norm.contains("pattern") && norm.contains("content")) {
            const auto& pattern = norm["pattern"];
            if (pattern.contains("String")) {
                replace_pattern_ = pattern["String"].get<std::string>();
            }
            replace_content_ = norm["content"].get<std::string>();
        }
    };

    if (norm_type == "Sequence") {
        const auto& normalizers = normalizer["normalizers"];
        for (const auto& norm : normalizers) {
            if (!norm.contains("type")) continue;
            std::string type = norm["type"];
            if (type == "Prepend") {
                if (norm.contains("prepend")) {
                    prepend_text_ = norm["prepend"].get<std::string>();
                }
            } else if (type == "Replace") {
                parse_replace(norm);
            }
        }
    } else if (norm_type == "Replace") {
        // Top-level Replace (Gemma): no Prepend step, no prefix space
        parse_replace(normalizer);
        // prepend_text_ intentionally left empty — Gemma does NOT prefix ▁
    }
}

std::string SimpleBpeTokenizer::apply_normalizer(const std::string& text) const {
    if (text.empty()) {
        return text;
    }

    std::string result = text;

    // Step 1: Prepend text if configured
    if (!prepend_text_.empty()) {
        result = prepend_text_ + result;
    }

    // Step 2: Replace pattern if configured
    if (!replace_pattern_.empty() && !replace_content_.empty()) {
        std::string::size_type pos = 0;
        while ((pos = result.find(replace_pattern_, pos)) != std::string::npos) {
            result.replace(pos, replace_pattern_.length(), replace_content_);
            pos += replace_content_.length();
        }
    }

    return result;
}

// ── ByteLevel pre-tokenizer ────────────────────────────────────────────────────
//
// Used by GPT-2 / Llama-3 family models.  Every input byte is mapped to a
// printable Unicode codepoint before BPE runs, so the vocabulary never needs
// byte-fallback tokens like <0xHH>.  Decoding just reverses the mapping.

const std::vector<std::string>& SimpleBpeTokenizer::byte_to_unicode_table() {
    static std::vector<std::string> table;
    if (!table.empty()) return table;

    // Bytes whose codepoints = themselves (printable ASCII + Latin-1 ranges)
    std::set<int> safe;
    for (int b = 33; b <= 126; ++b) safe.insert(b);
    for (int b = 161; b <= 172; ++b) safe.insert(b);
    for (int b = 174; b <= 255; ++b) safe.insert(b);

    // UTF-8 encoder for codepoints < 0x10000
    auto utf8 = [](uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    };

    table.resize(256);
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        table[b] = safe.count(b) ? utf8((uint32_t)b) : utf8(256u + (uint32_t)n++);
    }
    return table;
}

const std::unordered_map<std::string, uint8_t>& SimpleBpeTokenizer::unicode_to_byte_map() {
    static std::unordered_map<std::string, uint8_t> map;
    if (!map.empty()) return map;
    const auto& fwd = byte_to_unicode_table();
    for (int b = 0; b < 256; ++b) map[fwd[b]] = static_cast<uint8_t>(b);
    return map;
}

std::string SimpleBpeTokenizer::apply_byte_level_encoding(const std::string& text) {
    const auto& tbl = byte_to_unicode_table();
    std::string result;
    result.reserve(text.size() * 2); // worst case: all 2-byte
    for (unsigned char c : text) result += tbl[c];
    return result;
}

// Split raw text into word-like segments following the GPT-4 / Llama-3 regex:
//   '(?i:[sdmt]|ll|ve|re)                    — contractions
//   | [^\r\n\p{L}\p{N}]?\p{L}+              — optional 1 non-letter/digit/newline + letters
//   | \p{N}{1,3}                              — 1–3 digits
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*              — optional space + non-word chars
//   | \s*[\r\n]                               — newline
//   | \s+(?!\S)                               — trailing whitespace
//   | \s+                                     — remaining whitespace
//
// Approximation: uses C character classes for ASCII; non-ASCII multibyte sequences
// are treated as letters (covers common Unicode scripts correctly for most prompts).
std::vector<std::string> SimpleBpeTokenizer::byte_level_split_raw(const std::string& text) {
    std::vector<std::string> result;
    if (text.empty()) return result;

    const size_t n = text.size();
    size_t i = 0;

    // Character classification helpers that treat non-ASCII bytes as "letter"
    auto is_ascii_letter = [](unsigned char c) -> bool {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    };
    auto is_digit = [](unsigned char c) -> bool { return c >= '0' && c <= '9'; };
    auto is_space = [](unsigned char c) -> bool { return c == ' ' || c == '\t'; };
    auto is_newline = [](unsigned char c) -> bool { return c == '\n' || c == '\r'; };
    // Non-ASCII start byte (0xC0+) = likely a Unicode letter/symbol
    auto is_multibyte_start = [](unsigned char c) -> bool { return c >= 0xC0; };
    auto is_continuation = [](unsigned char c) -> bool { return (c & 0xC0) == 0x80; };
    auto is_letter = [&](unsigned char c) -> bool {
        return is_ascii_letter(c) || is_multibyte_start(c);
    };
    auto is_letter_or_digit = [&](unsigned char c) -> bool {
        return is_letter(c) || is_digit(c);
    };

    // Consume a full UTF-8 codepoint starting at `pos` and append to `out`.
    auto consume_codepoint = [&](size_t& pos, std::string& out) {
        out += text[pos++];
        while (pos < n && is_continuation((unsigned char)text[pos])) out += text[pos++];
    };

    while (i < n) {
        unsigned char c = (unsigned char)text[i];
        std::string seg;

        // ── 1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive) ──
        if (c == '\'') {
            static const char* cnts[] = {"'ll","'ve","'re","'t","'s","'m","'d",nullptr};
            bool matched = false;
            for (int k = 0; cnts[k]; ++k) {
                size_t len = std::strlen(cnts[k]);
                if (i + len <= n) {
                    bool ok = true;
                    for (size_t j = 0; j < len && ok; ++j)
                        ok = (std::tolower((unsigned char)text[i+j]) == (unsigned char)cnts[k][j]);
                    if (ok) {
                        result.push_back(text.substr(i, len));
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }
            if (matched) continue;
            // Fall through: treat bare ' as punctuation
        }

        // ── 5. Newline (possibly with leading whitespace): \s*[\r\n] ──
        if (is_newline(c)) {
            seg += text[i++];
            result.push_back(seg);
            continue;
        }

        // ── 2. Optional non-letter/digit/newline + letter sequence ──
        // Covers: "word", " word", ",word", etc.
        {
            // Look ahead: is there a letter coming (possibly after one non-letter/digit)?
            bool has_prefix = false;
            size_t look = i;
            unsigned char lc = (unsigned char)text[look];

            if (!is_letter_or_digit(lc) && !is_newline(lc) && !is_space(lc)) {
                // One non-letter/digit/newline prefix char (e.g. apostrophe handled above, punctuation)
                size_t tmp = look;
                consume_codepoint(tmp, seg);
                lc = tmp < n ? (unsigned char)text[tmp] : 0;
                if (lc && is_letter(lc)) { look = tmp; has_prefix = true; }
                else { seg.clear(); } // not letter next — handle as punctuation below
            } else if (is_space(lc)) {
                // Space: optional before letter (pattern 2) or punctuation (pattern 4)
                size_t after_space = look + 1;
                if (after_space < n && is_letter((unsigned char)text[after_space])) {
                    seg += text[i++]; // consume the space
                    look = i;
                    lc = (unsigned char)text[look];
                    has_prefix = true;
                } else {
                    // Not space+letter — fall to whitespace/punctuation handling below
                }
            }

            if (look < n && is_letter((unsigned char)text[look])) {
                i = look;
                while (i < n && is_letter((unsigned char)text[i]))
                    consume_codepoint(i, seg);
                result.push_back(seg);
                continue;
            }
            // seg may have a prefix char that didn't lead to letters — fall through
            seg.clear();
            i = static_cast<size_t>(look > i ? i : i); // reset look
        }

        c = (unsigned char)text[i]; // refresh after possible look-ahead reset

        // ── 6/7. Whitespace ──
        if (is_space(c)) {
            // Check: space + punctuation (non-letter/digit/space/newline) → pattern 4
            if (i + 1 < n) {
                unsigned char next = (unsigned char)text[i+1];
                if (!is_space(next) && !is_newline(next) && !is_letter_or_digit(next)) {
                    seg += text[i++]; // consume space
                    // consume non-word chars
                    while (i < n) {
                        unsigned char nc = (unsigned char)text[i];
                        if (is_space(nc) || is_newline(nc) || is_letter_or_digit(nc)) break;
                        consume_codepoint(i, seg);
                    }
                    // consume trailing newlines
                    while (i < n && is_newline((unsigned char)text[i])) seg += text[i++];
                    result.push_back(seg);
                    continue;
                }
            }
            // Standalone whitespace
            while (i < n && is_space((unsigned char)text[i])) seg += text[i++];
            result.push_back(seg);
            continue;
        }

        // ── 3. Digit sequence (1–3 digits) ──
        if (is_digit(c)) {
            int cnt = 0;
            while (i < n && is_digit((unsigned char)text[i]) && cnt < 3) {
                seg += text[i++]; ++cnt;
            }
            result.push_back(seg);
            continue;
        }

        // ── 4. Punctuation / other non-word chars (no leading space) ──
        while (i < n) {
            unsigned char nc = (unsigned char)text[i];
            if (is_space(nc) || is_newline(nc) || is_letter_or_digit(nc) || nc == '\'') break;
            consume_codepoint(i, seg);
        }
        // consume trailing newlines (per pattern 4)
        while (i < n && is_newline((unsigned char)text[i])) seg += text[i++];
        if (!seg.empty()) result.push_back(seg);
    }

    return result;
}

} // namespace compute