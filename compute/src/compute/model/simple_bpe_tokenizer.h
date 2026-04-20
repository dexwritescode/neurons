#pragma once

#include "../core/compute_types.h"
#include "tokenizer_config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <utility>
#include <queue>
#include <nlohmann/json.hpp>

namespace compute {

/**
 * Simple BPE (Byte Pair Encoding) tokenizer implementation
 * Optimized for TinyLlama and similar models with DEFAULT pre-tokenization
 *
 * Features:
 * - Loads vocabulary and merges from HuggingFace tokenizer.json
 * - Implements core BPE algorithm with priority queue
 * - Simple whitespace-based pre-tokenization (LLAMA_VOCAB_PRE_TYPE_DEFAULT)
 * - Special token handling (BOS/EOS/UNK/PAD)
 * - Chat template support via TokenizerConfig
 * - Pure C++ implementation for fast inference
 */
class SimpleBpeTokenizer {
public:
    /**
     * Factory method to create tokenizer from model directory
     * @param model_dir Path to model directory containing tokenizer files
     * @return Result containing tokenizer or error
     */
    static Result<SimpleBpeTokenizer> from_model_dir(const std::filesystem::path& model_dir);

    /**
     * Encode text to token IDs
     * @param text Input text to tokenize
     * @param add_special_tokens Whether to add BOS/EOS tokens based on config
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;

    /**
     * Decode token IDs to text
     * @param token_ids Vector of token IDs to decode
     * @param skip_special_tokens Whether to skip special tokens in output
     * @return Decoded text string
     */
    std::string decode(const std::vector<int>& token_ids, bool skip_special_tokens = true) const;

    /**
     * Apply chat template to conversation
     * @param messages Vector of (role, content) pairs
     * @param add_generation_prompt Whether to add generation prompt at end
     * @return Formatted chat string ready for encoding
     */
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = false) const;

    // ── ByteLevel (GPT-2 / Llama-3) static helpers ───────────────────────────
    // Exposed publicly so unit tests can verify table correctness without a model.

    /** 256-entry GPT-2 byte→Unicode table. Index b holds the UTF-8 string for byte b. */
    static const std::vector<std::string>& byte_to_unicode_table();

    /** Reverse table: UTF-8 encoded Unicode char → original byte. */
    static const std::unordered_map<std::string, uint8_t>& unicode_to_byte_map();

    /** Map every byte in `text` through the byte→Unicode table. */
    static std::string apply_byte_level_encoding(const std::string& text);

    /** Split raw text on GPT-2/Llama-3 word boundaries (before byte encoding). */
    static std::vector<std::string> byte_level_split_raw(const std::string& text);

    // Accessors
    const TokenizerConfig& config() const { return config_; }
    size_t vocab_size() const { return vocab_.size(); }
    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int unk_token_id() const { return unk_token_id_; }
    int pad_token_id() const { return pad_token_id_; }

    /**
     * Get token string for a token ID (for debugging)
     * @param token_id Token ID to look up
     * @return Token string or empty string if not found
     */
    std::string get_token_string(int token_id) const {
        auto it = vocab_reverse_.find(token_id);
        return (it != vocab_reverse_.end()) ? it->second : "";
    }

    /** Look up a token string in the vocabulary; returns -1 if not found. */
    int find_token_id(const std::string& token) const {
        auto it = vocab_.find(token);
        return (it != vocab_.end()) ? it->second : -1;
    }

private:
    /**
     * Private constructor - use from_model_dir() factory method
     */
    explicit SimpleBpeTokenizer(TokenizerConfig config);

    /**
     * Load vocabulary and merges from tokenizer.json
     * @param tokenizer_json_path Path to tokenizer.json file
     * @return Result indicating success or error
     */
    Result<void> load_vocabulary_from_json(const std::filesystem::path& tokenizer_json_path);

    /**
     * Setup special token IDs by looking up token strings in vocabulary
     */
    void setup_special_token_ids();

    /**
     * Pre-tokenize text using simple whitespace splitting
     * @param text Input text
     * @return Vector of text segments to process with BPE
     */
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /**
     * Apply BPE algorithm to a single word/segment
     * @param word Input word to tokenize
     * @return Vector of token IDs for this word
     */
    std::vector<int> encode_word(const std::string& word) const;


    /**
     * Check if a token ID represents a special token
     * @param token_id Token ID to check
     * @return True if token is special (BOS/EOS/UNK/PAD)
     */
    bool is_special_token(int token_id) const;

    // Integration with existing TokenizerConfig (keeps all chat template logic)
    TokenizerConfig config_;

    // BPE-specific data structures
    std::unordered_map<std::string, int> vocab_;           // token -> id mapping
    std::unordered_map<int, std::string> vocab_reverse_;   // id -> token mapping
    std::vector<std::pair<std::string, std::string>> merges_; // merge rules in priority order

    // Fast merge lookup: (token1, token2) -> merge_rank
    // Custom hash function for string pairs
    struct PairHash {
        std::size_t operator()(const std::pair<std::string, std::string>& p) const {
            std::size_t h1 = std::hash<std::string>{}(p.first);
            std::size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 << 1); // Simple hash combination
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;

    // Special token IDs (extracted from vocab using config token strings)
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    int unk_token_id_ = -1;
    int pad_token_id_ = -1;

    // Normalizer/pre-tokenizer configuration (loaded from tokenizer.json)
    // These are applied BEFORE BPE tokenization
    std::string prepend_text_;       // Text to prepend to all input (e.g., "▁")
    std::string replace_pattern_;    // Pattern to replace (e.g., " ")
    std::string replace_content_;    // Replacement text (e.g., "▁")
    bool use_metaspace_ = false;     // True if pre_tokenizer is Metaspace (Mistral)
    bool use_byte_level_ = false;    // True if pre_tokenizer is ByteLevel (GPT-2 / Llama-3)

    // Added tokens (special tokens like [INST], [/INST]) — matched before BPE.
    // Sorted longest-first to avoid prefix conflicts.
    std::vector<std::pair<std::string, int>> added_tokens_sorted_;

    /**
     * Apply normalizer pipeline to text (prepend + replace operations)
     * @param text Input text
     * @return Normalized text ready for BPE processing
     */
    std::string apply_normalizer(const std::string& text) const;

    /**
     * Load normalizer configuration from tokenizer.json
     * @param j JSON object from tokenizer.json
     */
    void load_normalizer_config(const nlohmann::json& j);

private:
    // BPE algorithm data structures (inspired by HuggingFace tokenizers + llama.cpp)

    /**
     * Symbol represents a text segment during BPE processing
     * Uses linked list structure for efficient merging
     */
    struct Symbol {
        std::string text;     // Text content of this symbol
        int prev = -1;        // Index of previous symbol (-1 if none)
        int next = -1;        // Index of next symbol (-1 if none)
        size_t len;           // Length of text (optimization)

        Symbol(std::string text_) : text(std::move(text_)), len(text.length()) {}
    };

    /**
     * Merge represents a potential BPE merge operation
     * Used in priority queue for BPE algorithm
     */
    struct Merge {
        int left;             // Left symbol index to merge
        int right;            // Right symbol index to merge
        int rank;             // Priority rank (lower = higher priority)
        int new_token_id;     // Token ID after merging
        // Snapshot of left/right symbol text at queue-time. When a higher-priority
        // merge consumes one of these symbols, its text changes; we must reject
        // this merge rather than blindly concatenating stale text.
        std::string left_text;
        std::string right_text;

        // Priority queue ordering: lower rank = higher priority
        bool operator>(const Merge& other) const {
            return rank > other.rank || (rank == other.rank && left > other.left);
        }
    };

    /**
     * Apply a merge operation to the symbols list
     * @param symbols Symbol list to modify
     * @param merge Merge operation to apply
     */
    void apply_merge(std::vector<Symbol>& symbols, const Merge& merge) const;

    /**
     * Check if a merge operation is still valid
     * @param symbols Current symbol list
     * @param merge Merge to validate
     * @return True if merge can still be applied
     */
    bool is_merge_valid(const std::vector<Symbol>& symbols, const Merge& merge) const;

    /**
     * Add potential new merges around a position
     * @param symbols Symbol list
     * @param merge_queue Priority queue to add merges to
     * @param pos Position around which to look for new merges
     */
    void add_potential_merges(const std::vector<Symbol>& symbols,
                             std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                             int pos) const;

    /**
     * Add a merge to queue if the merge rule exists
     * @param symbols Symbol list
     * @param merge_queue Priority queue to add merge to
     * @param left_pos Position of left symbol
     */
    void add_merge_if_exists(const std::vector<Symbol>& symbols,
                            std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                            int left_pos) const;

    /**
     * Add potential merges around a merged position
     * @param symbols Symbol list
     * @param merge_queue Priority queue
     * @param pos Position that was just merged
     */
    void add_potential_merges_around(const std::vector<Symbol>& symbols,
                                    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>>& merge_queue,
                                    int pos) const;

    /**
     * Convert final symbols to token IDs
     * @param symbols Final symbol list after BPE
     * @return Vector of token IDs
     */
    std::vector<int> symbols_to_token_ids(const std::vector<Symbol>& symbols) const;
};

} // namespace compute