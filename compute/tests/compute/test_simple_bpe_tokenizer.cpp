#include <gtest/gtest.h>
#include "compute/model/simple_bpe_tokenizer.h"
#include "test_config.h"
#include <filesystem>

namespace compute {

class SimpleBpeTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use configured paths from CMake
        tinyllama_model_dir = TINYLLAMA_MODEL_DIR;
    }

    std::filesystem::path tinyllama_model_dir;
};

TEST_F(SimpleBpeTokenizerTest, LoadTokenizerFromModelDir) {
    if (!std::filesystem::exists(tinyllama_model_dir))
        GTEST_SKIP() << "Model not found: " << tinyllama_model_dir;

    // Verify required files exist
    auto tokenizer_json = tinyllama_model_dir / "tokenizer.json";
    auto tokenizer_config_json = tinyllama_model_dir / "tokenizer_config.json";

    if (!std::filesystem::exists(tokenizer_json))
        GTEST_SKIP() << "tokenizer.json not found: " << tokenizer_json;
    if (!std::filesystem::exists(tokenizer_config_json))
        GTEST_SKIP() << "tokenizer_config.json not found: " << tokenizer_config_json;

    // Load tokenizer
    auto result = SimpleBpeTokenizer::from_model_dir(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value())
        << "Failed to load SimpleBpeTokenizer: " << result.error().message;

    const auto& tokenizer = *result;

    // Basic validation
    EXPECT_GT(tokenizer.vocab_size(), 0);
    EXPECT_EQ(tokenizer.vocab_size(), 32000); // TinyLlama vocab size

    // Check special tokens were found
    EXPECT_NE(tokenizer.bos_token_id(), -1) << "BOS token not found";
    EXPECT_NE(tokenizer.eos_token_id(), -1) << "EOS token not found";

    std::cout << "Tokenizer loaded successfully:" << std::endl;
    std::cout << "  Vocab size: " << tokenizer.vocab_size() << std::endl;
    std::cout << "  BOS token ID: " << tokenizer.bos_token_id() << std::endl;
    std::cout << "  EOS token ID: " << tokenizer.eos_token_id() << std::endl;
    std::cout << "  UNK token ID: " << tokenizer.unk_token_id() << std::endl;
    std::cout << "  PAD token ID: " << tokenizer.pad_token_id() << std::endl;
}

TEST_F(SimpleBpeTokenizerTest, BasicEncodingDecoding) {
    auto result = SimpleBpeTokenizer::from_model_dir(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load tokenizer";

    const auto& tokenizer = *result;

    // Test simple encoding - debug the tokenization process
    std::string test_text = "Hello world";
    auto tokens = tokenizer.encode(test_text, false); // Don't add special tokens for now

    EXPECT_FALSE(tokens.empty()) << "Encoding produced no tokens";

    // Test decoding
    auto decoded = tokenizer.decode(tokens, true); // Skip special tokens
    EXPECT_FALSE(decoded.empty()) << "Decoding produced empty string";

    std::cout << "Encoding test:" << std::endl;
    std::cout << "  Input: \"" << test_text << "\"" << std::endl;
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  Decoded: \"" << decoded << "\"" << std::endl;

    // Debug: Check what each token represents
    std::cout << "  Token details:" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string token_str = tokenizer.get_token_string(tokens[i]);
        if (!token_str.empty()) {
            std::cout << "    " << tokens[i] << " -> \"" << token_str << "\"" << std::endl;
        } else {
            std::cout << "    " << tokens[i] << " -> <NOT FOUND>" << std::endl;
        }
    }
}

TEST_F(SimpleBpeTokenizerTest, SpecialTokenHandling) {
    auto result = SimpleBpeTokenizer::from_model_dir(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load tokenizer";

    const auto& tokenizer = *result;

    // Test with special tokens
    std::string test_text = "Hello";
    auto tokens_with_special = tokenizer.encode(test_text, true);
    auto tokens_without_special = tokenizer.encode(test_text, false);

    std::cout << "Special token test:" << std::endl;
    std::cout << "  Without special tokens: " << tokens_without_special.size() << " tokens" << std::endl;
    std::cout << "  With special tokens: " << tokens_with_special.size() << " tokens" << std::endl;

    // With special tokens should have more tokens (BOS/EOS)
    if (tokenizer.config().add_bos_token || tokenizer.config().add_eos_token) {
        EXPECT_GT(tokens_with_special.size(), tokens_without_special.size());
    }
}

TEST_F(SimpleBpeTokenizerTest, PreTokenization) {
    auto result = SimpleBpeTokenizer::from_model_dir(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load tokenizer";

    const auto& tokenizer = *result;

    // Test various pre-tokenization cases
    std::vector<std::string> test_cases = {
        "hello world",
        "The quick brown fox",
        "can't won't don't",
        "  multiple   spaces  ",
        "punctuation!@#$%",
        ""
    };

    for (const auto& test : test_cases) {
        auto tokens = tokenizer.encode(test, false);
        auto decoded = tokenizer.decode(tokens, true);

        std::cout << "Test: \"" << test << "\" -> " << tokens.size() << " tokens" << std::endl;

        // Basic sanity checks
        if (!test.empty()) {
            EXPECT_FALSE(tokens.empty()) << "Empty test should not produce tokens: \"" << test << "\"";
        }
    }
}

TEST_F(SimpleBpeTokenizerTest, ChatTemplate) {
    auto result = SimpleBpeTokenizer::from_model_dir(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value()) << "Failed to load tokenizer";

    const auto& tokenizer = *result;

    // Test chat template
    std::vector<std::pair<std::string, std::string>> conversation = {
        {"user", "Hello, how are you?"},
        {"assistant", "I'm doing well, thank you!"}
    };

    auto formatted = tokenizer.apply_chat_template(conversation, false);
    EXPECT_FALSE(formatted.empty()) << "Chat template should produce formatted text";

    std::cout << "Chat template test:" << std::endl;
    std::cout << "  Formatted: \"" << formatted << "\"" << std::endl;

    // Test encoding the formatted chat
    auto tokens = tokenizer.encode(formatted, true);
    EXPECT_FALSE(tokens.empty()) << "Formatted chat should encode to tokens";

    std::cout << "  Encoded to " << tokens.size() << " tokens" << std::endl;
}

// Regression for a priority-queue BPE bug where a queued merge became stale
// after a higher-priority merge rewrote one of its symbols. The old code
// validated adjacency but not symbol text, so the stale merge blindly
// concatenated "▁" + "▁what" → "▁▁what" (not in vocab) and fell back to
// 10 raw byte-fallback tokens instead of the correct 2 tokens.
//
// The trigger is any input that produces "▁▁<word>" after normalization,
// which happens for Mistral's "[INST] <user_msg> [/INST]" template where the
// chunk after [INST] starts with a space.
//
// This test requires the Mistral-7B v0.3 model (any quant) to hit the exact
// merge table that exposes the bug. TinyLlama's merges don't trip it.
TEST_F(SimpleBpeTokenizerTest, MistralDoubleMetaspaceBpeRegression) {
    const std::filesystem::path mistral_dir =
        "/Users/dex/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-8bit";
    if (!std::filesystem::exists(mistral_dir)) {
        GTEST_SKIP() << "Mistral-7B-Instruct-v0.3-8bit not downloaded";
    }
    auto tok_result = SimpleBpeTokenizer::from_model_dir(mistral_dir);
    ASSERT_TRUE(tok_result.has_value()) << tok_result.error().message;
    const auto& tok = *tok_result;

    // " what" after normalize(prepend ▁, replace " " with ▁) → "▁▁what".
    // Correct BPE: [▁(29473), ▁what(1535)]. Broken BPE produced 10 bytes.
    auto ids = tok.encode(" what", /*add_special_tokens=*/false);
    std::cout << "DIAG: encode(' what') returned " << ids.size() << " tokens:";
    for (int id : ids) std::cout << " " << id;
    std::cout << std::endl;
    EXPECT_EQ(ids.size(), 2u)
        << "BPE regressed: expected 2 tokens for ' what', got " << ids.size();
    if (ids.size() == 2) {
        EXPECT_EQ(ids[0], 29473) << "first token should be ▁";
        EXPECT_EQ(ids[1], 1535)  << "second token should be ▁what";
    }

    // Full Mistral [INST] prompt: verify it matches HuggingFace tokenizers'
    // output (13 tokens for this exact string). If any tokenizer change drifts
    // from HF semantics, this will catch it before the model sees bad input.
    auto full_ids = tok.encode("[INST] what is the capital of france? [/INST]",
                               /*add_special_tokens=*/true);
    std::cout << "DIAG: full encode returned " << full_ids.size() << " tokens:";
    for (int id : full_ids) std::cout << " " << id;
    std::cout << std::endl;
    const std::vector<int> hf_reference = {
        1, 3, 29473, 1535, 1117, 1040, 6333, 1070, 1872, 1385, 29572, 29473, 4
    };
    EXPECT_EQ(full_ids, hf_reference)
        << "Tokenization diverged from HuggingFace reference.";
}

// ── ByteLevel pre-tokenizer unit tests ───────────────────────────────────────
// These tests exercise the static ByteLevel helpers and do NOT require any
// model files to be present on disk.

TEST(ByteLevelTest, ByteToUnicodeTableSize) {
    const auto& tbl = SimpleBpeTokenizer::byte_to_unicode_table();
    ASSERT_EQ(tbl.size(), 256u);
}

TEST(ByteLevelTest, ByteToUnicodeSpaceMapsToGamma) {
    // GPT-2 maps space (0x20 = 32) to U+0120 'Ġ' (UTF-8: 0xC4 0xA0)
    const auto& tbl = SimpleBpeTokenizer::byte_to_unicode_table();
    const std::string& space_enc = tbl[0x20];
    ASSERT_EQ(space_enc.size(), 2u);
    EXPECT_EQ((unsigned char)space_enc[0], 0xC4);
    EXPECT_EQ((unsigned char)space_enc[1], 0xA0);
}

TEST(ByteLevelTest, ByteToUnicodeNewlineMapsToC010A) {
    // Newline (0x0A = 10) maps to U+010A 'Ċ' (UTF-8: 0xC4 0x8A)
    const auto& tbl = SimpleBpeTokenizer::byte_to_unicode_table();
    const std::string& nl_enc = tbl[0x0A];
    ASSERT_EQ(nl_enc.size(), 2u);
    EXPECT_EQ((unsigned char)nl_enc[0], 0xC4);
    EXPECT_EQ((unsigned char)nl_enc[1], 0x8A);
}

TEST(ByteLevelTest, ByteToUnicodePrintableAsciiIsSelf) {
    // Printable ASCII 33-126 should map to themselves (single byte)
    const auto& tbl = SimpleBpeTokenizer::byte_to_unicode_table();
    for (int b = 33; b <= 126; ++b) {
        EXPECT_EQ(tbl[b].size(), 1u) << "Expected single byte for ASCII " << b;
        EXPECT_EQ((unsigned char)tbl[b][0], (unsigned char)b)
            << "Expected ASCII byte " << b << " to map to itself";
    }
}

TEST(ByteLevelTest, UnicodeToBytRoundtrip) {
    // For every byte 0-255, apply byte→unicode then unicode→byte and get back the same byte
    const auto& fwd = SimpleBpeTokenizer::byte_to_unicode_table();
    const auto& rev = SimpleBpeTokenizer::unicode_to_byte_map();
    for (int b = 0; b < 256; ++b) {
        auto it = rev.find(fwd[b]);
        ASSERT_NE(it, rev.end()) << "No reverse mapping for byte " << b;
        EXPECT_EQ(it->second, (uint8_t)b) << "Roundtrip failed for byte " << b;
    }
}

TEST(ByteLevelTest, ApplyByteLevelEncodingAscii) {
    // ASCII text should have letters/digits/punctuation unchanged;
    // space should become Ġ (2-byte UTF-8: 0xC4 0xA0)
    std::string text = "Hello World";
    std::string encoded = SimpleBpeTokenizer::apply_byte_level_encoding(text);
    // "Hello" stays as-is (printable ASCII), space → Ġ, "World" stays
    EXPECT_NE(encoded.find("Hello"), std::string::npos);
    EXPECT_NE(encoded.find("World"), std::string::npos);
    // Space must not appear literally — it becomes Ġ
    EXPECT_EQ(encoded.find(' '), std::string::npos);
    // Ġ (0xC4 0xA0) must appear
    std::string gamma = "\xC4\xA0";
    EXPECT_NE(encoded.find(gamma), std::string::npos);
}

TEST(ByteLevelTest, SplitRawSimpleWords) {
    // "Hello world" → ["Hello", " world"]
    auto segs = SimpleBpeTokenizer::byte_level_split_raw("Hello world");
    ASSERT_EQ(segs.size(), 2u);
    EXPECT_EQ(segs[0], "Hello");
    EXPECT_EQ(segs[1], " world");
}

TEST(ByteLevelTest, SplitRawPunctuation) {
    // "Hello, world!" → ["Hello", ",", " world", "!"]
    auto segs = SimpleBpeTokenizer::byte_level_split_raw("Hello, world!");
    ASSERT_GE(segs.size(), 3u);
    EXPECT_EQ(segs[0], "Hello");
    // comma is a separate segment
    bool has_comma = false;
    for (const auto& s : segs) if (s == ",") has_comma = true;
    EXPECT_TRUE(has_comma) << "Expected ',' as separate segment";
    // " world" should be one segment
    bool has_world = false;
    for (const auto& s : segs) if (s == " world") has_world = true;
    EXPECT_TRUE(has_world) << "Expected ' world' as one segment";
}

TEST(ByteLevelTest, SplitRawContraction) {
    // "it's" → ["it", "'s"]
    auto segs = SimpleBpeTokenizer::byte_level_split_raw("it's");
    ASSERT_GE(segs.size(), 2u);
    EXPECT_EQ(segs[0], "it");
    EXPECT_EQ(segs[1], "'s");
}

TEST(ByteLevelTest, SplitRawDigits) {
    // "the 42nd" → ["the", " ", "42", "nd"]  (digits are separate, max 3)
    auto segs = SimpleBpeTokenizer::byte_level_split_raw("the 42nd");
    bool has_the = false, has_42 = false;
    for (const auto& s : segs) {
        if (s == " the" || s == "the") has_the = true;
        if (s == "42") has_42 = true;
    }
    EXPECT_TRUE(has_the) << "Expected 'the' segment";
    EXPECT_TRUE(has_42) << "Expected '42' as separate segment";
}

TEST(ByteLevelTest, SplitRawLongDigitsChunked) {
    // "1234" → ["123", "4"] (max 3 digits per segment)
    auto segs = SimpleBpeTokenizer::byte_level_split_raw("1234");
    ASSERT_EQ(segs.size(), 2u);
    EXPECT_EQ(segs[0], "123");
    EXPECT_EQ(segs[1], "4");
}

TEST(ByteLevelTest, EncodeDecodeRoundtrip) {
    // Encode a string with byte-level then decode — should get back the original
    const std::string original = "Hello, World! 42";
    const auto& fwd = SimpleBpeTokenizer::byte_to_unicode_table();
    const auto& rev = SimpleBpeTokenizer::unicode_to_byte_map();

    // Encode
    std::string encoded;
    for (unsigned char c : original) encoded += fwd[c];

    // Decode back
    std::string decoded;
    size_t i = 0;
    while (i < encoded.size()) {
        unsigned char c = (unsigned char)encoded[i];
        std::string cp;
        if (c < 0x80) { cp = std::string(1, encoded[i++]); }
        else if ((c & 0xE0) == 0xC0) { cp = encoded.substr(i, 2); i += 2; }
        else { cp = encoded.substr(i, 3); i += 3; }
        auto it = rev.find(cp);
        ASSERT_NE(it, rev.end()) << "Missing reverse mapping for codepoint";
        decoded += (char)it->second;
    }

    EXPECT_EQ(decoded, original);
}

} // namespace compute