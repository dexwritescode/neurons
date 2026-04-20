#include <gtest/gtest.h>
#include "compute/model/simple_bpe_tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using namespace compute;
using json = nlohmann::json;

class TokenizerComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load the tokenizer
        auto model_dir = std::filesystem::path("/Users/dex/.neurons/models/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit");
        auto tokenizer_result = SimpleBpeTokenizer::from_model_dir(model_dir);

        ASSERT_TRUE(tokenizer_result.has_value()) << "Failed to load tokenizer: "
            << tokenizer_result.error().message;

        tokenizer_ = std::move(*tokenizer_result);

        // Load Python baseline results
        std::ifstream baseline_file("tokenizer_baseline_results.json");
        ASSERT_TRUE(baseline_file.is_open()) << "Failed to open tokenizer_baseline_results.json";
        baseline_file >> baseline_results_;
    }

    std::optional<SimpleBpeTokenizer> tokenizer_;
    json baseline_results_;

    SimpleBpeTokenizer& tokenizer() { return *tokenizer_; }
};

TEST_F(TokenizerComparisonTest, CompareSpecialTokens) {
    std::cout << "\n=== Special Tokens Comparison ===" << std::endl;

    // Expected values from Python baseline
    EXPECT_EQ(tokenizer().bos_token_id(), 1) << "BOS token ID mismatch";
    EXPECT_EQ(tokenizer().eos_token_id(), 2) << "EOS token ID mismatch";
    EXPECT_EQ(tokenizer().unk_token_id(), 0) << "UNK token ID mismatch";
    EXPECT_EQ(tokenizer().pad_token_id(), 2) << "PAD token ID mismatch";
    EXPECT_EQ(tokenizer().vocab_size(), 32000) << "Vocab size mismatch";

    std::cout << "✅ Special tokens match Python baseline" << std::endl;
}

TEST_F(TokenizerComparisonTest, CompareEncodingNoSpecialTokens) {
    std::cout << "\n=== Encoding Comparison (No Special Tokens) ===" << std::endl;

    for (const auto& [text, expected_obj] : baseline_results_.items()) {
        std::cout << "\nTest: " << text << std::endl;

        // Get expected tokens from Python baseline
        std::vector<int> expected_tokens = expected_obj["tokens_no_special"].get<std::vector<int>>();

        // Encode with C++ tokenizer (no special tokens)
        std::vector<int> actual_tokens = tokenizer().encode(text, false);

        std::cout << "  Expected: [";
        for (size_t i = 0; i < expected_tokens.size(); ++i) {
            std::cout << expected_tokens[i];
            if (i < expected_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Actual:   [";
        for (size_t i = 0; i < actual_tokens.size(); ++i) {
            std::cout << actual_tokens[i];
            if (i < actual_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Compare
        ASSERT_EQ(actual_tokens.size(), expected_tokens.size())
            << "Token count mismatch for: " << text;

        for (size_t i = 0; i < expected_tokens.size(); ++i) {
            EXPECT_EQ(actual_tokens[i], expected_tokens[i])
                << "Token mismatch at position " << i << " for text: " << text;
        }

        if (actual_tokens == expected_tokens) {
            std::cout << "  ✅ Match!" << std::endl;
        } else {
            std::cout << "  ❌ Mismatch!" << std::endl;
        }
    }
}

TEST_F(TokenizerComparisonTest, CompareEncodingWithSpecialTokens) {
    std::cout << "\n=== Encoding Comparison (With Special Tokens) ===" << std::endl;

    for (const auto& [text, expected_obj] : baseline_results_.items()) {
        std::cout << "\nTest: " << text << std::endl;

        // Get expected tokens from Python baseline
        std::vector<int> expected_tokens = expected_obj["tokens_with_special"].get<std::vector<int>>();

        // Encode with C++ tokenizer (with special tokens)
        std::vector<int> actual_tokens = tokenizer().encode(text, true);

        std::cout << "  Expected: [";
        for (size_t i = 0; i < expected_tokens.size(); ++i) {
            std::cout << expected_tokens[i];
            if (i < expected_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  Actual:   [";
        for (size_t i = 0; i < actual_tokens.size(); ++i) {
            std::cout << actual_tokens[i];
            if (i < actual_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Compare
        ASSERT_EQ(actual_tokens.size(), expected_tokens.size())
            << "Token count mismatch for: " << text;

        for (size_t i = 0; i < expected_tokens.size(); ++i) {
            EXPECT_EQ(actual_tokens[i], expected_tokens[i])
                << "Token mismatch at position " << i << " for text: " << text;
        }

        if (actual_tokens == expected_tokens) {
            std::cout << "  ✅ Match!" << std::endl;
        } else {
            std::cout << "  ❌ Mismatch!" << std::endl;
        }
    }
}

TEST_F(TokenizerComparisonTest, CompareDecoding) {
    std::cout << "\n=== Decoding Comparison ===" << std::endl;

    for (const auto& [text, expected_obj] : baseline_results_.items()) {
        std::cout << "\nTest: " << text << std::endl;

        // Get tokens and expected decoded text
        std::vector<int> tokens = expected_obj["tokens_no_special"].get<std::vector<int>>();
        std::string expected_decoded = expected_obj["decoded_no_special"].get<std::string>();

        // Decode with C++ tokenizer
        std::string actual_decoded = tokenizer().decode(tokens, false);

        std::cout << "  Expected: " << expected_decoded << std::endl;
        std::cout << "  Actual:   " << actual_decoded << std::endl;

        EXPECT_EQ(actual_decoded, expected_decoded)
            << "Decoded text mismatch for tokens of: " << text;

        if (actual_decoded == expected_decoded) {
            std::cout << "  ✅ Match!" << std::endl;
        } else {
            std::cout << "  ❌ Mismatch!" << std::endl;
        }
    }
}

TEST_F(TokenizerComparisonTest, RoundTripTest) {
    std::cout << "\n=== Round Trip Test (Encode -> Decode) ===" << std::endl;

    std::vector<std::string> test_texts = {
        "Hello",
        "Hello world",
        "The quick brown fox jumps over the lazy dog"
    };

    for (const auto& text : test_texts) {
        std::cout << "\nOriginal: " << text << std::endl;

        // Encode then decode
        auto tokens = tokenizer().encode(text, false);
        auto decoded = tokenizer().decode(tokens, false);

        std::cout << "Tokens: [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i];
            if (i < tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Decoded: " << decoded << std::endl;

        EXPECT_EQ(decoded, text) << "Round trip failed for: " << text;

        if (decoded == text) {
            std::cout << "  ✅ Round trip successful!" << std::endl;
        } else {
            std::cout << "  ❌ Round trip failed!" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}