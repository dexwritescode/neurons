#include <gtest/gtest.h>
#include "compute/model/tokenizer_config.h"
#include "test_config.h"
#include <filesystem>

using namespace compute;

class TokenizerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use the TinyLlama model directory from CMake configuration
        model_dir_ = std::filesystem::path(TINYLLAMA_MODEL_DIR);

        // Verify the model directory exists
        if (!std::filesystem::exists(model_dir_)) {
            GTEST_SKIP() << "TinyLlama model directory not found at: " << model_dir_;
        }
    }

    std::filesystem::path model_dir_;
};

TEST_F(TokenizerConfigTest, LoadValidConfig) {
    // Load tokenizer configuration from TinyLlama model
    auto result = TokenizerConfig::from_config_file(model_dir_);

    ASSERT_TRUE(result.has_value()) << "Failed to load tokenizer config: " << result.error().message;

    const auto& config = *result;

    // Verify required fields are loaded
    EXPECT_EQ(config.tokenizer_class, "LlamaTokenizer");
    EXPECT_EQ(config.model_max_length, 2048);
    EXPECT_TRUE(config.add_bos_token);
    EXPECT_FALSE(config.add_eos_token);

    // Verify special tokens
    EXPECT_EQ(config.bos_token, "<s>");
    EXPECT_EQ(config.eos_token, "</s>");
    EXPECT_EQ(config.unk_token, "<unk>");
    EXPECT_EQ(config.pad_token, "</s>");

    // Verify optional fields
    EXPECT_FALSE(config.legacy);
    EXPECT_FALSE(config.clean_up_tokenization_spaces);
    EXPECT_EQ(config.padding_side, "right");
    EXPECT_FALSE(config.use_default_system_prompt);

    // Verify chat template is present
    EXPECT_FALSE(config.chat_template.empty());

    // Verify added_tokens_decoder contains expected special tokens
    EXPECT_TRUE(config.added_tokens_decoder.contains(0)); // <unk>
    EXPECT_TRUE(config.added_tokens_decoder.contains(1)); // <s>
    EXPECT_TRUE(config.added_tokens_decoder.contains(2)); // </s>

    EXPECT_EQ(config.added_tokens_decoder.at(0).content, "<unk>");
    EXPECT_EQ(config.added_tokens_decoder.at(1).content, "<s>");
    EXPECT_EQ(config.added_tokens_decoder.at(2).content, "</s>");

    // Verify special token properties
    EXPECT_TRUE(config.added_tokens_decoder.at(0).special);
    EXPECT_TRUE(config.added_tokens_decoder.at(1).special);
    EXPECT_TRUE(config.added_tokens_decoder.at(2).special);
}

TEST_F(TokenizerConfigTest, LoadFromDirectory) {
    // Test loading from directory (should find tokenizer_config.json automatically)
    auto result = TokenizerConfig::from_config_file(model_dir_);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->tokenizer_class, "LlamaTokenizer");
}

TEST_F(TokenizerConfigTest, LoadFromSpecificFile) {
    // Test loading from specific file path
    auto config_file = model_dir_ / "tokenizer_config.json";
    auto result = TokenizerConfig::from_config_file(config_file);

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->tokenizer_class, "LlamaTokenizer");
}

TEST_F(TokenizerConfigTest, FileNotFound) {
    // Test error handling for non-existent file
    auto result = TokenizerConfig::from_config_file("/nonexistent/path/tokenizer_config.json");

    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::InvalidInput);
}

TEST_F(TokenizerConfigTest, ChatTemplateFormatting) {
    auto result = TokenizerConfig::from_config_file(model_dir_);
    ASSERT_TRUE(result.has_value());

    const auto& config = *result;

    // Verify chat template contains expected role markers
    EXPECT_TRUE(config.chat_template.find("<|user|>") != std::string::npos);
    EXPECT_TRUE(config.chat_template.find("<|assistant|>") != std::string::npos);
    EXPECT_TRUE(config.chat_template.find("<|system|>") != std::string::npos);
}