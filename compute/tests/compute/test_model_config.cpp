#include <gtest/gtest.h>
#include "compute/model/model_config.h"
#include "test_config.h"
#include <filesystem>

namespace compute {

class ModelConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        tinyllama_config_path = TINYLLAMA_CONFIG_FILE;
        test_resources_dir = TEST_RESOURCES_DIR;
    }

    std::filesystem::path test_resources_dir;
    std::filesystem::path tinyllama_config_path;
};

TEST_F(ModelConfigTest, ParseValidTinyLlamaConfig) {
    if (!std::filesystem::exists(tinyllama_config_path))
        GTEST_SKIP() << "Model not found: " << tinyllama_config_path;

    auto result = ModelConfig::from_config_file(tinyllama_config_path);
    ASSERT_TRUE(result.has_value()) << "Failed to parse config: " << result.error().message;

    const auto& config = *result;

    // Test core architecture values from actual TinyLlama config
    EXPECT_EQ(config.vocab_size, 32000);
    EXPECT_EQ(config.hidden_size, 2048);
    EXPECT_EQ(config.num_hidden_layers, 22);
    EXPECT_EQ(config.num_attention_heads, 32);
    EXPECT_EQ(config.num_key_value_heads, 4);
    EXPECT_EQ(config.intermediate_size, 5632);
    EXPECT_EQ(config.max_position_embeddings, 2048);

    // Test normalization and activation
    EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-05f);
    EXPECT_FLOAT_EQ(config.rope_theta, 10000.0f);
    EXPECT_EQ(config.hidden_act, "silu");

    // Test behavior flags
    EXPECT_FALSE(config.attention_bias);
    EXPECT_FALSE(config.tie_word_embeddings);

    // Test metadata
    EXPECT_EQ(config.model_type, "llama");
    EXPECT_EQ(config.torch_dtype, "bfloat16");
    EXPECT_FALSE(config.architectures.empty());
    EXPECT_EQ(config.architectures[0], "LlamaForCausalLM");

    // Test optional special tokens
    ASSERT_TRUE(config.bos_token_id.has_value());
    EXPECT_EQ(*config.bos_token_id, 1);
    ASSERT_TRUE(config.eos_token_ids.has_value());
    EXPECT_EQ(config.primary_eos_token_id(), 2);
    EXPECT_TRUE(config.is_eos_token(2));
    EXPECT_FALSE(config.is_eos_token(128001));

    // Test quantization
    ASSERT_TRUE(config.quantization.has_value());
    EXPECT_EQ(config.quantization->group_size, 64);
    EXPECT_EQ(config.quantization->bits, 4);
}

TEST_F(ModelConfigTest, ConfigValidation) {
    auto result = ModelConfig::from_config_file(tinyllama_config_path);
    ASSERT_TRUE(result.has_value());

    const auto& config = *result;
    EXPECT_TRUE(config.is_valid());
    EXPECT_TRUE(config.is_llama_architecture());
}

TEST_F(ModelConfigTest, GroupedQueryAttention) {
    auto result = ModelConfig::from_config_file(tinyllama_config_path);
    ASSERT_TRUE(result.has_value());

    const auto& config = *result;
    EXPECT_TRUE(config.uses_grouped_query_attention());
    EXPECT_EQ(config.effective_num_kv_heads(), 4);
    EXPECT_LT(config.num_key_value_heads, config.num_attention_heads);
}

TEST_F(ModelConfigTest, ConfigToString) {
    auto result = ModelConfig::from_config_file(tinyllama_config_path);
    ASSERT_TRUE(result.has_value());

    const auto& config = *result;
    std::string config_str = config.to_string();

    // Check that key values appear in the string
    EXPECT_NE(config_str.find("vocab_size: 32000"), std::string::npos);
    EXPECT_NE(config_str.find("hidden_size: 2048"), std::string::npos);
    EXPECT_NE(config_str.find("uses_gqa: true"), std::string::npos);
}

TEST_F(ModelConfigTest, ParseFromJsonString) {
    // Create minimal valid JSON
    std::string json_str = R"({
        "vocab_size": 1000,
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2048,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": false,
        "tie_word_embeddings": false,
        "model_type": "llama",
        "torch_dtype": "float16",
        "architectures": ["LlamaForCausalLM"]
    })";

    auto result = ModelConfig::from_json_string(json_str);
    ASSERT_TRUE(result.has_value()) << "Failed to parse JSON: " << result.error().message;

    const auto& config = *result;
    EXPECT_EQ(config.vocab_size, 1000);
    EXPECT_EQ(config.hidden_size, 512);
    EXPECT_TRUE(config.is_valid());
    EXPECT_FALSE(config.uses_grouped_query_attention()); // Same heads
}

TEST_F(ModelConfigTest, ParseMissingRequiredField) {
    // JSON missing required field
    std::string json_str = R"({
        "hidden_size": 512,
        "num_hidden_layers": 12
    })";

    auto result = ModelConfig::from_json_string(json_str);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("Missing required field"), std::string::npos);
}

TEST_F(ModelConfigTest, ParseInvalidJson) {
    std::string invalid_json = R"({ "invalid": json })";

    auto result = ModelConfig::from_json_string(invalid_json);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("JSON parse error"), std::string::npos);
}

TEST_F(ModelConfigTest, ParseNonExistentFile) {
    auto result = ModelConfig::from_config_file("/nonexistent/config.json");
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("does not exist"), std::string::npos);
}

TEST_F(ModelConfigTest, InvalidArchitecture) {
    std::string json_str = R"({
        "vocab_size": 1000,
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2048,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": false,
        "tie_word_embeddings": false,
        "model_type": "gpt",
        "torch_dtype": "float16",
        "architectures": ["GPTForCausalLM"]
    })";

    auto result = ModelConfig::from_json_string(json_str);
    ASSERT_TRUE(result.has_value());

    const auto& config = *result;
    EXPECT_FALSE(config.is_llama_architecture());
    EXPECT_FALSE(config.is_valid()); // Should be invalid due to architecture
}

} // namespace compute