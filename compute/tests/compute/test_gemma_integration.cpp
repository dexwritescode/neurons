#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../src/compute/model/language_model.h"
#include "../../src/compute/core/compute_backend.h"
#include <filesystem>
#include <string>

namespace {

// Shared backend for all Gemma integration tests.
// Initialized once per test suite to avoid repeated MLX setup.
class GemmaIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto b = compute::BackendFactory::create(compute::BackendType::MLX);
        if (!b) { backend_ = nullptr; return; }
        backend_ = std::move(*b);
        backend_->initialize();
    }

    static void TearDownTestSuite() {
        if (backend_) backend_->cleanup();
        backend_.reset();
    }

    static std::unique_ptr<compute::ComputeBackend> backend_;
};

std::unique_ptr<compute::ComputeBackend> GemmaIntegrationTest::backend_ = nullptr;

// ── Config parsing ────────────────────────────────────────────────────────────

TEST_F(GemmaIntegrationTest, ConfigParsesGemmaFields) {
    const auto model_path = std::filesystem::path(std::getenv("HOME"))
        / ".neurons/models/mlx-community/gemma-3-1b-it-qat-4bit";
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Gemma-3-1b-it-qat-4bit not downloaded";
    }

    auto config = compute::ModelConfig::from_config_file(model_path / "config.json");
    ASSERT_TRUE(config.has_value()) << config.error().message;

    EXPECT_EQ(config->model_type, "gemma3_text");
    EXPECT_TRUE(config->is_gemma_architecture());
    EXPECT_TRUE(config->is_supported_architecture());
    EXPECT_TRUE(config->is_valid());

    // Gemma3-specific fields
    ASSERT_TRUE(config->head_dim.has_value());
    EXPECT_EQ(*config->head_dim, 256u);

    ASSERT_TRUE(config->query_pre_attn_scalar.has_value());
    EXPECT_FLOAT_EQ(*config->query_pre_attn_scalar, 256.0f);

    ASSERT_TRUE(config->sliding_window.has_value());
    EXPECT_EQ(*config->sliding_window, 512u);

    ASSERT_TRUE(config->sliding_window_pattern.has_value());
    EXPECT_EQ(*config->sliding_window_pattern, 6);

    ASSERT_TRUE(config->rope_local_base_freq.has_value());
    EXPECT_FLOAT_EQ(*config->rope_local_base_freq, 10000.0f);

    // Layer classification (pattern=6: global at indices 5,11,17,23)
    EXPECT_TRUE(config->is_local_layer(0));
    EXPECT_TRUE(config->is_local_layer(4));
    EXPECT_FALSE(config->is_local_layer(5));  // global
    EXPECT_TRUE(config->is_local_layer(6));
    EXPECT_FALSE(config->is_local_layer(11)); // global

    // Computed helpers
    EXPECT_EQ(config->effective_head_dim(), 256u);
    EXPECT_FLOAT_EQ(config->effective_attention_scale(), 1.0f / 16.0f); // 1/sqrt(256)

    EXPECT_EQ(config->hidden_size, 1152u);
    EXPECT_EQ(config->num_hidden_layers, 26u);
    EXPECT_EQ(config->num_attention_heads, 4u);
    EXPECT_EQ(config->num_key_value_heads, 1u);
    EXPECT_EQ(config->vocab_size, 262144u);
    EXPECT_FALSE(config->tie_word_embeddings);

    // hidden_activation mapped to hidden_act
    EXPECT_EQ(config->hidden_act, "gelu_pytorch_tanh");
}

// ── End-to-end generation ─────────────────────────────────────────────────────

TEST_F(GemmaIntegrationTest, GenerateCapitalOfFrance) {
    if (!backend_) GTEST_SKIP() << "MLX backend not available";

    const auto model_path = std::filesystem::path(std::getenv("HOME"))
        / ".neurons/models/mlx-community/gemma-3-1b-it-qat-4bit";
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "Gemma-3-1b-it-qat-4bit not downloaded";
    }

    auto model = compute::LanguageModel::load(model_path, backend_.get());
    ASSERT_TRUE(model.has_value()) << model.error().message;

    EXPECT_EQ((*model)->model_type(), "gemma3_text");

    // Gemma3 chat template: BOS added by tokenizer (add_special_tokens=true)
    const std::string prompt =
        "<start_of_turn>user\n"
        "What is the capital of France?<end_of_turn>\n"
        "<start_of_turn>model\n";

    auto ids = (*model)->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(ids.empty());

    std::string output;
    compute::SamplingParams params;
    params.temperature = 0.0f;  // greedy

    auto tokens = (*model)->generate(ids, 64, params, [&](int tok) {
        output += (*model)->tokenizer().decode({tok});
        return true;
    });
    ASSERT_TRUE(tokens.has_value()) << tokens.error().message;

    EXPECT_THAT(output, ::testing::HasSubstr("Paris"));
}

} // namespace
