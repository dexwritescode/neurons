#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../src/compute/model/language_model.h"
#include "../../src/compute/core/compute_backend.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

class GemmaIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_path_ = std::filesystem::path(std::getenv("HOME"))
            / ".neurons/models/mlx-community/gemma-3-1b-it-qat-4bit";

        auto b = compute::BackendFactory::create(compute::BackendType::MLX);
        if (!b) { skip_reason_ = b.error().message; return; }
        backend_ = std::move(*b);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        if (!std::filesystem::exists(model_path_)) {
            skip_reason_ = "Gemma-3-1b-it-qat-4bit not downloaded";
            return;
        }

        auto m = compute::LanguageModel::load(model_path_, backend_.get());
        if (!m) { skip_reason_ = m.error().message; return; }
        model_ = std::move(*m);
    }

    static void TearDownTestSuite() {
        model_.reset();
        if (backend_) backend_->cleanup();
        backend_.reset();
        skip_reason_.clear();
    }

    void SetUp() override {
        if (!skip_reason_.empty())
            GTEST_SKIP() << skip_reason_;
    }

    static std::filesystem::path                    model_path_;
    static std::string                              skip_reason_;
    static std::unique_ptr<compute::ComputeBackend> backend_;
    static std::unique_ptr<compute::LanguageModel>  model_;
};

std::filesystem::path                    GemmaIntegrationTest::model_path_;
std::string                              GemmaIntegrationTest::skip_reason_;
std::unique_ptr<compute::ComputeBackend> GemmaIntegrationTest::backend_;
std::unique_ptr<compute::LanguageModel>  GemmaIntegrationTest::model_;

// ── Config parsing ────────────────────────────────────────────────────────────

TEST_F(GemmaIntegrationTest, ConfigParsesGemmaFields) {
    auto config = compute::ModelConfig::from_config_file(model_path_ / "config.json");
    ASSERT_TRUE(config.has_value()) << config.error().message;

    EXPECT_EQ(config->model_type, "gemma3_text");
    EXPECT_TRUE(config->is_gemma_architecture());
    EXPECT_TRUE(config->is_supported_architecture());
    EXPECT_TRUE(config->is_valid());

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

    EXPECT_TRUE(config->is_local_layer(0));
    EXPECT_TRUE(config->is_local_layer(4));
    EXPECT_FALSE(config->is_local_layer(5));
    EXPECT_TRUE(config->is_local_layer(6));
    EXPECT_FALSE(config->is_local_layer(11));

    EXPECT_EQ(config->effective_head_dim(), 256u);
    EXPECT_FLOAT_EQ(config->effective_attention_scale(), 1.0f / 16.0f);

    EXPECT_EQ(config->hidden_size, 1152u);
    EXPECT_EQ(config->num_hidden_layers, 26u);
    EXPECT_EQ(config->num_attention_heads, 4u);
    EXPECT_EQ(config->num_key_value_heads, 1u);
    EXPECT_EQ(config->vocab_size, 262144u);
    EXPECT_FALSE(config->tie_word_embeddings);

    EXPECT_EQ(config->hidden_act, "gelu_pytorch_tanh");
}

// ── End-to-end generation ─────────────────────────────────────────────────────

TEST_F(GemmaIntegrationTest, GenerateCapitalOfFrance) {
    EXPECT_EQ(model_->model_type(), "gemma3_text");

    const std::string prompt =
        "<start_of_turn>user\n"
        "What is the capital of France?<end_of_turn>\n"
        "<start_of_turn>model\n";

    auto ids = model_->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(ids.empty());

    std::string output;
    compute::SamplingParams params;
    params.temperature = 0.0f;

    auto tokens = model_->generate(ids, 64, params, [&](int tok) {
        output += model_->tokenizer().decode({tok});
        return true;
    });
    ASSERT_TRUE(tokens.has_value()) << tokens.error().message;

    EXPECT_THAT(output, ::testing::HasSubstr("Paris"));
}

TEST_F(GemmaIntegrationTest, GenerateThroughput) {
    const std::string prompt =
        "<start_of_turn>user\n"
        "Write a detailed paragraph about the history of France, "
        "including the French Revolution, Napoleon, and the World Wars.<end_of_turn>\n"
        "<start_of_turn>model\n";

    auto ids = model_->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(ids.empty());

    compute::SamplingParams greedy;
    greedy.temperature = 0.0f;

    // Warmup: trigger mx::compile before measuring steady-state decode.
    model_->generate(ids, /*max_new_tokens=*/8, greedy, [](int) { return true; });

    int token_count = 0;
    auto start = std::chrono::steady_clock::now();
    auto result = model_->generate(ids, /*max_new_tokens=*/128, greedy,
        [&](int /*tok*/) { ++token_count; return true; });
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_GT(token_count, 0) << "Model produced no tokens";

    double tok_s = token_count * 1000.0 / elapsed_ms;
    std::cout << "Gemma 3 1B throughput: " << tok_s << " tok/s ("
              << token_count << " tokens in " << elapsed_ms << " ms)" << std::endl;

    // Baseline (debug build, warmed): ~60 tok/s. Floor = baseline / 2.
    EXPECT_GE(tok_s, 30.0) << "throughput regression: " << tok_s << " tok/s";
}

} // namespace
