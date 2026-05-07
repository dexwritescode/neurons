#include <gtest/gtest.h>
#include "compute/model/language_model.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

class Qwen3TransformerMoeIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_dir_ = QWEN3_30B_MODEL_DIR;
        if (!std::filesystem::exists(model_dir_)) {
            skip_reason_ = std::string("Qwen3-30B-A3B model not found at ") + QWEN3_30B_MODEL_DIR
                         + " — download mlx-community/Qwen3-30B-A3B-4bit first";
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto model_result = LanguageModel::load(model_dir_, backend_.get());
        if (!model_result) {
            skip_reason_ = "Failed to load Qwen3-30B: " + model_result.error().message;
            return;
        }
        model_ = std::move(*model_result);

        std::cout << "Loaded model: " << model_->model_type()
                  << " hidden=" << model_->config().hidden_size
                  << " layers=" << model_->config().num_hidden_layers << std::endl;
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

    static std::filesystem::path           model_dir_;
    static std::string                     skip_reason_;
    static std::unique_ptr<ComputeBackend> backend_;
    static std::unique_ptr<LanguageModel>  model_;
};

std::filesystem::path           Qwen3TransformerMoeIntegrationTest::model_dir_;
std::string                     Qwen3TransformerMoeIntegrationTest::skip_reason_;
std::unique_ptr<ComputeBackend> Qwen3TransformerMoeIntegrationTest::backend_;
std::unique_ptr<LanguageModel>  Qwen3TransformerMoeIntegrationTest::model_;

TEST_F(Qwen3TransformerMoeIntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(model_->model_type(), "qwen3_moe");
    EXPECT_EQ(model_->config().num_hidden_layers, 48u);
    EXPECT_EQ(model_->config().hidden_size, 2048u);
    EXPECT_EQ(model_->config().num_attention_heads, 32u);
    EXPECT_EQ(model_->config().num_key_value_heads, 4u);
    ASSERT_TRUE(model_->config().num_experts.has_value());
    EXPECT_EQ(*model_->config().num_experts, 128u);
    ASSERT_TRUE(model_->config().num_experts_per_tok.has_value());
    EXPECT_EQ(*model_->config().num_experts_per_tok, 8u);
    std::cout << "✓ Qwen3-30B-A3B config validated" << std::endl;
}

TEST_F(Qwen3TransformerMoeIntegrationTest, GenerateCapitalOfFrance) {
    const std::string prompt =
        "<|im_start|>user\n"
        "What is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto token_ids = model_->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";
    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded;
    const int eos_id = model_->tokenizer().eos_token_id();

    auto result = model_->generate(token_ids, /*max_new_tokens=*/128, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded = model_->tokenizer().decode(gen_so_far);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_FALSE(decoded.empty()) << "Model produced no output";
    std::cout << "Generated (" << gen_so_far.size() << " tokens): \"" << decoded << "\"" << std::endl;

    bool mentions_paris = decoded.find("Paris") != std::string::npos ||
                          decoded.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris) << "Expected Paris, got: \"" << decoded << "\"";
}

TEST_F(Qwen3TransformerMoeIntegrationTest, GenerateThroughput) {
    const std::string prompt =
        "<|im_start|>user\n"
        "Write a detailed paragraph about the history of France, "
        "including the French Revolution, Napoleon, and the World Wars.<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto token_ids = model_->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    // Warmup: trigger mx::compile before measuring steady-state decode.
    model_->generate(token_ids, /*max_new_tokens=*/8, greedy, [](int) { return true; });

    int token_count = 0;
    auto start = std::chrono::steady_clock::now();
    auto result = model_->generate(token_ids, /*max_new_tokens=*/128, greedy,
        [&](int /*tok*/) { ++token_count; return true; });
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_GT(token_count, 0) << "Model produced no tokens";

    double tok_s = token_count * 1000.0 / elapsed_ms;
    std::cout << "Qwen3-30B-A3B throughput: " << tok_s << " tok/s ("
              << token_count << " tokens in " << elapsed_ms << " ms)" << std::endl;

    // Baseline (release, warmed): ~26 tok/s. Floor = baseline / 2.
    EXPECT_GE(tok_s, 13.0) << "throughput regression: " << tok_s << " tok/s";
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
