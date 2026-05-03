#include <gtest/gtest.h>
#include "compute/model/language_model.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

class Qwen3MoeIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_dir_ = QWEN3MOE_MODEL_DIR;
        if (!std::filesystem::exists(model_dir_)) {
            skip_reason_ = std::string("Qwen3.5 MoE model not found at ") + QWEN3MOE_MODEL_DIR
                         + " — download mlx-community/Qwen3-30B-A3B-4bit first";
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto model_result = LanguageModel::load(model_dir_, backend_.get());
        if (!model_result) {
            skip_reason_ = "Failed to load Qwen3 MoE: " + model_result.error().message;
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

std::filesystem::path           Qwen3MoeIntegrationTest::model_dir_;
std::string                     Qwen3MoeIntegrationTest::skip_reason_;
std::unique_ptr<ComputeBackend> Qwen3MoeIntegrationTest::backend_;
std::unique_ptr<LanguageModel>  Qwen3MoeIntegrationTest::model_;

TEST_F(Qwen3MoeIntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(model_->model_type(), "qwen3_5_moe");
    EXPECT_EQ(model_->config().num_hidden_layers, 40u);
    std::cout << "✓ Qwen3 MoE config validated" << std::endl;
}

TEST_F(Qwen3MoeIntegrationTest, GenerateCapitalOfFrance) {
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

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
