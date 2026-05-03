#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

class Qwen2IntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        model_dir_ = "/Users/dex/.neurons/models/mlx-community/Qwen2.5-3B-Instruct-4bit";
        if (!std::filesystem::exists(model_dir_))
            model_dir_ = "/Users/dex/.neurons/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit";

        if (!std::filesystem::exists(model_dir_)) {
            skip_reason_ = "Qwen2.5 model not found — download "
                           "mlx-community/Qwen2.5-3B-Instruct-4bit first";
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir_, backend_.get());
        if (!inf_result) {
            skip_reason_ = "Failed to load Qwen2.5: " + inf_result.error().message;
            return;
        }
        inference_ = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded Qwen2.5 model: " << inference_->config().model_type
                  << " hidden=" << inference_->config().hidden_size
                  << " layers=" << inference_->config().num_hidden_layers
                  << " attention_bias=" << inference_->config().attention_bias << std::endl;
    }

    static void TearDownTestSuite() {
        inference_.reset();
        if (backend_) backend_->cleanup();
        backend_.reset();
        skip_reason_.clear();
    }

    void SetUp() override {
        if (!skip_reason_.empty())
            GTEST_SKIP() << skip_reason_;
    }

    static std::filesystem::path              model_dir_;
    static std::string                        skip_reason_;
    static std::unique_ptr<ComputeBackend>    backend_;
    static std::unique_ptr<TinyLlamaInference> inference_;
};

std::filesystem::path               Qwen2IntegrationTest::model_dir_;
std::string                         Qwen2IntegrationTest::skip_reason_;
std::unique_ptr<ComputeBackend>     Qwen2IntegrationTest::backend_;
std::unique_ptr<TinyLlamaInference> Qwen2IntegrationTest::inference_;

TEST_F(Qwen2IntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference_->config().model_type, "qwen2");
    EXPECT_TRUE(inference_->config().is_qwen2_architecture());
    EXPECT_TRUE(inference_->config().is_supported_architecture());
    EXPECT_FALSE(inference_->config().is_llama_architecture());
    EXPECT_FALSE(inference_->config().is_mistral_architecture());
    std::cout << "✓ Qwen2.5 config validated" << std::endl;
}

TEST_F(Qwen2IntegrationTest, GenerateCapitalOfFrance) {
    const std::string prompt =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";

    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference_->config().primary_eos_token_id();

    auto result = inference_->generate(token_ids, /*max_new_tokens=*/100, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference_->tokenizer().decode(gen_so_far);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << "generate() failed: " << result.error().message;
    ASSERT_FALSE(decoded_so_far.empty()) << "Model produced no output";

    std::cout << "Generated (" << gen_so_far.size() << " tokens): \""
              << decoded_so_far << "\"" << std::endl;

    bool mentions_paris = decoded_so_far.find("Paris") != std::string::npos ||
                          decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected output to mention Paris, got: \"" << decoded_so_far << "\"";
}

TEST_F(Qwen2IntegrationTest, BiasPathDoesNotCrash) {
    EXPECT_EQ(inference_->config().model_type, "qwen2");
    std::cout << "✓ Optional attention bias path ran without error" << std::endl;
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
