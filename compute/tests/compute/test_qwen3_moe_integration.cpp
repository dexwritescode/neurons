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
    void SetUp() override {
        model_dir = QWEN3MOE_MODEL_DIR;
        if (!std::filesystem::exists(model_dir)) {
            GTEST_SKIP() << "Qwen3.5 MoE model not found at " << model_dir
                         << " — download mlx-community/Qwen3-30B-A3B-4bit first";
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto model_result = LanguageModel::load(model_dir, backend.get());
        ASSERT_TRUE(model_result.has_value())
            << "Failed to load Qwen3 MoE: " << model_result.error().message;
        model = std::move(*model_result);

        std::cout << "Loaded model: " << model->model_type()
                  << " hidden=" << model->config().hidden_size
                  << " layers=" << model->config().num_hidden_layers << std::endl;
    }

    void TearDown() override {
        model.reset();
        if (backend) backend->cleanup();
    }

    std::filesystem::path model_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<LanguageModel> model;
};

TEST_F(Qwen3MoeIntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(model->model_type(), "qwen3_5_moe");
    EXPECT_EQ(model->config().num_hidden_layers, 40u);
    std::cout << "✓ Qwen3 MoE config validated" << std::endl;
}

TEST_F(Qwen3MoeIntegrationTest, GenerateCapitalOfFrance) {
    // Qwen3 ChatML template (no system prompt, thinking disabled via /no_think)
    const std::string prompt =
        "<|im_start|>user\n"
        "What is the capital of France? /no_think<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto token_ids = model->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";
    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded;
    const int eos_id = model->tokenizer().eos_token_id();

    auto result = model->generate(token_ids, /*max_new_tokens=*/64, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded = model->tokenizer().decode(gen_so_far);
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
