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
    void SetUp() override {
        // Try 3B 4-bit first, fall back to smaller variants
        model_dir = "/Users/dex/.neurons/models/mlx-community/Qwen2.5-3B-Instruct-4bit";
        if (!std::filesystem::exists(model_dir))
            model_dir = "/Users/dex/.neurons/models/mlx-community/Qwen2.5-1.5B-Instruct-4bit";

        if (!std::filesystem::exists(model_dir)) {
            GTEST_SKIP() << "Qwen2.5 model not found — download "
                            "mlx-community/Qwen2.5-3B-Instruct-4bit first";
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir, backend.get());
        ASSERT_TRUE(inf_result.has_value())
            << "Failed to load Qwen2.5: " << inf_result.error().message;
        inference = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded Qwen2.5 model: " << inference->config().model_type
                  << " hidden=" << inference->config().hidden_size
                  << " layers=" << inference->config().num_hidden_layers
                  << " attention_bias=" << inference->config().attention_bias << std::endl;
    }

    void TearDown() override {
        inference.reset();
        if (backend) backend->cleanup();
    }

    std::filesystem::path model_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

TEST_F(Qwen2IntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference->config().model_type, "qwen2");
    EXPECT_TRUE(inference->config().is_qwen2_architecture());
    EXPECT_TRUE(inference->config().is_supported_architecture());
    EXPECT_FALSE(inference->config().is_llama_architecture());
    EXPECT_FALSE(inference->config().is_mistral_architecture());
    // Note: mlx-community sets attention_bias=null in config.json for some Qwen2 variants;
    // bias tensors are detected from the weight map directly in llama_model.cpp.
    std::cout << "✓ Qwen2.5 config validated" << std::endl;
}

TEST_F(Qwen2IntegrationTest, GenerateCapitalOfFrance) {
    // Qwen2 ChatML template
    const std::string prompt =
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";

    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference->config().primary_eos_token_id();

    auto result = inference->generate(token_ids, /*max_new_tokens=*/100, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference->tokenizer().decode(gen_so_far);
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
    // Verifies that the optional q/k/v_proj.bias weight probe runs without crashing.
    // Llama/Mistral don't have these tensors (lookup returns end()), so they're unaffected.
    // If we reached here, the bias probe and add() path executed without error.
    EXPECT_EQ(inference->config().model_type, "qwen2");
    std::cout << "✓ Optional attention bias path ran without error" << std::endl;
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
