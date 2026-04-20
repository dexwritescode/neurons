#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

// Tests that LlamaModel can load and run inference on standard HuggingFace
// bf16 safetensors models (no quantization, no *.scales / *.biases tensors).
// Uses TinyLlama-1.1B-Chat-v1.0 which ships as a single bfloat16 safetensors.
class Fp16Bf16IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_dir = std::filesystem::path(std::getenv("HOME"))
                    / ".neurons/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0";

        if (!std::filesystem::exists(model_dir)) {
            GTEST_SKIP() << "TinyLlama-1.1B-Chat-v1.0 (bf16) not found — "
                            "download from HuggingFace: TinyLlama/TinyLlama-1.1B-Chat-v1.0";
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir, backend.get());
        ASSERT_TRUE(inf_result.has_value())
            << "Failed to load bf16 model: " << inf_result.error().message;
        inference = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded bf16 model: " << inference->config().model_type
                  << " hidden=" << inference->config().hidden_size
                  << " layers=" << inference->config().num_hidden_layers
                  << " quantized=" << inference->config().quantization.has_value()
                  << std::endl;
    }

    void TearDown() override {
        inference.reset();
        if (backend) backend->cleanup();
    }

    std::filesystem::path model_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

TEST_F(Fp16Bf16IntegrationTest, ConfigIsUnquantized) {
    EXPECT_FALSE(inference->config().quantization.has_value())
        << "bf16 model should not have quantization config";
    EXPECT_EQ(inference->config().model_type, "llama");
    EXPECT_TRUE(inference->config().is_supported_architecture());
    std::cout << "✓ bf16 model config is unquantized" << std::endl;
}

TEST_F(Fp16Bf16IntegrationTest, GenerateCapitalOfFrance) {
    // TinyLlama Zephyr chat template
    const std::string prompt =
        "<|system|>\nYou are a helpful assistant.</s>\n"
        "<|user|>\nWhat is the capital of France?</s>\n"
        "<|assistant|>\n";

    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";

    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference->config().primary_eos_token_id();

    auto result = inference->generate(token_ids, /*max_new_tokens=*/64, greedy,
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

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
