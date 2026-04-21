#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <vector>
#include <iostream>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#include <mlx/mlx.h>

namespace compute {

class ForwardPassTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_dir    = TINYLLAMA_MODEL_DIR;
        baseline_dir = std::filesystem::path(TEST_RESOURCES_DIR).parent_path() / "baselines" / "output";

        if (!std::filesystem::exists(model_dir))
            GTEST_SKIP() << "Model not found: " << model_dir;

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir, backend.get());
        ASSERT_TRUE(inf_result.has_value()) << inf_result.error().message;
        inference = std::make_unique<TinyLlamaInference>(std::move(*inf_result));
    }

    void TearDown() override {
        inference.reset();
        if (backend) backend->cleanup();
    }

    std::filesystem::path model_dir;
    std::filesystem::path baseline_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

// Verify forward_logits() produces the same greedy next token as Python MLX
TEST_F(ForwardPassTest, GreedyTokenMatchesPython) {
    // Same token IDs used in the Python baseline script
    // "What is the capital of France?" (with BOS=1)
    std::vector<int> token_ids = {1, 1724, 338, 278, 7483, 310, 3444, 29973};

    std::cout << "Running forward_logits for " << token_ids.size() << " tokens..." << std::endl;

    auto logits_result = inference->forward(token_ids);
    ASSERT_TRUE(logits_result.has_value()) << logits_result.error().message;

    const auto& logits = *logits_result;
    ASSERT_EQ(logits.size(), inference->config().vocab_size)
        << "Logits size should be vocab_size";

    // Find greedy (argmax) token
    int greedy_token = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    std::cout << "C++ greedy next token: " << greedy_token << std::endl;

    // Correct greedy token is 2 (EOS) for this bare 8-token input without chat template.
    // Token 13 was from the old incorrect baseline (pre-RoPE-fix).
    const int python_greedy = 2;
    EXPECT_EQ(greedy_token, python_greedy)
        << "Greedy token mismatch: C++ got " << greedy_token
        << ", Python got " << python_greedy;

    // Top logit should be finite and reasonable
    float top_logit = logits[greedy_token];
    EXPECT_TRUE(std::isfinite(top_logit)) << "Top logit is not finite";
    EXPECT_GT(top_logit, -100.0f) << "Top logit unreasonably small";
    EXPECT_LT(top_logit, 100.0f)  << "Top logit unreasonably large";

    std::cout << "✓ Greedy token matches Python baseline (token=" << greedy_token
              << ", logit=" << top_logit << ")" << std::endl;
}

// Verify generate() produces coherent text for a known prompt
TEST_F(ForwardPassTest, GenerateCoherentOutput) {
    // Tokenize a simple prompt using the model's tokenizer
    const std::string prompt = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);

    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";
    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    // Generate up to 30 tokens with greedy decoding (temperature=0)
    compute::SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;
    auto generated_result = inference->generate(token_ids, /*max_new_tokens=*/30, greedy);
    ASSERT_TRUE(generated_result.has_value()) << generated_result.error().message;

    const auto& generated_ids = *generated_result;
    ASSERT_FALSE(generated_ids.empty()) << "generate() returned no tokens";

    // Decode generated tokens
    std::string output = inference->tokenizer().decode(generated_ids);
    std::cout << "Generated output: \"" << output << "\"" << std::endl;

    // The model should mention Paris in its response
    bool mentions_paris = output.find("Paris") != std::string::npos ||
                          output.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected 'Paris' in output, got: \"" << output << "\"";
}

} // namespace compute

#endif // MLX_BACKEND_ENABLED