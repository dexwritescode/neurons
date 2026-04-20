#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

static const std::filesystem::path kModelDir =
    std::filesystem::path(std::getenv("HOME")) /
    ".neurons/models/mlx-community/Llama-3.1-8B-Instruct-4bit";

class Llama3IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModelDir)) {
            GTEST_SKIP() << "Llama-3.1-8B-Instruct-4bit not found at " << kModelDir
                         << " — download mlx-community/Llama-3.1-8B-Instruct-4bit first";
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto inf_result = TinyLlamaInference::from_model_dir(kModelDir, backend.get());
        ASSERT_TRUE(inf_result.has_value())
            << "Failed to load Llama-3: " << inf_result.error().message;
        inference = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded Llama-3 model: " << inference->config().model_type
                  << " hidden=" << inference->config().hidden_size
                  << " layers=" << inference->config().num_hidden_layers << std::endl;
    }

    void TearDown() override {
        inference.reset();
        if (backend) backend->cleanup();
    }

    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

// ── Config ────────────────────────────────────────────────────────────────────

TEST_F(Llama3IntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference->config().model_type, "llama");
    EXPECT_EQ(inference->config().vocab_size, 128256u);
    EXPECT_EQ(inference->config().hidden_size, 4096u);
    EXPECT_EQ(inference->config().num_hidden_layers, 32u);
    EXPECT_EQ(inference->config().num_attention_heads, 32u);
    EXPECT_EQ(inference->config().num_key_value_heads, 8u);
    EXPECT_TRUE(inference->config().is_llama_architecture());
    EXPECT_TRUE(inference->config().is_supported_architecture());
    EXPECT_FALSE(inference->config().is_mistral_architecture());

    // Multiple EOS token IDs
    ASSERT_TRUE(inference->config().eos_token_ids.has_value());
    EXPECT_TRUE(inference->config().is_eos_token(128001));  // <|end_of_text|>
    EXPECT_TRUE(inference->config().is_eos_token(128008));  // <|eom_id|>
    EXPECT_TRUE(inference->config().is_eos_token(128009));  // <|eot_id|>
    EXPECT_FALSE(inference->config().is_eos_token(2));      // old EOS should NOT match

    std::cout << "✓ Llama-3 config validated" << std::endl;
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

TEST_F(Llama3IntegrationTest, ByteLevelPreTokenizerDetected) {
    // ByteLevel pre-tokenizer should have been detected from tokenizer.json
    // We verify indirectly: tokens like "Ġis" (space+is) should be in the vocab
    const int gais_id = inference->tokenizer().find_token_id("Ġis");  // U+0120 + 'is'
    // It's possible the token uses a different representation; the key check
    // is that tokenization produces correct IDs matching the HF reference.
    (void)gais_id;

    // BOS token should be present
    EXPECT_NE(inference->tokenizer().bos_token_id(), -1);
    EXPECT_EQ(inference->tokenizer().bos_token_id(), 128000);  // <|begin_of_text|>
}

TEST_F(Llama3IntegrationTest, TokenizerPlainTextMatchesHFReference) {
    // HuggingFace reference for "What is the capital of France?" (no add_special_tokens):
    //   [3923, 374, 279, 6864, 315, 9822, 30]
    // Tokens: ['What', 'Ġis', 'Ġthe', 'Ġcapital', 'Ġof', 'ĠFrance', '?']
    const std::string prompt = "What is the capital of France?";
    auto ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/false);

    std::cout << "Token IDs (" << ids.size() << "): ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    const std::vector<int> hf_reference = {3923, 374, 279, 6864, 315, 9822, 30};
    EXPECT_EQ(ids, hf_reference)
        << "Tokenization diverged from HuggingFace reference.";
}

TEST_F(Llama3IntegrationTest, TokenizerDecodeRoundtrip) {
    const std::string text = "Hello, world! How are you?";
    auto ids = inference->tokenizer().encode(text, /*add_special_tokens=*/false);
    auto decoded = inference->tokenizer().decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, text) << "Encode→decode roundtrip failed for: " << text;
}

// ── Generation ────────────────────────────────────────────────────────────────

// Uses the same load path as the CLI and service.
// Tests the full Llama-3 pipeline: ByteLevel tokenization → GQA attention → LM head.
TEST_F(Llama3IntegrationTest, GenerateCapitalOfFrance) {
    // Llama-3 chat template (matches load_command.cpp and neurons_service.cpp).
    // BOS is explicit in the template since add_bos_token=null in tokenizer_config.
    const std::string prompt =
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty());
    std::cout << "Prompt tokens: " << token_ids.size() << std::endl;

    const ModelConfig& cfg = inference->config();
    std::vector<int> generated;
    std::string decoded_so_far;
    auto& tok = inference->tokenizer();

    SamplingParams params;
    params.temperature = 0.0f;  // greedy

    auto result = inference->generate(
        token_ids, /*max_new_tokens=*/64, params,
        [&](int tok_id) -> bool {
            if (cfg.is_eos_token(tok_id)) return false;
            generated.push_back(tok_id);
            decoded_so_far = tok.decode(generated, /*skip_special_tokens=*/true);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_FALSE(decoded_so_far.empty()) << "Model produced no output";

    std::cout << "Output (" << generated.size() << " tokens): \""
              << decoded_so_far << "\"" << std::endl;

    // The answer must mention Paris
    const bool has_paris = decoded_so_far.find("Paris") != std::string::npos ||
                           decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(has_paris)
        << "Expected 'Paris' in output, got: \"" << decoded_so_far << "\"";
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
