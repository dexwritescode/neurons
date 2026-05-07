#include <gtest/gtest.h>
#include "compute/model/llama_model.h"
#include "compute/core/compute_backend.h"
#include <chrono>
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
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(kModelDir)) {
            skip_reason_ = "Llama-3.1-8B-Instruct-4bit not found at " + kModelDir.string()
                         + " — download mlx-community/Llama-3.1-8B-Instruct-4bit first";
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto inf_result = LlamaModel::from_model_dir(kModelDir, backend_.get());
        if (!inf_result) {
            skip_reason_ = "Failed to load Llama-3: " + inf_result.error().message;
            return;
        }
        inference_ = std::make_unique<LlamaModel>(std::move(*inf_result));

        std::cout << "Loaded Llama-3 model: " << inference_->config().model_type
                  << " hidden=" << inference_->config().hidden_size
                  << " layers=" << inference_->config().num_hidden_layers << std::endl;
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

    static std::string                         skip_reason_;
    static std::unique_ptr<ComputeBackend>     backend_;
    static std::unique_ptr<LlamaModel> inference_;
};

std::string                         Llama3IntegrationTest::skip_reason_;
std::unique_ptr<ComputeBackend>     Llama3IntegrationTest::backend_;
std::unique_ptr<LlamaModel> Llama3IntegrationTest::inference_;

// ── Config ────────────────────────────────────────────────────────────────────

TEST_F(Llama3IntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference_->config().model_type, "llama");
    EXPECT_EQ(inference_->config().vocab_size, 128256u);
    EXPECT_EQ(inference_->config().hidden_size, 4096u);
    EXPECT_EQ(inference_->config().num_hidden_layers, 32u);
    EXPECT_EQ(inference_->config().num_attention_heads, 32u);
    EXPECT_EQ(inference_->config().num_key_value_heads, 8u);
    EXPECT_TRUE(inference_->config().is_llama_architecture());
    EXPECT_TRUE(inference_->config().is_supported_architecture());
    EXPECT_FALSE(inference_->config().is_mistral_architecture());

    ASSERT_TRUE(inference_->config().eos_token_ids.has_value());
    EXPECT_TRUE(inference_->config().is_eos_token(128001));
    EXPECT_TRUE(inference_->config().is_eos_token(128008));
    EXPECT_TRUE(inference_->config().is_eos_token(128009));
    EXPECT_FALSE(inference_->config().is_eos_token(2));

    std::cout << "✓ Llama-3 config validated" << std::endl;
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

TEST_F(Llama3IntegrationTest, ByteLevelPreTokenizerDetected) {
    const int gais_id = inference_->tokenizer().find_token_id("Ġis");
    (void)gais_id;
    EXPECT_NE(inference_->tokenizer().bos_token_id(), -1);
    EXPECT_EQ(inference_->tokenizer().bos_token_id(), 128000);
}

TEST_F(Llama3IntegrationTest, TokenizerPlainTextMatchesHFReference) {
    const std::string prompt = "What is the capital of France?";
    auto ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/false);

    std::cout << "Token IDs (" << ids.size() << "): ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    const std::vector<int> hf_reference = {3923, 374, 279, 6864, 315, 9822, 30};
    EXPECT_EQ(ids, hf_reference)
        << "Tokenization diverged from HuggingFace reference.";
}

TEST_F(Llama3IntegrationTest, TokenizerDecodeRoundtrip) {
    const std::string text = "Hello, world! How are you?";
    auto ids = inference_->tokenizer().encode(text, /*add_special_tokens=*/false);
    auto decoded = inference_->tokenizer().decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded, text) << "Encode→decode roundtrip failed for: " << text;
}

// ── Generation ────────────────────────────────────────────────────────────────

TEST_F(Llama3IntegrationTest, GenerateCapitalOfFrance) {
    const std::string prompt =
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty());
    std::cout << "Prompt tokens: " << token_ids.size() << std::endl;

    const ModelConfig& cfg = inference_->config();
    std::vector<int> generated;
    std::string decoded_so_far;
    auto& tok = inference_->tokenizer();

    SamplingParams params;
    params.temperature = 0.0f;

    auto result = inference_->generate(
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

    const bool has_paris = decoded_so_far.find("Paris") != std::string::npos ||
                           decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(has_paris)
        << "Expected 'Paris' in output, got: \"" << decoded_so_far << "\"";
}

TEST_F(Llama3IntegrationTest, GenerateThroughput) {
    const std::string prompt =
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Write a detailed paragraph about the history of France, "
        "including the French Revolution, Napoleon, and the World Wars."
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n";

    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/false);
    ASSERT_FALSE(token_ids.empty());

    SamplingParams greedy;
    greedy.temperature = 0.0f;

    // Warmup: trigger mx::compile before measuring steady-state decode.
    inference_->generate(token_ids, /*max_new_tokens=*/8, greedy, [](int) { return true; });

    int token_count = 0;
    auto start = std::chrono::steady_clock::now();
    auto result = inference_->generate(token_ids, /*max_new_tokens=*/128, greedy,
        [&](int /*tok*/) { ++token_count; return true; });
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_GT(token_count, 0) << "Model produced no tokens";

    double tok_s = token_count * 1000.0 / elapsed_ms;
    std::cout << "Llama-3.1 8B throughput: " << tok_s << " tok/s ("
              << token_count << " tokens in " << elapsed_ms << " ms)" << std::endl;

    // Baseline (debug build, warmed): ~68 tok/s. Floor = baseline / 2.
    EXPECT_GE(tok_s, 33.0) << "throughput regression: " << tok_s << " tok/s";
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
