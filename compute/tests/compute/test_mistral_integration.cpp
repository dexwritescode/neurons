#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <iostream>
#include <string>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

class MistralIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Try 8-bit first, fall back to 4-bit
        model_dir_ = "/Users/dex/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-8bit";
        if (!std::filesystem::exists(model_dir_))
            model_dir_ = MISTRAL_MODEL_DIR;

        if (!std::filesystem::exists(model_dir_)) {
            skip_reason_ = "Mistral model not found at " + model_dir_.string()
                         + " — download mlx-community/Mistral-7B-Instruct-v0.3-4bit first";
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir_, backend_.get());
        if (!inf_result) {
            skip_reason_ = "Failed to load Mistral: " + inf_result.error().message;
            return;
        }
        inference_ = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded Mistral model: " << inference_->config().model_type
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

    static std::filesystem::path              model_dir_;
    static std::string                        skip_reason_;
    static std::unique_ptr<ComputeBackend>    backend_;
    static std::unique_ptr<TinyLlamaInference> inference_;
};

std::filesystem::path               MistralIntegrationTest::model_dir_;
std::string                         MistralIntegrationTest::skip_reason_;
std::unique_ptr<ComputeBackend>     MistralIntegrationTest::backend_;
std::unique_ptr<TinyLlamaInference> MistralIntegrationTest::inference_;

// Verify the model loads with correct Mistral architecture config
TEST_F(MistralIntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference_->config().model_type, "mistral");
    EXPECT_EQ(inference_->config().hidden_size, 4096u);
    EXPECT_EQ(inference_->config().num_hidden_layers, 32u);
    EXPECT_EQ(inference_->config().num_attention_heads, 32u);
    EXPECT_EQ(inference_->config().num_key_value_heads, 8u);
    EXPECT_EQ(inference_->config().vocab_size, 32768u);
    EXPECT_TRUE(inference_->config().is_mistral_architecture());
    EXPECT_TRUE(inference_->config().is_supported_architecture());
    EXPECT_FALSE(inference_->config().is_llama_architecture());
    std::cout << "✓ Mistral config validated" << std::endl;
}

// Reproduces the exact GUI path: load via LanguageModel::load (which moves
// the tokenizer into LlamaModel) and then call encode() on the resulting
// tokenizer reference.
TEST_F(MistralIntegrationTest, TokenizerViaLanguageModelLoadMatchesHFReference) {
    auto ids = inference_->tokenizer().encode(
        "[INST] what is the capital of france? [/INST]", /*add_special_tokens=*/true);
    std::cout << "DIAG via LanguageModel::load: " << ids.size() << " tokens:";
    for (int id : ids) std::cout << " " << id;
    std::cout << std::endl;
    const std::vector<int> hf_reference = {
        1, 3, 29473, 1535, 1117, 1040, 6333, 1070, 1872, 1385, 29572, 29473, 4
    };
    EXPECT_EQ(ids, hf_reference)
        << "Tokenizer via LanguageModel::load diverged from HF reference.";
}

// Verify the tokenizer encodes/decodes Mistral tokens correctly
TEST_F(MistralIntegrationTest, TokenizerBasicRoundtrip) {
    const std::string text = "Hello, world!";
    auto ids = inference_->tokenizer().encode(text);
    ASSERT_FALSE(ids.empty()) << "Tokenizer produced empty output for: " << text;

    std::string decoded = inference_->tokenizer().decode(ids);
    std::cout << "Encoded '" << text << "' → " << ids.size() << " tokens → '" << decoded << "'" << std::endl;
    EXPECT_NE(decoded.find("Hello"), std::string::npos) << "Decoded text missing 'Hello'";
}

// The critical end-to-end test: load → tokenize → generate → verify output quality
TEST_F(MistralIntegrationTest, GenerateCapitalOfFrance) {
    const std::string prompt = "[INST] What is the capital of France? [/INST]";
    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/true);

    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Token count: " << token_ids.size() << std::endl;
    std::cout << "Token IDs: ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "\nToken pieces: ";
    for (int id : token_ids) std::cout << "[" << inference_->tokenizer().decode({id}) << "]";
    std::cout << std::endl;

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference_->tokenizer().eos_token_id();
    auto result = inference_->generate(token_ids, /*max_new_tokens=*/100, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference_->tokenizer().decode(gen_so_far);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << "generate() failed: " << result.error().message;
    ASSERT_FALSE(decoded_so_far.empty()) << "Model produced no output";

    std::cout << "Token count generated: " << gen_so_far.size() << std::endl;
    std::cout << "Generated output: \"" << decoded_so_far << "\"" << std::endl;

    bool mentions_paris = decoded_so_far.find("Paris") != std::string::npos ||
                          decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected output to mention Paris, got: \"" << decoded_so_far << "\"";
}

// Sanity check: model should produce coherent (non-garbage) text
TEST_F(MistralIntegrationTest, OutputIsCoherentText) {
    const std::string prompt = "[INST] Say the word 'hello' [/INST]";
    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::string output;
    inference_->generate(token_ids, /*max_new_tokens=*/20, greedy,
        [&](int tok) { output += inference_->tokenizer().decode({tok}); return true; });

    std::cout << "Coherence check output: \"" << output << "\"" << std::endl;

    int printable = 0, total = static_cast<int>(output.size());
    for (char c : output) if (c >= 32 && c < 127) ++printable;
    if (total > 0) {
        float ratio = static_cast<float>(printable) / total;
        EXPECT_GT(ratio, 0.8f) << "Output contains too many non-printable characters";
    }
}

// Diagnostic: print top-5 logits at each decode step
TEST_F(MistralIntegrationTest, DiagnosticTokenLogitTrace) {
    const std::string prompt = "[INST] What is the capital of France? [/INST]";
    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    std::cout << "\n=== C++ Logit Trace (compare with Python MLX) ===\n";
    std::cout << "Prompt tokens (" << token_ids.size() << "):";
    for (int id : token_ids) std::cout << " " << id;
    std::cout << "\n\n";

    const std::vector<int> python_reference = {1183, 6333, 1070, 5611, 1117, 6233, 29491, 3761, 29493, 1146, 29510, 29481, 3046, 1066, 5807};

    auto prefill_result = inference_->prefill(token_ids);
    ASSERT_TRUE(prefill_result.has_value()) << prefill_result.error().message;

    std::vector<int> generated;
    const int eos_id = inference_->config().primary_eos_token_id();
    std::vector<float> current_logits = *prefill_result;

    for (int step = 0; step < 15; ++step) {
        std::vector<std::pair<float, int>> top;
        for (size_t i = 0; i < current_logits.size(); ++i)
            top.push_back({current_logits[i], static_cast<int>(i)});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });

        int chosen = top[0].second;
        std::cout << "Step " << (step+1) << ": top tokens = ";
        for (int k = 0; k < 5; ++k) {
            std::cout << "[" << top[k].second << "="
                      << inference_->tokenizer().decode({top[k].second})
                      << "(" << top[k].first << ")] ";
        }
        if (step < (int)python_reference.size()) {
            int ref = python_reference[step];
            bool match = (chosen == ref);
            std::cout << "→ chose " << chosen
                      << " (python=" << ref << ") "
                      << (match ? "✓" : "✗ DIVERGED");
        }
        std::cout << "\n";

        generated.push_back(chosen);
        if (chosen == eos_id) break;

        auto next = inference_->decode(chosen);
        if (!next.has_value()) break;
        current_logits = *next;
    }

    std::cout << "\nC++ output: " << inference_->tokenizer().decode(generated) << "\n";
    std::cout << "================================================\n\n";
}

// Exercises the same sampling path as the app
TEST_F(MistralIntegrationTest, GenerateCapitalOfFranceWithSampling) {
    const std::string prompt = "[INST] what is the capital of france? [/INST]";
    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Token IDs (" << token_ids.size() << "): ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "\n";
    std::cout << "Token pieces: ";
    for (int id : token_ids) std::cout << "[" << inference_->tokenizer().decode({id}) << "]";
    std::cout << std::endl;

    SamplingParams app_params;
    app_params.temperature = 0.7f;
    app_params.top_p       = 0.9f;
    app_params.top_k       = 40;
    app_params.rep_penalty = 1.1f;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference_->tokenizer().eos_token_id();

    auto result = inference_->generate(token_ids, /*max_new_tokens=*/200, app_params,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference_->tokenizer().decode(gen_so_far);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_FALSE(decoded_so_far.empty()) << "Model produced no output";

    std::cout << "Sampled output (" << gen_so_far.size() << " tokens): \""
              << decoded_so_far << "\"" << std::endl;

    bool mentions_paris = decoded_so_far.find("Paris") != std::string::npos ||
                          decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected output to mention Paris, got: \"" << decoded_so_far << "\"";

    int printable = 0;
    for (char c : decoded_so_far) if (c >= 32 && c < 127) ++printable;
    float ratio = static_cast<float>(printable) / static_cast<float>(decoded_so_far.size());
    EXPECT_GT(ratio, 0.85f) << "Output contains too many non-printable chars";
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
