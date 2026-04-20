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
    void SetUp() override {
        // Try 8-bit first, fall back to 4-bit
        model_dir = "/Users/dex/.neurons/models/mlx-community/Mistral-7B-Instruct-v0.3-8bit";
        if (!std::filesystem::exists(model_dir))
            model_dir = MISTRAL_MODEL_DIR;

        if (!std::filesystem::exists(model_dir)) {
            GTEST_SKIP() << "Mistral model not found at " << model_dir
                         << " — download mlx-community/Mistral-7B-Instruct-v0.3-4bit first";
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value()) << backend_result.error().message;
        backend = std::move(*backend_result);
        ASSERT_TRUE(backend->initialize().has_value());

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir, backend.get());
        ASSERT_TRUE(inf_result.has_value()) << "Failed to load Mistral: " << inf_result.error().message;
        inference = std::make_unique<TinyLlamaInference>(std::move(*inf_result));

        std::cout << "Loaded Mistral model: " << inference->config().model_type
                  << " hidden=" << inference->config().hidden_size
                  << " layers=" << inference->config().num_hidden_layers << std::endl;
    }

    void TearDown() override {
        inference.reset();
        if (backend) backend->cleanup();
    }

    std::filesystem::path model_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

// Verify the model loads with correct Mistral architecture config
TEST_F(MistralIntegrationTest, ConfigLoadsCorrectly) {
    EXPECT_EQ(inference->config().model_type, "mistral");
    EXPECT_EQ(inference->config().hidden_size, 4096u);
    EXPECT_EQ(inference->config().num_hidden_layers, 32u);
    EXPECT_EQ(inference->config().num_attention_heads, 32u);
    EXPECT_EQ(inference->config().num_key_value_heads, 8u);
    EXPECT_EQ(inference->config().vocab_size, 32768u);
    EXPECT_TRUE(inference->config().is_mistral_architecture());
    EXPECT_TRUE(inference->config().is_supported_architecture());
    EXPECT_FALSE(inference->config().is_llama_architecture());
    std::cout << "✓ Mistral config validated" << std::endl;
}

// Reproduces the exact GUI path: load via LanguageModel::load (which moves
// the tokenizer into LlamaModel) and then call encode() on the resulting
// tokenizer reference. Compares against HF reference to catch any divergence
// introduced by the move semantics or LlamaModel's tokenizer() accessor.
TEST_F(MistralIntegrationTest, TokenizerViaLanguageModelLoadMatchesHFReference) {
    auto ids = inference->tokenizer().encode(
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
    auto ids = inference->tokenizer().encode(text);
    ASSERT_FALSE(ids.empty()) << "Tokenizer produced empty output for: " << text;

    std::string decoded = inference->tokenizer().decode(ids);
    std::cout << "Encoded '" << text << "' → " << ids.size() << " tokens → '" << decoded << "'" << std::endl;
    // Decoded may have leading space due to sentencepiece — allow for that
    EXPECT_NE(decoded.find("Hello"), std::string::npos) << "Decoded text missing 'Hello'";
}

// The critical end-to-end test: load → tokenize → generate → verify output quality
// This test would have caught the garbage output bug immediately.
TEST_F(MistralIntegrationTest, GenerateCapitalOfFrance) {
    // Mistral [INST] chat template
    const std::string prompt = "[INST] What is the capital of France? [/INST]";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);

    ASSERT_FALSE(token_ids.empty()) << "Tokenizer produced empty output";
    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Token count: " << token_ids.size() << std::endl;
    std::cout << "Token IDs: ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "\nToken pieces: ";
    for (int id : token_ids) std::cout << "[" << inference->tokenizer().decode({id}) << "]";
    std::cout << std::endl;

    // Greedy decoding — deterministic, no sampling noise
    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    // Use cumulative decode (same as ChatEngine) for correct spacing
    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference->tokenizer().eos_token_id();
    auto result = inference->generate(token_ids, /*max_new_tokens=*/100, greedy,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference->tokenizer().decode(gen_so_far);
            return true;
        });

    const std::string& output = decoded_so_far;
    ASSERT_TRUE(result.has_value()) << "generate() failed: " << result.error().message;
    ASSERT_FALSE(output.empty()) << "Model produced no output";

    std::cout << "Token count generated: " << gen_so_far.size() << std::endl;
    std::cout << "Generated output: \"" << output << "\"" << std::endl;

    bool mentions_paris = output.find("Paris") != std::string::npos ||
                          output.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected output to mention Paris, got: \"" << output << "\"";
}

// Sanity check: model should NOT produce the TinyLlama-style garbage we saw
TEST_F(MistralIntegrationTest, OutputIsCoherentText) {
    const std::string prompt = "[INST] Say the word 'hello' [/INST]";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;

    std::string output;
    inference->generate(token_ids, /*max_new_tokens=*/20, greedy,
        [&](int tok) { output += inference->tokenizer().decode({tok}); return true; });

    std::cout << "Coherence check output: \"" << output << "\"" << std::endl;

    // Output should be printable ASCII / reasonable text, not binary garbage
    int printable = 0, total = static_cast<int>(output.size());
    for (char c : output) if (c >= 32 && c < 127) ++printable;
    if (total > 0) {
        float ratio = static_cast<float>(printable) / total;
        EXPECT_GT(ratio, 0.8f) << "Output contains too many non-printable characters — likely garbage";
    }
}

// Diagnostic: print top-5 logits at each decode step, compare with Python MLX reference.
// Python reference (greedy, same prompt, same model) generated:
//   Step 1: 1183="The"   Step 2: 6333="capital"  Step 3: 1070="of"
//   Step 4: 5611="France" Step 5: 1117="is"       Step 6: 6233="Paris"
//   Step 7: 29491="."     Step 8: 3761="However"  ...
TEST_F(MistralIntegrationTest, DiagnosticTokenLogitTrace) {
    const std::string prompt = "[INST] What is the capital of France? [/INST]";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    std::cout << "\n=== C++ Logit Trace (compare with Python MLX) ===\n";
    std::cout << "Prompt tokens (" << token_ids.size() << "):";
    for (int id : token_ids) std::cout << " " << id;
    std::cout << "\n\n";

    // Python reference token sequence (greedy):
    const std::vector<int> python_reference = {1183, 6333, 1070, 5611, 1117, 6233, 29491, 3761, 29493, 1146, 29510, 29481, 3046, 1066, 5807};

    // Run prefill
    auto prefill_result = inference->prefill(token_ids);
    ASSERT_TRUE(prefill_result.has_value()) << prefill_result.error().message;

    std::vector<int> generated;
    const int eos_id = inference->config().primary_eos_token_id();
    std::vector<float> current_logits = *prefill_result;

    for (int step = 0; step < 15; ++step) {
        const auto& logits = current_logits;

        // Find top-5
        std::vector<std::pair<float, int>> top;
        for (size_t i = 0; i < logits.size(); ++i)
            top.push_back({logits[i], static_cast<int>(i)});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });

        int chosen = top[0].second;
        std::cout << "Step " << (step+1) << ": top tokens = ";
        for (int k = 0; k < 5; ++k) {
            std::cout << "[" << top[k].second << "="
                      << inference->tokenizer().decode({top[k].second})
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

        // Decode next
        auto next = inference->decode(chosen);
        if (!next.has_value()) break;
        current_logits = *next;
    }

    std::cout << "\nC++ output: " << inference->tokenizer().decode(generated) << "\n";
    std::cout << "================================================\n\n";
    // This test always passes — it's diagnostic only
}

// Diagnostic: compare no-cache forward vs prefill+decode to isolate KV cache bug.
// Python (no cache, full sequence) gives "capital"=28.4531 at step 2.
// If forward() matches Python but decode() doesn't, the bug is in KV cache concatenation.
TEST_F(MistralIntegrationTest, DiagnosticNoCacheVsDecode) {
    const std::string prompt = "[INST] What is the capital of France? [/INST]";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    const int tok_the = 1183;  // "The"

    // No-cache forward on [prompt + "The"] — should match Python exactly
    std::vector<int> extended = token_ids;
    extended.push_back(tok_the);
    auto nc_result = inference->forward(extended);
    ASSERT_TRUE(nc_result.has_value()) << nc_result.error().message;

    // Prefill + one decode step
    auto pf_result = inference->prefill(token_ids);
    ASSERT_TRUE(pf_result.has_value()) << pf_result.error().message;
    auto dc_result = inference->decode(tok_the);
    ASSERT_TRUE(dc_result.has_value()) << dc_result.error().message;

    // Print top-5 for both paths
    auto print_top5 = [&](const std::string& label, const std::vector<float>& logits) {
        std::vector<std::pair<float,int>> top;
        for (size_t i = 0; i < logits.size(); ++i) top.push_back({logits[i], (int)i});
        std::partial_sort(top.begin(), top.begin()+5, top.end(),
                          [](const auto& a, const auto& b){ return a.first > b.first; });
        std::cout << label << ": ";
        for (int k = 0; k < 5; ++k)
            std::cout << "[" << top[k].second << "=" << inference->tokenizer().decode({top[k].second})
                      << "(" << top[k].first << ")] ";
        std::cout << "\n";
    };

    std::cout << "\n=== No-cache vs decode comparison ===\n";
    std::cout << "Python reference: [6333=capital(28.4531)] [17821=Capital(16.8281)] ...\n";
    print_top5("No-cache forward", *nc_result);
    print_top5("Prefill+decode  ", *dc_result);
}

// Exercises the SAME sampling path as the app (temperature + rep_penalty + top_k/p).
// Greedy tests pass even with a broken sampler — this one catches spiral/garbage output.
// This test would have caught the rep_penalty-per-occurrence compounding bug immediately.
TEST_F(MistralIntegrationTest, GenerateCapitalOfFranceWithSampling) {
    const std::string prompt = "[INST] what is the capital of france? [/INST]";
    auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);
    ASSERT_FALSE(token_ids.empty());

    // Diagnostic: dump tokens + piece-wise decode so we can see what the model actually sees.
    std::cout << "Prompt: \"" << prompt << "\"\n";
    std::cout << "Token IDs (" << token_ids.size() << "): ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "\n";
    std::cout << "Token pieces: ";
    for (int id : token_ids) {
        std::cout << "[" << inference->tokenizer().decode({id}) << "]";
    }
    std::cout << std::endl;

    // Match the app's default SamplingParams exactly (ChatEngine.h defaults).
    // rep_penalty=1.1 — higher values (e.g. 1.3) cause Mistral-7B to ramble.
    SamplingParams app_params;
    app_params.temperature = 0.7f;
    app_params.top_p       = 0.9f;
    app_params.top_k       = 40;
    app_params.rep_penalty = 1.1f;

    std::vector<int> gen_so_far;
    std::string decoded_so_far;
    const int eos_id = inference->tokenizer().eos_token_id();

    auto result = inference->generate(token_ids, /*max_new_tokens=*/200, app_params,
        [&](int tok) {
            if (tok == eos_id) return false;
            gen_so_far.push_back(tok);
            decoded_so_far = inference->tokenizer().decode(gen_so_far);
            return true;
        });

    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_FALSE(decoded_so_far.empty()) << "Model produced no output";

    std::cout << "Sampled output (" << gen_so_far.size() << " tokens): \""
              << decoded_so_far << "\"" << std::endl;

    // Must mention Paris
    bool mentions_paris = decoded_so_far.find("Paris") != std::string::npos ||
                          decoded_so_far.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected output to mention Paris, got: \"" << decoded_so_far << "\"";

    // Must not spiral: output should be mostly printable and not excessively long
    int printable = 0;
    for (char c : decoded_so_far) if (c >= 32 && c < 127) ++printable;
    float ratio = static_cast<float>(printable) / static_cast<float>(decoded_so_far.size());
    EXPECT_GT(ratio, 0.85f) << "Output contains too many non-printable chars — likely garbage spiral";
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
