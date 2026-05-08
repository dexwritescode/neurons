#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../../src/compute/model/tool_runner.h"
#include "../../src/compute/model/language_model.h"
#include "../../src/compute/model/simple_bpe_tokenizer.h"
#include "../../src/compute/model/model_config.h"

#include <atomic>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

// ── StubLanguageModel ─────────────────────────────────────────────────────────
//
// Controlled LanguageModel for testing ToolRunner in isolation.
//
// Tool call format:  "<<TOOL:name:args_json>>"
// Tool result format: "<<RESULT:name:result_json>>"
//
// Pre-configure what text each generate() turn should produce via add_turn().
// The tokenizer is loaded from a real model directory so encode/decode works.

class StubLanguageModel : public compute::LanguageModel {
public:
    static constexpr const char* kModelPath =
        "/Users/dex/.neurons/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0";

    explicit StubLanguageModel(compute::SimpleBpeTokenizer tok)
        : tok_(std::move(tok)) {
        // Minimal ModelConfig — no EOS tokens so the stub generate() drives termination.
        config_.vocab_size            = 32000;
        config_.hidden_size           = 1;
        config_.num_hidden_layers     = 1;
        config_.num_attention_heads   = 1;
        config_.num_key_value_heads   = 1;
        config_.intermediate_size     = 1;
        config_.max_position_embeddings = 4096;
        config_.rms_norm_eps          = 1e-5f;
        config_.rope_theta            = 10000.f;
        config_.hidden_act            = "silu";
        config_.attention_bias        = false;
        config_.tie_word_embeddings   = false;
        config_.model_type            = "stub";
        config_.torch_dtype           = "float32";
        config_.architectures         = {"StubForCausalLM"};
    }

    // Pre-encode text so generate() can call on_token with the right IDs.
    void add_turn(const std::string& text) {
        turns_.push_back(tok_.encode(text, /*add_special_tokens=*/false));
    }

    // ── LanguageModel interface ───────────────────────────────────────────────

    compute::Result<std::vector<int>> generate(
            const std::vector<int>&,
            size_t,
            compute::SamplingParams,
            std::function<bool(int)> on_token) override {
        std::vector<int> produced;
        if (turn_ < static_cast<int>(turns_.size())) {
            for (int id : turns_[turn_]) {
                if (!on_token(id)) break;
                produced.push_back(id);
            }
            ++turn_;
        }
        return produced;
    }

    bool supports_tool_use() const override { return true; }

    std::optional<ToolCall> detect_tool_call(const std::string& text) const override {
        const auto pos = text.find("<<TOOL:");
        if (pos == std::string::npos) return std::nullopt;
        const auto end = text.find(">>", pos);
        if (end == std::string::npos) return std::nullopt;
        const auto inner = text.substr(pos + 7, end - pos - 7);
        const auto colon = inner.find(':');
        if (colon == std::string::npos) return std::nullopt;
        ToolCall tc;
        tc.name           = inner.substr(0, colon);
        tc.arguments_json = inner.substr(colon + 1);
        return tc;
    }

    std::string format_tool_result(const std::string& name,
                                   const std::string& result_json) const override {
        return " <<RESULT:" + name + ":" + result_json + ">>";
    }

    const compute::SimpleBpeTokenizer& tokenizer()    const override { return tok_; }
    const compute::ModelConfig&        config()        const override { return config_; }
    const std::string&                 model_type()   const override { return config_.model_type; }
    size_t                             num_parameters() const override { return 0; }

private:
    compute::SimpleBpeTokenizer    tok_;
    compute::ModelConfig           config_;
    std::vector<std::vector<int>>  turns_;
    int                            turn_ = 0;
};

// ── Fixture ───────────────────────────────────────────────────────────────────

class ToolRunnerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        const std::filesystem::path path{StubLanguageModel::kModelPath};
        if (!std::filesystem::exists(path)) {
            skip_reason_ = "TinyLlama not downloaded";
            return;
        }
        auto tok = compute::SimpleBpeTokenizer::from_model_dir(path);
        if (!tok.has_value()) {
            skip_reason_ = "Failed to load tokenizer: " + tok.error().message;
            return;
        }
        tok_ = std::make_unique<compute::SimpleBpeTokenizer>(std::move(*tok));
    }

    static void TearDownTestSuite() { tok_.reset(); }

    void SetUp() override {
        if (!skip_reason_.empty()) GTEST_SKIP() << skip_reason_;
    }

    std::unique_ptr<StubLanguageModel> make_model() {
        auto tok = compute::SimpleBpeTokenizer::from_model_dir(
            std::filesystem::path{StubLanguageModel::kModelPath});
        return std::make_unique<StubLanguageModel>(std::move(*tok));
    }

    static std::unique_ptr<compute::SimpleBpeTokenizer> tok_;
    static std::string                                   skip_reason_;
};

std::unique_ptr<compute::SimpleBpeTokenizer> ToolRunnerTest::tok_;
std::string                                  ToolRunnerTest::skip_reason_;

// ── Tests ─────────────────────────────────────────────────────────────────────

// No tool callback — single turn, all deltas streamed, correct token count.
TEST_F(ToolRunnerTest, NoToolCb_SingleTurnStreamsAllTokens) {
    auto model = make_model();
    model->add_turn("Hello world");

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model, {1}, 256, {},
        [&](const std::string& d) { output += d; return true; },
        /*tool_cb=*/nullptr,
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result.value(), 0u);
    EXPECT_THAT(output, ::testing::HasSubstr("Hello"));
}

// Detects a tool call, invokes tool_cb, injects result, continues second turn.
TEST_F(ToolRunnerTest, SingleToolTurn_ToolCbCalledAndResultInjected) {
    auto model = make_model();
    model->add_turn("Let me check. <<TOOL:lookup:{\"query\":\"capitals\"}>>");
    model->add_turn("Paris is the answer.");

    std::string output;
    int tool_calls = 0;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model, {1}, 256, {},
        [&](const std::string& d) { output += d; return true; },
        [&](const compute::LanguageModel::ToolCall& call) -> std::optional<std::string> {
            ++tool_calls;
            EXPECT_EQ(call.name, "lookup");
            return R"({"result":"Paris"})";
        },
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(tool_calls, 1);
    EXPECT_THAT(output, ::testing::HasSubstr("Paris is the answer"));
}

// When tool_cb returns nullopt the denied error is injected and generation continues.
TEST_F(ToolRunnerTest, ToolDenied_ErrorInjectedAndGenerationContinues) {
    auto model = make_model();
    model->add_turn("<<TOOL:secret:{}>>");
    model->add_turn("I cannot access that.");

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model, {1}, 256, {},
        [&](const std::string& d) { output += d; return true; },
        [&](const compute::LanguageModel::ToolCall&) -> std::optional<std::string> {
            return std::nullopt;  // deny
        },
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(output, ::testing::HasSubstr("I cannot access that"));
}

// Returning false from token_cb cancels generation.
TEST_F(ToolRunnerTest, TokenCbReturnsFalse_StopsGeneration) {
    auto model = make_model();
    model->add_turn("This is a very long response that should be cut short.");

    int delta_count = 0;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model, {1}, 256, {},
        [&](const std::string&) -> bool { return ++delta_count < 3; },
        /*tool_cb=*/nullptr,
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result.value(), 20u);  // stopped early, not all tokens generated
}

// Setting the cancelled flag stops generation.
TEST_F(ToolRunnerTest, CancelledFlag_StopsGeneration) {
    auto model = make_model();
    model->add_turn("Long response that will be cancelled mid-way through generation.");

    std::atomic<bool> cancelled{false};
    int delta_count = 0;

    auto result = compute::ToolRunner{}.run(
        *model, {1}, 256, {},
        [&](const std::string&) -> bool {
            if (++delta_count == 2) cancelled.store(true);
            return true;
        },
        /*tool_cb=*/nullptr,
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result.value(), 20u);
}

} // namespace
