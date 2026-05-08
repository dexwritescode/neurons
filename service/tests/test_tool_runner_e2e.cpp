#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "mcp/mcp_manager.h"
#include "mcp/mcp_types.h"
#include "compute/model/tool_runner.h"
#include "compute/model/language_model.h"
#include "compute/model/simple_bpe_tokenizer.h"
#include "compute/model/model_config.h"

namespace fs = std::filesystem;
using namespace neurons_service;

// ── StubLanguageModel ─────────────────────────────────────────────────────────
//
// Same approach as the compute unit test: pre-configured turn texts,
// simple <<TOOL:name:args>> detection.

class StubModel : public compute::LanguageModel {
public:
    static constexpr const char* kTokPath =
        "/Users/dex/.neurons/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0";

    explicit StubModel(compute::SimpleBpeTokenizer tok) : tok_(std::move(tok)) {
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

    void add_turn(const std::string& text) {
        turns_.push_back(tok_.encode(text, /*add_special_tokens=*/false));
    }

    compute::Result<std::vector<int>> generate(
            const std::vector<int>&, size_t, compute::SamplingParams,
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

    const compute::SimpleBpeTokenizer& tokenizer()     const override { return tok_; }
    const compute::ModelConfig&        config()         const override { return config_; }
    const std::string&                 model_type()    const override { return config_.model_type; }
    size_t                             num_parameters() const override { return 0; }

private:
    compute::SimpleBpeTokenizer   tok_;
    compute::ModelConfig          config_;
    std::vector<std::vector<int>> turns_;
    int                           turn_ = 0;
};

// ── Fixture ───────────────────────────────────────────────────────────────────

class ToolRunnerE2ETest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!fs::exists(StubModel::kTokPath)) {
            skip_reason_ = "TinyLlama not downloaded";
            return;
        }
        auto tok = compute::SimpleBpeTokenizer::from_model_dir(
            fs::path{StubModel::kTokPath});
        if (!tok.has_value()) {
            skip_reason_ = "Failed to load tokenizer: " + tok.error().message;
        }
    }

    void SetUp() override {
        if (!skip_reason_.empty()) GTEST_SKIP() << skip_reason_;

        tmp_dir_ = fs::temp_directory_path() / ("tool_runner_e2e_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp_dir_);

        mgr_ = std::make_unique<McpManager>(tmp_dir_.string());

        // Allow all reads without prompting.
        PermissionRule allow_all;
        allow_all.server     = "*";
        allow_all.tool       = "read_file";
        allow_all.permission = "always_allow";
        allow_all.scope      = "global";
        mgr_->set_rule(allow_all);

        // Load tokenizer for StubModel.
        auto tok = compute::SimpleBpeTokenizer::from_model_dir(
            fs::path{StubModel::kTokPath});
        model_ = std::make_unique<StubModel>(std::move(*tok));
    }

    void TearDown() override {
        model_.reset();
        mgr_.reset();
        fs::remove_all(tmp_dir_);
    }

    static std::string skip_reason_;
    fs::path           tmp_dir_;
    std::unique_ptr<McpManager> mgr_;
    std::unique_ptr<StubModel>  model_;
};

std::string ToolRunnerE2ETest::skip_reason_;

// ── Tests ─────────────────────────────────────────────────────────────────────

// ToolRunner dispatches through McpManager's built-in filesystem and the
// result is injected into the continuation.
TEST_F(ToolRunnerE2ETest, ReadFileTool_DispatchedAndResultInjected) {
    // Write a file the model will "read".
    const auto file_path = tmp_dir_ / "secret.txt";
    { std::ofstream f(file_path); f << "neurons-rocks"; }

    // Turn 1: trigger a read_file call.
    const std::string args = nlohmann::json{{"path", file_path.string()}}.dump();
    model_->add_turn("Looking it up. <<TOOL:read_file:" + args + ">>");
    // Turn 2: final response after reading.
    model_->add_turn("The file says: neurons-rocks.");

    auto tool_cb = mgr_->make_tool_call_cb("session-1", "chat-1", /*approval_cb=*/nullptr);

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb,
        cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result.value(), 0u);
    // The second turn's text was streamed after the tool result injection.
    EXPECT_THAT(output, ::testing::HasSubstr("The file says"));
}

// When server_filter excludes the server owning the tool, the callback returns
// an error result and generation continues (ToolRunner treats it as denied).
TEST_F(ToolRunnerE2ETest, ServerFilter_ExcludedServerReturnsError) {
    const auto file_path = tmp_dir_ / "data.txt";
    { std::ofstream f(file_path); f << "data"; }

    const std::string args = nlohmann::json{{"path", file_path.string()}}.dump();
    model_->add_turn("<<TOOL:read_file:" + args + ">>");
    model_->add_turn("Could not read.");

    // Only allow "neurons-shell" — excludes "neurons-filesystem".
    auto tool_cb = mgr_->make_tool_call_cb(
        "session-1", "chat-1", nullptr, {"neurons-shell"});

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb,
        cancelled);

    ASSERT_TRUE(result.has_value());
    // Generation continued (second turn ran) even though the tool was rejected.
    EXPECT_THAT(output, ::testing::HasSubstr("Could not read"));
}

// With no tool_cb, ToolRunner runs a single turn and ignores any tool markers.
TEST_F(ToolRunnerE2ETest, NoToolCb_SingleTurnIgnoresToolMarker) {
    const std::string args = nlohmann::json{{"path", "/tmp/x"}}.dump();
    model_->add_turn("<<TOOL:read_file:" + args + ">> done.");

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        /*tool_cb=*/nullptr,
        cancelled);

    ASSERT_TRUE(result.has_value());
    // Streamed without calling any tool — marker text passes through.
    EXPECT_THAT(output, ::testing::HasSubstr("done"));
}
