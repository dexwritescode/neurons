#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include "compute/model/chat_template.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <string>
#include <vector>

#include "mcp/mcp_manager.h"
#include "mcp/mcp_types.h"
#include "compute/model/tool_runner.h"
#include "compute/model/language_model.h"
#include "compute/model/simple_bpe_tokenizer.h"
#include "compute/model/model_config.h"
#include "compute/core/compute_backend.h"
#include "compute/core/compute_types.h"

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
        // Populate tool_to_server_ for the built-in servers.
        mgr_->connect_enabled();

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
    EXPECT_GT(result->gen_tokens, 0u);
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

// write_file defaults to always_ask. When the approval callback returns true the
// tool runs and the file is created.
TEST_F(ToolRunnerE2ETest, AlwaysAsk_ApprovalApproved_ToolExecuted) {
    const auto out_file = tmp_dir_ / "approved_output.txt";
    const std::string args = nlohmann::json{
        {"path", out_file.string()}, {"content", "approved"}}.dump();
    model_->add_turn("Writing now. <<TOOL:write_file:" + args + ">>");
    model_->add_turn("File written successfully.");

    std::string approved_tool;
    ApprovalCb approval_cb = [&](const ToolApprovalRequest& req) -> std::future<bool> {
        approved_tool = req.tool;
        std::promise<bool> p;
        p.set_value(true);
        return p.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("session-1", "chat-1", approval_cb);

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(approved_tool, "write_file");
    EXPECT_TRUE(std::filesystem::exists(out_file)) << "File should exist after approved write";
    EXPECT_THAT(output, ::testing::HasSubstr("File written successfully"));
}

// When the approval callback returns false the tool does not run and the error
// result is injected — generation continues from the next turn.
TEST_F(ToolRunnerE2ETest, AlwaysAsk_ApprovalDenied_ErrorInjectedAndToolNotRun) {
    const auto out_file = tmp_dir_ / "denied_output.txt";
    const std::string args = nlohmann::json{
        {"path", out_file.string()}, {"content", "should not appear"}}.dump();
    model_->add_turn("Writing now. <<TOOL:write_file:" + args + ">>");
    model_->add_turn("Could not write the file.");

    bool was_asked = false;
    ApprovalCb approval_cb = [&](const ToolApprovalRequest&) -> std::future<bool> {
        was_asked = true;
        std::promise<bool> p;
        p.set_value(false);
        return p.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("session-1", "chat-1", approval_cb);

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(was_asked);
    EXPECT_FALSE(std::filesystem::exists(out_file)) << "File must not be created when denied";
    EXPECT_THAT(output, ::testing::HasSubstr("Could not write"));
}

// allow_shell_fallback=true: unknown tool routes to neurons-shell run_command.
// Approval is requested and the command executes when approved.
TEST_F(ToolRunnerE2ETest, UnknownTool_FallsBackToShell_WhenApproved) {
    const std::string args = nlohmann::json{{"message", "hello-world"}}.dump();
    model_->add_turn("Running it. <<TOOL:echo_test:" + args + ">>");
    model_->add_turn("Command ran.");

    bool was_asked = false;
    ApprovalCb approval_cb = [&](const ToolApprovalRequest& req) -> std::future<bool> {
        was_asked = true;
        EXPECT_EQ(req.server, "neurons-shell");
        EXPECT_EQ(req.tool,   "echo_test");  // shows original tool name in UI
        EXPECT_TRUE(req.destructive);
        std::promise<bool> p;
        p.set_value(true);
        return p.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb(
        "session-1", "chat-1", approval_cb, {}, /*allow_shell_fallback=*/true);

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(was_asked) << "Approval should be requested for shell-fallback tool";
    EXPECT_THAT(output, ::testing::HasSubstr("Command ran"));
}

// allow_shell_fallback=false (default): unknown tool returns error, not shell execution.
TEST_F(ToolRunnerE2ETest, UnknownTool_NoFallback_WhenFlagOff) {
    const std::string args = nlohmann::json{{"message", "hello"}}.dump();
    model_->add_turn("<<TOOL:unknown_tool:" + args + ">>");
    model_->add_turn("Could not run.");

    auto tool_cb = mgr_->make_tool_call_cb("session-1", "chat-1");  // fallback off by default

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(output, ::testing::HasSubstr("Could not run"));
}

// allow_shell_fallback=true but no approval_cb: always_ask is auto-denied.
TEST_F(ToolRunnerE2ETest, UnknownTool_FallsBackToShell_DeniedWithoutApprovalCb) {
    const std::string args = nlohmann::json{{"message", "hello"}}.dump();
    model_->add_turn("<<TOOL:unknown_tool:" + args + ">>");
    model_->add_turn("Could not run.");

    auto tool_cb = mgr_->make_tool_call_cb(
        "session-1", "chat-1", /*approval_cb=*/nullptr, {}, /*allow_shell_fallback=*/true);

    std::string output;
    std::atomic<bool> cancelled{false};

    auto result = compute::ToolRunner{}.run(
        *model_, {1}, 512, {},
        [&](const std::string& d) { output += d; return true; },
        tool_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(output, ::testing::HasSubstr("Could not run"));
}

// Live inference: a real Qwen3 model generates a <tool_call>, McpManager
// dispatches it to the built-in filesystem server, and the result is injected
// before the model produces its final answer. Skipped if the model is absent.
TEST(ToolRunnerLiveTest, Qwen3_GeneratesToolCallAndContinues) {
#if !defined(MLX_BACKEND_ENABLED)
    GTEST_SKIP() << "MLX backend not compiled";
#else
    const auto model_path = std::filesystem::path(std::getenv("HOME"))
        / ".neurons/models/mlx-community/Qwen3-8B-4bit";
    if (!std::filesystem::exists(model_path))
        GTEST_SKIP() << "Qwen3-8B-4bit not downloaded";

    // Temp dir + secret file the model will read.
    const auto tmp = std::filesystem::temp_directory_path() / "tool_runner_live_test";
    std::filesystem::create_directories(tmp);
    const auto secret = tmp / "secret.txt";
    { std::ofstream f(secret); f << "neurons-live-42"; }

    auto backend_res = compute::BackendFactory::create(compute::BackendType::MLX);
    ASSERT_TRUE(backend_res.has_value());
    auto backend = std::move(*backend_res);
    ASSERT_TRUE(backend->initialize().has_value());

    auto model_res = compute::LanguageModel::load(model_path, backend.get());
    ASSERT_TRUE(model_res.has_value()) << model_res.error().message;
    auto model = std::move(*model_res);

    McpManager mgr(tmp.string());
    mgr.connect_enabled();  // populate tool_to_server_ for built-ins

    const std::string tools_json = mgr.tools_json();
    const std::string sys_prompt = model->format_tool_system_prompt(tools_json);
    const std::string user_msg   =
        "Read the file at " + secret.string() + " and tell me what it contains.";

    const std::string chat_text = compute::apply_chat_template(
        model->tokenizer(), sys_prompt, {{"user", user_msg}});
    const auto token_ids = model->tokenizer().encode(
        chat_text, /*add_special_tokens=*/false);

    auto tool_cb = mgr.make_tool_call_cb(
        "live-session", "live-chat", /*approval_cb=*/nullptr);

    int tool_calls = 0;
    compute::ToolCallCb counted_cb =
        [&](const compute::LanguageModel::ToolCall& call) -> std::optional<std::string> {
            ++tool_calls;
            return tool_cb(call);
        };

    std::string output;
    std::atomic<bool> cancelled{false};
    compute::SamplingParams params;
    params.temperature = 0.0f;  // greedy — deterministic

    auto result = compute::ToolRunner{}.run(
        *model, token_ids, 512, params,
        [&](const std::string& d) { output += d; return true; },
        counted_cb, cancelled);

    ASSERT_TRUE(result.has_value());
    EXPECT_GE(tool_calls, 1) << "Model should have called at least one tool.\nOutput: " << output;
    EXPECT_THAT(output, ::testing::HasSubstr("neurons-live-42"));

    backend->cleanup();
    std::filesystem::remove_all(tmp);
#endif
}
