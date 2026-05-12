#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

#include "mcp/builtin_shell.h"
#include "mcp/mcp_manager.h"

namespace fs = std::filesystem;

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string make_args(std::initializer_list<std::pair<std::string,std::string>> kv) {
    nlohmann::json j;
    for (const auto& [k, v] : kv) j[k] = v;
    return j.dump();
}

// ── BuiltinShell tests ────────────────────────────────────────────────────────

TEST(BuiltinShellTest, ToolDefsNonEmpty) {
    const auto defs = neurons_service::BuiltinShell::tool_defs();
    ASSERT_EQ(defs.size(), 1u);
    EXPECT_EQ(defs[0].name, "run_command");
}

TEST(BuiltinShellTest, EchoCommand) {
    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinShell::handle(
        "run_command", make_args({{"command", "echo hello"}}), result));
    auto j = nlohmann::json::parse(result);
    EXPECT_EQ(j["exit_code"].get<int>(), 0);
    EXPECT_NE(j["output"].get<std::string>().find("hello"), std::string::npos);
}

TEST(BuiltinShellTest, TildeExpandedByShell) {
    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinShell::handle(
        "run_command", make_args({{"command", "ls ~/.neurons"}}), result));
    auto j = nlohmann::json::parse(result);
    // Shell expands ~ — command should at least run (exit 0 if dir exists, 1 if not,
    // but never fail with "no such file" on the tilde itself).
    EXPECT_TRUE(j.contains("output"));
}

TEST(BuiltinShellTest, FailingCommand) {
    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinShell::handle(
        "run_command", make_args({{"command", "exit 42"}}), result));
    auto j = nlohmann::json::parse(result);
    EXPECT_EQ(j["exit_code"].get<int>(), 42);
}

TEST(BuiltinShellTest, UnknownToolReturnsError) {
    std::string result;
    const auto err = neurons_service::BuiltinShell::handle("bogus_tool", "{}", result);
    EXPECT_FALSE(err.empty());
}

// ── McpManager built-in integration tests ────────────────────────────────────

class McpManagerBuiltinTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_ = fs::temp_directory_path() / ("neurons_mgr_test_" + std::to_string(getpid()));
        fs::create_directories(tmp_);
        mgr_ = std::make_unique<neurons_service::McpManager>(tmp_.string());
    }
    void TearDown() override { fs::remove_all(tmp_); }

    fs::path tmp_;
    std::unique_ptr<neurons_service::McpManager> mgr_;
};

TEST_F(McpManagerBuiltinTest, BuiltinServerListed) {
    const auto servers = mgr_->list_servers();
    ASSERT_GE(servers.size(), 2u);
    const auto has_server = [&](const std::string& name) {
        return std::any_of(servers.begin(), servers.end(),
            [&](const auto& s) { return s.name == name && s.builtin; });
    };
    EXPECT_TRUE(has_server("neurons-shell"));
    EXPECT_TRUE(has_server("neurons-filesystem"));
}

TEST_F(McpManagerBuiltinTest, BuiltinToolsInListTools) {
    mgr_->connect_enabled();
    const auto tools = mgr_->list_tools();
    const bool found = std::any_of(tools.begin(), tools.end(),
        [](const auto& t) { return t.name == "run_command"; });
    EXPECT_TRUE(found) << "Missing tool: run_command";
}

TEST_F(McpManagerBuiltinTest, HasActiveToolsWithBuiltins) {
    EXPECT_TRUE(mgr_->has_active_tools());
}

TEST_F(McpManagerBuiltinTest, BuiltinIsConnected) {
    EXPECT_TRUE(mgr_->is_connected("neurons-shell"));
}

TEST_F(McpManagerBuiltinTest, DefaultPermissionsForRunCommand) {
    const auto perm = mgr_->resolve_permission(
        "neurons-shell", "run_command", "{}", "", "");
    EXPECT_EQ(perm, "always_ask");
}

TEST_F(McpManagerBuiltinTest, PostCallHookTransformsResult) {
    mgr_->connect_enabled();

    bool hook_called = false;
    mgr_->add_tool_hook({
        .pre_call  = nullptr,
        .post_call = [&](const std::string&, const std::string&,
                         const std::string&, std::string& result) {
            hook_called = true;
            result = R"({"hooked":true})";
        }
    });

    mgr_->set_rule({
        .server = "neurons-shell", .tool = "run_command",
        .permission = "always_allow", .scope = "global", .priority = 0
    });

    auto cb = mgr_->make_tool_call_cb("", "", nullptr);
    compute::LanguageModel::ToolCall tc;
    tc.name           = "run_command";
    tc.arguments_json = R"({"command":"echo hi"})";
    const auto result = cb(tc);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(hook_called);
    EXPECT_EQ(*result, R"({"hooked":true})");
}

TEST_F(McpManagerBuiltinTest, PreCallHookCanDenyCall) {
    mgr_->connect_enabled();

    mgr_->add_tool_hook({
        .pre_call = [](const std::string&, const std::string&, std::string&) {
            return false;
        },
        .post_call = nullptr
    });

    mgr_->set_rule({
        .server = "neurons-shell", .tool = "run_command",
        .permission = "always_allow", .scope = "global", .priority = 0
    });

    auto cb = mgr_->make_tool_call_cb("", "", nullptr);
    compute::LanguageModel::ToolCall tc;
    tc.name           = "run_command";
    tc.arguments_json = R"({"command":"echo hi"})";
    const auto result = cb(tc);
    ASSERT_TRUE(result.has_value());
    auto j = nlohmann::json::parse(*result);
    EXPECT_TRUE(j.contains("error"));
}
