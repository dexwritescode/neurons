#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>

#include "mcp/builtin_filesystem.h"
#include "mcp/builtin_shell.h"
#include "mcp/mcp_manager.h"

namespace fs = std::filesystem;

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string make_args(std::initializer_list<std::pair<std::string,std::string>> kv) {
    nlohmann::json j;
    for (const auto& [k, v] : kv) j[k] = v;
    return j.dump();
}

// ── BuiltinFilesystem tests ───────────────────────────────────────────────────

class BuiltinFilesystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_ = fs::temp_directory_path() / ("neurons_fs_test_" + std::to_string(getpid()));
        fs::create_directories(tmp_);
    }
    void TearDown() override {
        fs::remove_all(tmp_);
    }
    fs::path tmp_;
};

TEST_F(BuiltinFilesystemTest, ToolDefsNonEmpty) {
    const auto defs = neurons_service::BuiltinFilesystem::tool_defs();
    ASSERT_EQ(defs.size(), 4u);
    const std::vector<std::string> expected = {"read_file","write_file","list_dir","search_files"};
    for (size_t i = 0; i < defs.size(); ++i)
        EXPECT_EQ(defs[i].name, expected[i]);
}

TEST_F(BuiltinFilesystemTest, WriteAndReadFile) {
    const std::string path = (tmp_ / "hello.txt").string();
    const std::string content = "Hello, Neurons!\n";

    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "write_file", make_args({{"path", path}, {"content", content}}), result));
    auto wj = nlohmann::json::parse(result);
    EXPECT_EQ(wj["written"].get<size_t>(), content.size());

    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "read_file", make_args({{"path", path}}), result));
    auto rj = nlohmann::json::parse(result);
    EXPECT_EQ(rj["content"].get<std::string>(), content);
}

TEST_F(BuiltinFilesystemTest, ReadFileNotFound) {
    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "read_file", make_args({{"path", "/nonexistent/file.txt"}}), result));
    auto j = nlohmann::json::parse(result);
    EXPECT_TRUE(j.contains("error"));
}

TEST_F(BuiltinFilesystemTest, ListDir) {
    // Create a few files.
    for (const auto& name : {"a.txt", "b.cpp", "c.md"}) {
        std::ofstream(tmp_ / name) << name;
    }

    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "list_dir", make_args({{"path", tmp_.string()}}), result));
    auto j = nlohmann::json::parse(result);
    ASSERT_TRUE(j.contains("entries"));
    EXPECT_EQ(j["entries"].size(), 3u);
}

TEST_F(BuiltinFilesystemTest, SearchFiles) {
    fs::create_directories(tmp_ / "sub");
    std::ofstream(tmp_ / "foo.cpp") << "x";
    std::ofstream(tmp_ / "bar.hpp") << "x";
    std::ofstream(tmp_ / "sub" / "baz.cpp") << "x";

    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "search_files",
        make_args({{"pattern", "*.cpp"}, {"root", tmp_.string()}}),
        result));
    auto j = nlohmann::json::parse(result);
    ASSERT_TRUE(j.contains("matches"));
    EXPECT_EQ(j["matches"].size(), 2u);
}

TEST_F(BuiltinFilesystemTest, WriteCreatesParentDirs) {
    const std::string path = (tmp_ / "nested" / "dir" / "file.txt").string();
    std::string result;
    ASSERT_EQ("", neurons_service::BuiltinFilesystem::handle(
        "write_file", make_args({{"path", path}, {"content", "data"}}), result));
    EXPECT_TRUE(fs::exists(path));
}

TEST_F(BuiltinFilesystemTest, UnknownToolReturnsError) {
    std::string result;
    const auto err = neurons_service::BuiltinFilesystem::handle("bogus_tool", "{}", result);
    EXPECT_FALSE(err.empty());
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

TEST_F(McpManagerBuiltinTest, BuiltinServersListedFirst) {
    const auto servers = mgr_->list_servers();
    ASSERT_GE(servers.size(), 2u);
    EXPECT_EQ(servers[0].name, "neurons-filesystem");
    EXPECT_EQ(servers[1].name, "neurons-shell");
    EXPECT_TRUE(servers[0].builtin);
    EXPECT_TRUE(servers[1].builtin);
}

TEST_F(McpManagerBuiltinTest, BuiltinToolsInListTools) {
    mgr_->connect_enabled();
    const auto tools = mgr_->list_tools();
    const std::vector<std::string> expected =
        {"read_file", "write_file", "list_dir", "search_files", "run_command"};
    for (const auto& name : expected) {
        const bool found = std::any_of(tools.begin(), tools.end(),
            [&](const auto& t) { return t.name == name; });
        EXPECT_TRUE(found) << "Missing tool: " << name;
    }
}

TEST_F(McpManagerBuiltinTest, HasActiveToolsWithBuiltins) {
    EXPECT_TRUE(mgr_->has_active_tools());
}

TEST_F(McpManagerBuiltinTest, BuiltinIsConnected) {
    EXPECT_TRUE(mgr_->is_connected("neurons-filesystem"));
    EXPECT_TRUE(mgr_->is_connected("neurons-shell"));
}

TEST_F(McpManagerBuiltinTest, DefaultPermissionsForReadTools) {
    const auto perm = mgr_->resolve_permission(
        "neurons-filesystem", "read_file", "{}", "", "");
    EXPECT_EQ(perm, "allow_session");
}

TEST_F(McpManagerBuiltinTest, DefaultPermissionsForWriteTools) {
    const auto perm = mgr_->resolve_permission(
        "neurons-filesystem", "write_file", "{}", "", "");
    EXPECT_EQ(perm, "always_ask");
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

    // Dispatch via make_tool_call_cb with always_allow permission.
    mgr_->set_rule({
        .server = "neurons-filesystem", .tool = "read_file",
        .permission = "always_allow", .scope = "global", .priority = 0
    });

    auto cb = mgr_->make_tool_call_cb("", "", nullptr);
    compute::LanguageModel::ToolCall tc;
    tc.name          = "read_file";
    tc.arguments_json = R"({"path":"/nonexistent"})";
    const auto result = cb(tc);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(hook_called);
    EXPECT_EQ(*result, R"({"hooked":true})");
}

TEST_F(McpManagerBuiltinTest, PreCallHookCanDenyCall) {
    mgr_->connect_enabled();

    mgr_->add_tool_hook({
        .pre_call = [](const std::string&, const std::string&, std::string&) {
            return false;  // deny everything
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
