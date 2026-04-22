#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <unistd.h>

#include "mcp/mcp_manager.h"
#include "mcp/mcp_types.h"
#include "compute/model/language_model.h"

namespace fs = std::filesystem;
using namespace neurons_service;

// ── Fixture ───────────────────────────────────────────────────────────────────

class McpManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / ("mcp_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp_dir_);
        mgr_ = std::make_unique<McpManager>(tmp_dir_.string());
    }

    void TearDown() override {
        mgr_.reset();
        fs::remove_all(tmp_dir_);
    }

    McpServerConfig make_server(const std::string& name, bool enabled = true) {
        McpServerConfig cfg;
        cfg.name    = name;
        cfg.enabled = enabled;
        cfg.command = "/usr/bin/echo";
        return cfg;
    }

    PermissionRule make_rule(const std::string& server,
                             const std::string& tool,
                             const std::string& permission,
                             const std::string& scope    = "global",
                             int                priority = 0,
                             const std::string& args     = "") {
        PermissionRule r;
        r.server          = server;
        r.tool            = tool;
        r.permission      = permission;
        r.scope           = scope;
        r.priority        = priority;
        r.arg_constraints = args;
        return r;
    }

    fs::path                  tmp_dir_;
    std::unique_ptr<McpManager> mgr_;
};

// ── resolve_permission ────────────────────────────────────────────────────────

TEST_F(McpManagerTest, DefaultPermissionIsAlwaysAsk) {
    // No rules configured → default is always_ask
    EXPECT_EQ(mgr_->resolve_permission("my_server", "read_file", "{}", "", ""),
              "always_ask");
}

TEST_F(McpManagerTest, GlobalRuleMatchReturnsItsPermission) {
    mgr_->set_rule(make_rule("my_server", "read_file", "always_allow"));
    EXPECT_EQ(mgr_->resolve_permission("my_server", "read_file", "{}", "", ""),
              "always_allow");
}

TEST_F(McpManagerTest, GlobWildcardMatchesAnyServer) {
    mgr_->set_rule(make_rule("*", "*", "always_deny"));
    EXPECT_EQ(mgr_->resolve_permission("anything", "any_tool", "{}", "", ""),
              "always_deny");
}

TEST_F(McpManagerTest, GlobalRuleDoesNotMatchDifferentTool) {
    mgr_->set_rule(make_rule("my_server", "write_file", "always_deny"));
    EXPECT_EQ(mgr_->resolve_permission("my_server", "read_file", "{}", "", ""),
              "always_ask");  // falls through to default
}

TEST_F(McpManagerTest, SessionScopeWinsOverGlobal) {
    mgr_->set_rule(make_rule("srv", "tool", "always_deny",  "global"));
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "session:sess1"));
    EXPECT_EQ(mgr_->resolve_permission("srv", "tool", "{}", "sess1", ""),
              "always_allow");
}

TEST_F(McpManagerTest, ChatScopeWinsOverGlobal) {
    mgr_->set_rule(make_rule("srv", "tool", "always_deny",  "global"));
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "chat:chat1"));
    EXPECT_EQ(mgr_->resolve_permission("srv", "tool", "{}", "", "chat1"),
              "always_allow");
}

TEST_F(McpManagerTest, SessionScopeWinsOverChat) {
    mgr_->set_rule(make_rule("srv", "tool", "always_deny",  "chat:chat1"));
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "session:sess1"));
    EXPECT_EQ(mgr_->resolve_permission("srv", "tool", "{}", "sess1", "chat1"),
              "always_allow");
}

TEST_F(McpManagerTest, SessionScopeNotAppliedToOtherSession) {
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "session:sess1"));
    mgr_->set_rule(make_rule("srv", "tool", "always_deny",  "global"));
    EXPECT_EQ(mgr_->resolve_permission("srv", "tool", "{}", "sess2", ""),
              "always_deny");  // session:sess1 doesn't match sess2
}

TEST_F(McpManagerTest, PriorityOrderingWithinGlobal) {
    // priority 5 = "always_deny", priority 1 = "always_allow"  → priority 1 wins
    mgr_->set_rule(make_rule("srv", "tool", "always_deny",  "global", 5));
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "global", 1));
    EXPECT_EQ(mgr_->resolve_permission("srv", "tool", "{}", "", ""),
              "always_allow");
}

TEST_F(McpManagerTest, ArgConstraintMatchGrantsPermission) {
    PermissionRule r = make_rule("fs", "read", "always_allow", "global", 0,
                                 R"({"path":"/safe/*"})");
    mgr_->set_rule(r);
    EXPECT_EQ(mgr_->resolve_permission("fs", "read",
                                       R"({"path":"/safe/readme.md"})", "", ""),
              "always_allow");
}

TEST_F(McpManagerTest, ArgConstraintMismatchFallsThrough) {
    PermissionRule r = make_rule("fs", "read", "always_allow", "global", 0,
                                 R"({"path":"/safe/*"})");
    mgr_->set_rule(r);
    // path doesn't match the constraint → rule skipped → default always_ask
    EXPECT_EQ(mgr_->resolve_permission("fs", "read",
                                       R"({"path":"/etc/passwd"})", "", ""),
              "always_ask");
}

// ── match_constraints (tested via resolve_permission) ────────────────────────

TEST_F(McpManagerTest, EmptyConstraintsMatchAnyArgs) {
    mgr_->set_rule(make_rule("s", "t", "always_allow", "global", 0, ""));
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"any":"value"})", "", ""),
              "always_allow");
}

TEST_F(McpManagerTest, ConstraintMissingKeyDeniesMatch) {
    PermissionRule r = make_rule("s", "t", "always_allow", "global", 0,
                                 R"({"required_key":"*"})");
    mgr_->set_rule(r);
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"other":"val"})", "", ""),
              "always_ask");
}

TEST_F(McpManagerTest, GlobPatternMatchesPrefix) {
    // "ls*" matches any string starting with "ls" (space is not a path separator)
    PermissionRule r = make_rule("s", "t", "always_allow", "global", 0,
                                 R"({"cmd":"ls*"})");
    mgr_->set_rule(r);
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"cmd":"ls -la"})", "", ""),
              "always_allow");
    // A different prefix does not match
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"cmd":"rm -rf /"})", "", ""),
              "always_ask");
}

TEST_F(McpManagerTest, GlobPatternFNMPathnamePreventsSlashCrossing) {
    // FNM_PATHNAME: "*" in pattern does not match "/" in value.
    // "/safe/*" allows files directly under /safe/ but not in subdirectories.
    PermissionRule r = make_rule("s", "t", "always_allow", "global", 0,
                                 R"({"path":"/safe/*"})");
    mgr_->set_rule(r);
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"path":"/safe/file.txt"})", "", ""),
              "always_allow");
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"path":"/safe/sub/file.txt"})", "", ""),
              "always_ask");  // "*" won't cross the "/" between sub and file
}

TEST_F(McpManagerTest, ExactArgConstraintMatch) {
    PermissionRule r = make_rule("s", "t", "always_allow", "global", 0,
                                 R"({"action":"read"})");
    mgr_->set_rule(r);
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"action":"read"})", "", ""),
              "always_allow");
    EXPECT_EQ(mgr_->resolve_permission("s", "t", R"({"action":"write"})", "", ""),
              "always_ask");
}

// ── Rule CRUD ─────────────────────────────────────────────────────────────────

TEST_F(McpManagerTest, SetRuleAddsNewRule) {
    EXPECT_TRUE(mgr_->list_rules().empty());
    mgr_->set_rule(make_rule("s", "t", "always_allow"));
    EXPECT_EQ(mgr_->list_rules().size(), 1u);
}

TEST_F(McpManagerTest, SetRuleReplacesExistingWithSameKey) {
    mgr_->set_rule(make_rule("s", "t", "always_allow", "global"));
    mgr_->set_rule(make_rule("s", "t", "always_deny",  "global"));  // same key
    const auto rules = mgr_->list_rules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules[0].permission, "always_deny");
}

TEST_F(McpManagerTest, SetRuleDoesNotReplaceRuleWithDifferentScope) {
    mgr_->set_rule(make_rule("s", "t", "always_allow", "global"));
    mgr_->set_rule(make_rule("s", "t", "always_deny",  "session:x"));
    EXPECT_EQ(mgr_->list_rules().size(), 2u);
}

TEST_F(McpManagerTest, DeleteRuleRemovesMatchingRule) {
    mgr_->set_rule(make_rule("s", "t", "always_allow", "global"));
    EXPECT_EQ(mgr_->list_rules().size(), 1u);
    mgr_->delete_rule("s", "t", "global");
    EXPECT_TRUE(mgr_->list_rules().empty());
}

TEST_F(McpManagerTest, DeleteRuleNoOpIfNotFound) {
    mgr_->set_rule(make_rule("s", "t", "always_allow"));
    mgr_->delete_rule("s", "other_tool", "global");
    EXPECT_EQ(mgr_->list_rules().size(), 1u);  // untouched
}

TEST_F(McpManagerTest, ListRulesScopeFilter) {
    mgr_->set_rule(make_rule("s", "t", "always_allow", "global"));
    mgr_->set_rule(make_rule("s", "t", "always_deny",  "session:x"));
    EXPECT_EQ(mgr_->list_rules("global").size(),    1u);
    EXPECT_EQ(mgr_->list_rules("session:x").size(), 1u);
    EXPECT_EQ(mgr_->list_rules().size(),            2u);
}

TEST_F(McpManagerTest, RulesAreSortedByPriorityAfterSetRule) {
    mgr_->set_rule(make_rule("s", "t1", "always_allow", "global", 10));
    mgr_->set_rule(make_rule("s", "t2", "always_deny",  "global", 2));
    mgr_->set_rule(make_rule("s", "t3", "always_ask",   "global", 5));
    const auto rules = mgr_->list_rules("global");
    ASSERT_EQ(rules.size(), 3u);
    EXPECT_EQ(rules[0].priority, 2);
    EXPECT_EQ(rules[1].priority, 5);
    EXPECT_EQ(rules[2].priority, 10);
}

// ── Server config CRUD ────────────────────────────────────────────────────────

TEST_F(McpManagerTest, AddServerAppendsToList) {
    EXPECT_TRUE(mgr_->list_servers().empty());
    mgr_->add_server(make_server("my_server"));
    EXPECT_EQ(mgr_->list_servers().size(), 1u);
    EXPECT_EQ(mgr_->list_servers()[0].name, "my_server");
}

TEST_F(McpManagerTest, AddServerUpdatesExistingByName) {
    mgr_->add_server(make_server("srv"));
    McpServerConfig updated = make_server("srv");
    updated.enabled = false;
    mgr_->add_server(updated);
    const auto servers = mgr_->list_servers();
    ASSERT_EQ(servers.size(), 1u);
    EXPECT_FALSE(servers[0].enabled);
}

TEST_F(McpManagerTest, RemoveServerReturnsTrueOnSuccess) {
    mgr_->add_server(make_server("srv"));
    EXPECT_TRUE(mgr_->remove_server("srv"));
    EXPECT_TRUE(mgr_->list_servers().empty());
}

TEST_F(McpManagerTest, RemoveServerReturnsFalseIfNotFound) {
    EXPECT_FALSE(mgr_->remove_server("nonexistent"));
}

TEST_F(McpManagerTest, PushServersReplacesEntireList) {
    mgr_->add_server(make_server("old1"));
    mgr_->add_server(make_server("old2"));
    mgr_->push_servers({make_server("new1")});
    const auto servers = mgr_->list_servers();
    ASSERT_EQ(servers.size(), 1u);
    EXPECT_EQ(servers[0].name, "new1");
}

TEST_F(McpManagerTest, PushServersWithEmptyListClearsAll) {
    mgr_->add_server(make_server("srv"));
    mgr_->push_servers({});
    EXPECT_TRUE(mgr_->list_servers().empty());
}

// ── Persistence ───────────────────────────────────────────────────────────────

TEST_F(McpManagerTest, SaveAndLoadConfigRoundTrip) {
    McpServerConfig cfg = make_server("test_srv");
    cfg.args    = {"--port", "9000"};
    cfg.env     = {{"FOO", "bar"}};
    cfg.enabled = false;
    mgr_->add_server(cfg);
    mgr_->save_config();

    // Fresh manager loading from the same dir
    McpManager mgr2(tmp_dir_.string());
    mgr2.load_config();
    const auto servers = mgr2.list_servers();
    ASSERT_EQ(servers.size(), 1u);
    EXPECT_EQ(servers[0].name,    "test_srv");
    EXPECT_FALSE(servers[0].enabled);
    EXPECT_EQ(servers[0].args,    (std::vector<std::string>{"--port", "9000"}));
    EXPECT_EQ(servers[0].env.at("FOO"), "bar");
}

TEST_F(McpManagerTest, SaveAndLoadPermissionsRoundTrip) {
    mgr_->set_rule(make_rule("srv", "tool", "always_allow", "global"));
    // Session-scoped rule must NOT be persisted
    mgr_->set_rule(make_rule("srv", "tool2", "always_deny", "session:x"));
    mgr_->save_permissions();

    McpManager mgr2(tmp_dir_.string());
    mgr2.load_permissions();
    const auto rules = mgr2.list_rules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules[0].tool,       "tool");
    EXPECT_EQ(rules[0].permission, "always_allow");
    EXPECT_EQ(rules[0].scope,      "global");
}

TEST_F(McpManagerTest, LoadPermissionsPreservesInMemorySessionRules) {
    // Session rules loaded before load_permissions() must survive the reload
    mgr_->set_rule(make_rule("s", "t", "always_deny", "session:abc"));
    mgr_->save_permissions();   // nothing to persist (no global rules)
    mgr_->load_permissions();   // should not wipe session:abc
    ASSERT_EQ(mgr_->list_rules("session:abc").size(), 1u);
}

TEST_F(McpManagerTest, LoadConfigOnMissingFileIsNoop) {
    // No file written — load should silently succeed
    McpManager fresh(tmp_dir_.string());
    EXPECT_NO_THROW(fresh.load_config());
    EXPECT_TRUE(fresh.list_servers().empty());
}

TEST_F(McpManagerTest, LoadPermissionsOnMissingFileIsNoop) {
    McpManager fresh(tmp_dir_.string());
    EXPECT_NO_THROW(fresh.load_permissions());
    EXPECT_TRUE(fresh.list_rules().empty());
}

// ── make_tool_call_cb permission gates (no MCP subprocess required) ───────────

static compute::LanguageModel::ToolCall fake_call(const std::string& name,
                                                   const std::string& args = "{}") {
    compute::LanguageModel::ToolCall c;
    c.name           = name;
    c.arguments_json = args;
    return c;
}

// No connected client → tool not in tool_to_server_ → always_deny fires before
// the server lookup (production bug was fixed to ensure this order).
TEST_F(McpManagerTest, AlwaysDenyReturnsNulloptWithoutCallingCallback) {
    mgr_->set_rule(make_rule("*", "deny_tool", "always_deny", "global"));

    bool cb_called = false;
    ApprovalCb cb = [&](const ToolApprovalRequest&) -> std::future<bool> {
        cb_called = true;
        std::promise<bool> p; p.set_value(true); return p.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("", "", cb);
    EXPECT_FALSE(tool_cb(fake_call("deny_tool")).has_value());
    EXPECT_FALSE(cb_called);
}

// No connected client, always_allow: bypasses callback, returns error JSON
// because no client is connected (correct — can't execute with no server).
TEST_F(McpManagerTest, AlwaysAllowReturnsErrorJsonWhenNoClientWithoutCallingCallback) {
    mgr_->set_rule(make_rule("*", "allow_tool", "always_allow", "global"));

    bool cb_called = false;
    ApprovalCb cb = [&](const ToolApprovalRequest&) -> std::future<bool> {
        cb_called = true;
        std::promise<bool> p; p.set_value(true); return p.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("", "", cb);
    const auto result = tool_cb(fake_call("allow_tool"));

    EXPECT_FALSE(cb_called);
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("error"), std::string::npos);
}

// ── Approval flow integration test (real MCP subprocess) ─────────────────────
// Requires Python 3 — skipped automatically if python3 is not on PATH.
// Spawns the echo MCP server, connects to it, exercises the full
// always_ask promise/future handshake including the deny and approve paths.

static std::string find_python3() {
    const char* paths[] = {"/usr/bin/python3", "/usr/local/bin/python3",
                           "/opt/homebrew/bin/python3", nullptr};
    for (int i = 0; paths[i]; ++i)
        if (access(paths[i], X_OK) == 0) return paths[i];
    return "";
}

// Path to the echo server script relative to the source tree.
// CMake sets MCTS_FIXTURES_DIR via a compile definition (added in CMakeLists.txt).
#ifndef MCTS_FIXTURES_DIR
#  define MCTS_FIXTURES_DIR "."
#endif

class McpApprovalIntegrationTest : public McpManagerTest {
protected:
    void SetUp() override {
        McpManagerTest::SetUp();
        python3_ = find_python3();
        server_script_ = std::string(MCTS_FIXTURES_DIR) + "/mcp_echo_server.py";
    }
    std::string python3_;
    std::string server_script_;
};

TEST_F(McpApprovalIntegrationTest, AlwaysAskDeniedReturnsNullopt) {
    if (python3_.empty()) GTEST_SKIP() << "python3 not found";
    if (!fs::exists(server_script_)) GTEST_SKIP() << "echo server script not found";

    McpServerConfig cfg;
    cfg.name    = "echo";
    cfg.command = python3_;
    cfg.args    = {server_script_};
    mgr_->add_server(cfg);

    const std::string err = mgr_->connect_server("echo");
    ASSERT_TRUE(err.empty()) << "connect failed: " << err;

    // The server exposes one tool: "echo". Set it to always_ask.
    mgr_->set_rule(make_rule("echo", "echo", "always_ask", "global"));

    std::promise<void> cb_fired;
    std::promise<bool> decision;
    std::string        captured_id;

    ApprovalCb approval_cb = [&](const ToolApprovalRequest& req) -> std::future<bool> {
        captured_id = req.approval_id;
        EXPECT_EQ(req.server, "echo");
        EXPECT_EQ(req.tool,   "echo");
        cb_fired.set_value();
        return decision.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("sess", "", approval_cb);

    std::optional<std::string> result;
    std::thread worker([&] {
        result = tool_cb(fake_call("echo", R"({"text":"hello"})"));
    });

    cb_fired.get_future().wait();
    EXPECT_FALSE(captured_id.empty());

    decision.set_value(false);  // deny
    worker.join();

    EXPECT_FALSE(result.has_value());  // denied → nullopt
}

TEST_F(McpApprovalIntegrationTest, AlwaysAskApprovedExecutesTool) {
    if (python3_.empty()) GTEST_SKIP() << "python3 not found";
    if (!fs::exists(server_script_)) GTEST_SKIP() << "echo server script not found";

    McpServerConfig cfg;
    cfg.name    = "echo";
    cfg.command = python3_;
    cfg.args    = {server_script_};
    mgr_->add_server(cfg);

    ASSERT_TRUE(mgr_->connect_server("echo").empty());

    mgr_->set_rule(make_rule("echo", "echo", "always_ask", "global"));

    std::promise<void> cb_fired;
    std::promise<bool> decision;

    ApprovalCb approval_cb = [&](const ToolApprovalRequest&) -> std::future<bool> {
        cb_fired.set_value();
        return decision.get_future();
    };

    auto tool_cb = mgr_->make_tool_call_cb("sess", "", approval_cb);

    std::optional<std::string> result;
    std::thread worker([&] {
        result = tool_cb(fake_call("echo", R"({"text":"hello from test"})"));
    });

    cb_fired.get_future().wait();
    decision.set_value(true);  // approve
    worker.join();

    // Approved → tool executes → echo server returns the text argument.
    ASSERT_TRUE(result.has_value());
    EXPECT_NE(result->find("hello from test"), std::string::npos);
}
