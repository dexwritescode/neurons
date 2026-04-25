#include <gtest/gtest.h>
#include <filesystem>
#include <future>
#include <thread>

#include "neurons_service.h"
#include "mcp/mcp_types.h"

// Generated proto headers
#include "neurons.pb.h"
#include "neurons.grpc.pb.h"

namespace fs = std::filesystem;
using namespace neurons_service;

// ── Fixture ───────────────────────────────────────────────────────────────────

class McpRpcTest : public ::testing::Test {
protected:
    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / ("mcp_rpc_test_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp_dir_);

        // NeuronsServiceImpl accepts a null backend — MCP handlers never touch it.
        svc_ = std::make_unique<NeuronsServiceImpl>(
            tmp_dir_.string(), nullptr);
    }

    void TearDown() override {
        svc_.reset();
        fs::remove_all(tmp_dir_);
    }

    // Build a proto McpServerConfig with required fields
    neurons::McpServerConfig make_proto_server(const std::string& name,
                                               bool enabled = true) {
        neurons::McpServerConfig s;
        s.set_name(name);
        s.set_transport("stdio");
        s.set_command("/usr/bin/echo");
        s.set_enabled(enabled);
        return s;
    }

    neurons::PermissionRule make_proto_rule(const std::string& server,
                                            const std::string& tool,
                                            const std::string& permission,
                                            const std::string& scope = "global",
                                            int priority = 0) {
        neurons::PermissionRule r;
        r.set_server(server);
        r.set_tool(tool);
        r.set_permission(permission);
        r.set_scope(scope);
        r.set_priority(priority);
        return r;
    }

    // Count only user-configured servers (excluding built-ins).
    int user_server_count(const neurons::ListMcpServersResponse& resp) {
        int n = 0;
        for (int i = 0; i < resp.servers_size(); ++i) {
            const auto& name = resp.servers(i).name();
            if (name != "neurons-filesystem" && name != "neurons-shell") ++n;
        }
        return n;
    }

    fs::path tmp_dir_;
    std::unique_ptr<NeuronsServiceImpl> svc_;
};

// ── ListMcpServers ────────────────────────────────────────────────────────────

TEST_F(McpRpcTest, ListMcpServersInitiallyEmpty) {
    neurons::ListMcpServersRequest req;
    neurons::ListMcpServersResponse resp;
    const auto status = svc_->ListMcpServers(nullptr, &req, &resp);
    EXPECT_TRUE(status.ok());
    // Built-in servers are always present; no user-configured servers initially.
    EXPECT_EQ(user_server_count(resp), 0);
}

TEST_F(McpRpcTest, ListMcpServersReturnsConfiguredServers) {
    // Add two servers directly via McpManager, then list via RPC
    svc_->mcp_manager().add_server([]{
        McpServerConfig c;
        c.name = "srv1"; c.command = "/bin/a"; return c;
    }());
    svc_->mcp_manager().add_server([]{
        McpServerConfig c;
        c.name = "srv2"; c.command = "/bin/b"; return c;
    }());

    neurons::ListMcpServersRequest req;
    neurons::ListMcpServersResponse resp;
    ASSERT_TRUE(svc_->ListMcpServers(nullptr, &req, &resp).ok());
    EXPECT_EQ(user_server_count(resp), 2);
}

// ── AddMcpServer ─────────────────────────────────────────────────────────────

TEST_F(McpRpcTest, AddMcpServerSuccess) {
    neurons::AddMcpServerRequest req;
    *req.mutable_server() = make_proto_server("my_srv");
    neurons::AddMcpServerResponse resp;
    ASSERT_TRUE(svc_->AddMcpServer(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
    EXPECT_TRUE(resp.error().empty());

    // Verify it's now in the list
    neurons::ListMcpServersRequest list_req;
    neurons::ListMcpServersResponse list_resp;
    svc_->ListMcpServers(nullptr, &list_req, &list_resp);
    EXPECT_EQ(user_server_count(list_resp), 1);
    // Find the user server by name (builtins are also in the list)
    bool found = false;
    for (int i = 0; i < list_resp.servers_size(); ++i)
        if (list_resp.servers(i).name() == "my_srv") { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(McpRpcTest, AddMcpServerRejectsEmptyName) {
    neurons::AddMcpServerRequest req;
    neurons::McpServerConfig s;
    s.set_name("");  // empty name
    *req.mutable_server() = s;
    neurons::AddMcpServerResponse resp;
    ASSERT_TRUE(svc_->AddMcpServer(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
    EXPECT_FALSE(resp.error().empty());
}

// ── RemoveMcpServer ───────────────────────────────────────────────────────────

TEST_F(McpRpcTest, RemoveMcpServerSuccess) {
    // Add, then remove
    neurons::AddMcpServerRequest add_req;
    *add_req.mutable_server() = make_proto_server("to_remove");
    neurons::AddMcpServerResponse add_resp;
    svc_->AddMcpServer(nullptr, &add_req, &add_resp);

    neurons::RemoveMcpServerRequest rem_req;
    rem_req.set_name("to_remove");
    neurons::RemoveMcpServerResponse rem_resp;
    ASSERT_TRUE(svc_->RemoveMcpServer(nullptr, &rem_req, &rem_resp).ok());
    EXPECT_TRUE(rem_resp.success());
    EXPECT_TRUE(rem_resp.error().empty());

    neurons::ListMcpServersRequest list_req;
    neurons::ListMcpServersResponse list_resp;
    svc_->ListMcpServers(nullptr, &list_req, &list_resp);
    EXPECT_EQ(user_server_count(list_resp), 0);
}

TEST_F(McpRpcTest, RemoveMcpServerNotFoundReturnsError) {
    neurons::RemoveMcpServerRequest req;
    req.set_name("ghost");
    neurons::RemoveMcpServerResponse resp;
    ASSERT_TRUE(svc_->RemoveMcpServer(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
    EXPECT_FALSE(resp.error().empty());
}

// ── PushMcpServers ────────────────────────────────────────────────────────────

TEST_F(McpRpcTest, PushMcpServersReplacesEntireList) {
    // Pre-populate two servers
    neurons::AddMcpServerRequest add_req;
    neurons::AddMcpServerResponse add_resp;
    *add_req.mutable_server() = make_proto_server("old1");
    svc_->AddMcpServer(nullptr, &add_req, &add_resp);
    *add_req.mutable_server() = make_proto_server("old2");
    svc_->AddMcpServer(nullptr, &add_req, &add_resp);

    neurons::PushMcpServersRequest push_req;
    *push_req.add_servers() = make_proto_server("new1");
    neurons::PushMcpServersResponse push_resp;
    ASSERT_TRUE(svc_->PushMcpServers(nullptr, &push_req, &push_resp).ok());
    EXPECT_TRUE(push_resp.success());

    neurons::ListMcpServersRequest list_req;
    neurons::ListMcpServersResponse list_resp;
    svc_->ListMcpServers(nullptr, &list_req, &list_resp);
    EXPECT_EQ(user_server_count(list_resp), 1);
    bool found = false;
    for (int i = 0; i < list_resp.servers_size(); ++i)
        if (list_resp.servers(i).name() == "new1") { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(McpRpcTest, PushMcpServersWithEmptyListClearsAll) {
    neurons::AddMcpServerRequest add_req;
    neurons::AddMcpServerResponse add_resp;
    *add_req.mutable_server() = make_proto_server("srv");
    svc_->AddMcpServer(nullptr, &add_req, &add_resp);

    neurons::PushMcpServersRequest push_req;  // empty servers list
    neurons::PushMcpServersResponse push_resp;
    svc_->PushMcpServers(nullptr, &push_req, &push_resp);

    neurons::ListMcpServersRequest list_req;
    neurons::ListMcpServersResponse list_resp;
    svc_->ListMcpServers(nullptr, &list_req, &list_resp);
    EXPECT_EQ(user_server_count(list_resp), 0);
}

// ── ListPermissionRules ───────────────────────────────────────────────────────

TEST_F(McpRpcTest, ListPermissionRulesInitiallyEmpty) {
    neurons::ListPermissionRulesRequest req;
    neurons::ListPermissionRulesResponse resp;
    ASSERT_TRUE(svc_->ListPermissionRules(nullptr, &req, &resp).ok());
    EXPECT_EQ(resp.rules_size(), 0);
}

TEST_F(McpRpcTest, ListPermissionRulesReturnsAllRules) {
    // Insert via SetPermissionRule, then list
    neurons::SetPermissionRuleRequest set_req;
    *set_req.mutable_rule() = make_proto_rule("srv", "tool", "always_allow");
    neurons::SetPermissionRuleResponse set_resp;
    svc_->SetPermissionRule(nullptr, &set_req, &set_resp);

    neurons::ListPermissionRulesRequest list_req;
    neurons::ListPermissionRulesResponse list_resp;
    svc_->ListPermissionRules(nullptr, &list_req, &list_resp);
    ASSERT_EQ(list_resp.rules_size(), 1);
    EXPECT_EQ(list_resp.rules(0).permission(), "always_allow");
}

TEST_F(McpRpcTest, ListPermissionRulesScopeFilter) {
    neurons::SetPermissionRuleRequest req;
    neurons::SetPermissionRuleResponse set_resp;
    *req.mutable_rule() = make_proto_rule("s", "t1", "always_allow", "global");
    svc_->SetPermissionRule(nullptr, &req, &set_resp);
    *req.mutable_rule() = make_proto_rule("s", "t2", "always_deny", "session:x");
    svc_->SetPermissionRule(nullptr, &req, &set_resp);

    neurons::ListPermissionRulesRequest list_req;
    list_req.set_scope("global");
    neurons::ListPermissionRulesResponse list_resp;
    svc_->ListPermissionRules(nullptr, &list_req, &list_resp);
    EXPECT_EQ(list_resp.rules_size(), 1);
    EXPECT_EQ(list_resp.rules(0).tool(), "t1");
}

// ── SetPermissionRule ─────────────────────────────────────────────────────────

TEST_F(McpRpcTest, SetPermissionRuleSuccess) {
    neurons::SetPermissionRuleRequest req;
    *req.mutable_rule() = make_proto_rule("srv", "tool", "always_allow");
    neurons::SetPermissionRuleResponse resp;
    ASSERT_TRUE(svc_->SetPermissionRule(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
}

TEST_F(McpRpcTest, SetPermissionRuleRejectsEmptyRequest) {
    neurons::SetPermissionRuleRequest req;  // no rule set
    neurons::SetPermissionRuleResponse resp;
    ASSERT_TRUE(svc_->SetPermissionRule(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
    EXPECT_FALSE(resp.error().empty());
}

// ── DeletePermissionRule ──────────────────────────────────────────────────────

TEST_F(McpRpcTest, DeletePermissionRuleRemovesRule) {
    neurons::SetPermissionRuleRequest set_req;
    neurons::SetPermissionRuleResponse set_resp;
    *set_req.mutable_rule() = make_proto_rule("srv", "tool", "always_allow");
    svc_->SetPermissionRule(nullptr, &set_req, &set_resp);

    neurons::DeletePermissionRuleRequest del_req;
    del_req.set_server("srv");
    del_req.set_tool("tool");
    del_req.set_scope("global");
    neurons::DeletePermissionRuleResponse del_resp;
    ASSERT_TRUE(svc_->DeletePermissionRule(nullptr, &del_req, &del_resp).ok());
    EXPECT_TRUE(del_resp.success());

    neurons::ListPermissionRulesRequest list_req;
    neurons::ListPermissionRulesResponse list_resp;
    svc_->ListPermissionRules(nullptr, &list_req, &list_resp);
    EXPECT_EQ(list_resp.rules_size(), 0);
}

TEST_F(McpRpcTest, DeletePermissionRuleIsIdempotent) {
    // Deleting a rule that doesn't exist should still return success
    neurons::DeletePermissionRuleRequest req;
    req.set_server("ghost");
    req.set_tool("tool");
    req.set_scope("global");
    neurons::DeletePermissionRuleResponse resp;
    ASSERT_TRUE(svc_->DeletePermissionRule(nullptr, &req, &resp).ok());
    EXPECT_TRUE(resp.success());
}

// ── RespondToolApproval ───────────────────────────────────────────────────────
// The success path (approve/deny a live pending tool call) requires a loaded
// model running a Generate stream and is covered by integration tests.
// The unit-testable case is the error path: unknown approval_id.

TEST_F(McpRpcTest, RespondToolApprovalUnknownIdReturnsError) {
    neurons::ToolApprovalResponse req;
    req.set_approval_id("does-not-exist");
    req.set_approved(true);
    neurons::ToolApprovalResult resp;
    ASSERT_TRUE(svc_->RespondToolApproval(nullptr, &req, &resp).ok());
    EXPECT_FALSE(resp.success());
    EXPECT_FALSE(resp.error().empty());
}
