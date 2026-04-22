#pragma once

#include "mcp_client.h"
#include "mcp_types.h"
#include "compute/model/language_model.h"

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace neurons_service {

// Approval request issued to the caller when a tool needs always_ask permission.
struct ToolApprovalRequest {
    std::string approval_id;
    std::string server;
    std::string tool;
    std::string args_json;
    std::string description;
    bool        destructive = false;
};

// Callback types for the Generate stream.
// ApprovalCb: called when a tool requires explicit user approval.
// Returns a future that resolves to true (approved) or false (denied) when
// RespondToolApproval is called with the same approval_id.
using ApprovalCb = std::function<std::future<bool>(const ToolApprovalRequest&)>;

// ToolCallCb: return value from make_tool_call_cb().
// nullopt = denied (tool call silently dropped).
// string  = tool result JSON to inject into the context.
using ToolCallCb = std::function<
    std::optional<std::string>(const compute::LanguageModel::ToolCall&)>;

// Owns and coordinates MCP server connections.
// Aggregates tool lists across active servers, enforces per-tool permissions,
// and routes tool calls to the correct McpClient.
//
// Thread-safety: all public methods are guarded by an internal mutex.
class McpManager {
public:
    // config_dir: directory holding mcp_servers.json and mcp_permissions.json.
    // Defaults to ~/.neurons/ when empty.
    explicit McpManager(std::string config_dir = "");

    // ── Server config (persisted to mcp_servers.json) ────────────────────────

    void load_config();
    void save_config() const;

    void                         add_server(const McpServerConfig& cfg);
    bool                         remove_server(const std::string& name);
    std::vector<McpServerConfig> list_servers() const;

    // Replace entire server list (used by PushMcpServers inherit-mode).
    // Does not persist — in-memory only for the session.
    void push_servers(const std::vector<McpServerConfig>& servers);

    // ── Connections ───────────────────────────────────────────────────────────

    void        connect_enabled();
    std::string connect_server(const std::string& name);
    void        disconnect_server(const std::string& name);
    bool        is_connected(const std::string& name) const;

    // ── Tool queries ──────────────────────────────────────────────────────────

    std::vector<ToolDef> list_tools();
    std::string          tools_json();
    bool                 has_active_tools() const;

    // ── Permission rules ──────────────────────────────────────────────────────

    void load_permissions();
    void save_permissions() const;

    // Returns rules matching the given scope filter (empty = all scopes).
    std::vector<PermissionRule> list_rules(const std::string& scope_filter = "") const;

    // Add or replace a rule. Rules with the same (server, tool, scope) are replaced.
    void set_rule(const PermissionRule& rule);

    // Remove rules matching (server, tool, scope).
    void delete_rule(const std::string& server,
                     const std::string& tool,
                     const std::string& scope);

    // Resolve permission for a tool call given its parsed args_json.
    // Checks session overrides, then per-chat rules, then global rules.
    // Returns "always_allow" | "allow_session" | "always_ask" | "always_deny".
    std::string resolve_permission(const std::string& server,
                                   const std::string& tool,
                                   const std::string& args_json,
                                   const std::string& session_id,
                                   const std::string& chat_id) const;

    // ── Tool call callback ────────────────────────────────────────────────────

    // approval_cb: called when a tool's resolved permission is "always_ask".
    // Returns a ToolCallCb suitable for passing to generate_internal().
    ToolCallCb make_tool_call_cb(const std::string& session_id = "",
                                 const std::string& chat_id    = "",
                                 ApprovalCb         approval_cb = nullptr);

private:
    std::string servers_path()     const;
    std::string permissions_path() const;

    void rebuild_tool_map_locked();

    // Returns true if args_json satisfies constraint_json (glob matching).
    // Empty constraint_json always matches.
    static bool match_constraints(const std::string& constraint_json,
                                  const std::string& args_json);

    std::string config_dir_;

    std::vector<McpServerConfig>                                server_configs_;
    std::unordered_map<std::string, std::unique_ptr<McpClient>> clients_;
    std::unordered_map<std::string, std::string>                tool_to_server_;

    // All permission rules, sorted by priority on modification.
    std::vector<PermissionRule> rules_;

    mutable std::mutex mutex_;
};

} // namespace neurons_service
