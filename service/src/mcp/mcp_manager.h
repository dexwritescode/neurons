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
using ApprovalCb = std::function<std::future<bool>(const ToolApprovalRequest&)>;

using ToolCallCb = std::function<
    std::optional<std::string>(const compute::LanguageModel::ToolCall&)>;

// Hook called around every tool dispatch (built-in or external MCP server).
// Hooks are invoked in registration order.
//
// pre_call:  receives mutable args_json — hook can rewrite args or return false
//            to silently deny the call.
// post_call: receives mutable result_json — hook can strip tokens, transform, or
//            log the result (e.g. RTK-style output trimming).
struct ToolHook {
    std::function<bool(const std::string& server,
                       const std::string& tool,
                       std::string&       args_json)>   pre_call;

    std::function<void(const std::string& server,
                       const std::string& tool,
                       const std::string& args_json,
                       std::string&       result_json)> post_call;
};

// Owns and coordinates MCP server connections.
// Aggregates tool lists across active servers (including built-ins), enforces
// per-tool permissions, and routes tool calls to the correct handler.
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

    // Returns built-in servers followed by user-configured servers.
    std::vector<McpServerConfig> list_servers() const;

    // Replace entire user-configured server list (used by PushMcpServers).
    // Does not affect built-in servers. Does not persist — in-memory only.
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

    std::vector<PermissionRule> list_rules(const std::string& scope_filter = "") const;
    void set_rule(const PermissionRule& rule);
    void delete_rule(const std::string& server,
                     const std::string& tool,
                     const std::string& scope);

    std::string resolve_permission(const std::string& server,
                                   const std::string& tool,
                                   const std::string& args_json,
                                   const std::string& session_id,
                                   const std::string& chat_id) const;

    // ── Tool call callback ────────────────────────────────────────────────────

    ToolCallCb make_tool_call_cb(const std::string& session_id = "",
                                 const std::string& chat_id    = "",
                                 ApprovalCb         approval_cb = nullptr);

    // ── Tool hooks ────────────────────────────────────────────────────────────

    // Register a hook to run before/after every tool dispatch (built-in or external).
    // Hooks are called in registration order. Thread-safe.
    void add_tool_hook(ToolHook hook);

private:
    std::string servers_path()     const;
    std::string permissions_path() const;

    // Register the two built-in servers (neurons-filesystem, neurons-shell).
    // Called once from the constructor.
    void register_builtins();

    void rebuild_tool_map_locked();

    static bool match_constraints(const std::string& constraint_json,
                                  const std::string& args_json);

    // Dispatch a single tool call through hooks → handler → hooks.
    // Called with mutex_ NOT held (handler may do blocking I/O).
    std::string dispatch_tool(const std::string& server_name,
                              const std::string& tool_name,
                              std::string        args_json);

    std::string config_dir_;

    // User-configured external servers.
    std::vector<McpServerConfig>                                server_configs_;
    std::unordered_map<std::string, std::unique_ptr<McpClient>> clients_;
    std::unordered_map<std::string, std::string>                tool_to_server_;

    // Built-in in-process servers.
    using BuiltinHandler = std::function<std::string(
        const std::string& tool, const std::string& args_json, std::string& result_out)>;
    std::vector<McpServerConfig>                    builtin_configs_;
    std::unordered_map<std::string, BuiltinHandler> builtin_handlers_;
    std::vector<ToolDef>                            builtin_tool_defs_;

    // User-defined permission rules, sorted by priority on modification.
    // Built-in default rules are stored separately and not exposed via list_rules().
    std::vector<PermissionRule> rules_;
    std::vector<PermissionRule> builtin_rules_;

    // Tool hooks (pre/post dispatch).
    std::vector<ToolHook> hooks_;

    mutable std::mutex mutex_;
};

} // namespace neurons_service
