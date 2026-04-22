#pragma once

#include "mcp_client.h"
#include "mcp_types.h"
#include "compute/model/language_model.h"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace neurons_service {

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

    // ── Connections ───────────────────────────────────────────────────────────

    // Connect all servers that are enabled in the stored config.
    // Skips servers that are already connected.
    void connect_enabled();

    // Connect / disconnect a single server by name.
    // Returns "" on success or an error message.
    std::string connect_server(const std::string& name);
    void        disconnect_server(const std::string& name);
    bool        is_connected(const std::string& name) const;

    // ── Tool queries ──────────────────────────────────────────────────────────

    // Aggregate tool list across all connected servers.
    std::vector<ToolDef> list_tools();

    // JSON array of tool definitions in OpenAI function-calling format.
    // Suitable for passing to LanguageModel::format_tool_system_prompt().
    std::string tools_json();

    // ── Permissions (persisted to mcp_permissions.json) ──────────────────────

    void load_permissions();
    void save_permissions() const;

    McpPermission get_permission(const std::string& server,
                                 const std::string& tool) const;
    void          set_permission(const std::string& server,
                                 const std::string& tool,
                                 McpPermission      perm);

    // ── Tool call callback ────────────────────────────────────────────────────

    // Returns a ToolCallCb suitable for passing to NeuronsServiceImpl::generate_internal().
    // session_id: unique string per generation session — used to track
    //   "allow for session" grants that don't persist to disk.
    using ToolCallCb = std::function<
        std::optional<std::string>(const compute::LanguageModel::ToolCall&)>;

    ToolCallCb make_tool_call_cb(const std::string& session_id = "");

    // Check if any enabled server is connected (i.e., tools are available).
    bool has_active_tools() const;

private:
    std::string servers_path()     const;
    std::string permissions_path() const;

    // Rebuild the tool→server map from currently connected clients.
    // Must be called under mutex_.
    void rebuild_tool_map_locked();

    std::string config_dir_;

    std::vector<McpServerConfig>                               server_configs_;
    std::unordered_map<std::string, std::unique_ptr<McpClient>> clients_;

    // tool_name → server_name (rebuilt on connect/disconnect)
    std::unordered_map<std::string, std::string> tool_to_server_;

    // Persisted permissions key: "server_name:tool_name"
    std::unordered_map<std::string, McpPermission> saved_permissions_;

    // Per-session grants (not persisted) key: "session_id:server_name:tool_name"
    std::unordered_set<std::string> session_allowed_;

    mutable std::mutex mutex_;
};

} // namespace neurons_service
