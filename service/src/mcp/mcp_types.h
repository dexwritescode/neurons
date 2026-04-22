#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace neurons_service {

enum class McpTransport { Stdio, Sse };

struct McpServerConfig {
    std::string  name;
    McpTransport transport = McpTransport::Stdio;
    std::string              command;
    std::vector<std::string> args;
    std::string url;
    std::unordered_map<std::string, std::string> env;
    bool enabled = true;
};

// A permission rule. Rules are evaluated in priority order (lower = first).
// First match wins; no match defaults to always_ask.
//
// arg_constraints: JSON object whose keys are arg field names and values are
// glob patterns matched against the corresponding field in the tool's args_json.
// Empty string = match any invocation.
//
// scope:
//   "global"          — persisted in mcp_permissions.json
//   "session:<id>"    — in-memory, cleared when the session ends
//   "chat:<id>"       — persisted in the ChatSession JSON
struct PermissionRule {
    std::string server;           // exact name or "*"
    std::string tool;             // exact name or "*"
    std::string arg_constraints;  // JSON object string, empty = no constraint
    std::string permission;       // "always_allow" | "allow_session" | "always_ask" | "always_deny"
    std::string scope;            // "global" | "session:<id>" | "chat:<id>"
    int         priority = 0;     // lower evaluated first
};

// A tool exposed by an MCP server.
struct ToolDef {
    std::string server_name;
    std::string name;
    std::string description;
    std::string input_schema_json;
};

} // namespace neurons_service
