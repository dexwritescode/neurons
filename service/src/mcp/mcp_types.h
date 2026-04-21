#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace neurons_service {

enum class McpTransport { Stdio, Sse };

enum class McpPermission {
    AlwaysAsk,    // prompt on every call (default for write/exec tools)
    AllowSession, // allow for the rest of this generation session
    AlwaysAllow,  // persisted — never ask again
    AlwaysDeny,   // persisted — silently reject all calls to this tool
};

struct McpServerConfig {
    std::string  name;
    McpTransport transport = McpTransport::Stdio;
    // stdio transport
    std::string              command;
    std::vector<std::string> args;
    // sse transport
    std::string url;
    // common
    std::unordered_map<std::string, std::string> env;
    bool enabled = true;
};

// A tool exposed by an MCP server.
struct ToolDef {
    std::string server_name;       // owning server
    std::string name;
    std::string description;
    std::string input_schema_json; // JSON Schema object string
};

} // namespace neurons_service
