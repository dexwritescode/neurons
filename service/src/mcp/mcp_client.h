#pragma once

#include "mcp_types.h"
#include <memory>
#include <string>
#include <vector>

namespace neurons_service {

// JSON-RPC 2.0 MCP client.
// Supports stdio transport (subprocess with bidirectional pipes).
// SSE transport is stubbed — returns an error until implemented.
//
// Thread-safety: a single McpClient must not be called concurrently.
// McpManager serializes calls with its own mutex.
class McpClient {
public:
    ~McpClient();

    static std::unique_ptr<McpClient> create(const McpServerConfig& cfg);

    // Launch the subprocess and complete the MCP initialize handshake.
    // Returns empty string on success, error message on failure.
    std::string connect();
    void        disconnect();
    bool        is_connected() const { return connected_; }

    // Fetch tool definitions from this server.
    std::string list_tools(std::vector<ToolDef>& tools_out);

    // Invoke a tool.  args_json must be a JSON object string.
    // On success: result_out receives human-readable text, returns "".
    // On failure: returns an error message.
    std::string call_tool(const std::string& name,
                          const std::string& args_json,
                          std::string&       result_out);

    const McpServerConfig& config() const { return config_; }

private:
    explicit McpClient(const McpServerConfig& cfg);

    // Synchronous JSON-RPC call.  Blocks until the server replies.
    // Returns "" on success and fills result_json_out with the "result" field.
    std::string rpc_call(const std::string& method,
                         const std::string& params_json,
                         std::string&       result_json_out);

    // Read a full newline-terminated line from stdout_fd_.
    // Returns "" on EOF or error (sets connected_ = false).
    std::string read_line();

    McpServerConfig config_;
    bool  connected_ = false;
    int   stdin_fd_  = -1;  // write end — parent → child
    int   stdout_fd_ = -1;  // read end  — child  → parent
    pid_t child_pid_ = -1;
    int   next_id_   = 1;
};

} // namespace neurons_service
