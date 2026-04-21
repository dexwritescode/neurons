#include "mcp_client.h"

#include <nlohmann/json.hpp>
#include <cerrno>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

namespace neurons_service {

// ── Factory ───────────────────────────────────────────────────────────────────

McpClient::McpClient(const McpServerConfig& cfg) : config_(cfg) {}

McpClient::~McpClient() {
    disconnect();
}

std::unique_ptr<McpClient> McpClient::create(const McpServerConfig& cfg) {
    return std::unique_ptr<McpClient>(new McpClient(cfg));
}

// ── connect ───────────────────────────────────────────────────────────────────

std::string McpClient::connect() {
    if (connected_) return "";

    if (config_.transport == McpTransport::Sse) {
        return "SSE transport not yet implemented";
    }

    if (config_.command.empty()) {
        return "Server \"" + config_.name + "\": command is empty";
    }

    // Create bidirectional pipes: parent_write→child_stdin, child_stdout→parent_read
    int to_child[2], from_child[2];
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        return std::string("pipe() failed: ") + strerror(errno);
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(to_child[0]);  close(to_child[1]);
        close(from_child[0]); close(from_child[1]);
        return std::string("fork() failed: ") + strerror(errno);
    }

    if (pid == 0) {
        // ── Child ──────────────────────────────────────────────────────────────
        dup2(to_child[0],   STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        // Close all inherited fds except stdin/stdout/stderr
        close(to_child[0]);  close(to_child[1]);
        close(from_child[0]); close(from_child[1]);

        for (const auto& [k, v] : config_.env) {
            setenv(k.c_str(), v.c_str(), 1);
        }

        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(config_.command.c_str()));
        for (const auto& a : config_.args)
            argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);

        execvp(config_.command.c_str(), argv.data());
        _exit(1); // exec failed
    }

    // ── Parent ─────────────────────────────────────────────────────────────────
    close(to_child[0]);   // parent doesn't read from this end
    close(from_child[1]); // parent doesn't write to this end
    stdin_fd_  = to_child[1];   // parent writes here → child's stdin
    stdout_fd_ = from_child[0]; // parent reads here  ← child's stdout
    child_pid_ = pid;

    // MCP initialize handshake
    std::string init_result;
    auto err = rpc_call("initialize",
        R"({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"neurons","version":"1.0"}})",
        init_result);
    if (!err.empty()) {
        disconnect();
        return "MCP initialize failed: " + err;
    }

    // Send notifications/initialized (notification — no id, no response)
    const std::string notif = R"({"jsonrpc":"2.0","method":"notifications/initialized"})" "\n";
    if (write(stdin_fd_, notif.c_str(), notif.size()) < 0) {
        disconnect();
        return "Failed to send initialized notification";
    }

    connected_ = true;
    return "";
}

// ── disconnect ────────────────────────────────────────────────────────────────

void McpClient::disconnect() {
    if (stdin_fd_  != -1) { close(stdin_fd_);  stdin_fd_  = -1; }
    if (stdout_fd_ != -1) { close(stdout_fd_); stdout_fd_ = -1; }
    if (child_pid_ != -1) {
        int status;
        waitpid(child_pid_, &status, WNOHANG);
        child_pid_ = -1;
    }
    connected_ = false;
}

// ── list_tools ────────────────────────────────────────────────────────────────

std::string McpClient::list_tools(std::vector<ToolDef>& tools_out) {
    std::string raw;
    auto err = rpc_call("tools/list", "", raw);
    if (!err.empty()) return err;

    try {
        auto j = nlohmann::json::parse(raw);
        if (!j.contains("tools")) return "";  // empty list is valid

        for (const auto& t : j["tools"]) {
            ToolDef def;
            def.server_name       = config_.name;
            def.name              = t["name"].get<std::string>();
            def.description       = t.value("description", "");
            def.input_schema_json = t.contains("inputSchema")
                                     ? t["inputSchema"].dump() : "{}";
            tools_out.push_back(std::move(def));
        }
    } catch (const nlohmann::json::exception& e) {
        return std::string("tools/list parse error: ") + e.what();
    }
    return "";
}

// ── call_tool ─────────────────────────────────────────────────────────────────

std::string McpClient::call_tool(const std::string& name,
                                  const std::string& args_json,
                                  std::string&       result_out) {
    std::string params =
        R"({"name":")" + name + R"(","arguments":)" + args_json + "}";
    std::string raw;
    auto err = rpc_call("tools/call", params, raw);
    if (!err.empty()) return err;

    try {
        auto j = nlohmann::json::parse(raw);
        const bool is_error = j.value("isError", false);

        // Concatenate all text-type content blocks
        std::string text;
        if (j.contains("content")) {
            for (const auto& c : j["content"]) {
                if (c.value("type", "") == "text") {
                    text += c.value("text", "");
                }
            }
        }

        result_out = text.empty() ? raw : text;
        if (is_error) return "Tool \"" + name + "\" returned an error";
    } catch (const nlohmann::json::exception& e) {
        return std::string("tools/call parse error: ") + e.what();
    }
    return "";
}

// ── rpc_call (internal) ───────────────────────────────────────────────────────

std::string McpClient::rpc_call(const std::string& method,
                                 const std::string& params_json,
                                 std::string&       result_json_out) {
    const int id = next_id_++;

    // Build JSON-RPC 2.0 request
    std::string req = R"({"jsonrpc":"2.0","id":)" + std::to_string(id) +
                      R"(,"method":")" + method + '"';
    if (!params_json.empty()) {
        req += R"(,"params":)" + params_json;
    }
    req += "}\n";

    if (write(stdin_fd_, req.c_str(), req.size()) < 0) {
        disconnect();
        return std::string("write to subprocess failed: ") + strerror(errno);
    }

    // Read lines until we find the response for our id.
    // Skip notifications (no "id") and responses for other ids.
    for (int attempts = 0; attempts < 1000; ++attempts) {
        std::string line = read_line();
        if (line.empty()) {
            return "Subprocess disconnected waiting for " + method + " response";
        }
        try {
            auto j = nlohmann::json::parse(line);
            if (!j.contains("id") || j["id"] != id) continue; // notification/other

            if (j.contains("error")) {
                const auto& e = j["error"];
                return e.value("message", "Unknown RPC error");
            }
            result_json_out = j.contains("result") ? j["result"].dump() : "{}";
            return "";
        } catch (...) {
            // Malformed JSON — keep reading
        }
    }
    return "No response received for " + method;
}

// ── read_line ─────────────────────────────────────────────────────────────────

std::string McpClient::read_line() {
    std::string line;
    line.reserve(256);
    char c;
    while (true) {
        const ssize_t n = read(stdout_fd_, &c, 1);
        if (n <= 0) {
            connected_ = false;
            return "";
        }
        if (c == '\n') return line;
        line += c;
    }
}

} // namespace neurons_service
