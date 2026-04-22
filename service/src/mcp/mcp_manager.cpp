#include "mcp_manager.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace neurons_service {

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string default_config_dir() {
    const char* home = getenv("HOME");
    return home ? std::string(home) + "/.neurons" : ".neurons";
}

static std::string perm_to_string(McpPermission p) {
    switch (p) {
    case McpPermission::AlwaysAsk:    return "always_ask";
    case McpPermission::AllowSession: return "allow_session";
    case McpPermission::AlwaysAllow:  return "always_allow";
    case McpPermission::AlwaysDeny:   return "always_deny";
    }
    return "always_ask";
}

static McpPermission perm_from_string(const std::string& s) {
    if (s == "allow_session") return McpPermission::AllowSession;
    if (s == "always_allow")  return McpPermission::AlwaysAllow;
    if (s == "always_deny")   return McpPermission::AlwaysDeny;
    return McpPermission::AlwaysAsk;
}

// ── Constructor ───────────────────────────────────────────────────────────────

McpManager::McpManager(std::string config_dir)
    : config_dir_(config_dir.empty() ? default_config_dir() : std::move(config_dir))
{}

// ── Paths ─────────────────────────────────────────────────────────────────────

std::string McpManager::servers_path()     const { return config_dir_ + "/mcp_servers.json"; }
std::string McpManager::permissions_path() const { return config_dir_ + "/mcp_permissions.json"; }

// ── Config persistence ────────────────────────────────────────────────────────

void McpManager::load_config() {
    std::lock_guard lock(mutex_);
    server_configs_.clear();

    std::ifstream f(servers_path());
    if (!f.is_open()) return;  // file doesn't exist yet — start empty

    try {
        auto j = nlohmann::json::parse(f);
        for (const auto& s : j.value("servers", nlohmann::json::array())) {
            McpServerConfig cfg;
            cfg.name    = s.value("name", "");
            cfg.enabled = s.value("enabled", true);
            const std::string t = s.value("transport", "stdio");
            cfg.transport = (t == "sse") ? McpTransport::Sse : McpTransport::Stdio;
            cfg.command = s.value("command", "");
            if (s.contains("args"))
                cfg.args = s["args"].get<std::vector<std::string>>();
            cfg.url = s.value("url", "");
            if (s.contains("env"))
                cfg.env = s["env"].get<std::unordered_map<std::string, std::string>>();
            if (!cfg.name.empty()) server_configs_.push_back(std::move(cfg));
        }
    } catch (...) {}
}

void McpManager::save_config() const {
    std::lock_guard lock(mutex_);
    fs::create_directories(config_dir_);

    nlohmann::json root;
    root["servers"] = nlohmann::json::array();
    for (const auto& cfg : server_configs_) {
        nlohmann::json s;
        s["name"]      = cfg.name;
        s["enabled"]   = cfg.enabled;
        s["transport"] = (cfg.transport == McpTransport::Sse) ? "sse" : "stdio";
        s["command"]   = cfg.command;
        s["args"]      = cfg.args;
        s["url"]       = cfg.url;
        s["env"]       = cfg.env;
        root["servers"].push_back(std::move(s));
    }
    std::ofstream f(servers_path());
    f << root.dump(2);
}

// ── Server management ─────────────────────────────────────────────────────────

void McpManager::add_server(const McpServerConfig& cfg) {
    std::lock_guard lock(mutex_);
    // Replace if name already exists
    for (auto& existing : server_configs_) {
        if (existing.name == cfg.name) { existing = cfg; return; }
    }
    server_configs_.push_back(cfg);
}

bool McpManager::remove_server(const std::string& name) {
    std::lock_guard lock(mutex_);
    const auto before = server_configs_.size();
    server_configs_.erase(
        std::remove_if(server_configs_.begin(), server_configs_.end(),
                       [&](const McpServerConfig& c) { return c.name == name; }),
        server_configs_.end());
    disconnect_server(name);
    return server_configs_.size() < before;
}

std::vector<McpServerConfig> McpManager::list_servers() const {
    std::lock_guard lock(mutex_);
    return server_configs_;
}

// ── Connection management ─────────────────────────────────────────────────────

void McpManager::connect_enabled() {
    std::vector<std::string> to_connect;
    {
        std::lock_guard lock(mutex_);
        for (const auto& cfg : server_configs_) {
            if (cfg.enabled && clients_.find(cfg.name) == clients_.end()) {
                to_connect.push_back(cfg.name);
            }
        }
    }
    for (const auto& name : to_connect) connect_server(name);
}

std::string McpManager::connect_server(const std::string& name) {
    std::lock_guard lock(mutex_);

    if (clients_.count(name)) return "";  // already connected

    const McpServerConfig* cfg = nullptr;
    for (const auto& c : server_configs_) {
        if (c.name == name) { cfg = &c; break; }
    }
    if (!cfg) return "Server \"" + name + "\" not found in config";

    auto client = McpClient::create(*cfg);
    auto err = client->connect();
    if (!err.empty()) return err;

    clients_[name] = std::move(client);
    rebuild_tool_map_locked();
    return "";
}

void McpManager::disconnect_server(const std::string& name) {
    // Called both under lock and without — caller must hold mutex_ or pass via remove_server.
    clients_.erase(name);
    rebuild_tool_map_locked();
}

bool McpManager::is_connected(const std::string& name) const {
    std::lock_guard lock(mutex_);
    return clients_.count(name) > 0;
}

// ── Tool queries ──────────────────────────────────────────────────────────────

void McpManager::rebuild_tool_map_locked() {
    tool_to_server_.clear();
    for (auto& [server_name, client] : clients_) {
        std::vector<ToolDef> tools;
        client->list_tools(tools);
        for (const auto& t : tools)
            tool_to_server_[t.name] = server_name;
    }
}

std::vector<ToolDef> McpManager::list_tools() {
    std::lock_guard lock(mutex_);
    std::vector<ToolDef> all;
    for (auto& [name, client] : clients_) {
        client->list_tools(all);
    }
    return all;
}

std::string McpManager::tools_json() {
    const auto tools = list_tools();
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& t : tools) {
        nlohmann::json tool;
        tool["type"]     = "function";
        tool["function"] = {
            {"name",        t.name},
            {"description", t.description},
            {"parameters",  nlohmann::json::parse(
                                t.input_schema_json.empty() ? "{}" : t.input_schema_json)},
        };
        arr.push_back(std::move(tool));
    }
    return arr.dump();
}

bool McpManager::has_active_tools() const {
    std::lock_guard lock(mutex_);
    return !clients_.empty();
}

// ── Permissions ───────────────────────────────────────────────────────────────

void McpManager::load_permissions() {
    std::lock_guard lock(mutex_);
    saved_permissions_.clear();
    std::ifstream f(permissions_path());
    if (!f.is_open()) return;
    try {
        auto j = nlohmann::json::parse(f);
        for (auto it = j.begin(); it != j.end(); ++it) {
            saved_permissions_[it.key()] = perm_from_string(it.value().get<std::string>());
        }
    } catch (...) {}
}

void McpManager::save_permissions() const {
    std::lock_guard lock(mutex_);
    fs::create_directories(config_dir_);
    nlohmann::json j;
    for (const auto& [k, v] : saved_permissions_) {
        j[k] = perm_to_string(v);
    }
    std::ofstream f(permissions_path());
    f << j.dump(2);
}

McpPermission McpManager::get_permission(const std::string& server,
                                          const std::string& tool) const {
    std::lock_guard lock(mutex_);
    const std::string key = server + ":" + tool;
    auto it = saved_permissions_.find(key);
    return (it != saved_permissions_.end()) ? it->second : McpPermission::AlwaysAsk;
}

void McpManager::set_permission(const std::string& server,
                                 const std::string& tool,
                                 McpPermission      perm) {
    std::lock_guard lock(mutex_);
    saved_permissions_[server + ":" + tool] = perm;
}

// ── Tool call callback ────────────────────────────────────────────────────────

McpManager::ToolCallCb McpManager::make_tool_call_cb(const std::string& session_id) {
    return [this, session_id](const compute::LanguageModel::ToolCall& call)
               -> std::optional<std::string> {

        std::unique_lock lock(mutex_);

        // Route: find which server owns this tool
        auto server_it = tool_to_server_.find(call.name);
        if (server_it == tool_to_server_.end()) {
            // Unknown tool — return an error result rather than denying silently
            return R"({"error":"Tool \")" + call.name + R"(\" not found on any connected MCP server"})";
        }
        const std::string server_name = server_it->second;

        // Permission check
        const std::string perm_key    = server_name + ":" + call.name;
        const std::string session_key = session_id + ":" + perm_key;
        const bool session_grant      = session_allowed_.count(session_key) > 0;

        McpPermission perm = McpPermission::AlwaysAsk;
        auto perm_it = saved_permissions_.find(perm_key);
        if (perm_it != saved_permissions_.end()) perm = perm_it->second;

        if (perm == McpPermission::AlwaysDeny) return std::nullopt;  // denied

        // For service context: treat AlwaysAsk as AllowSession until L.3 adds
        // interactive approval over gRPC.  CLI wires its own approval handler.
        const bool allowed = (perm == McpPermission::AlwaysAllow)
                          || (perm == McpPermission::AllowSession)
                          || session_grant
                          || (perm == McpPermission::AlwaysAsk);  // auto-allow in service

        if (!allowed) return std::nullopt;

        // Grant session permission for future calls in this session
        if (perm == McpPermission::AlwaysAsk && !session_id.empty()) {
            session_allowed_.insert(session_key);
        }

        // Dispatch to the client (release lock during the blocking call)
        auto client_it = clients_.find(server_name);
        if (client_it == clients_.end()) {
            return R"({"error":"Server \")" + server_name + R"(\" is not connected"})";
        }
        McpClient* client = client_it->second.get();
        lock.unlock();

        std::string result;
        const auto err = client->call_tool(call.name, call.arguments_json, result);
        if (!err.empty()) {
            return R"({"error":")" + err + R"("})";
        }
        return result;
    };
}

} // namespace neurons_service
