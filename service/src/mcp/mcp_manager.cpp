#include "mcp_manager.h"
#include "builtin_filesystem.h"
#include "builtin_shell.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <fnmatch.h>   // POSIX glob matching

namespace fs = std::filesystem;

namespace neurons_service {

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string default_config_dir() {
    const char* home = getenv("HOME");
    return home ? std::string(home) + "/.neurons" : ".neurons";
}

static bool glob_match(const std::string& pattern, const std::string& value) {
    if (pattern == "*") return true;
    return fnmatch(pattern.c_str(), value.c_str(), FNM_PATHNAME) == 0;
}

// ── Constructor ───────────────────────────────────────────────────────────────

McpManager::McpManager(std::string config_dir)
    : config_dir_(config_dir.empty() ? default_config_dir() : std::move(config_dir))
{
    register_builtins();
}

// ── Built-in server registration ──────────────────────────────────────────────

void McpManager::register_builtins() {
    struct BuiltinDef {
        std::string name;
        BuiltinHandler handler;
        std::vector<ToolDef> (*tool_defs_fn)();
    };

    const BuiltinDef defs[] = {
        { "neurons-filesystem", BuiltinFilesystem::handle, BuiltinFilesystem::tool_defs },
        { "neurons-shell",      BuiltinShell::handle,      BuiltinShell::tool_defs      },
    };

    for (const auto& def : defs) {
        McpServerConfig cfg;
        cfg.name    = def.name;
        cfg.builtin = true;
        cfg.enabled = true;
        builtin_configs_.push_back(cfg);
        builtin_handlers_[def.name] = def.handler;
        for (auto& td : def.tool_defs_fn())
            builtin_tool_defs_.push_back(std::move(td));
    }

    // Default permission rules for built-in tools.  Stored in builtin_rules_ so
    // they don't appear in list_rules() (which shows only user-defined rules).
    // resolve_permission() checks builtin_rules_ after user rules as a fallback.
    auto add_default = [&](const std::string& server, const std::string& tool,
                           const std::string& perm) {
        PermissionRule r;
        r.server     = server;
        r.tool       = tool;
        r.permission = perm;
        r.scope      = "global";
        r.priority   = 0;
        builtin_rules_.push_back(std::move(r));
    };

    add_default("neurons-filesystem", "read_file",    "allow_session");
    add_default("neurons-filesystem", "list_dir",     "allow_session");
    add_default("neurons-filesystem", "search_files", "allow_session");
    add_default("neurons-filesystem", "write_file",   "always_ask");
    add_default("neurons-shell",      "run_command",  "always_ask");
}

// ── Paths ─────────────────────────────────────────────────────────────────────

std::string McpManager::servers_path()     const { return config_dir_ + "/mcp_servers.json"; }
std::string McpManager::permissions_path() const { return config_dir_ + "/mcp_permissions.json"; }

// ── Config persistence ────────────────────────────────────────────────────────

void McpManager::load_config() {
    std::lock_guard lock(mutex_);
    server_configs_.clear();

    std::ifstream f(servers_path());
    if (!f.is_open()) return;

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
    clients_.erase(name);
    rebuild_tool_map_locked();
    return server_configs_.size() < before;
}

std::vector<McpServerConfig> McpManager::list_servers() const {
    std::lock_guard lock(mutex_);
    // Built-ins appear first so the GUI can always show them at the top.
    auto result = builtin_configs_;
    result.insert(result.end(), server_configs_.begin(), server_configs_.end());
    return result;
}

void McpManager::push_servers(const std::vector<McpServerConfig>& servers) {
    std::lock_guard lock(mutex_);
    server_configs_ = servers;
    // Disconnect any clients no longer in the list
    std::unordered_map<std::string, std::unique_ptr<McpClient>> kept;
    for (auto& cfg : server_configs_) {
        auto it = clients_.find(cfg.name);
        if (it != clients_.end()) kept[cfg.name] = std::move(it->second);
    }
    clients_ = std::move(kept);
    rebuild_tool_map_locked();
}

// ── Connection management ─────────────────────────────────────────────────────

void McpManager::connect_enabled() {
    std::vector<std::string> to_connect;
    {
        std::lock_guard lock(mutex_);
        for (const auto& cfg : server_configs_)
            if (cfg.enabled && !clients_.count(cfg.name))
                to_connect.push_back(cfg.name);
        // Built-ins are always connected; rebuild map to include their tools.
        rebuild_tool_map_locked();
    }
    for (const auto& name : to_connect) connect_server(name);
}

std::string McpManager::connect_server(const std::string& name) {
    // Built-ins are always in-process — nothing to connect.
    {
        std::lock_guard lock(mutex_);
        if (builtin_handlers_.count(name)) return "";
    }

    std::lock_guard lock(mutex_);
    if (clients_.count(name)) return "";

    const McpServerConfig* cfg = nullptr;
    for (const auto& c : server_configs_)
        if (c.name == name) { cfg = &c; break; }
    if (!cfg) return "Server \"" + name + "\" not found in config";

    auto client = McpClient::create(*cfg);
    auto err    = client->connect();
    if (!err.empty()) return err;

    clients_[name] = std::move(client);
    rebuild_tool_map_locked();
    return "";
}

void McpManager::disconnect_server(const std::string& name) {
    std::lock_guard lock(mutex_);
    // Built-ins cannot be disconnected.
    if (builtin_handlers_.count(name)) return;
    clients_.erase(name);
    rebuild_tool_map_locked();
}

bool McpManager::is_connected(const std::string& name) const {
    std::lock_guard lock(mutex_);
    // Enabled built-ins are always considered connected.
    for (const auto& cfg : builtin_configs_)
        if (cfg.name == name && cfg.enabled) return true;
    return clients_.count(name) > 0;
}

// ── Tool queries ──────────────────────────────────────────────────────────────

void McpManager::rebuild_tool_map_locked() {
    tool_to_server_.clear();
    // Built-in tools (enabled servers only).
    for (const auto& cfg : builtin_configs_) {
        if (!cfg.enabled) continue;
        for (const auto& t : builtin_tool_defs_)
            if (t.server_name == cfg.name)
                tool_to_server_[t.name] = cfg.name;
    }
    // External MCP clients.
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
    // Built-in tools first.
    for (const auto& cfg : builtin_configs_) {
        if (!cfg.enabled) continue;
        for (const auto& t : builtin_tool_defs_)
            if (t.server_name == cfg.name) all.push_back(t);
    }
    // External MCP client tools.
    for (auto& [name, client] : clients_) client->list_tools(all);
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
    if (!clients_.empty()) return true;
    for (const auto& cfg : builtin_configs_)
        if (cfg.enabled) return true;
    return false;
}

// ── Permission rules ──────────────────────────────────────────────────────────

void McpManager::load_permissions() {
    std::lock_guard lock(mutex_);
    // Remove only global rules (session/chat rules are transient)
    rules_.erase(std::remove_if(rules_.begin(), rules_.end(),
                     [](const PermissionRule& r){ return r.scope == "global"; }),
                 rules_.end());

    std::ifstream f(permissions_path());
    if (!f.is_open()) return;
    try {
        auto j = nlohmann::json::parse(f);
        for (const auto& r : j.value("rules", nlohmann::json::array())) {
            PermissionRule rule;
            rule.server          = r.value("server", "*");
            rule.tool            = r.value("tool", "*");
            rule.arg_constraints = r.value("arg_constraints", "");
            rule.permission      = r.value("permission", "always_ask");
            rule.scope           = "global";
            rule.priority        = r.value("priority", 0);
            rules_.push_back(std::move(rule));
        }
        std::stable_sort(rules_.begin(), rules_.end(),
                         [](const PermissionRule& a, const PermissionRule& b){
                             return a.priority < b.priority; });
    } catch (...) {}
}

void McpManager::save_permissions() const {
    std::lock_guard lock(mutex_);
    fs::create_directories(config_dir_);

    nlohmann::json root;
    root["rules"] = nlohmann::json::array();
    for (const auto& r : rules_) {
        if (r.scope != "global") continue;  // only persist global rules
        nlohmann::json jr;
        jr["server"]          = r.server;
        jr["tool"]            = r.tool;
        jr["arg_constraints"] = r.arg_constraints;
        jr["permission"]      = r.permission;
        jr["priority"]        = r.priority;
        root["rules"].push_back(std::move(jr));
    }
    std::ofstream f(permissions_path());
    f << root.dump(2);
}

std::vector<PermissionRule> McpManager::list_rules(const std::string& scope_filter) const {
    std::lock_guard lock(mutex_);
    if (scope_filter.empty()) return rules_;
    std::vector<PermissionRule> out;
    for (const auto& r : rules_)
        if (r.scope == scope_filter) out.push_back(r);
    return out;
}

void McpManager::set_rule(const PermissionRule& rule) {
    std::lock_guard lock(mutex_);
    // Replace existing rule with same (server, tool, scope)
    for (auto& r : rules_) {
        if (r.server == rule.server && r.tool == rule.tool && r.scope == rule.scope) {
            r = rule;
            std::stable_sort(rules_.begin(), rules_.end(),
                             [](const PermissionRule& a, const PermissionRule& b){
                                 return a.priority < b.priority; });
            return;
        }
    }
    rules_.push_back(rule);
    std::stable_sort(rules_.begin(), rules_.end(),
                     [](const PermissionRule& a, const PermissionRule& b){
                         return a.priority < b.priority; });
}

void McpManager::delete_rule(const std::string& server,
                              const std::string& tool,
                              const std::string& scope) {
    std::lock_guard lock(mutex_);
    rules_.erase(std::remove_if(rules_.begin(), rules_.end(),
                     [&](const PermissionRule& r){
                         return r.server == server && r.tool == tool && r.scope == scope;
                     }), rules_.end());
}

// ── Constraint matching ───────────────────────────────────────────────────────

bool McpManager::match_constraints(const std::string& constraint_json,
                                   const std::string& args_json) {
    if (constraint_json.empty()) return true;
    try {
        const auto constraints = nlohmann::json::parse(constraint_json);
        const auto args        = nlohmann::json::parse(args_json.empty() ? "{}" : args_json);
        for (auto it = constraints.begin(); it != constraints.end(); ++it) {
            const auto& key     = it.key();
            const auto& pattern = it.value().get<std::string>();
            if (!args.contains(key)) return false;
            const auto& val = args[key];
            const std::string val_str = val.is_string() ? val.get<std::string>() : val.dump();
            if (!glob_match(pattern, val_str)) return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

// ── Permission resolution ─────────────────────────────────────────────────────

std::string McpManager::resolve_permission(const std::string& server,
                                            const std::string& tool,
                                            const std::string& args_json,
                                            const std::string& session_id,
                                            const std::string& chat_id) const {
    std::lock_guard lock(mutex_);

    // Scope priority: session > chat > global
    const std::string session_scope = session_id.empty() ? "" : "session:" + session_id;
    const std::string chat_scope    = chat_id.empty()    ? "" : "chat:"    + chat_id;

    for (const std::string* scope : {&session_scope, &chat_scope}) {
        if (scope->empty()) continue;
        for (const auto& r : rules_) {
            if (r.scope != *scope) continue;
            if (!glob_match(r.server, server)) continue;
            if (!glob_match(r.tool, tool))     continue;
            if (!match_constraints(r.arg_constraints, args_json)) continue;
            return r.permission;
        }
    }
    // Global user rules (already sorted by priority)
    for (const auto& r : rules_) {
        if (r.scope != "global") continue;
        if (!glob_match(r.server, server)) continue;
        if (!glob_match(r.tool, tool))     continue;
        if (!match_constraints(r.arg_constraints, args_json)) continue;
        return r.permission;
    }
    // Built-in default rules (fallback, not exposed via list_rules)
    for (const auto& r : builtin_rules_) {
        if (!glob_match(r.server, server)) continue;
        if (!glob_match(r.tool, tool))     continue;
        return r.permission;
    }
    return "always_ask";  // ultimate default
}

// ── Tool hooks ────────────────────────────────────────────────────────────────

void McpManager::add_tool_hook(ToolHook hook) {
    std::lock_guard lock(mutex_);
    hooks_.push_back(std::move(hook));
}

// ── Unified tool dispatch (hooks → handler → hooks) ───────────────────────────

std::string McpManager::dispatch_tool(const std::string& server_name,
                                       const std::string& tool_name,
                                       std::string        args_json) {
    // Pre-call hooks (may rewrite args or deny the call).
    {
        std::lock_guard lock(mutex_);
        for (auto& hook : hooks_) {
            if (hook.pre_call && !hook.pre_call(server_name, tool_name, args_json))
                return R"({"error":"Call denied by pre-call hook"})";
        }
    }

    std::string result;

    // Built-in in-process handler.
    BuiltinHandler builtin;
    {
        std::lock_guard lock(mutex_);
        auto it = builtin_handlers_.find(server_name);
        if (it != builtin_handlers_.end()) builtin = it->second;
    }
    if (builtin) {
        const auto err = builtin(tool_name, args_json, result);
        if (!err.empty()) result = R"({"error":")" + err + R"("})";
    } else {
        // External MCP subprocess.
        McpClient* client = nullptr;
        {
            std::lock_guard lock(mutex_);
            auto it = clients_.find(server_name);
            if (it == clients_.end())
                return R"({"error":"Server not connected"})";
            client = it->second.get();
        }
        const auto err = client->call_tool(tool_name, args_json, result);
        if (!err.empty()) result = R"({"error":")" + err + R"("})";
    }

    // Post-call hooks (may transform or trim result).
    {
        std::lock_guard lock(mutex_);
        for (auto& hook : hooks_) {
            if (hook.post_call) hook.post_call(server_name, tool_name, args_json, result);
        }
    }

    return result;
}

// ── Tool call callback ────────────────────────────────────────────────────────

ToolCallCb McpManager::make_tool_call_cb(const std::string& session_id,
                                          const std::string& chat_id,
                                          ApprovalCb         approval_cb) {
    return [this, session_id, chat_id, approval_cb](
               const compute::LanguageModel::ToolCall& call) -> std::optional<std::string> {

        // Look up which server owns this tool.
        bool tool_found = false;
        std::string server_name;
        {
            std::lock_guard lock(mutex_);
            auto it = tool_to_server_.find(call.name);
            if (it != tool_to_server_.end()) {
                tool_found  = true;
                server_name = it->second;
            }
        }
        if (!tool_found) {
            // Unknown tool — still check for an explicit deny rule so callers can
            // block unknown tools via a wildcard rule without getting an error string.
            const std::string early = resolve_permission(
                "", call.name, call.arguments_json, session_id, chat_id);
            if (early == "always_deny") return std::nullopt;
            return R"({"error":"Tool not found on any connected MCP server"})";
        }

        const std::string perm = resolve_permission(
            server_name, call.name, call.arguments_json, session_id, chat_id);

        if (perm == "always_deny") return std::nullopt;

        if (perm == "always_ask") {
            if (!approval_cb) return std::nullopt;

            ToolApprovalRequest req;
            req.approval_id = [] {
                static std::atomic<uint64_t> counter{0};
                return "approval-" + std::to_string(++counter);
            }();
            req.server      = server_name;
            req.tool        = call.name;
            req.args_json   = call.arguments_json;
            req.description = "Tool call: " + server_name + "." + call.name;
            req.destructive = call.name.find("write")  != std::string::npos ||
                              call.name.find("delete")  != std::string::npos ||
                              call.name.find("run")     != std::string::npos ||
                              call.name.find("exec")    != std::string::npos;

            auto future = approval_cb(req);
            if (!future.get()) return std::nullopt;
        }

        // Unified dispatch through hooks.
        return dispatch_tool(server_name, call.name, call.arguments_json);
    };
}

} // namespace neurons_service
