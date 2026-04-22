#include "mcp_command.h"
#include "mcp/mcp_client.h"
#include "mcp/mcp_manager.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace neurons::cli {

namespace fs = std::filesystem;

// ── Constructor ───────────────────────────────────────────────────────────────

McpCommand::McpCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "mcp";
    description_  = "Manage Model Context Protocol (MCP) servers";
}

// ── setup_command ─────────────────────────────────────────────────────────────

void McpCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);

    // ── mcp add ───────────────────────────────────────────────────────────────
    auto* add = cmd->add_subcommand("add", "Register an MCP server");
    add->add_option("name", add_name_, "Server name (unique)")->required();
    add->add_option("--command,-c", add_command_,
                    "Executable to launch (stdio transport)")
       ->envname("MCP_COMMAND");
    add->add_option("--arg,-a", add_args_,
                    "Arguments for the command (repeatable)");
    add->add_option("--url,-u", add_url_,
                    "Endpoint URL (SSE transport)");
    add->add_flag("--sse", add_sse_,
                  "Use SSE transport instead of stdio");
    add->add_option("--env,-e", add_env_,
                    "Environment variables as KEY=VALUE (repeatable)");
    add->callback([this]() { subcommand_ = "add"; });

    // ── mcp remove ────────────────────────────────────────────────────────────
    auto* rem = cmd->add_subcommand("remove", "Unregister an MCP server");
    rem->add_option("name", target_name_, "Server name")->required();
    rem->callback([this]() { subcommand_ = "remove"; });

    // ── mcp list ──────────────────────────────────────────────────────────────
    auto* lst = cmd->add_subcommand("list", "List configured MCP servers");
    lst->callback([this]() { subcommand_ = "list"; });

    // ── mcp test ──────────────────────────────────────────────────────────────
    auto* tst = cmd->add_subcommand("test",
        "Connect to a server and list its tools");
    tst->add_option("name", target_name_, "Server name")->required();
    tst->callback([this]() { subcommand_ = "test"; });

    cmd->require_subcommand(1);
    cmd->callback([this]() { std::exit(execute()); });
}

// ── execute ───────────────────────────────────────────────────────────────────

int McpCommand::execute() {
    if (subcommand_ == "add")    return do_add();
    if (subcommand_ == "remove") return do_remove();
    if (subcommand_ == "list")   return do_list();
    if (subcommand_ == "test")   return do_test();
    return 1;
}

// ── do_add ────────────────────────────────────────────────────────────────────

int McpCommand::do_add() {
    if (add_name_.empty()) {
        std::cerr << "Server name is required.\n"; return 1;
    }
    if (!add_sse_ && add_command_.empty()) {
        std::cerr << "Stdio transport requires --command. "
                     "Use --sse for SSE transport.\n";
        return 1;
    }
    if (add_sse_ && add_url_.empty()) {
        std::cerr << "SSE transport requires --url.\n"; return 1;
    }

    neurons_service::McpServerConfig cfg;
    cfg.name      = add_name_;
    cfg.transport = add_sse_ ? neurons_service::McpTransport::Sse
                             : neurons_service::McpTransport::Stdio;
    cfg.command   = add_command_;
    cfg.args      = add_args_;
    cfg.url       = add_url_;
    cfg.enabled   = true;

    for (const auto& kv : add_env_) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) {
            std::cerr << "Invalid --env value (expected KEY=VALUE): " << kv << "\n";
            return 1;
        }
        cfg.env[kv.substr(0, eq)] = kv.substr(eq + 1);
    }

    neurons_service::McpManager mgr;
    mgr.load_config();
    mgr.add_server(cfg);
    mgr.save_config();

    std::cout << "Added MCP server \"" << add_name_ << "\".\n";
    return 0;
}

// ── do_remove ─────────────────────────────────────────────────────────────────

int McpCommand::do_remove() {
    neurons_service::McpManager mgr;
    mgr.load_config();
    if (!mgr.remove_server(target_name_)) {
        std::cerr << "Server \"" << target_name_ << "\" not found.\n";
        return 1;
    }
    mgr.save_config();
    std::cout << "Removed MCP server \"" << target_name_ << "\".\n";
    return 0;
}

// ── do_list ───────────────────────────────────────────────────────────────────

int McpCommand::do_list() {
    neurons_service::McpManager mgr;
    mgr.load_config();
    const auto servers = mgr.list_servers();

    if (servers.empty()) {
        std::cout << "No MCP servers configured.\n"
                     "Add one with: neurons mcp add <name> --command <cmd>\n";
        return 0;
    }

    // Simple table output
    const std::string sep(60, '-');
    std::cout << sep << "\n";
    std::printf("%-20s %-8s %-6s  %s\n",
                "NAME", "TRANSPORT", "STATUS", "COMMAND/URL");
    std::cout << sep << "\n";

    for (const auto& s : servers) {
        const std::string transport =
            (s.transport == neurons_service::McpTransport::Sse) ? "sse" : "stdio";
        const std::string status = s.enabled ? "on" : "off";
        std::string endpoint = s.transport == neurons_service::McpTransport::Sse
            ? s.url : s.command;
        if (!s.args.empty()) {
            for (const auto& a : s.args) endpoint += " " + a;
        }
        std::printf("%-20s %-8s %-6s  %s\n",
                    s.name.c_str(), transport.c_str(),
                    status.c_str(), endpoint.c_str());
    }
    std::cout << sep << "\n";
    return 0;
}

// ── do_test ───────────────────────────────────────────────────────────────────

int McpCommand::do_test() {
    neurons_service::McpManager mgr;
    mgr.load_config();

    const auto servers = mgr.list_servers();
    const neurons_service::McpServerConfig* cfg = nullptr;
    for (const auto& s : servers) {
        if (s.name == target_name_) { cfg = &s; break; }
    }
    if (!cfg) {
        std::cerr << "Server \"" << target_name_ << "\" not found.\n"; return 1;
    }

    std::cout << "Connecting to \"" << target_name_ << "\"... " << std::flush;
    auto client = neurons_service::McpClient::create(*cfg);
    const auto err = client->connect();
    if (!err.empty()) {
        std::cout << "FAILED\n";
        std::cerr << err << "\n";
        return 1;
    }
    std::cout << "OK\n";

    std::vector<neurons_service::ToolDef> tools;
    const auto list_err = client->list_tools(tools);
    if (!list_err.empty()) {
        std::cerr << "tools/list failed: " << list_err << "\n";
        return 1;
    }

    if (tools.empty()) {
        std::cout << "Server exposes no tools.\n";
        return 0;
    }

    std::cout << "\n" << tools.size() << " tool(s):\n\n";
    for (const auto& t : tools) {
        std::cout << "  " << t.name << "\n";
        if (!t.description.empty())
            std::cout << "    " << t.description << "\n";
    }
    return 0;
}

} // namespace neurons::cli
