#pragma once

#include "base_command.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

// `neurons mcp` — manage Model Context Protocol servers.
//
// Subcommands:
//   add    — register a new MCP server
//   remove — unregister a server
//   list   — show configured servers
//   test   — connect and list tools exposed by a server
class McpCommand : public BaseCommand {
public:
    explicit McpCommand(NeuronsConfig* config);
    ~McpCommand() override = default;

    void setup_command(CLI::App& app) override;
    int  execute() override;

private:
    int do_add();
    int do_remove();
    int do_list();
    int do_test();

    NeuronsConfig* config_;
    std::string subcommand_;

    // add options
    std::string add_name_;
    std::string add_command_;
    std::vector<std::string> add_args_;
    std::string add_url_;
    bool add_sse_ = false;
    std::vector<std::string> add_env_; // KEY=VALUE pairs

    // remove / test
    std::string target_name_;
};

} // namespace neurons::cli
