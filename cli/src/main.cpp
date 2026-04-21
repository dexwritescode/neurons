#include <iostream>
#include <memory>
#include "cli/core/cli_app.h"
#include "cli/commands/download_command.h"
#include "cli/commands/list_command.h"
#include "cli/commands/load_command.h"
#include "cli/commands/search_command.h"
#include "cli/commands/config_command.h"
#include "cli/commands/token_command.h"
#include "cli/commands/node_command.h"
#include "cli/commands/chat_command.h"
#include "cli/commands/server_command.h"
#include "cli/commands/mcp_command.h"
#include "cli/config/neurons_config.h"

int main(int argc, char** argv) {
    auto config = std::make_unique<neurons::cli::NeuronsConfig>();

    neurons::cli::CliApp app;

    app.add_command(std::make_unique<neurons::cli::DownloadCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::ListCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::LoadCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::SearchCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::ConfigCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::TokenCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::NodeCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::ChatCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::ServerCommand>(config.get()));
    app.add_command(std::make_unique<neurons::cli::McpCommand>(config.get()));

    app.setup();
    return app.run(argc, argv);
}
