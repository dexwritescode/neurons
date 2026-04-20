#pragma once

#include "base_command.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class TokenCommand : public BaseCommand {
public:
    explicit TokenCommand(NeuronsConfig* config);
    ~TokenCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    NeuronsConfig* config_;
    std::string subcommand_;
    std::string token_value_;
};

} // namespace neurons::cli
