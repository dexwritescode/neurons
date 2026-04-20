#pragma once

#include "base_command.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class NodeCommand : public BaseCommand {
public:
    explicit NodeCommand(NeuronsConfig* config);
    ~NodeCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    NeuronsConfig* config_;
    std::string subcommand_;

    // add options
    std::string add_name_;
    std::string add_host_;
    int add_port_ = 50051;

    // remove / use options
    std::string target_id_;
};

} // namespace neurons::cli
