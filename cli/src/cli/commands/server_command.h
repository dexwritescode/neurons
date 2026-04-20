#pragma once

#include "base_command.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

// Starts the neurons-service binary (HTTP + gRPC) as a subprocess,
// streaming its logs to stdout. The binary must be in the same directory
// as the cli executable, or on PATH.
class ServerCommand : public BaseCommand {
public:
    explicit ServerCommand(NeuronsConfig* config);
    ~ServerCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    NeuronsConfig* config_;
    int  grpc_port_ = 50051;
    int  http_port_ = 8080;
    std::string model_;
};

} // namespace neurons::cli
