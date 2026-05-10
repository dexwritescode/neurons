#include "server_command.h"
#include "service_runner.h"

#include <iostream>

namespace neurons::cli {

ServerCommand::ServerCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "server";
    description_  = "Start the Neurons inference server (HTTP + gRPC)";
}

void ServerCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);
    cmd->add_option("--grpc-port", grpc_port_, "gRPC port (default: 50051)");
    cmd->add_option("--http-port", http_port_, "OpenAI HTTP port (default: 8080; 0 = disabled)");
    cmd->add_option("--model,-m",  model_,     "Model to auto-load on startup");
    cmd->callback([this]() { std::exit(execute()); });
}

int ServerCommand::execute() {
    (void)config_;
    std::cout << "Starting inference server (gRPC :" << grpc_port_
              << ", HTTP :" << http_port_ << ")\n";
    std::cout << "OpenAI endpoint: http://localhost:" << http_port_ << "/v1\n\n";
    return neurons_service::run(static_cast<uint16_t>(grpc_port_), http_port_, model_);
}

} // namespace neurons::cli
