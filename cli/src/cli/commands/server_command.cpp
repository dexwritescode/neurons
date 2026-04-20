#include "server_command.h"

#include <filesystem>
#include <iostream>
#include <unistd.h>
#include <cstdlib>

#if defined(__APPLE__)
#  include <mach-o/dyld.h>
#endif

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
    // Look for neurons-service next to this binary, or on PATH.
    std::filesystem::path self;
    {
        char buf[4096] = {};
#if defined(__APPLE__)
        uint32_t size = static_cast<uint32_t>(sizeof(buf));
        _NSGetExecutablePath(buf, &size);
#elif defined(__linux__)
        readlink("/proc/self/exe", buf, sizeof(buf) - 1);
#endif
        self = std::filesystem::path(buf).parent_path();
    }

    std::string service_bin;
    const auto candidate = self / "neurons_service";
    if (std::filesystem::exists(candidate)) {
        service_bin = candidate.string();
    } else {
        // Fall back to PATH
        service_bin = "neurons_service";
    }

    // Build argv for exec
    std::string grpc_port_s = std::to_string(grpc_port_);
    std::string http_port_s = std::to_string(http_port_);

    std::vector<std::string> args_storage = {
        service_bin,
        "--port",      grpc_port_s,
        "--http-port", http_port_s,
    };
    if (!model_.empty()) {
        args_storage.push_back("--model");
        args_storage.push_back(model_);
    }

    std::vector<char*> argv;
    for (auto& s : args_storage) argv.push_back(s.data());
    argv.push_back(nullptr);

    std::cout << "Starting " << service_bin
              << " (gRPC :" << grpc_port_ << ", HTTP :" << http_port_ << ")\n";
    std::cout << "OpenAI endpoint: http://localhost:" << http_port_ << "/v1\n\n";

    execvp(argv[0], argv.data());

    // execvp only returns on error
    std::cerr << "Failed to exec '" << service_bin << "': " << strerror(errno) << "\n";
    std::cerr << "Make sure neurons_service is built and in PATH or next to the cli binary.\n";
    return 1;
}

} // namespace neurons::cli
