#include "service_runner.h"
#include "neurons_service.h"
#include "http_server.h"
#include "logger.h"
#include "compute/core/compute_backend.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace neurons_service {

int run(uint16_t grpc_port, int http_port, const std::string& model_path) {
    std::string models_dir;
    if (const char* home = std::getenv("HOME")) {
        auto neurons_dir = std::filesystem::path(home) / ".neurons";
        models_dir = (neurons_dir / "models").string();
        try {
            std::filesystem::create_directories(neurons_dir / "models");
            std::filesystem::create_directories(neurons_dir / "chats");
        } catch (...) {}
    } else {
        models_dir = "./.neurons/models";
    }

    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    NeuronsServiceImpl service(models_dir, nullptr);

    // Initialize MCP tool servers so the gRPC/GUI path can dispatch tool calls.
    service.mcp_manager().load_config();
    service.mcp_manager().load_permissions();
    service.mcp_manager().connect_enabled();

    grpc::ServerBuilder builder;
    const std::string address = "0.0.0.0:" + std::to_string(grpc_port);
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "Failed to start server on " << address << "\n";
        return 1;
    }

    std::cout << "neurons-service listening on " << address << "\n" << std::flush;

    std::unique_ptr<OpenAiHttpServer> http_server;
    if (http_port > 0) {
        http_server = std::make_unique<OpenAiHttpServer>(service, Logger::global());
        if (http_server->start(http_port)) {
            service.set_http_port(http_port);
            std::cout << "OpenAI HTTP server on port " << http_port
                      << " — base URL: http://localhost:" << http_port << "/v1\n" << std::flush;
        } else {
            http_server.reset();
        }
    }

    // MLX must be initialised on the main thread (Metal requires it).
    // The gRPC thread pool is already running by this point — that ordering is intentional.
    auto backend_result = compute::BackendFactory::create(compute::BackendType::MLX);
    if (!backend_result.has_value()) {
        std::cerr << "Failed to create backend: " << backend_result.error().message << "\n";
        return 1;
    }
    auto init_result = (*backend_result)->initialize();
    if (!init_result.has_value()) {
        std::cerr << "Failed to initialize backend: " << init_result.error().message << "\n";
        return 1;
    }
    service.set_backend(std::move(*backend_result));
    std::cout << "Backend initialized\n" << std::flush;

    if (!model_path.empty()) {
        std::cout << "Loading model: " << model_path << " ...\n" << std::flush;
        std::string load_error;
        if (service.load_model_internal(model_path, load_error)) {
            std::cout << "Model loaded successfully\n" << std::flush;
        } else {
            std::cerr << "Warning: model load failed: " << load_error << "\n" << std::flush;
        }
    }

    server->Wait();
    return 0;
}

} // namespace neurons_service
