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

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [--port PORT] [--http-port PORT] [--model MODEL_DIR]\n"
              << "  --port PORT       gRPC port (default: 50051)\n"
              << "  --http-port PORT  OpenAI HTTP port (default: 8080; 0 = disabled)\n"
              << "  --model MODEL_DIR Model directory to auto-load on startup\n"
              << "  --help            Show this message\n";
}

int main(int argc, char** argv) {
    uint16_t    port       = 50051;
    int         http_port  = 8080;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--http-port" && i + 1 < argc) {
            http_port = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Default models directory: ~/.neurons/models/
    std::string models_dir;
    if (const char* home = std::getenv("HOME")) {
        auto neurons_dir = std::filesystem::path(home) / ".neurons";
        models_dir = (neurons_dir / "models").string();
        // Ensure data directories exist on first run
        try {
            std::filesystem::create_directories(neurons_dir / "models");
            std::filesystem::create_directories(neurons_dir / "chats");
        } catch (...) {}
    } else {
        models_dir = "./.neurons/models";
    }

    // Enable gRPC server reflection (lets grpcurl discover services)
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    // Build service without a backend yet — backend must be initialised on the
    // main thread AFTER gRPC's thread pool is created (BuildAndStart).  If we
    // call BackendFactory::create(MLX) before BuildAndStart, Metal/GCD holds
    // resources that prevent gRPC from spawning its worker threads.
    neurons_service::NeuronsServiceImpl service(models_dir, nullptr);

    grpc::ServerBuilder builder;
    const std::string address = "0.0.0.0:" + std::to_string(port);
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // Start server FIRST — gRPC thread pool is now running.
    auto server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "Failed to start server on " << address << "\n";
        return 1;
    }

    std::cout << "neurons-service listening on " << address << "\n" << std::flush;

    // Start OpenAI-compatible HTTP server (if enabled)
    std::unique_ptr<neurons_service::OpenAiHttpServer> http_server;
    if (http_port > 0) {
        http_server = std::make_unique<neurons_service::OpenAiHttpServer>(
            service, neurons_service::Logger::global());
        if (http_server->start(http_port)) {
            service.set_http_port(http_port);
            std::cout << "OpenAI HTTP server on port " << http_port
                      << " — base URL: http://localhost:" << http_port << "/v1\n" << std::flush;
        } else {
            http_server.reset();
        }
    }

    // NOW initialise MLX on the main thread (Metal requires main-thread init).
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

    // Auto-load model if specified on command line.
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
