#include "service_runner.h"

#include <cstdlib>
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
    uint16_t    port      = 50051;
    int         http_port = 8080;
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

    return neurons_service::run(port, http_port, model_path);
}
