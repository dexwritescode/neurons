#pragma once

#include <cstdint>
#include <string>

namespace neurons_service {

// Starts the gRPC + HTTP inference server and blocks until shutdown.
// Returns the process exit code (0 on clean shutdown).
int run(uint16_t grpc_port, int http_port, const std::string& model_path);

} // namespace neurons_service
