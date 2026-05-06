#pragma once

#include <expected>
#include <string>

namespace compute {

// Error handling
enum class ErrorCode {
    Success,
    InvalidInput,
    InvalidModel,
    BackendNotAvailable,
    ComputeError,
    UnknownError
};

struct Error {
    ErrorCode code;
    std::string message;

    Error(ErrorCode c, std::string msg) : code(c), message(std::move(msg)) {}
};

template<typename T>
using Result = std::expected<T, Error>;

// Backend types
enum class BackendType {
    MLX,    // Apple MLX framework (Metal-accelerated on Apple Silicon)
    Auto    // Let system choose best available
};

// Forward declarations
class ComputeBackend;

} // namespace compute
