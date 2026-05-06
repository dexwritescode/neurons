#include "mlx_backend.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

MLXBackend::MLXBackend() : m_initialized(false) {}

MLXBackend::~MLXBackend() {
    cleanup();
}

BackendType MLXBackend::type() const {
    return BackendType::MLX;
}

std::string MLXBackend::name() const {
    return "MLX (Apple Silicon)";
}

bool MLXBackend::is_available() const {
    try {
        auto test_array = mx::array({1.0f});
        return true;
    } catch (...) {
        return false;
    }
}

Result<void> MLXBackend::initialize() {
    if (m_initialized) return {};

    if (!is_available())
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "MLX backend not available"});

    try {
        mx::set_default_device(mx::Device::gpu);
        auto test = mx::array({1.0f, 2.0f});
        mx::eval(test);
        m_initialized = true;
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
            std::string("MLX initialization failed: ") + e.what()});
    }
}

void MLXBackend::cleanup() {
    if (m_initialized) {
        mx::clear_cache();
        m_initialized = false;
    }
}

size_t MLXBackend::preferred_batch_size() const { return 1024; }
bool MLXBackend::supports_async() const { return false; }

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
