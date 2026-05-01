#include "compute_backend.h"
#include "backends/mlx/mlx_backend.h"
#include <vector>
#include <memory>

namespace compute {

// BackendFactory implementation
Result<std::unique_ptr<ComputeBackend>> BackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::Auto:
        case BackendType::MLX:
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
        {
            auto mlx_backend = std::make_unique<MLXBackend>();
            if (mlx_backend->is_available()) {
                auto init_result = mlx_backend->initialize();
                if (init_result) {
                    return std::move(mlx_backend);
                }
                return std::unexpected(Error{ErrorCode::BackendNotAvailable, "MLX backend failed to initialize"});
            }
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "MLX not available"});
        }
#else
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "MLX backend only available on Apple Silicon with MLX enabled"});
#endif

        case BackendType::Metal:
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Metal backend not yet implemented; use MLX"});

        default:
            return std::unexpected(Error{ErrorCode::UnknownError, "Unknown backend type"});
    }
}

std::vector<BackendType> BackendFactory::available_backends() {
    std::vector<BackendType> backends;
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    backends.push_back(BackendType::MLX);
#endif
    return backends;
}

BackendType BackendFactory::best_available_backend() {
    auto available = available_backends();
    return available.empty() ? BackendType::Auto : available[0];
}

// BackendManager implementation
BackendManager& BackendManager::instance() {
    static BackendManager instance;
    return instance;
}

Result<void> BackendManager::initialize() {
    if (initialized_) {
        return {};
    }
    create_available_backends();
    if (!backends_.empty()) {
        default_backend_ = backends_[0].get();
    }
    initialized_ = true;
    return {};
}

void BackendManager::cleanup() {
    if (!initialized_) return;
    for (auto& backend : backends_) {
        backend->cleanup();
    }
    backends_.clear();
    default_backend_ = nullptr;
    initialized_ = false;
}

ComputeBackend* BackendManager::get_backend(BackendType type) {
    if (!initialized_) {
        auto init_result = initialize();
        if (!init_result) return nullptr;
    }
    for (auto& backend : backends_) {
        if (backend->type() == type) {
            return backend.get();
        }
    }
    return nullptr;
}

ComputeBackend* BackendManager::get_default_backend() {
    if (!initialized_) {
        auto init_result = initialize();
        if (!init_result) return nullptr;
    }
    return default_backend_;
}

void BackendManager::create_available_backends() {
    for (auto type : BackendFactory::available_backends()) {
        auto backend_result = BackendFactory::create(type);
        if (backend_result) {
            auto& backend = *backend_result;
            auto init_result = backend->initialize();
            if (init_result) {
                backends_.push_back(std::move(backend));
            }
        }
    }
}

} // namespace compute
