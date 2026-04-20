#include "compute_backend.h"
#include "../backends/simd/neon_backend.h"
#if defined(__APPLE__) && defined(__aarch64__)
#include "../backends/metal/metal_backend.h"
#if defined(MLX_BACKEND_ENABLED)
#include "../backends/mlx/mlx_backend.h"
#endif
#endif
#include <vector>
#include <memory>

namespace compute {

// BackendFactory implementation
Result<std::unique_ptr<ComputeBackend>> BackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::SimdNeon:
#ifdef __ARM_NEON
            return std::make_unique<NeonBackend>();
#else
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "NEON backend only available on ARM platforms with NEON support"});
#endif
            
        case BackendType::Auto: {
            // Try Metal first (GPU acceleration), then NEON (CPU SIMD)
#if defined(__APPLE__) && defined(__aarch64__)
            auto metal_backend = std::make_unique<MetalBackend>();
            if (metal_backend->is_available()) {
                return std::move(metal_backend);
            }
#endif
#ifdef __ARM_NEON
            auto neon_backend = std::make_unique<NeonBackend>();
            if (neon_backend->is_available()) {
                auto init_result = neon_backend->initialize();
                if (init_result) {
                    return std::move(neon_backend);
                }
            }
#endif
            // No viable backend found
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, 
                "No compute backend available. LLM inference requires SIMD support (NEON/Metal on this platform)"});
        }
        
        case BackendType::Metal:
#if defined(__APPLE__) && defined(__aarch64__)
            {
                auto metal_backend = std::make_unique<MetalBackend>();
                if (metal_backend->is_available()) {
                    auto init_result = metal_backend->initialize();
                    if (init_result) {
                        return std::move(metal_backend);
                    }
                    return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Metal backend failed to initialize"});
                }
                return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Metal device not available"});
            }
#else
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Metal backend only available on Apple Silicon"});
#endif

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
            
        default:
            return std::unexpected(Error{ErrorCode::UnknownError, "Unknown backend type"});
    }
}

std::vector<BackendType> BackendFactory::available_backends() {
    std::vector<BackendType> backends;
    
    // Check MLX availability (Apple Silicon with MLX enabled)
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    backends.push_back(BackendType::MLX);
#endif
    
    // Check Metal availability (Apple Silicon only)
#if defined(__APPLE__) && defined(__aarch64__)
    backends.push_back(BackendType::Metal);
#endif
    
    // Check NEON availability at compile time
#ifdef __ARM_NEON
    backends.push_back(BackendType::SimdNeon);
#endif
    
    return backends;
}

BackendType BackendFactory::best_available_backend() {
    // For now, just return the first available
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
        return {}; // Already initialized
    }
    
    // Create available backends
    create_available_backends();
    
    // Set default backend
    if (!backends_.empty()) {
        default_backend_ = backends_[0].get();
    }
    
    initialized_ = true;
    return {};
}

void BackendManager::cleanup() {
    if (!initialized_) return;
    
    // Clean up all backends
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
    
    // Find backend of requested type
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
    // Create all available backends
    auto available_types = BackendFactory::available_backends();
    
    for (auto type : available_types) {
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
