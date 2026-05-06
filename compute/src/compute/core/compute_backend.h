#pragma once

#include "compute_types.h"
#include <memory>
#include <string>
#include <vector>

namespace compute {

// Thin lifecycle handle for compute backends.
// Math operations are performed directly via backend-native APIs (e.g. mx::array for MLX).
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    // Backend identification
    virtual BackendType type() const = 0;
    virtual std::string name() const = 0;
    virtual bool is_available() const = 0;

    // Lifecycle
    virtual Result<void> initialize() = 0;
    virtual void cleanup() = 0;

    // Performance hints
    virtual size_t preferred_batch_size() const { return 1024; }
    virtual bool supports_async() const { return false; }
};

// Factory for creating backends
class BackendFactory {
public:
    static Result<std::unique_ptr<ComputeBackend>> create(BackendType type);
    static std::vector<BackendType> available_backends();
    static BackendType best_available_backend();
};

// Backend manager — singleton that manages backend lifecycle
class BackendManager {
public:
    static BackendManager& instance();

    Result<void> initialize();
    void cleanup();

    ComputeBackend* get_backend(BackendType type);
    ComputeBackend* get_default_backend();

private:
    BackendManager() = default;
    std::vector<std::unique_ptr<ComputeBackend>> backends_;
    ComputeBackend* default_backend_ = nullptr;
    bool initialized_ = false;

    void create_available_backends();
};

} // namespace compute
