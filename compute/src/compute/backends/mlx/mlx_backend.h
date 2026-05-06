#pragma once

#include "../../core/compute_backend.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;

namespace compute {

class MLXBackend : public ComputeBackend {
public:
    MLXBackend();
    ~MLXBackend();

    BackendType type() const override;
    std::string name() const override;
    bool is_available() const override;

    Result<void> initialize() override;
    void cleanup() override;

private:
    bool m_initialized;
};

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
