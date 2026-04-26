#pragma once

#include "../../core/compute_backend.h"
#include "mlx_buffer.h"
#include "mlx_utils.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;

namespace compute {

// Forward declaration for MLX buffer wrapper
class MLXBuffer;

// MLX-based backend for Apple Silicon using MLX framework
class MLXBackend : public ComputeBackend {
public:
    MLXBackend();
    ~MLXBackend();
    
    BackendType type() const override;
    std::string name() const override;
    bool is_available() const override;
    
    Result<void> initialize() override;
    void cleanup() override;
    
    // Tensor creation
    Tensor create_tensor(std::span<const float> data, std::vector<size_t> shape) override;
    Tensor create_tensor(std::vector<size_t> shape) override;
    Tensor wrap_native_tensor(void* native_tensor, std::vector<size_t> shape) override;
    
    // Core operations using MLX
    Tensor dot_product(const Tensor& a, const Tensor& b) override;
    Tensor matrix_scalar_add(const Tensor& input, float scalar) override;
    
    // Core operations with proper error handling
    Result<Tensor> matmul(const Tensor& a, const Tensor& b) override;
    Result<Tensor> dequantize(
        const Tensor& w,
        const Tensor& scales,
        const Tensor& biases,
        int group_size = 64,
        int bits = 4
    ) override;
    Result<Tensor> quantized_matmul(
        const Tensor& x,
        const Tensor& w,
        const Tensor& scales,
        const Tensor* biases = nullptr,
        bool transpose = true,
        int group_size = 64,
        int bits = 4,
        const std::string& mode = "affine"
    ) override;
    Result<Tensor> add(const Tensor& a, const Tensor& b) override;
    Result<Tensor> multiply(const Tensor& a, const Tensor& b) override;
    Result<Tensor> softmax(const Tensor& input, int dim = -1) override;
    Result<Tensor> silu(const Tensor& input) override;
    Result<Tensor> gelu(const Tensor& input) override;
    Result<Tensor> sigmoid(const Tensor& input) override;
    Result<Tensor> conv1d(const Tensor& input, const Tensor& weight,
                          int stride = 1, int padding = 0, int groups = 1) override;
    Result<Tensor> transpose(const Tensor& input) override;
    Result<Tensor> swapaxes(const Tensor& input, int axis1, int axis2) override;
    Result<Tensor> reshape(const Tensor& input, const std::vector<size_t>& new_shape) override;
    Result<Tensor> concatenate(const std::vector<Tensor>& tensors, int axis = 0) override;

    // Additional tensor operations for transformer inference
    Result<Tensor> mean(const Tensor& input, int axis = -1, bool keepdims = false) override;
    Result<Tensor> rsqrt(const Tensor& input) override;
    Result<Tensor> slice(const Tensor& input, int start, int stop, int axis = 0) override;
    Result<Tensor> repeat(const Tensor& input, int repeats, int axis) override;
    Result<Tensor> triu(const Tensor& input, int k = 0) override;

    // Optimized transformer operations using MLX fast implementations
    Result<Tensor> rms_norm(const Tensor& input, const Tensor& weight, float eps) override;
    Result<Tensor> rope(const Tensor& input, int dims, float theta, int offset) override;
    Result<Tensor> scaled_dot_product_attention(
        const Tensor& queries,
        const Tensor& keys,
        const Tensor& values,
        float scale,
        const std::string& mask = ""
    ) override;

    // Utility operations
    Result<void> extract(const Tensor& tensor, std::span<float> output) override;
    Result<void> evaluate_all() override;
    
    // Model loading (stub for now)
    std::unordered_map<std::string, Tensor> load_model(const std::string& path) override;
    
    // Performance hints optimized for MLX
    size_t preferred_batch_size() const override;
    bool supports_async() const override;

private:
    bool m_initialized;
};

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)