#pragma once

#include "compute_types.h"
#include "tensor.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <span>

namespace compute {

// Abstract base for all compute backends using new Tensor-based API
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
    
    // Tensor creation (directly in backend-optimal format)
    
    /**
     * Create tensor from raw data (single conversion to backend format)
     */
    virtual Tensor create_tensor(std::span<const float> data, std::vector<size_t> shape) = 0;
    
    /**
     * Create uninitialized tensor (backend allocates optimal memory)
     */
    virtual Tensor create_tensor(std::vector<size_t> shape) = 0;
    
    /**
     * Wrap existing backend-native tensor (e.g., from model loading)
     * @param native_tensor Backend-specific tensor pointer (mx::array*, etc.)
     * @param shape Tensor shape
     */
    virtual Tensor wrap_native_tensor(void* native_tensor, std::vector<size_t> shape) = 0;
    
    // Core operations (stay in backend-native format)
    
    /**
     * Dot product of two vectors (returns scalar tensor)
     */
    virtual Tensor dot_product(const Tensor& a, const Tensor& b) = 0;
    
    /**
     * Add scalar to all elements of tensor
     */
    virtual Tensor matrix_scalar_add(const Tensor& input, float scalar) = 0;
    
    /**
     * Matrix multiplication
     */
    virtual Result<Tensor> matmul(const Tensor& a, const Tensor& b) = 0;

    /**
     * Dequantize a quantized tensor back to float
     * @param w Quantized weight matrix
     * @param scales Scale factors
     * @param biases Zero-point biases
     * @param group_size Elements per quantization group
     * @param bits Quantization bits
     */
    virtual Result<Tensor> dequantize(
        const Tensor& w,
        const Tensor& scales,
        const Tensor& biases,
        int group_size = 64,
        int bits = 4
    ) = 0;

    /**
     * Quantized matrix multiplication (universal quantization support)
     * @param x Input activation tensor
     * @param w Quantized weight matrix
     * @param scales Scale factors for dequantization
     * @param biases Optional bias terms (can be nullptr)
     * @param transpose Whether to transpose w during computation
     * @param group_size Elements per quantization group (default 64)
     * @param bits Quantization bits (default 4, supports 1-8)
     * @param mode Quantization mode ("affine" default, "symmetric")
     */
    virtual Result<Tensor> quantized_matmul(
        const Tensor& x,
        const Tensor& w,
        const Tensor& scales,
        const Tensor* biases = nullptr,
        bool transpose = true,
        int group_size = 64,
        int bits = 4,
        const std::string& mode = "affine"
    ) = 0;

    /**
     * Element-wise addition
     */
    virtual Result<Tensor> add(const Tensor& a, const Tensor& b) = 0;

    /**
     * Element-wise multiplication
     */
    virtual Result<Tensor> multiply(const Tensor& a, const Tensor& b) = 0;

    /**
     * Softmax along specified dimension
     */
    virtual Result<Tensor> softmax(const Tensor& input, int dim = -1) = 0;

    /**
     * SiLU (Swish) activation function: x * sigmoid(x)
     */
    virtual Result<Tensor> silu(const Tensor& input) = 0;

    /**
     * GELU activation (tanh approximation): used by Gemma GeGLU FFN
     */
    virtual Result<Tensor> gelu(const Tensor& input) = 0;

    /**
     * Sigmoid activation: 1 / (1 + exp(-x))
     */
    virtual Result<Tensor> sigmoid(const Tensor& input) = 0;

    /**
     * 1D convolution (channels-last layout: input [N, L, C_in], weight [C_out, kW, C_in/groups])
     * @param input   [N, L, C_in] or [L, C_in] (batch dim added if missing)
     * @param weight  [C_out, kernel_size, C_in/groups]
     * @param stride  Convolution stride (default 1)
     * @param padding Zero-padding applied symmetrically on both sides (default 0)
     * @param groups  Number of groups for grouped/depthwise conv (default 1)
     */
    virtual Result<Tensor> conv1d(
        const Tensor& input,
        const Tensor& weight,
        int stride  = 1,
        int padding = 0,
        int groups  = 1) = 0;

    /**
     * Transpose tensor (general transpose - reverses dimension order)
     */
    virtual Result<Tensor> transpose(const Tensor& input) = 0;

    /**
     * Swap two axes of tensor (for attention mechanism)
     */
    virtual Result<Tensor> swapaxes(const Tensor& input, int axis1, int axis2) = 0;

    /**
     * Reshape tensor to new shape (total elements must remain the same)
     */
    virtual Result<Tensor> reshape(const Tensor& input, const std::vector<size_t>& new_shape) = 0;

    /**
     * Concatenate tensors along specified axis
     */
    virtual Result<Tensor> concatenate(const std::vector<Tensor>& tensors, int axis = 0) = 0;

    // Additional tensor operations for transformer inference

    /**
     * Compute mean along specified axis
     * @param input Input tensor
     * @param axis Axis to reduce (default -1 for last axis)
     * @param keepdims Whether to keep reduced dimension (default false)
     */
    virtual Result<Tensor> mean(const Tensor& input, int axis = -1, bool keepdims = false) = 0;

    /**
     * Reciprocal square root: 1/sqrt(x)
     * @param input Input tensor
     */
    virtual Result<Tensor> rsqrt(const Tensor& input) = 0;

    /**
     * Extract slice from tensor along specified axis
     * @param input Input tensor
     * @param start Start index
     * @param stop Stop index (exclusive)
     * @param axis Axis to slice along (default 0)
     */
    virtual Result<Tensor> slice(const Tensor& input, int start, int stop, int axis = 0) = 0;

    /**
     * Repeat tensor elements along specified axis
     * @param input Input tensor
     * @param repeats Number of repetitions
     * @param axis Axis to repeat along
     */
    virtual Result<Tensor> repeat(const Tensor& input, int repeats, int axis) = 0;

    /**
     * Upper triangular matrix (for causal masking)
     * @param input Input tensor
     * @param k Diagonal offset (0 = main diagonal, >0 = above, <0 = below)
     */
    virtual Result<Tensor> triu(const Tensor& input, int k = 0) = 0;

    // Optimized transformer operations

    /**
     * RMSNorm layer normalization (fused implementation)
     * @param input Input tensor
     * @param weight Scale weights
     * @param eps Epsilon for numerical stability
     */
    virtual Result<Tensor> rms_norm(const Tensor& input, const Tensor& weight, float eps) = 0;

    /**
     * Rotary Position Embedding (RoPE) - fused implementation
     * @param input Input tensor
     * @param dims Number of dimensions to apply RoPE to
     * @param theta Base for frequency computation (typically 10000.0)
     * @param offset Position offset for the sequence
     */
    virtual Result<Tensor> rope(const Tensor& input, int dims, float theta, int offset) = 0;

    /**
     * Scaled dot-product attention (fused implementation)
     * @param queries Query tensor
     * @param keys Key tensor
     * @param values Value tensor
     * @param scale Scaling factor (typically 1/sqrt(head_dim))
     * @param mask Attention mask type ("causal" for autoregressive, "" for none)
     */
    virtual Result<Tensor> scaled_dot_product_attention(
        const Tensor& queries,
        const Tensor& keys,
        const Tensor& values,
        float scale,
        const std::string& mask = ""
    ) = 0;

    // Utility operations
    
    /**
     * Extract tensor data to CPU buffer (triggers computation for lazy backends)
     * @param tensor Source tensor
     * @param output Destination CPU buffer
     */
    virtual Result<void> extract(const Tensor& tensor, std::span<float> output) = 0;
    
    /**
     * Force evaluation of all pending operations (for lazy backends)
     */
    virtual Result<void> evaluate_all() = 0;
    
    // Model loading integration (implementation in Phase 2+)

    /**
     * Load model tensors from file (backend-specific format)
     * @param path Path to model file (.safetensors, .gguf, etc.)
     * @return Map of tensor names to tensors
     */
    virtual std::unordered_map<std::string, Tensor> load_model(const std::string& path) = 0;
    
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

// Backend manager - singleton that manages backend lifecycle
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