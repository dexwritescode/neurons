#include "neon_backend.h"
#include "simd_utils.h"
#include <arm_neon.h>
#include <algorithm>

namespace compute {

BackendType NeonBackend::type() const {
    return BackendType::SimdNeon;
}

std::string NeonBackend::name() const {
    return "NEON SIMD Backend (ARM64)";
}

bool NeonBackend::is_available() const {
#ifdef __ARM_NEON
    return true;
#else
    return false;
#endif
}

Result<void> NeonBackend::initialize() {
    if (!is_available()) {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "NEON SIMD not available on this platform"});
    }
    return {};
}

void NeonBackend::cleanup() {
    // Nothing to clean up for NEON
}

// Tensor creation methods
Tensor NeonBackend::create_tensor(std::span<const float> data, std::vector<size_t> shape) {
    auto buffer = std::make_shared<CpuBuffer>(data, BackendType::SimdNeon);
    return Tensor(buffer, shape);
}

Tensor NeonBackend::create_tensor(std::vector<size_t> shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    auto buffer = std::make_shared<CpuBuffer>(total_size, BackendType::SimdNeon);
    return Tensor(buffer, shape);
}

Tensor NeonBackend::wrap_native_tensor(void* native_tensor, std::vector<size_t> shape) {
    // For CPU backend, native tensor is std::vector<float>*
    auto* vec_ptr = static_cast<std::vector<float>*>(native_tensor);
    auto buffer = std::make_shared<CpuBuffer>(std::move(*vec_ptr), BackendType::SimdNeon);
    return Tensor(buffer, shape);
}

// Core operations using new Tensor API
Tensor NeonBackend::dot_product(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    if (!a.is_vector() || !b.is_vector()) {
        throw std::invalid_argument("Both inputs must be vectors");
    }
    
    // Get CPU buffers
    auto* buf_a = get_cpu_buffer(a);
    auto* buf_b = get_cpu_buffer(b);
    
    // Perform SIMD dot product
    float result = simd_dot_product(buf_a->data_ptr(), buf_b->data_ptr(), a.size());
    
    // Create scalar result tensor
    std::vector<float> result_data = {result};
    auto result_buffer = std::make_shared<CpuBuffer>(std::move(result_data), BackendType::SimdNeon);
    return Tensor(result_buffer, {1}); // Scalar tensor
}

Tensor NeonBackend::matrix_scalar_add(const Tensor& input, float scalar) {
    // Get input buffer
    auto* buf_input = get_cpu_buffer(input);
    
    // Create output buffer
    auto result_buffer = std::make_shared<CpuBuffer>(input.size(), BackendType::SimdNeon);
    auto* input_data = buf_input->data_ptr();
    auto* output_data = result_buffer->data_ptr();
    size_t size = input.size();
    
    // NEON-optimized scalar addition
    float32x4_t scalar_vec = vdupq_n_f32(scalar);
    size_t simd_size = size & ~3; // Round down to multiple of 4
    
    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t input_vec = vld1q_f32(&input_data[i]);
        float32x4_t result_vec = vaddq_f32(input_vec, scalar_vec);
        vst1q_f32(&output_data[i], result_vec);
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        output_data[i] = input_data[i] + scalar;
    }
    
    return Tensor(result_buffer, input.shape());
}

// Core operations with proper error handling
Result<Tensor> NeonBackend::matmul(const Tensor& a, const Tensor& b) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON matmul not implemented yet - use dot_product for vectors"});
}

Result<Tensor> NeonBackend::dequantize(
    const Tensor& /*w*/,
    const Tensor& /*scales*/,
    const Tensor& /*biases*/,
    int /*group_size*/,
    int /*bits*/
) {
    return std::unexpected(Error{ErrorCode::BackendNotAvailable,
                               "NeonBackend does not support dequantize"});
}

Result<Tensor> NeonBackend::quantized_matmul(
    const Tensor& x,
    const Tensor& w,
    const Tensor& scales,
    const Tensor* biases,
    bool transpose,
    int group_size,
    int bits,
    const std::string& mode
) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON quantized_matmul not implemented - quantization not supported on CPU backend"});
}

Result<Tensor> NeonBackend::add(const Tensor& a, const Tensor& b) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON add not implemented yet - use matrix_scalar_add"});
}

Result<Tensor> NeonBackend::multiply(const Tensor& a, const Tensor& b) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON multiply not implemented yet - use MLX backend for element-wise operations"});
}

Result<Tensor> NeonBackend::softmax(const Tensor& input, int dim) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON softmax not implemented yet"});
}

Result<Tensor> NeonBackend::silu(const Tensor& input) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON silu not implemented - use MLX backend for activation functions"});
}

Result<Tensor> NeonBackend::gelu(const Tensor& input) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON gelu not implemented - use MLX backend"});
}

Result<Tensor> NeonBackend::transpose(const Tensor& input) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON transpose not implemented yet"});
}

Result<Tensor> NeonBackend::swapaxes(const Tensor& input, int axis1, int axis2) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON swapaxes not implemented yet"});
}

Result<Tensor> NeonBackend::reshape(const Tensor& input, const std::vector<size_t>& new_shape) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON reshape not implemented yet - use MLX backend for tensor manipulation"});
}

Result<Tensor> NeonBackend::concatenate(const std::vector<Tensor>& tensors, int axis) {
    return std::unexpected(Error{ErrorCode::ComputeError, "NEON concatenate not implemented yet - use MLX backend for tensor manipulation"});
}

// Utility operations
Result<void> NeonBackend::extract(const Tensor& tensor, std::span<float> output) {
    if (tensor.backend_type() != BackendType::SimdNeon) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Tensor not from NEON backend"});
    }
    
    if (output.size() != tensor.size()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Output buffer size mismatch"});
    }
    
    auto* buf = get_cpu_buffer(tensor);
    std::copy(buf->data_ptr(), buf->data_ptr() + tensor.size(), output.begin());
    
    return {};
}

Result<void> NeonBackend::evaluate_all() {
    // CPU operations are eager - nothing to evaluate
    return {};
}

std::unordered_map<std::string, Tensor> NeonBackend::load_model(const std::string& path) {
    throw std::runtime_error("Model loading not implemented yet");
}

// Performance hints
size_t NeonBackend::preferred_batch_size() const {
    return 4096; // Optimized for NEON 4-element vectors
}

bool NeonBackend::supports_async() const {
    return false; // CPU backend doesn't need async
}

Result<Tensor> NeonBackend::mean(const Tensor& input, int axis, bool keepdims) {
    return std::unexpected(Error{ErrorCode::ComputeError, "mean() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::rsqrt(const Tensor& input) {
    return std::unexpected(Error{ErrorCode::ComputeError, "rsqrt() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::slice(const Tensor& input, int start, int stop, int axis) {
    return std::unexpected(Error{ErrorCode::ComputeError, "slice() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::repeat(const Tensor& input, int repeats, int axis) {
    return std::unexpected(Error{ErrorCode::ComputeError, "repeat() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::triu(const Tensor& input, int k) {
    return std::unexpected(Error{ErrorCode::ComputeError, "triu() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::rms_norm(const Tensor& input, const Tensor& weight, float eps) {
    return std::unexpected(Error{ErrorCode::ComputeError, "rms_norm() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::rope(const Tensor& input, int dims, float theta, int offset) {
    return std::unexpected(Error{ErrorCode::ComputeError, "rope() not implemented for NEON backend - use MLX backend"});
}

Result<Tensor> NeonBackend::scaled_dot_product_attention(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    float scale,
    const std::string& mask) {
    return std::unexpected(Error{ErrorCode::ComputeError, "scaled_dot_product_attention() not implemented for NEON backend - use MLX backend"});
}

// Private helper methods
CpuBuffer* NeonBackend::get_cpu_buffer(const Tensor& tensor) const {
    if (tensor.backend_type() != BackendType::SimdNeon) {
        throw std::invalid_argument("Tensor is not from NEON backend");
    }
    
    return static_cast<CpuBuffer*>(tensor.buffer().get());
}

std::vector<size_t> NeonBackend::compute_matmul_shape(const std::vector<size_t>& a_shape, 
                                                     const std::vector<size_t>& b_shape) const {
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Matrix dimension mismatch");
    }
    
    return {a_shape[0], b_shape[1]};
}

} // namespace compute
