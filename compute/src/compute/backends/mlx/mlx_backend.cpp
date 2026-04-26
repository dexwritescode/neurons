#include "mlx_backend.h"
#include "../../model/model_loader.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

namespace compute {

MLXBackend::MLXBackend() : m_initialized(false) {
}

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
    // Simple check - just try to create a basic MLX array
    // If MLX is working, this should succeed
    try {
        auto test_array = mx::array({1.0f});
        return true;
    } catch (...) {
        return false;
    }
}

Result<void> MLXBackend::initialize() {
    if (m_initialized) {
        return {};
    }
    
    if (!is_available()) {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, 
                                   "MLX backend not available"});
    }
    
    try {
        // Set MLX to use GPU by default
        mx::set_default_device(mx::Device::gpu);
        
        // Test basic functionality
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
        // Clear MLX memory cache
        mx::clear_cache();
        m_initialized = false;
    }
}

// Tensor creation
Tensor MLXBackend::create_tensor(std::span<const float> data, std::vector<size_t> shape) {
    auto buffer = std::make_shared<MLXBuffer>(data, shape);
    return Tensor(buffer, shape);
}

Tensor MLXBackend::create_tensor(std::vector<size_t> shape) {
    auto buffer = std::make_shared<MLXBuffer>(shape);
    return Tensor(buffer, shape);
}

Tensor MLXBackend::wrap_native_tensor(void* native_tensor, std::vector<size_t> shape) {
    // For MLX backend, native tensor is mx::array*
    auto* array_ptr = static_cast<mx::array*>(native_tensor);
    auto buffer = std::make_shared<MLXBuffer>(*array_ptr);
    return Tensor(buffer, shape);
}

// Core operations
Tensor MLXBackend::dot_product(const Tensor& a, const Tensor& b) {
    // Validate inputs
    if (a.size() != b.size()) {
        throw std::runtime_error("MLX dot_product: Vector sizes must match");
    }

    if (a.shape().size() != 1 || b.shape().size() != 1) {
        throw std::runtime_error("MLX dot_product: Both inputs must be vectors");
    }

    // Use MLX inner product for vectors - creates lazy computation graph
    mx::array result = mx::inner(a.to_mlx(), b.to_mlx());

    auto result_buffer = std::make_shared<MLXBuffer>(result);
    return Tensor(result_buffer, {1});
}

Tensor MLXBackend::matrix_scalar_add(const Tensor& input, float scalar) {
    // Use MLX broadcasting for scalar addition - creates lazy computation graph
    mx::array result = input.to_mlx() + scalar;

    auto result_buffer = std::make_shared<MLXBuffer>(result);
    return Tensor(result_buffer, input.shape());
}

// Core operations with proper error handling
Result<Tensor> MLXBackend::matmul(const Tensor& a, const Tensor& b) {
    // Validate input tensors are 2D matrices
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX matmul: only 2D matrices supported"});
    }

    // Check dimension compatibility: (m, k) x (k, n) -> (m, n)
    if (a.shape()[1] != b.shape()[0]) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX matmul: incompatible matrix dimensions"});
    }

    try {
        mx::array result = mx::matmul(a.to_mlx(), b.to_mlx());

        // Get result shape from MLX array
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX matmul failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::dequantize(
    const Tensor& w,
    const Tensor& scales,
    const Tensor& biases,
    int group_size,
    int bits
) {
    if (w.backend_type() != BackendType::MLX ||
        scales.backend_type() != BackendType::MLX ||
        biases.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "MLX dequantize: all tensors must be from MLX backend"});
    }

    try {
        mx::array result = mx::dequantize(
            w.to_mlx(), scales.to_mlx(), biases.to_mlx(), group_size, bits);
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());
        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX dequantize failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::quantized_matmul(
    const Tensor& x,
    const Tensor& w,
    const Tensor& scales,
    const Tensor* biases,
    bool transpose,
    int group_size,
    int bits,
    const std::string& mode
) {
    // Validate input tensors are from MLX backend
    if (x.backend_type() != BackendType::MLX ||
        w.backend_type() != BackendType::MLX ||
        scales.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "MLX quantized_matmul: all tensors must be from MLX backend"});
    }

    try {
        // Prepare optional biases for MLX call
        std::optional<mx::array> mlx_biases = std::nullopt;
        if (biases != nullptr && biases->backend_type() == BackendType::MLX) {
            mlx_biases = biases->to_mlx();
        }

        // Call MLX quantized matrix multiplication - MLX validates all parameters
        mx::array result = mx::quantized_matmul(
            x.to_mlx(),
            w.to_mlx(),
            scales.to_mlx(),
            mlx_biases,
            transpose,
            group_size,
            bits,
            mode
        );

        // Get result shape from MLX array
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX quantized_matmul failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::add(const Tensor& a, const Tensor& b) {
    // Validate tensors have compatible shapes for broadcasting
    if (a.backend_type() != BackendType::MLX || b.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX add: tensors must be from MLX backend"});
    }

    try {
        mx::array result = a.to_mlx() + b.to_mlx();

        // Get result shape from MLX array
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX add failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::multiply(const Tensor& a, const Tensor& b) {
    VALIDATE_MLX_TENSOR(a, b);

    return mlx_utils::mlx_tensor_op(mlx_utils::broadcast_shape(a.shape(), b.shape()), [&]() {
        auto mlx_a = mlx_utils::to_mlx_auto(a);
        auto mlx_b = mlx_utils::to_mlx_auto(b);
        return mlx_a * mlx_b;  // Element-wise multiplication
    });
}

Result<Tensor> MLXBackend::softmax(const Tensor& input, int dim) {
    VALIDATE_MLX_TENSOR(input);

    return mlx_utils::mlx_tensor_op(input.shape(), [&]() {
        auto mlx_input = mlx_utils::to_mlx_auto(input);
        return (dim == -1)
            ? mx::softmax(mlx_input, static_cast<int>(input.shape().size()) - 1, true)
            : mx::softmax(mlx_input, dim, true);
    });
}

Result<Tensor> MLXBackend::silu(const Tensor& input) {
    VALIDATE_MLX_TENSOR(input);

    return mlx_utils::mlx_tensor_op(input.shape(), [&]() {
        auto mlx_input = mlx_utils::to_mlx_auto(input);
        // SiLU = x * sigmoid(x)
        return mlx_input * mx::sigmoid(mlx_input);
    });
}

Result<Tensor> MLXBackend::gelu(const Tensor& input) {
    VALIDATE_MLX_TENSOR(input);

    return mlx_utils::mlx_tensor_op(input.shape(), [&]() {
        auto mlx_input = mlx_utils::to_mlx_auto(input);
        // GELU tanh approximation (gelu_pytorch_tanh) used by Gemma GeGLU:
        //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c  = 0.7978845608028654f;  // sqrt(2 / pi)
        const float k  = 0.044715f;
        auto x3        = mlx_input * mlx_input * mlx_input;
        auto inner     = mlx_input + k * x3;
        auto tanh_part = mx::tanh(c * inner);
        return 0.5f * mlx_input * (1.0f + tanh_part);
    });
}

Result<Tensor> MLXBackend::sigmoid(const Tensor& input) {
    VALIDATE_MLX_TENSOR(input);

    return mlx_utils::mlx_tensor_op(input.shape(), [&]() {
        auto mlx_input = mlx_utils::to_mlx_auto(input);
        return mx::sigmoid(mlx_input);
    });
}

Result<Tensor> MLXBackend::conv1d(
    const Tensor& input, const Tensor& weight,
    int stride, int padding, int groups)
{
    VALIDATE_MLX_TENSOR(input);
    VALIDATE_MLX_TENSOR(weight);

    auto mlx_input  = mlx_utils::to_mlx_auto(input);
    auto mlx_weight = mlx_utils::to_mlx_auto(weight);

    // mx::conv1d expects [N, L, C_in]; add batch dim if missing.
    bool added_batch = false;
    if (mlx_input.ndim() == 2) {
        mlx_input  = mx::expand_dims(mlx_input, 0);
        added_batch = true;
    }

    auto out_shape = input.shape();  // placeholder — recomputed after call
    return mlx_utils::mlx_tensor_op(out_shape, [&]() {
        auto out = mx::conv1d(mlx_input, mlx_weight, stride, padding, /*dilation=*/1, groups);
        if (added_batch) out = mx::squeeze(out, 0);
        return out;
    });
}

Result<Tensor> MLXBackend::transpose(const Tensor& input) {
    if (input.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX transpose: tensor must be from MLX backend"});
    }

    try {
        // General transpose (reverses dimension order)
        mx::array result = mx::transpose(input.to_mlx());

        // Get result shape from MLX array
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX transpose failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::swapaxes(const Tensor& input, int axis1, int axis2) {
    if (input.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX swapaxes: tensor must be from MLX backend"});
    }

    const auto& shape = input.shape();
    int ndim = static_cast<int>(shape.size());

    // Normalize negative indices
    if (axis1 < 0) axis1 += ndim;
    if (axis2 < 0) axis2 += ndim;

    // Validate axes
    if (axis1 < 0 || axis1 >= ndim || axis2 < 0 || axis2 >= ndim) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX swapaxes: axis out of bounds"});
    }

    try {
        // Use MLX swapaxes for efficient axis swapping
        mx::array result = mx::swapaxes(input.to_mlx(), axis1, axis2);

        // Get result shape from MLX array
        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX swapaxes failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::reshape(const Tensor& input, const std::vector<size_t>& new_shape) {
    VALIDATE_MLX_TENSOR(input);

    // Validate that total number of elements remains the same
    size_t input_size = input.size();
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        if (dim == 0) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "MLX reshape: shape dimensions cannot be zero"});
        }
        new_size *= dim;
    }

    if (input_size != new_size) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "MLX reshape: total elements must remain the same (input: " +
            std::to_string(input_size) + ", new: " + std::to_string(new_size) + ")"});
    }

    return mlx_utils::mlx_tensor_op(new_shape, [&]() {
        auto mlx_input = mlx_utils::to_mlx_auto(input);

        // Convert size_t vector to MLX Shape (SmallVector<int>)
        mx::Shape mlx_shape;
        mlx_shape.reserve(new_shape.size());
        for (size_t dim : new_shape) {
            mlx_shape.push_back(static_cast<int>(dim));
        }

        return mx::reshape(mlx_input, mlx_shape);
    });
}

Result<Tensor> MLXBackend::concatenate(const std::vector<Tensor>& tensors, int axis) {
    // Validate input
    if (tensors.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX concatenate: empty tensor list"});
    }

    // Validate all tensors are from MLX backend
    for (const auto& tensor : tensors) {
        if (auto error = mlx_utils::validate_single_mlx_tensor(tensor)) {
            return std::unexpected(*error);
        }
    }

    // Get reference shape for validation
    const auto& ref_shape = tensors[0].shape();
    int ndim = static_cast<int>(ref_shape.size());

    // Normalize negative axis
    if (axis < 0) {
        axis += ndim;
    }

    // Validate axis bounds
    if (axis < 0 || axis >= ndim) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "MLX concatenate: axis out of bounds"});
    }

    if (tensors.size() == 1) {
        // Single tensor - just return a copy (after validation)
        return tensors[0];
    }

    // Validate all tensors have compatible shapes
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& current_shape = tensors[i].shape();

        if (current_shape.size() != ref_shape.size()) {
            return std::unexpected(Error{ErrorCode::InvalidInput,
                "MLX concatenate: all tensors must have same number of dimensions"});
        }

        // Check all dimensions except the concatenation axis
        for (int dim = 0; dim < ndim; ++dim) {
            if (dim != axis && current_shape[dim] != ref_shape[dim]) {
                return std::unexpected(Error{ErrorCode::InvalidInput,
                    "MLX concatenate: tensor shapes must match except along concatenation axis"});
            }
        }
    }

    // Compute output shape
    std::vector<size_t> output_shape = ref_shape;
    for (size_t i = 1; i < tensors.size(); ++i) {
        output_shape[axis] += tensors[i].shape()[axis];
    }

    return mlx_utils::mlx_tensor_op(output_shape, [&]() {
        // Convert tensors to MLX arrays
        std::vector<mx::array> mlx_arrays;
        mlx_arrays.reserve(tensors.size());
        for (const auto& tensor : tensors) {
            mlx_arrays.push_back(mlx_utils::to_mlx_auto(tensor));
        }

        return mx::concatenate(mlx_arrays, axis);
    });
}

// Utility operations
Result<void> MLXBackend::extract(const Tensor& tensor, std::span<float> output) {
    if (tensor.backend_type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Tensor not from MLX backend"});
    }

    if (output.size() != tensor.size()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Output buffer size mismatch"});
    }

    const float* data = tensor.data_f32();
    std::copy(data, data + tensor.size(), output.begin());

    return {};
}

Result<void> MLXBackend::evaluate_all() {
    return {}; // MLX operations are lazy by default
}

std::unordered_map<std::string, Tensor> MLXBackend::load_model(const std::string& path) {
    try {
        // Use ModelLoader to load the complete model
        auto result = ModelLoader::load_model(std::filesystem::path(path), this);

        if (!result) {
            throw std::runtime_error("Failed to load model: " + result.error().message);
        }

        // Extract tensor map from the result pair (config, tensors)
        return std::move(result->second);
    } catch (const std::exception& e) {
        throw std::runtime_error("MLX load_model failed: " + std::string(e.what()));
    }
}

size_t MLXBackend::preferred_batch_size() const {
    return 2048;  // MLX optimized batch size
}

bool MLXBackend::supports_async() const {
    return true;  // MLX supports async operations
}

Result<Tensor> MLXBackend::mean(const Tensor& input, int axis, bool keepdims) {
    try {
        mx::array input_array = input.to_mlx();
        mx::array result = mx::mean(input_array, axis, keepdims);

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX mean failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::rsqrt(const Tensor& input) {
    try {
        mx::array input_array = input.to_mlx();
        mx::array result = mx::rsqrt(input_array);

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX rsqrt failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::slice(const Tensor& input, int start, int stop, int axis) {
    try {
        mx::array input_array = input.to_mlx();

        // Use split to extract slice along axis, then take first piece
        std::vector<mx::array> splits = mx::split(input_array, {start, stop}, axis);
        mx::array result = splits[1]; // Middle piece is what we want

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX slice failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::repeat(const Tensor& input, int repeats, int axis) {
    try {
        mx::array input_array = input.to_mlx();
        mx::array result = mx::repeat(input_array, repeats, axis);

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX repeat failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::triu(const Tensor& input, int k) {
    try {
        mx::array input_array = input.to_mlx();
        mx::array result = mx::triu(input_array, k);

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX triu failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::rms_norm(const Tensor& input, const Tensor& weight, float eps) {
    try {
        mx::array input_array = input.to_mlx();
        mx::array weight_array = weight.to_mlx();

        // Use MLX optimized fused RMSNorm
        mx::array result = mx::fast::rms_norm(input_array, weight_array, eps);

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX rms_norm failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::rope(const Tensor& input, int dims, float theta, int offset) {
    try {
        mx::array input_array = input.to_mlx();

        // MLX fast::rope Metal kernel dispatches with:
        //   B = shape[0], T = shape[-2], N = product(shape[1..ndim-3])
        // For 3D input [n_heads, seq, head_dim], the kernel sees B=n_heads, T=seq, N=1.
        // When seq=1 (decode), the "single" fast path activates with grid=(dims/2, N=1, 1),
        // processing only N=1 head instead of n_heads — corrupting all heads except the first.
        //
        // Fix: always present rope with a 4D tensor [1, n_heads, seq, head_dim], matching
        // the Python mlx-lm convention. This makes N=n_heads so all heads are processed.
        bool added_batch_dim = false;
        if (input_array.ndim() == 3) {
            auto s = input_array.shape();
            input_array = mx::reshape(input_array, {1, s[0], s[1], s[2]});
            added_batch_dim = true;
        }

        // MLX fast::rope parameters:
        // - dims: number of dimensions to apply RoPE to
        // - traditional: false for modern RoPE formulation (same as Python)
        // - base: theta value (optional, default 10000.0)
        // - scale: scaling factor (default 1.0)
        // - offset: position offset
        mx::array result = mx::fast::rope(
            input_array,
            dims,
            false,                   // modern RoPE (not traditional)
            std::optional<float>(theta),  // base frequency
            1.0f,                    // scale
            offset                   // position offset
        );

        // Remove the batch dimension we added
        if (added_batch_dim) {
            result = mx::squeeze(result, 0);
        }

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX rope failed: ") + e.what()});
    }
}

Result<Tensor> MLXBackend::scaled_dot_product_attention(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    float scale,
    const std::string& mask) {
    try {
        mx::array q_array = queries.to_mlx();
        mx::array k_array = keys.to_mlx();
        mx::array v_array = values.to_mlx();

        // MLX fast::scaled_dot_product_attention expects rank 4: [B, n_heads, seq, head_dim]
        // If inputs are rank 3 [n_heads, seq, head_dim], add batch dimension via reshape
        // (not expand_dims, which creates a non-contiguous view with stride=1 at dim 0 —
        //  that fails the sdpa_vector q_copy_unless check and triggers an incorrect copy path)
        bool added_batch_dim = false;
        if (q_array.ndim() == 3) {
            auto q_shape = q_array.shape();
            auto k_shape = k_array.shape();
            auto v_shape = v_array.shape();
            q_array = mx::reshape(q_array, {1, q_shape[0], q_shape[1], q_shape[2]});
            k_array = mx::reshape(k_array, {1, k_shape[0], k_shape[1], k_shape[2]});
            v_array = mx::reshape(v_array, {1, v_shape[0], v_shape[1], v_shape[2]});
            added_batch_dim = true;
        }

        // MLX fast::scaled_dot_product_attention parameters:
        // - queries, keys, values: input tensors
        // - scale: scaling factor for attention scores
        // - mask: attention mask ("causal" for autoregressive, "" for none)
        mx::array result = mx::fast::scaled_dot_product_attention(
            q_array,
            k_array,
            v_array,
            scale,
            mask
        );

        // Remove batch dimension if we added it
        if (added_batch_dim) {
            result = mx::squeeze(result, 0);
        }

        auto mlx_shape = result.shape();
        std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

        auto result_buffer = std::make_shared<MLXBuffer>(result);
        return Tensor(result_buffer, result_shape);
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX scaled_dot_product_attention failed: ") + e.what()});
    }
}

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
