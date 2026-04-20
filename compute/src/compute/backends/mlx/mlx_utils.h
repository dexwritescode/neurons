#pragma once

#include "../../core/compute_types.h"
#include "../../core/tensor.h"
#include "mlx_buffer.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#include <expected>
#include <concepts>

namespace mx = mlx::core;

namespace compute::mlx_utils {

// Helper to validate a single tensor
constexpr auto validate_single_mlx_tensor(const Tensor& tensor) -> std::optional<Error> {
    if (tensor.backend_type() != BackendType::MLX) {
        return Error{ErrorCode::InvalidInput, "Tensor must be from MLX backend"};
    }
    return std::nullopt;
}

// Variadic validation using fold expression (C++17)
template<typename... Tensors>
constexpr auto validate_mlx_tensors(const Tensors&... tensors) -> std::optional<Error> {
    std::optional<Error> error = std::nullopt;
    ((error = validate_single_mlx_tensor(tensors), error.has_value()) || ...);
    return error;
}

// Macro for validating one or more MLX tensors
#define VALIDATE_MLX_TENSOR(...) \
    do { \
        if (auto error = compute::mlx_utils::validate_mlx_tensors(__VA_ARGS__)) { \
            return std::unexpected(*error); \
        } \
    } while(0)

// Concept to detect types that can convert to mx::array
template<typename T>
concept MLXConvertible = requires(const T& t) {
    { t.to_mlx() } -> std::convertible_to<mx::array>;
};

// Automatic conversion function
template<MLXConvertible T>
constexpr auto to_mlx_auto(const T& tensor) -> mx::array {
    return tensor.to_mlx();
}

// Overload for mx::array (identity)
constexpr auto to_mlx_auto(const mx::array& array) -> const mx::array& {
    return array;
}

// Generic wrapper for MLX operations that may throw
template<typename Func, typename... Args>
auto mlx_safe(Func&& func, Args&&... args) noexcept
    -> std::expected<std::invoke_result_t<Func, Args...>, Error> {
    try {
        return std::forward<Func>(func)(std::forward<Args>(args)...);
    } catch (const std::invalid_argument& e) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   std::string("MLX invalid argument: ") + e.what()});
    } catch (const std::runtime_error& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX runtime error: ") + e.what()});
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   std::string("MLX error: ") + e.what()});
    } catch (...) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   "MLX unknown error"});
    }
}

// Specialized wrapper for operations returning mx::array -> Tensor
// This overload automatically converts arguments using ADL and concepts
template<typename Func, MLXConvertible... Args>
auto mlx_tensor_op(const std::vector<size_t>& expected_shape, Func&& func, const Args&... args) noexcept
    -> Result<Tensor> {
    auto result = mlx_safe([&]() {
        return std::forward<Func>(func)(to_mlx_auto(args)...);
    });

    if (!result) {
        return std::unexpected(result.error());
    }

    // Convert mx::array to Tensor
    auto mlx_array = *result;
    auto mlx_shape = mlx_array.shape();
    std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

    auto result_buffer = std::make_shared<MLXBuffer>(mlx_array);
    return Tensor(result_buffer, result_shape);
}

// Fallback for lambda-based operations (original implementation)
template<typename Func>
auto mlx_tensor_op(const std::vector<size_t>& expected_shape, Func&& func) noexcept
    -> Result<Tensor> {
    auto result = mlx_safe(std::forward<Func>(func));
    if (!result) {
        return std::unexpected(result.error());
    }

    // Convert mx::array to Tensor
    auto mlx_array = *result;
    auto mlx_shape = mlx_array.shape();
    std::vector<size_t> result_shape(mlx_shape.begin(), mlx_shape.end());

    auto result_buffer = std::make_shared<MLXBuffer>(mlx_array);
    return Tensor(result_buffer, result_shape);
}

// Helper to compute broadcast shape for binary operations
inline std::vector<size_t> broadcast_shape(const std::vector<size_t>& a_shape,
                                         const std::vector<size_t>& b_shape) {
    // For now, return the larger shape - MLX will handle broadcasting
    // This is a simplified implementation; MLX does the actual broadcast computation
    return a_shape.size() >= b_shape.size() ? a_shape : b_shape;
}

} // namespace compute::mlx_utils

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
