#pragma once

#include "../../core/tensor.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;

namespace compute {

/**
 * MLX-based buffer implementation using mx::array
 * Wraps MLX lazy-evaluated arrays for tensor operations
 */
class MLXBuffer : public BackendBuffer {
private:
    mx::array mlx_array_;
    mutable bool evaluated_ = false;
    
public:
    /**
     * Create buffer wrapping existing MLX array
     */
    explicit MLXBuffer(mx::array array)
        : mlx_array_(std::move(array)), evaluated_(false) {}
    
    /**
     * Create buffer from data (creates MLX array)
     */
    MLXBuffer(std::span<const float> data, const std::vector<size_t>& shape) 
        : mlx_array_(create_array_from_data(data, shape)), evaluated_(false) {
    }
    
    /**
     * Create uninitialized buffer with given shape
     */
    MLXBuffer(const std::vector<size_t>& shape) 
        : mlx_array_(create_zeros_array(shape)), evaluated_(false) {
    }
    
    ~MLXBuffer() override = default;
    
    void* get_data() override {
        evaluate();
        // Convert to float32 if needed (e.g., from bfloat16)
        // MLX .data<T>() only works if the array dtype matches T exactly
        if (mlx_array_.dtype() != mx::float32) {
            mlx_array_ = mx::astype(mlx_array_, mx::float32);
            mx::eval(mlx_array_);
            evaluated_ = true;  // Re-mark as evaluated after conversion
        }
        return mlx_array_.data<float>();
    }
    
    size_t get_size() const override {
        return mlx_array_.nbytes();
    }
    
    BackendType get_backend_type() const override {
        return BackendType::MLX;
    }
    
    void evaluate() override {
        if (!evaluated_) {
            mx::eval(mlx_array_);  // Force MLX computation
            evaluated_ = true;
        }
    }
    
    void evaluate() const {
        if (!evaluated_) {
            mx::eval(mlx_array_);  // Force MLX computation
            evaluated_ = true;
        }
    }
    
    // MLX-specific access
    const mx::array& mlx_array() const { return mlx_array_; }
    mx::array& mlx_array() { return mlx_array_; }
    
    float* data_ptr() { 
        evaluate();
        return mlx_array_.data<float>(); 
    }
    
    const float* data_ptr() const { 
        evaluate();
        return mlx_array_.data<float>(); 
    }
    
    size_t num_elements() const { return mlx_array_.size(); }

private:
    // Helper methods to create MLX arrays
    static mx::array create_array_from_data(std::span<const float> data, const std::vector<size_t>& shape) {
        // Convert shape to MLX format (Shape is SmallVector<int>)
        mx::Shape mlx_shape;
        mlx_shape.reserve(shape.size());
        for (size_t dim : shape) {
            mlx_shape.push_back(static_cast<int>(dim));
        }
        
        // Create MLX array using iterator constructor
        return mx::array(data.begin(), mlx_shape, mx::float32);
    }
    
    static mx::array create_zeros_array(const std::vector<size_t>& shape) {
        // Convert shape to MLX format (Shape is SmallVector<int>)
        mx::Shape mlx_shape;
        mlx_shape.reserve(shape.size());
        for (size_t dim : shape) {
            mlx_shape.push_back(static_cast<int>(dim));
        }
        
        return mx::zeros(mlx_shape, mx::float32);
    }
};

} // namespace compute

#endif // defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)