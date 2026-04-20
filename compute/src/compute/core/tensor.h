#pragma once

#include "compute_types.h"
#include <memory>
#include <vector>
#include <span>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

namespace compute {

// Forward declarations
class BackendBuffer;
class ComputeBackend;

/**
 * Backend-managed memory buffer abstraction
 * Each backend implements this to manage memory in their optimal format
 */
class BackendBuffer {
public:
    virtual ~BackendBuffer() = default;
    
    /**
     * Get raw pointer to data (may trigger computation for lazy backends)
     * @return Pointer to buffer data
     */
    virtual void* get_data() = 0;
    
    /**
     * Get buffer size in bytes
     */
    virtual size_t get_size() const = 0;
    
    /**
     * Get backend type that owns this buffer
     */
    virtual BackendType get_backend_type() const = 0;
    
    /**
     * Force evaluation of any pending computations (for lazy backends)
     */
    virtual void evaluate() = 0;
};

/**
 * Unified tensor that lives in backend-native format
 * Eliminates TensorView abstraction - tensors are created directly in backend format
 */
class Tensor {
private:
    std::shared_ptr<BackendBuffer> buffer_;
    std::vector<size_t> shape_;
    
public:
    /**
     * Construct tensor with backend buffer and shape
     */
    Tensor(std::shared_ptr<BackendBuffer> buffer, std::vector<size_t> shape)
        : buffer_(std::move(buffer)), shape_(std::move(shape)) {}
    
    // Move constructor and assignment
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    
    // Copy constructor and assignment
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    
    /**
     * Get typed pointer to tensor data
     * May trigger computation for lazy backends
     */
    template<typename T>
    T* data() {
        buffer_->evaluate();  // Ensure computation is complete
        return static_cast<T*>(buffer_->get_data());
    }
    
    template<typename T>
    const T* data() const {
        const_cast<BackendBuffer*>(buffer_.get())->evaluate();
        return static_cast<const T*>(buffer_->get_data());
    }
    
    /**
     * Get raw pointer (convenience for float tensors)
     */
    float* data_f32() { return data<float>(); }
    const float* data_f32() const { return data<float>(); }
    
    /**
     * Get backend type
     */
    BackendType backend_type() const { 
        return buffer_->get_backend_type(); 
    }
    
    /**
     * Get tensor shape
     */
    const std::vector<size_t>& shape() const { return shape_; }
    
    /**
     * Get total number of elements
     */
    size_t size() const {
        size_t total = 1;
        for (size_t dim : shape_) {
            total *= dim;
        }
        return total;
    }
    
    /**
     * Get size in bytes (assumes float32)
     */
    size_t byte_size() const {
        return size() * sizeof(float);
    }
    
    // Convenience shape accessors
    bool is_scalar() const { return shape_.empty() || (shape_.size() == 1 && shape_[0] == 1); }
    bool is_vector() const { return shape_.size() == 1; }
    bool is_matrix() const { return shape_.size() == 2; }
    
    size_t length() const { 
        return is_vector() ? shape_[0] : 0; 
    }
    
    size_t rows() const { 
        return is_matrix() ? shape_[0] : 0; 
    }
    
    size_t cols() const { 
        return is_matrix() ? shape_[1] : 0; 
    }
    
    /**
     * Get underlying buffer (for backend-specific operations)
     */
    std::shared_ptr<BackendBuffer> buffer() const { return buffer_; }
    
    /**
     * Force evaluation of any pending computations
     */
    void evaluate() const {
        buffer_->evaluate();
    }

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    /**
     * Get MLX array for MLX backend tensors
     * Throws if tensor is not from MLX backend
     */
    mx::array& to_mlx();
    const mx::array& to_mlx() const;
#endif
};

} // namespace compute