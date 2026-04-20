#pragma once

#include "../../core/tensor.h"
#include <vector>

namespace compute {

/**
 * CPU-based buffer implementation for SIMD backends
 * Simple wrapper around std::vector<float> for CPU memory
 */
class CpuBuffer : public BackendBuffer {
private:
    std::vector<float> data_;
    BackendType backend_type_;
    
public:
    /**
     * Create buffer with data (copy from input)
     */
    CpuBuffer(std::span<const float> data, BackendType backend_type)
        : data_(data.begin(), data.end()), backend_type_(backend_type) {}
    
    /**
     * Create uninitialized buffer
     */
    CpuBuffer(size_t size, BackendType backend_type)
        : data_(size), backend_type_(backend_type) {}
    
    /**
     * Create buffer from existing vector (move)
     */
    CpuBuffer(std::vector<float> data, BackendType backend_type)
        : data_(std::move(data)), backend_type_(backend_type) {}
    
    void* get_data() override {
        return data_.data();
    }
    
    size_t get_size() const override {
        return data_.size() * sizeof(float);
    }
    
    BackendType get_backend_type() const override {
        return backend_type_;
    }
    
    void evaluate() override {
        // CPU operations are eager - nothing to evaluate
    }
    
    // CPU-specific access
    std::vector<float>& cpu_data() { return data_; }
    const std::vector<float>& cpu_data() const { return data_; }
    
    float* data_ptr() { return data_.data(); }
    const float* data_ptr() const { return data_.data(); }
    
    size_t num_elements() const { return data_.size(); }
};

} // namespace compute