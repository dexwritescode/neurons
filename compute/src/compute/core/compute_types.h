#pragma once

#include <span>
#include <memory>
#include <expected>
#include <string>
#include <vector>

namespace compute {

// Error handling
enum class ErrorCode {
    Success,
    InvalidInput,
    InvalidArgument,
    InvalidModel,
    BackendNotAvailable,
    InsufficientMemory,
    ComputeError,
    TensorNotFound,
    NotImplemented,
    UnknownError
};

struct Error {
    ErrorCode code;
    std::string message;
    
    Error(ErrorCode c, std::string msg) : code(c), message(std::move(msg)) {}
};

template<typename T>
using Result = std::expected<T, Error>;

// Backend types
enum class BackendType {
    Metal,
    MLX,    // Apple MLX framework (Metal-accelerated on Apple Silicon)
    Auto    // Let system choose best available
};

// Forward declarations  
class ComputeBackend;
class ComputeGraph;
class ComputeGraphBuilder;
class Tensor;
class BackendBuffer;

// Node ID for tracking operations in the computation graph
using NodeId = size_t;

// Symbolic reference to a scalar value in the computation graph
class SymbolicScalar {
public:
    explicit SymbolicScalar(NodeId node_id) : node_id_(node_id) {}
    
    // Get the node ID for dependency tracking
    NodeId node_id() const { return node_id_; }

private:
    NodeId node_id_;
};

// Symbolic reference to a tensor in the computation graph  
class SymbolicTensor {
public:
    explicit SymbolicTensor(NodeId node_id, std::vector<size_t> shape)
        : node_id_(node_id), shape_(std::move(shape)) {}
    
    // Get the node ID for dependency tracking
    NodeId node_id() const { return node_id_; }
    
    // Get the expected output shape
    const std::vector<size_t>& shape() const { return shape_; }
    
    // Convenience accessors (same as TensorView)
    bool is_vector() const { return shape_.size() == 1; }
    bool is_matrix() const { return shape_.size() == 2; }
    size_t length() const { return is_vector() ? shape_[0] : 0; }
    size_t rows() const { return is_matrix() ? shape_[0] : 0; }
    size_t cols() const { return is_matrix() ? shape_[1] : 0; }

private:
    NodeId node_id_;
    std::vector<size_t> shape_;
};

} // namespace compute