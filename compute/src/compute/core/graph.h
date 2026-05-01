#pragma once

#include "compute_types.h"
#include "compute_backend.h"
#include <memory>
#include <variant>
#include <vector>
#include <unordered_map>

namespace compute {

// Operation types for the computation graph
enum class OpType {
    DotProduct,
    MatrixScalarAdd
};

// Input types for operations (can be immediate values or symbolic references)
using ScalarInput = std::variant<float, SymbolicScalar>;
using TensorInput = std::variant<Tensor, SymbolicTensor>;

// Parameters for different operations  
struct DotProductParams {
    TensorInput input_a;
    TensorInput input_b;
    std::span<float> output;  // Where to write the scalar result
};

struct MatrixScalarAddParams {
    TensorInput input_tensor;
    ScalarInput scalar;
    std::span<float> output;  // Where to write the result data
    std::vector<size_t> output_shape;
};

using OpParams = std::variant<DotProductParams, MatrixScalarAddParams>;

// Represents a single operation in the computation graph
struct GraphNode {
    NodeId id;
    OpType type;
    OpParams params;
    std::vector<NodeId> dependencies;  // Which nodes must execute before this one
    
    GraphNode(NodeId node_id, OpType op_type, OpParams op_params) 
        : id(node_id), type(op_type), params(std::move(op_params)) {}
};

// Forward declaration for fluent interface
class ComputeGraphBuilder;

// Execution result of a computation graph
class ComputeResult {
public:
    ComputeResult(Result<std::vector<float>> data, std::vector<size_t> shape)
        : data_(std::move(data)), shape_(std::move(shape)) {}
    
    // Get the result data
    Result<std::span<const float>> data() const {
        if (!data_) return std::unexpected(data_.error());
        return std::span<const float>(data_->data(), data_->size());
    }
    
    // Get result shape
    const std::vector<size_t>& shape() const { return shape_; }
    
    // Convenience for scalar results (dot product)
    Result<float> scalar() const {
        if (!data_) return std::unexpected(data_.error());
        if (data_->size() != 1) {
            return std::unexpected(Error{ErrorCode::InvalidInput, "Result is not a scalar"});
        }
        return (*data_)[0];
    }

private:
    Result<std::vector<float>> data_;
    std::vector<size_t> shape_;
};

// The computation graph that can be executed
class ComputeGraph {
public:
    explicit ComputeGraph(BackendType backend_type = BackendType::Auto)
        : backend_type_(backend_type), next_node_id_(0) {}
    
    // Execute the computation graph with dependency resolution
    Result<ComputeResult> execute();
    
    // For building the graph (used by ComputeGraphBuilder)
    NodeId add_node(GraphNode node);
    
    // Get node information
    const GraphNode* get_node(NodeId id) const;
    
    // Get next node ID (for ComputeGraphBuilder)
    NodeId get_next_node_id() const { return next_node_id_; }
    
private:
    std::vector<GraphNode> nodes_;
    BackendType backend_type_;
    NodeId next_node_id_;
    
    // Intermediate results storage during execution
    std::unordered_map<NodeId, std::vector<float>> intermediate_scalars_;
    std::unordered_map<NodeId, std::vector<float>> intermediate_tensors_;
    
    Result<ComputeBackend*> get_backend();
    std::vector<NodeId> topological_sort() const;
    Result<void> execute_node(const GraphNode& node, ComputeBackend* backend);
    
    // Helper methods for resolving inputs
    Result<Tensor> resolve_tensor_input(const TensorInput& input);
    Result<float> resolve_scalar_input(const ScalarInput& input);
};

// Builder for fluently constructing computation graphs
class ComputeGraphBuilder {
public:
    explicit ComputeGraphBuilder(BackendType backend = BackendType::Auto)
        : graph_(std::make_unique<ComputeGraph>(backend)) {}
    
    // Dot product operation - returns symbolic scalar
    SymbolicScalar dot_product(TensorInput a, TensorInput b, std::span<float> output) {
        DotProductParams params{std::move(a), std::move(b), output};
        GraphNode node(graph_->get_next_node_id(), OpType::DotProduct, std::move(params));
        
        // Add dependencies based on inputs
        add_dependencies(node);
        
        NodeId node_id = graph_->add_node(std::move(node));
        return SymbolicScalar(node_id);
    }
    
    // Convenience overload for immediate tensor values
    SymbolicScalar dot_product(Tensor a, Tensor b, std::span<float> output) {
        return dot_product(TensorInput{a}, TensorInput{b}, output);
    }
    
    // Matrix scalar addition operation - writes result to output buffer
    void matrix_scalar_add(TensorInput tensor, ScalarInput scalar, std::span<float> output, std::vector<size_t> output_shape) {
        MatrixScalarAddParams params{std::move(tensor), std::move(scalar), output, std::move(output_shape)};
        GraphNode node(graph_->get_next_node_id(), OpType::MatrixScalarAdd, std::move(params));
        
        // Add dependencies based on inputs
        add_dependencies(node);
        
        graph_->add_node(std::move(node));
    }
    
    // Convenience overloads for immediate values
    void matrix_scalar_add(Tensor tensor, float scalar, std::span<float> output, std::vector<size_t> output_shape) {
        matrix_scalar_add(TensorInput{tensor}, ScalarInput{scalar}, output, std::move(output_shape));
    }
    
    void matrix_scalar_add(Tensor tensor, SymbolicScalar scalar, std::span<float> output, std::vector<size_t> output_shape) {
        matrix_scalar_add(TensorInput{tensor}, ScalarInput{scalar}, output, std::move(output_shape));
    }
    
    // Execute the built graph
    Result<ComputeResult> execute() {
        return graph_->execute();
    }
    
    // Get the built graph (transfers ownership)
    std::unique_ptr<ComputeGraph> build() {
        return std::move(graph_);
    }

private:
    std::unique_ptr<ComputeGraph> graph_;
    
    // Helper to add dependencies based on symbolic inputs
    void add_dependencies(GraphNode& node) {
        // Extract dependencies from OpParams
        std::visit([&node, this](const auto& params) {
            using ParamType = std::decay_t<decltype(params)>;
            
            if constexpr (std::is_same_v<ParamType, DotProductParams>) {
                add_tensor_dependency(node, params.input_a);
                add_tensor_dependency(node, params.input_b);
            } else if constexpr (std::is_same_v<ParamType, MatrixScalarAddParams>) {
                add_tensor_dependency(node, params.input_tensor);
                add_scalar_dependency(node, params.scalar);
            }
        }, node.params);
    }
    
    void add_tensor_dependency(GraphNode& node, const TensorInput& input) {
        if (std::holds_alternative<SymbolicTensor>(input)) {
            const auto& symbolic = std::get<SymbolicTensor>(input);
            node.dependencies.push_back(symbolic.node_id());
        }
    }
    
    void add_scalar_dependency(GraphNode& node, const ScalarInput& input) {
        if (std::holds_alternative<SymbolicScalar>(input)) {
            const auto& symbolic = std::get<SymbolicScalar>(input);
            node.dependencies.push_back(symbolic.node_id());
        }
    }
};

// Convenience factory functions
inline ComputeGraphBuilder graph(BackendType backend = BackendType::Auto) {
    return ComputeGraphBuilder(backend);
}

inline ComputeGraphBuilder simd_graph() {
    return ComputeGraphBuilder(BackendType::MLX);
}

inline ComputeGraphBuilder metal_graph() {
    return ComputeGraphBuilder(BackendType::Metal);
}

} // namespace compute
