#include "graph.h"
#include "compute_backend.h"
#include <algorithm>
#include <queue>

namespace compute {

// ComputeGraph implementations
NodeId ComputeGraph::add_node(GraphNode node) {
    NodeId id = next_node_id_++;
    node.id = id;  // Ensure node has correct ID
    nodes_.push_back(std::move(node));
    return id;
}

const GraphNode* ComputeGraph::get_node(NodeId id) const {
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
                          [id](const GraphNode& node) { return node.id == id; });
    return it != nodes_.end() ? &(*it) : nullptr;
}

Result<ComputeResult> ComputeGraph::execute() {
    // Get compute backend
    auto backend_result = get_backend();
    if (!backend_result) {
        return std::unexpected(backend_result.error());
    }
    ComputeBackend* backend = *backend_result;
    
    // Clear intermediate results from previous executions
    intermediate_scalars_.clear();
    intermediate_tensors_.clear();
    
    // Get execution order via topological sort
    auto execution_order = topological_sort();
    
    // Execute nodes in dependency order
    for (NodeId node_id : execution_order) {
        const GraphNode* node = get_node(node_id);
        if (!node) {
            return std::unexpected(Error{ErrorCode::UnknownError, "Node not found during execution"});
        }
        
        auto exec_result = execute_node(*node, backend);
        if (!exec_result) {
            return std::unexpected(exec_result.error());
        }
    }
    
    // For now, return empty result - in the future this could return final outputs
    std::vector<float> result_data = {0.0f};
    return ComputeResult(Result<std::vector<float>>(std::move(result_data)), {1});
}

Result<ComputeBackend*> ComputeGraph::get_backend() {
    auto& manager = BackendManager::instance();
    auto init_result = manager.initialize();
    if (!init_result) {
        return std::unexpected(init_result.error());
    }
    
    ComputeBackend* backend = nullptr;
    if (backend_type_ == BackendType::Auto) {
        backend = manager.get_default_backend();
    } else {
        backend = manager.get_backend(backend_type_);
    }
    
    if (!backend) {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Requested backend not available"});
    }
    
    return backend;
}

std::vector<NodeId> ComputeGraph::topological_sort() const {
    std::vector<NodeId> result;
    std::unordered_map<NodeId, int> in_degree;
    std::queue<NodeId> ready_queue;
    
    // Initialize in-degree count for all nodes
    for (const auto& node : nodes_) {
        in_degree[node.id] = node.dependencies.size();
        
        // If node has no dependencies, it's ready to execute
        if (node.dependencies.empty()) {
            ready_queue.push(node.id);
        }
    }
    
    // Process nodes in topological order
    while (!ready_queue.empty()) {
        NodeId current = ready_queue.front();
        ready_queue.pop();
        result.push_back(current);
        
        // Find all nodes that depend on the current node
        for (const auto& node : nodes_) {
            // Check if this node depends on the current node
            auto it = std::find(node.dependencies.begin(), node.dependencies.end(), current);
            if (it != node.dependencies.end()) {
                // Reduce in-degree
                in_degree[node.id]--;
                
                // If in-degree becomes 0, node is ready to execute
                if (in_degree[node.id] == 0) {
                    ready_queue.push(node.id);
                }
            }
        }
    }
    
    // Check for cycles (if we didn't process all nodes)
    if (result.size() != nodes_.size()) {
        // There's a cycle in the dependency graph
        // For now, just return the partial result - in the future we should handle this error
        // TODO: Return error result indicating cyclic dependency
    }
    
    return result;
}

Result<void> ComputeGraph::execute_node(const GraphNode& node, ComputeBackend* backend) {
    return std::visit([this, &node, backend](const auto& params) -> Result<void> {
        using ParamType = std::decay_t<decltype(params)>;
        
        if constexpr (std::is_same_v<ParamType, DotProductParams>) {
            // Resolve tensor inputs (could be immediate or symbolic)
            auto tensor_a_result = resolve_tensor_input(params.input_a);
            if (!tensor_a_result) return std::unexpected(tensor_a_result.error());
            
            auto tensor_b_result = resolve_tensor_input(params.input_b);
            if (!tensor_b_result) return std::unexpected(tensor_b_result.error());
            
            // Execute dot product
            auto result_tensor = backend->dot_product(*tensor_a_result, *tensor_b_result);
            
            // Extract result to output buffer
            auto extract_result = backend->extract(result_tensor, params.output);
            if (!extract_result) return std::unexpected(extract_result.error());
            
            // Store intermediate result for potential use by other nodes
            if (params.output.size() == 1) {
                intermediate_scalars_[node.id] = {params.output[0]};
            }
            
            return {};
            
        } else if constexpr (std::is_same_v<ParamType, MatrixScalarAddParams>) {
            // Resolve tensor input
            auto tensor_result = resolve_tensor_input(params.input_tensor);
            if (!tensor_result) return std::unexpected(tensor_result.error());
            
            // Resolve scalar input (could be immediate value or symbolic)
            auto scalar_result = resolve_scalar_input(params.scalar);
            if (!scalar_result) return std::unexpected(scalar_result.error());
            
            // Execute matrix scalar add
            auto result_tensor = backend->matrix_scalar_add(*tensor_result, *scalar_result);
            
            // Extract result to output buffer
            auto extract_result = backend->extract(result_tensor, params.output);
            if (!extract_result) return std::unexpected(extract_result.error());
            
            // Store intermediate result for potential use by other nodes
            std::vector<float> result_data(params.output.begin(), params.output.end());
            intermediate_tensors_[node.id] = std::move(result_data);
            
            return {};
            
        } else {
            return std::unexpected(Error{ErrorCode::UnknownError, "Unknown operation type"});
        }
    }, node.params);
}

Result<Tensor> ComputeGraph::resolve_tensor_input(const TensorInput& input) {
    return std::visit([this](const auto& value) -> Result<Tensor> {
        using ValueType = std::decay_t<decltype(value)>;
        
        if constexpr (std::is_same_v<ValueType, Tensor>) {
            // Immediate tensor value
            return value;
            
        } else if constexpr (std::is_same_v<ValueType, SymbolicTensor>) {
            // Symbolic reference - need to get from intermediate results
            auto it = intermediate_tensors_.find(value.node_id());
            if (it == intermediate_tensors_.end()) {
                return std::unexpected(Error{ErrorCode::ComputeError, 
                    "Symbolic tensor result not found - dependency not satisfied"});
            }
            
            // For now, we need a backend to create a tensor from the stored data
            // This is a limitation of the current design - we should improve this
            return std::unexpected(Error{ErrorCode::ComputeError, 
                "Symbolic tensor resolution not fully implemented yet"});
            
        } else {
            return std::unexpected(Error{ErrorCode::UnknownError, "Unknown tensor input type"});
        }
    }, input);
}

Result<float> ComputeGraph::resolve_scalar_input(const ScalarInput& input) {
    return std::visit([this](const auto& value) -> Result<float> {
        using ValueType = std::decay_t<decltype(value)>;
        
        if constexpr (std::is_same_v<ValueType, float>) {
            // Immediate scalar value
            return value;
            
        } else if constexpr (std::is_same_v<ValueType, SymbolicScalar>) {
            // Symbolic reference - need to get from intermediate results
            auto it = intermediate_scalars_.find(value.node_id());
            if (it == intermediate_scalars_.end()) {
                return std::unexpected(Error{ErrorCode::ComputeError, 
                    "Symbolic scalar result not found - dependency not satisfied"});
            }
            
            if (it->second.empty()) {
                return std::unexpected(Error{ErrorCode::ComputeError, "Scalar result is empty"});
            }
            
            return it->second[0];
            
        } else {
            return std::unexpected(Error{ErrorCode::UnknownError, "Unknown scalar input type"});
        }
    }, input);
}

} // namespace compute