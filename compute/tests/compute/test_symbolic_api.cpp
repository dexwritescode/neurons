#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "compute/core/compute_types.h"
#include "compute/core/compute_backend.h"
#include "compute/core/graph.h"

using namespace compute;

TEST(SymbolicApiTest, TensorBasicFunctionality) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Get a backend to create tensors
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        GTEST_SKIP() << "No backend available for testing";
    }
    
    auto& backend = *backend_result;
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result);
    
    auto tensor = backend->create_tensor(std::span<const float>(data), {2, 2});
    
    EXPECT_EQ(tensor.size(), 4);
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);
    
    auto tensor_data = tensor.data<float>();
    EXPECT_EQ(tensor_data[0], 1.0f);
    EXPECT_EQ(tensor_data[3], 4.0f);
    
    backend->cleanup();
}

TEST(SymbolicApiTest, SymbolicScalarCreation) {
    auto builder = ComputeGraphBuilder();
    
    std::vector<float> vec_a = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec_b = {4.0f, 5.0f, 6.0f};
    
    // Get a backend to create tensors
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        GTEST_SKIP() << "No backend available for testing";
    }
    
    auto& backend = *backend_result;
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result);
    
    auto tensor_a = backend->create_tensor(std::span<const float>(vec_a), {3});
    auto tensor_b = backend->create_tensor(std::span<const float>(vec_b), {3});
    
    float result;
    auto symbolic = builder.dot_product(tensor_a, tensor_b, std::span<float>(&result, 1));
    
    // Test symbolic scalar properties
    EXPECT_EQ(symbolic.node_id(), 0);  // First node should have ID 0
    
    backend->cleanup();
}

TEST(SymbolicApiTest, ComputationGraphBuildingWithDependencies) {
    auto builder = ComputeGraphBuilder();
    
    // Get a backend to create tensors
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        GTEST_SKIP() << "No backend available for testing";
    }
    
    auto& backend = *backend_result;
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result);
    
    // Create test data
    std::vector<float> vec_a = {1.0f, 2.0f};
    std::vector<float> vec_b = {3.0f, 4.0f}; 
    std::vector<float> matrix_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output_data(4);
    
    auto tensor_a = backend->create_tensor(std::span<const float>(vec_a), {2});
    auto tensor_b = backend->create_tensor(std::span<const float>(vec_b), {2});
    auto matrix = backend->create_tensor(std::span<const float>(matrix_data), {2, 2});
    
    // Build computation graph with dependencies
    float scalar_result;
    auto dot_result = builder.dot_product(tensor_a, tensor_b, std::span<float>(&scalar_result, 1));
    
    // This should create a dependency on the dot_product result
    builder.matrix_scalar_add(matrix, dot_result, std::span<float>(output_data), {2, 2});
    
    // Test that we can build the graph without errors
    auto graph_ptr = builder.build();
    EXPECT_NE(graph_ptr, nullptr);
    
    backend->cleanup();
}

TEST(SymbolicApiTest, MultipleOperationTypesInGraph) {
    auto builder = ComputeGraphBuilder();
    
    // Get a backend to create tensors
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        GTEST_SKIP() << "No backend available for testing";
    }
    
    auto& backend = *backend_result;
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result);
    
    // Test data
    std::vector<float> vec_data = {2.0f, 3.0f, 4.0f};
    std::vector<float> matrix_data = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> output1(4), output2(4);
    
    auto vector = backend->create_tensor(std::span<const float>(vec_data), {3});
    auto matrix = backend->create_tensor(std::span<const float>(matrix_data), {2, 2});
    
    // Build complex graph
    float scalar1, scalar2;
    auto dot1 = builder.dot_product(vector, vector, std::span<float>(&scalar1, 1));
    auto dot2 = builder.dot_product(vector, vector, std::span<float>(&scalar2, 1)); 
    
    builder.matrix_scalar_add(matrix, dot1, std::span<float>(output1), {2, 2});
    builder.matrix_scalar_add(matrix, dot2, std::span<float>(output2), {2, 2});
    
    // Should build successfully
    auto graph_ptr = builder.build();
    EXPECT_NE(graph_ptr, nullptr);
    
    backend->cleanup();
}

TEST(SymbolicApiTest, ConvenienceOverloadsWorkCorrectly) {
    auto builder = ComputeGraphBuilder();
    
    // Get a backend to create tensors
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        GTEST_SKIP() << "No backend available for testing";
    }
    
    auto& backend = *backend_result;
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result);
    
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(4);
    
    auto tensor = backend->create_tensor(std::span<const float>(data), {2, 2});
    
    // Test immediate value overload
    builder.matrix_scalar_add(tensor, 5.0f, std::span<float>(output), {2, 2});
    
    // Test symbolic reference overload
    float scalar_result;
    auto symbolic = builder.dot_product(tensor, tensor, std::span<float>(&scalar_result, 1));
    builder.matrix_scalar_add(tensor, symbolic, std::span<float>(output), {2, 2});
    
    // Should compile and build without errors
    auto graph_ptr = builder.build();
    EXPECT_NE(graph_ptr, nullptr);
    
    backend->cleanup();
}