#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

#include "compute/core/compute_types.h"
#include "compute/core/compute_backend.h"
#include "compute/core/graph.h"
#include "compute/backends/simd/neon_backend.h"

using namespace compute;

TEST(NeonExecutionTest, EndToEndDotProductExecution) {
    // Test with new Tensor API using NEON backend directly
    auto backend = std::make_unique<NeonBackend>();
    if (!backend->is_available()) {
        GTEST_SKIP() << "NEON backend not available on this platform";
    }

    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result.has_value()) << "Failed to initialize NEON backend: " << init_result.error().message;
        
    // Test vectors: dot product should be 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    std::vector<float> vec_a = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec_b = {4.0f, 5.0f, 6.0f};
    
    // Create tensors using new API
    auto tensor_a = backend->create_tensor(std::span<const float>(vec_a), {3});
    auto tensor_b = backend->create_tensor(std::span<const float>(vec_b), {3});
    
    // Execute dot product
    auto result_tensor = backend->dot_product(tensor_a, tensor_b);
    
    // Extract result
    std::vector<float> result_data(1);
    auto extract_result = backend->extract(result_tensor, std::span<float>(result_data));
    ASSERT_TRUE(extract_result.has_value()) << "Failed to extract result: " << extract_result.error().message;
    
    // Check the result
    EXPECT_FLOAT_EQ(result_data[0], 32.0f);
    
    backend->cleanup();
}

TEST(NeonExecutionTest, EndToEndMatrixScalarAdditionExecution) {
    // Test with new Tensor API using NEON backend directly
    auto backend = std::make_unique<NeonBackend>();
    if (!backend->is_available()) {
        GTEST_SKIP() << "NEON backend not available on this platform";
    }
    
    auto init_result = backend->initialize();
    ASSERT_TRUE(init_result.has_value()) << "Failed to initialize NEON backend: " << init_result.error().message;
    
    // Test matrix: add 10 to each element
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Create input tensor using new API
    auto input_tensor = backend->create_tensor(std::span<const float>(input_data), {2, 2});
    
    // Execute scalar addition
    auto result_tensor = backend->matrix_scalar_add(input_tensor, 10.0f);
    
    // Extract results
    std::vector<float> output_data(4);
    auto extract_result = backend->extract(result_tensor, std::span<float>(output_data));
    ASSERT_TRUE(extract_result.has_value()) << "Failed to extract result: " << extract_result.error().message;
    
    // Check results
    EXPECT_FLOAT_EQ(output_data[0], 11.0f);
    EXPECT_FLOAT_EQ(output_data[1], 12.0f);
    EXPECT_FLOAT_EQ(output_data[2], 13.0f);
    EXPECT_FLOAT_EQ(output_data[3], 14.0f);
    
    backend->cleanup();
}

TEST(NeonExecutionTest, ComplexComputationGraphWithDependencies) {
    // Explicitly test with NEON backend (the original working backend)
    auto backend_result = BackendFactory::create(BackendType::SimdNeon);
    if (!backend_result) {
        // Skip if NEON not available
        GTEST_SKIP() << "NEON backend not available on this platform";
    }
    
    auto builder = ComputeGraphBuilder(BackendType::SimdNeon);
    
    // Build a complex graph: 
    // 1. Compute dot product of two vectors
    // 2. Use that result to add to a matrix
    
    std::vector<float> vec_a = {1.0f, 2.0f};  // dot product = 1*3 + 2*4 = 11
    std::vector<float> vec_b = {3.0f, 4.0f};
    std::vector<float> matrix_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output_data(4);
    
    // Get backend to create tensors
    auto& backend = *backend_result;
    auto tensor_a = backend->create_tensor(std::span<const float>(vec_a), {2});
    auto tensor_b = backend->create_tensor(std::span<const float>(vec_b), {2});
    auto input_matrix = backend->create_tensor(std::span<const float>(matrix_data), {2, 2});
    
    // Build computation graph with dependency
    float dot_result;
    auto symbolic_scalar = builder.dot_product(tensor_a, tensor_b, std::span<float>(&dot_result, 1));
    
    // This operation depends on the dot product result
    builder.matrix_scalar_add(input_matrix, symbolic_scalar, std::span<float>(output_data), {2, 2});
    
    auto execution_result = builder.execute();
    if (!execution_result) {
        std::cerr << "NEON execution failed: " << execution_result.error().message << std::endl;
    }
    ASSERT_TRUE(execution_result);
    
    // Check intermediate result (dot product)
    EXPECT_FLOAT_EQ(dot_result, 11.0f);
    
    // Check final results (matrix + scalar)
    EXPECT_FLOAT_EQ(output_data[0], 12.0f); // 1 + 11
    EXPECT_FLOAT_EQ(output_data[1], 13.0f); // 2 + 11
    EXPECT_FLOAT_EQ(output_data[2], 14.0f); // 3 + 11
    EXPECT_FLOAT_EQ(output_data[3], 15.0f); // 4 + 11
}

TEST(NeonExecutionTest, BackendAvailabilityAndPerformance) {
    // Check that NEON backend is available on Apple Silicon
    auto available_backends = BackendFactory::available_backends();
    
    bool neon_available = std::find(available_backends.begin(), available_backends.end(), 
                                   BackendType::SimdNeon) != available_backends.end();
    
#ifdef __ARM_NEON
    EXPECT_TRUE(neon_available);
#endif
    
    // Test that Auto backend selection works
    auto backend_result = BackendFactory::create(BackendType::Auto);
    ASSERT_TRUE(backend_result);
    
    auto& backend = *backend_result;
    EXPECT_TRUE(backend->is_available());
}

TEST(NeonExecutionTest, BackendManagerFunctionality) {
    auto& manager = BackendManager::instance();
    
    // Test initialization
    auto init_result = manager.initialize();
    ASSERT_TRUE(init_result);
    
    // Test getting backends
    auto* default_backend = manager.get_default_backend();
    EXPECT_NE(default_backend, nullptr);
    
    // Test backend info
    EXPECT_FALSE(default_backend->name().empty());
    
    // Cleanup
    manager.cleanup();
}
