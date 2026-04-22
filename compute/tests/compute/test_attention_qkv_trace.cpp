#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <fstream>
#include <cmath>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#include "compute/backends/mlx/mlx_buffer.h"
#include <mlx/mlx.h>
#endif

namespace compute {

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

class AttentionQKVTraceTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_dir = TINYLLAMA_MODEL_DIR;
        baseline_dir = std::filesystem::path(TEST_RESOURCES_DIR).parent_path() / "baselines" / "output";

        if (!std::filesystem::exists(model_dir))
            GTEST_SKIP() << "Model not found: " << model_dir;

        // Create MLX backend
        auto backend_result = BackendFactory::create(BackendType::MLX);
        ASSERT_TRUE(backend_result.has_value())
            << "Failed to create MLX backend: " << backend_result.error().message;

        backend = std::move(*backend_result);
        auto init_result = backend->initialize();
        ASSERT_TRUE(init_result.has_value())
            << "Failed to initialize MLX backend: " << init_result.error().message;

        // Load TinyLlama model
        auto inference_result = TinyLlamaInference::from_model_dir(model_dir, backend.get());
        ASSERT_TRUE(inference_result.has_value())
            << "Failed to load TinyLlama model: " << inference_result.error().message;

        inference = std::make_unique<TinyLlamaInference>(std::move(*inference_result));
    }

    void TearDown() override {
        inference.reset();
        if (backend) {
            backend->cleanup();
        }
    }

    // Helper: Compare two MLX arrays with tolerance
    void compare_arrays(const mx::array& cpp_array, const mx::array& python_array,
                       const std::string& name, float rtol = 1e-3f, float atol = 1e-4f) {
        ASSERT_EQ(cpp_array.shape().size(), python_array.shape().size())
            << name << ": Shape dimension mismatch";

        for (size_t i = 0; i < cpp_array.shape().size(); ++i) {
            ASSERT_EQ(cpp_array.shape()[i], python_array.shape()[i])
                << name << ": Shape mismatch at dimension " << i;
        }

        // Convert both to float32 for comparison
        auto cpp_f32 = mx::astype(cpp_array, mx::float32);
        auto py_f32 = mx::astype(python_array, mx::float32);
        mx::eval(cpp_f32, py_f32);

        // Compute relative and absolute differences
        auto diff = mx::abs(cpp_f32 - py_f32);
        auto rel_diff = diff / (mx::abs(py_f32) + 1e-8f);

        mx::eval(diff, rel_diff);

        auto max_abs_diff = mx::max(diff).item<float>();
        auto max_rel_diff = mx::max(rel_diff).item<float>();

        EXPECT_LE(max_abs_diff, atol)
            << name << ": Max absolute difference " << max_abs_diff << " exceeds tolerance " << atol;
        EXPECT_LE(max_rel_diff, rtol)
            << name << ": Max relative difference " << max_rel_diff << " exceeds tolerance " << rtol;

        if (max_abs_diff <= atol && max_rel_diff <= rtol) {
            std::cout << "✓ " << name << " matches (max_abs_diff=" << max_abs_diff
                     << ", max_rel_diff=" << max_rel_diff << ")" << std::endl;
        }
    }

    std::filesystem::path model_dir;
    std::filesystem::path baseline_dir;
    std::unique_ptr<ComputeBackend> backend;
    std::unique_ptr<TinyLlamaInference> inference;
};

TEST_F(AttentionQKVTraceTest, CompareWithPythonBaseline) {
    // Load Python baseline
    auto baseline_file = baseline_dir / "attention_full_baseline.safetensors";
    ASSERT_TRUE(std::filesystem::exists(baseline_file))
        << "Baseline file not found: " << baseline_file;

    std::cout << "Loading baseline from: " << baseline_file << std::endl;
    auto baseline_data = mx::load_safetensors(baseline_file.string());
    auto& baseline_arrays = baseline_data.first;

    // Get input from baseline to ensure exact same input
    auto input_mlx = baseline_arrays.at("input");
    mx::eval(input_mlx);

    std::cout << "Input shape: [" << input_mlx.shape()[0] << ", "
              << input_mlx.shape()[1] << "]" << std::endl;
    std::cout << "Input dtype: " << input_mlx.dtype() << std::endl;

    // Wrap as Tensor (need to convert to bfloat16 to match model dtype)
    auto input_bf16 = mx::astype(input_mlx, mx::bfloat16);
    mx::eval(input_bf16);
    auto* input_array_ptr = new mx::array(input_bf16);
    auto input_tensor = backend->wrap_native_tensor(input_array_ptr, {5, 2048});

    // Call attention_layer - this is what we're testing!
    const int layer_idx = 0;
    std::cout << "\nRunning C++ attention_layer..." << std::endl;
    auto attention_output = inference->attention_layer(input_tensor, layer_idx);

    // Expect success now that implementation is complete
    ASSERT_TRUE(attention_output.has_value())
        << "attention_layer failed: " << attention_output.error().message;

    std::cout << "✓ attention_layer completed successfully" << std::endl;

    // Extract C++ output as MLX array
    auto cpp_output = const_cast<Tensor&>(*attention_output).to_mlx();
    mx::eval(cpp_output);

    std::cout << "C++ output shape: [" << cpp_output.shape()[0] << ", "
              << cpp_output.shape()[1] << "]" << std::endl;
    std::cout << "C++ output dtype: " << cpp_output.dtype() << std::endl;

    // Get Python baseline output
    auto python_output = baseline_arrays.at("final_output");

    std::cout << "Python output shape: [" << python_output.shape()[0] << ", "
              << python_output.shape()[1] << "]" << std::endl;

    // Compare outputs
    // Note: We use relaxed tolerances because of bfloat16 quantization
    std::cout << "\nComparing C++ vs Python outputs..." << std::endl;
    compare_arrays(cpp_output, python_output, "final_output",
                   1e-2f,  // rtol: 1% relative tolerance (relaxed for bfloat16)
                   1e-2f); // atol: 0.01 absolute tolerance

    std::cout << "\n✓ Full attention layer test passed!" << std::endl;
}

#endif // MLX_BACKEND_ENABLED

} // namespace compute