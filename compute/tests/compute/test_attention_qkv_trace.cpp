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
    static void SetUpTestSuite() {
        model_dir_    = TINYLLAMA_MODEL_DIR;
        baseline_dir_ = std::filesystem::path(TEST_RESOURCES_DIR).parent_path() / "baselines" / "output";

        if (!std::filesystem::exists(model_dir_)) {
            skip_reason_ = "Model not found: " + model_dir_.string();
            return;
        }

        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (!backend_result) { skip_reason_ = backend_result.error().message; return; }
        backend_ = std::move(*backend_result);
        if (!backend_->initialize()) { skip_reason_ = "Backend init failed"; return; }

        auto inference_result = TinyLlamaInference::from_model_dir(model_dir_, backend_.get());
        if (!inference_result) { skip_reason_ = inference_result.error().message; return; }
        inference_ = std::make_unique<TinyLlamaInference>(std::move(*inference_result));
    }

    static void TearDownTestSuite() {
        inference_.reset();
        if (backend_) backend_->cleanup();
        backend_.reset();
        skip_reason_.clear();
    }

    void SetUp() override {
        if (!skip_reason_.empty())
            GTEST_SKIP() << skip_reason_;
    }

    void compare_arrays(const mx::array& cpp_array, const mx::array& python_array,
                        const std::string& name, float rtol = 1e-3f, float atol = 1e-4f) {
        ASSERT_EQ(cpp_array.shape().size(), python_array.shape().size())
            << name << ": Shape dimension mismatch";
        for (size_t i = 0; i < cpp_array.shape().size(); ++i) {
            ASSERT_EQ(cpp_array.shape()[i], python_array.shape()[i])
                << name << ": Shape mismatch at dimension " << i;
        }

        auto cpp_f32 = mx::astype(cpp_array, mx::float32);
        auto py_f32  = mx::astype(python_array, mx::float32);
        mx::eval(cpp_f32, py_f32);

        auto diff     = mx::abs(cpp_f32 - py_f32);
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

    static std::filesystem::path              model_dir_;
    static std::filesystem::path              baseline_dir_;
    static std::string                        skip_reason_;
    static std::unique_ptr<ComputeBackend>    backend_;
    static std::unique_ptr<TinyLlamaInference> inference_;
};

std::filesystem::path               AttentionQKVTraceTest::model_dir_;
std::filesystem::path               AttentionQKVTraceTest::baseline_dir_;
std::string                         AttentionQKVTraceTest::skip_reason_;
std::unique_ptr<ComputeBackend>     AttentionQKVTraceTest::backend_;
std::unique_ptr<TinyLlamaInference> AttentionQKVTraceTest::inference_;

TEST_F(AttentionQKVTraceTest, CompareWithPythonBaseline) {
    auto baseline_file = baseline_dir_ / "attention_full_baseline.safetensors";
    ASSERT_TRUE(std::filesystem::exists(baseline_file))
        << "Baseline file not found: " << baseline_file;

    std::cout << "Loading baseline from: " << baseline_file << std::endl;
    auto baseline_data   = mx::load_safetensors(baseline_file.string());
    auto& baseline_arrays = baseline_data.first;

    auto input_mlx = baseline_arrays.at("input");
    mx::eval(input_mlx);

    std::cout << "Input shape: [" << input_mlx.shape()[0] << ", "
              << input_mlx.shape()[1] << "]" << std::endl;
    std::cout << "Input dtype: " << input_mlx.dtype() << std::endl;

    auto input_bf16 = mx::astype(input_mlx, mx::bfloat16);
    mx::eval(input_bf16);
    auto* input_array_ptr = new mx::array(input_bf16);
    auto input_tensor = backend_->wrap_native_tensor(input_array_ptr, {5, 2048});

    const int layer_idx = 0;
    std::cout << "\nRunning C++ attention_layer..." << std::endl;
    auto attention_output = inference_->attention_layer(input_tensor, layer_idx);

    ASSERT_TRUE(attention_output.has_value())
        << "attention_layer failed: " << attention_output.error().message;

    std::cout << "✓ attention_layer completed successfully" << std::endl;

    auto cpp_output = const_cast<Tensor&>(*attention_output).to_mlx();
    mx::eval(cpp_output);

    std::cout << "C++ output shape: [" << cpp_output.shape()[0] << ", "
              << cpp_output.shape()[1] << "]" << std::endl;

    auto python_output = baseline_arrays.at("final_output");
    std::cout << "Python output shape: [" << python_output.shape()[0] << ", "
              << python_output.shape()[1] << "]" << std::endl;

    std::cout << "\nComparing C++ vs Python outputs..." << std::endl;
    compare_arrays(cpp_output, python_output, "final_output", 1e-2f, 1e-2f);

    std::cout << "\n✓ Full attention layer test passed!" << std::endl;
}

#endif // MLX_BACKEND_ENABLED

} // namespace compute
