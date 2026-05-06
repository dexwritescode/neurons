#include <gtest/gtest.h>
#include "compute/model/tinyllama_inference.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>
#include <vector>
#include <iostream>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#include <mlx/mlx.h>

namespace compute {

class ForwardPassTest : public ::testing::Test {
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

        auto inf_result = TinyLlamaInference::from_model_dir(model_dir_, backend_.get());
        if (!inf_result) { skip_reason_ = inf_result.error().message; return; }
        inference_ = std::make_unique<TinyLlamaInference>(std::move(*inf_result));
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

    static std::filesystem::path              model_dir_;
    static std::filesystem::path              baseline_dir_;
    static std::string                        skip_reason_;
    static std::unique_ptr<ComputeBackend>    backend_;
    static std::unique_ptr<TinyLlamaInference> inference_;
};

std::filesystem::path               ForwardPassTest::model_dir_;
std::filesystem::path               ForwardPassTest::baseline_dir_;
std::string                         ForwardPassTest::skip_reason_;
std::unique_ptr<ComputeBackend>     ForwardPassTest::backend_;
std::unique_ptr<TinyLlamaInference> ForwardPassTest::inference_;

TEST_F(ForwardPassTest, GenerateCoherentOutput) {
    const std::string prompt = "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    auto token_ids = inference_->tokenizer().encode(prompt, /*add_special_tokens=*/true);

    ASSERT_FALSE(token_ids.empty());
    std::cout << "Prompt token count: " << token_ids.size() << std::endl;

    compute::SamplingParams greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;
    auto generated_result = inference_->generate(token_ids, /*max_new_tokens=*/30, greedy);
    ASSERT_TRUE(generated_result.has_value()) << generated_result.error().message;

    const auto& generated_ids = *generated_result;
    ASSERT_FALSE(generated_ids.empty());

    std::string output = inference_->tokenizer().decode(generated_ids);
    std::cout << "Generated output: \"" << output << "\"" << std::endl;

    bool mentions_paris = output.find("Paris") != std::string::npos ||
                          output.find("paris") != std::string::npos;
    EXPECT_TRUE(mentions_paris)
        << "Expected 'Paris' in output, got: \"" << output << "\"";
}

} // namespace compute

#endif // MLX_BACKEND_ENABLED
