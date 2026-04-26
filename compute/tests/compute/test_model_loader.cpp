#include <gtest/gtest.h>
#include "compute/model/model_loader.h"
#include "compute/core/compute_backend.h"
#include "test_config.h"
#include <filesystem>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#endif

namespace compute {

// Create a minimal backend mock for testing
class MockBackend : public ComputeBackend {
public:
    BackendType type() const override { return BackendType::MLX; }
    std::string name() const override { return "Mock"; }
    bool is_available() const override { return false; }
    Result<void> initialize() override { return {}; }
    void cleanup() override {}

    // Create dummy tensors with nullptr buffer - these won't actually work but satisfy the interface
    Tensor create_tensor(std::span<const float> data, std::vector<size_t> shape) override {
        return Tensor(nullptr, shape);
    }
    Tensor create_tensor(std::vector<size_t> shape) override {
        return Tensor(nullptr, shape);
    }
    Tensor wrap_native_tensor(void*, std::vector<size_t> shape) override {
        return Tensor(nullptr, shape);
    }

    Tensor dot_product(const Tensor& a, const Tensor& b) override {
        return Tensor(nullptr, {1});
    }
    Tensor matrix_scalar_add(const Tensor& input, float) override {
        return Tensor(nullptr, input.shape());
    }

    Result<Tensor> matmul(const Tensor&, const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> dequantize(const Tensor&, const Tensor&, const Tensor&, int, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> quantized_matmul(const Tensor&, const Tensor&, const Tensor&, const Tensor*, bool, int, int, const std::string&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> add(const Tensor&, const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> multiply(const Tensor&, const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> softmax(const Tensor&, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> silu(const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> gelu(const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> sigmoid(const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> conv1d(const Tensor&, const Tensor&, int, int, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> transpose(const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> swapaxes(const Tensor&, int, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> reshape(const Tensor&, const std::vector<size_t>&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> concatenate(const std::vector<Tensor>&, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> mean(const Tensor&, int, bool) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> rsqrt(const Tensor&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> slice(const Tensor&, int, int, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> repeat(const Tensor&, int, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> triu(const Tensor&, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> rms_norm(const Tensor&, const Tensor&, float) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> rope(const Tensor&, int, float, int) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<Tensor> scaled_dot_product_attention(const Tensor&, const Tensor&, const Tensor&, float, const std::string&) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }

    Result<void> extract(const Tensor&, std::span<float>) override {
        return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
    }
    Result<void> evaluate_all() override { return {}; }

    std::unordered_map<std::string, Tensor> load_model(const std::string&) override { return {}; }
};

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use configured paths from CMake
        tinyllama_model_dir = TINYLLAMA_MODEL_DIR;
        test_resources_dir = TEST_RESOURCES_DIR;

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
        // Create MLX backend for testing if available
        auto backend_result = BackendFactory::create(BackendType::MLX);
        if (backend_result.has_value()) {
            backend = std::move(*backend_result);
            auto init_result = backend->initialize();
            mlx_available = init_result.has_value();
        } else {
            mlx_available = false;
        }
#else
        mlx_available = false;
#endif
    }

    void TearDown() override {
        if (backend) {
            backend->cleanup();
        }
    }

    std::filesystem::path tinyllama_model_dir;
    std::filesystem::path test_resources_dir;
    std::unique_ptr<ComputeBackend> backend;
    bool mlx_available = false;
};

TEST_F(ModelLoaderTest, LoadConfigOnly) {
    if (!std::filesystem::exists(tinyllama_model_dir))
        GTEST_SKIP() << "Model not found: " << tinyllama_model_dir;

    auto result = ModelLoader::load_config(tinyllama_model_dir);
    ASSERT_TRUE(result.has_value())
        << "Failed to load config: " << result.error().message;

    const auto& config = *result;
    EXPECT_EQ(config.vocab_size, 32000);
    EXPECT_EQ(config.hidden_size, 2048);
    EXPECT_TRUE(config.is_valid());
    EXPECT_TRUE(config.is_llama_architecture());
}

TEST_F(ModelLoaderTest, LoadConfigNonExistentDirectory) {
    auto result = ModelLoader::load_config("/nonexistent/directory");
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("does not exist"), std::string::npos);
}

TEST_F(ModelLoaderTest, LoadConfigMissingConfigFile) {
    // Create temporary directory without config.json
    auto temp_dir = std::filesystem::temp_directory_path() / "test_model_no_config";
    std::filesystem::create_directories(temp_dir);

    auto result = ModelLoader::load_config(temp_dir);
    EXPECT_FALSE(result.has_value());

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

TEST_F(ModelLoaderTest, FindSafetensorsFiles) {
    if (!std::filesystem::exists(tinyllama_model_dir))
        GTEST_SKIP() << "Model not found: " << tinyllama_model_dir;

    // This tests file discovery indirectly by checking that load_model finds the files
    // We'll use a mock backend that claims to be something other than MLX
    class NonMLXMockBackend : public ComputeBackend {
    public:
        BackendType type() const override { return BackendType::Metal; } // Not MLX
        std::string name() const override { return "NonMLXMock"; }
        bool is_available() const override { return false; }
        Result<void> initialize() override { return {}; }
        void cleanup() override {}

        Tensor create_tensor(std::span<const float>, std::vector<size_t> shape) override {
            return Tensor(nullptr, shape);
        }
        Tensor create_tensor(std::vector<size_t> shape) override {
            return Tensor(nullptr, shape);
        }
        Tensor wrap_native_tensor(void*, std::vector<size_t> shape) override {
            return Tensor(nullptr, shape);
        }
        Tensor dot_product(const Tensor&, const Tensor&) override {
            return Tensor(nullptr, {1});
        }
        Tensor matrix_scalar_add(const Tensor&, float) override {
            return Tensor(nullptr, {1});
        }

        Result<Tensor> matmul(const Tensor&, const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> dequantize(const Tensor&, const Tensor&, const Tensor&, int, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> quantized_matmul(const Tensor&, const Tensor&, const Tensor&, const Tensor*, bool, int, int, const std::string&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> add(const Tensor&, const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> multiply(const Tensor&, const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> softmax(const Tensor&, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> silu(const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> gelu(const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> sigmoid(const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> conv1d(const Tensor&, const Tensor&, int, int, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> transpose(const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> swapaxes(const Tensor&, int, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> reshape(const Tensor&, const std::vector<size_t>&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> concatenate(const std::vector<Tensor>&, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> mean(const Tensor&, int, bool) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> rsqrt(const Tensor&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> slice(const Tensor&, int, int, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> repeat(const Tensor&, int, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> triu(const Tensor&, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> rms_norm(const Tensor&, const Tensor&, float) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> rope(const Tensor&, int, float, int) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<Tensor> scaled_dot_product_attention(const Tensor&, const Tensor&, const Tensor&, float, const std::string&) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<void> extract(const Tensor&, std::span<float>) override {
            return std::unexpected(Error{ErrorCode::BackendNotAvailable, "Mock"});
        }
        Result<void> evaluate_all() override { return {}; }
        std::unordered_map<std::string, Tensor> load_model(const std::string&) override { return {}; }
    };

    NonMLXMockBackend mock_backend;

    // This should fail because ModelLoader requires MLX backend for safetensors
    auto result = ModelLoader::load_model(tinyllama_model_dir, &mock_backend);

    // Should fail because non-MLX backend can't load safetensors
    EXPECT_FALSE(result.has_value());

    // But error should NOT be about missing safetensors files - should be about backend type
    EXPECT_EQ(result.error().message.find("No .safetensors files found"), std::string::npos);
    EXPECT_NE(result.error().message.find("MLX backend"), std::string::npos);
}

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

TEST_F(ModelLoaderTest, LoadCompleteModelWithMLX) {
    if (!mlx_available) {
        GTEST_SKIP() << "MLX backend not available";
    }

    if (!std::filesystem::exists(tinyllama_model_dir))
        GTEST_SKIP() << "Model not found: " << tinyllama_model_dir;

    ASSERT_NE(backend, nullptr) << "MLX backend not initialized";

    auto result = ModelLoader::load_model(tinyllama_model_dir, backend.get());
    ASSERT_TRUE(result.has_value())
        << "Failed to load model: " << result.error().message;

    const auto& [config, tensors] = *result;

    // Verify config is correct
    EXPECT_EQ(config.vocab_size, 32000);
    EXPECT_EQ(config.hidden_size, 2048);
    EXPECT_TRUE(config.is_valid());

    // Verify tensors were loaded
    EXPECT_FALSE(tensors.empty()) << "No tensors loaded";

    // Check for expected tensor patterns (basic smoke test)
    bool found_embedding = false;
    bool found_attention = false;
    bool found_mlp = false;
    bool found_norm = false;

    for (const auto& [name, tensor] : tensors) {
        if (name.find("embed_tokens") != std::string::npos) {
            found_embedding = true;
        }
        if (name.find("self_attn") != std::string::npos) {
            found_attention = true;
        }
        if (name.find("mlp") != std::string::npos) {
            found_mlp = true;
        }
        if (name.find("norm") != std::string::npos) {
            found_norm = true;
        }
    }

    EXPECT_TRUE(found_embedding) << "No embedding tensors found";
    EXPECT_TRUE(found_attention) << "No attention tensors found";
    EXPECT_TRUE(found_mlp) << "No MLP tensors found";
    EXPECT_TRUE(found_norm) << "No normalization tensors found";

    std::cout << "Successfully loaded " << tensors.size() << " tensors" << std::endl;
}

#endif

TEST_F(ModelLoaderTest, LoadModelNullBackend) {
    auto result = ModelLoader::load_model(tinyllama_model_dir, nullptr);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("Backend cannot be null"), std::string::npos);
}

TEST_F(ModelLoaderTest, LoadModelNonExistentDirectory) {
    MockBackend mock_backend;

    auto result = ModelLoader::load_model("/nonexistent/directory", &mock_backend);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("does not exist"), std::string::npos);
}

TEST_F(ModelLoaderTest, LoadModelDirectoryWithoutSafetensors) {
    MockBackend mock_backend;

    // Create temporary directory with only config.json
    auto temp_dir = std::filesystem::temp_directory_path() / "test_model_no_safetensors";
    std::filesystem::create_directories(temp_dir);

    // Copy config file
    std::filesystem::copy_file(test_resources_dir / "tinyllama_config.json",
                              temp_dir / "config.json");

    auto result = ModelLoader::load_model(temp_dir, &mock_backend);
    EXPECT_FALSE(result.has_value());
    EXPECT_NE(result.error().message.find("No .safetensors files found"), std::string::npos);

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

} // namespace compute