#include <gtest/gtest.h>
#include "compute/model/model_loader.h"
#include "test_config.h"
#include <filesystem>

namespace compute {

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        tinyllama_model_dir = TINYLLAMA_MODEL_DIR;
        test_resources_dir  = TEST_RESOURCES_DIR;
    }

    std::filesystem::path tinyllama_model_dir;
    std::filesystem::path test_resources_dir;
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
    auto temp_dir = std::filesystem::temp_directory_path() / "test_model_no_config";
    std::filesystem::create_directories(temp_dir);

    auto result = ModelLoader::load_config(temp_dir);
    EXPECT_FALSE(result.has_value());

    std::filesystem::remove_all(temp_dir);
}

} // namespace compute
