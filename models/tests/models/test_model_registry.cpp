#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include "models/registry/model_registry.h"

using namespace models::registry;

class MockFileSystem : public FileSystemInterface {
public:
    // Mock data using standard C++ containers
    std::unordered_set<std::string> existingPaths;
    std::unordered_set<std::string> directories;
    std::unordered_map<std::string, std::vector<std::string>> directoryContents;

    bool exists(const std::string& path) const override {
        return existingPaths.find(path) != existingPaths.end();
    }

    bool isDir(const std::string& path) const override {
        return directories.find(path) != directories.end();
    }

    std::vector<std::string> entryList(const std::string& path,
                                      const std::vector<std::string>& nameFilters) const override {
        auto it = directoryContents.find(path);
        if (it == directoryContents.end()) {
            return {};
        }

        const std::vector<std::string>& allFiles = it->second;

        if (nameFilters.empty()) {
            return allFiles;
        }

        std::vector<std::string> filteredFiles;
        for (const std::string& file : allFiles) {
            for (const std::string& filter : nameFilters) {
                if (filter.starts_with("*.")) {
                    std::string extension = filter.substr(1); // Remove *
                    if (file.ends_with(extension)) {
                        filteredFiles.push_back(file);
                        break;
                    }
                } else if (file == filter) {
                    filteredFiles.push_back(file);
                    break;
                }
            }
        }
        return filteredFiles;
    }

    std::string absolutePath(const std::string& path) const override {
        return path; // Simplified for testing
    }
};

class ModelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for each test
    }

    void TearDown() override {
        // Common cleanup for each test
    }

    void setupMockFileSystem(MockFileSystem& fs, const std::string& modelPath, ModelFormat format) {
        fs.existingPaths.insert(modelPath);
        fs.directories.insert(modelPath);

        std::vector<std::string> files;
        switch (format) {
        case ModelFormat::SAFETENSORS:
            files = {"weights.00.safetensors", "config.json", "tokenizer_config.json",
                     "tokenizer.json", "special_tokens_map.json"};
            break;
        case ModelFormat::GGUF:
            files = {"model.gguf"};
            break;
        case ModelFormat::UNKNOWN:
            files = {"README.md"}; // No model files
            break;
        }

        fs.directoryContents[modelPath] = files;

        // Add config files to existing paths for safetensors
        if (format == ModelFormat::SAFETENSORS) {
            for (const std::string& file : files) {
                std::filesystem::path filePath(modelPath);
                filePath /= file;
                fs.existingPaths.insert(filePath.string());
            }
        }
    }
};

TEST_F(ModelRegistryTest, SafetensorsModelDetection) {
    MockFileSystem mockFs;
    const std::string modelsDir = "/test/models";
    const std::string modelName = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit";
    const std::string modelPath = modelsDir + "/" + modelName;

    setupMockFileSystem(mockFs, modelPath, ModelFormat::SAFETENSORS);

    ModelRegistry registry(modelsDir, &mockFs);
    const ModelLocation location = registry.locateModel(modelName);

    EXPECT_TRUE(location.isValid());
    EXPECT_EQ(location.modelName, modelName);
    EXPECT_EQ(location.modelPath, modelPath);
    EXPECT_EQ(location.format, ModelFormat::SAFETENSORS);
    EXPECT_EQ(location.modelFiles.size(), 1);
    EXPECT_TRUE(location.modelFiles[0].ends_with("weights.00.safetensors"));
    EXPECT_FALSE(location.configPath.empty());
    EXPECT_FALSE(location.tokenizerConfigPath.empty());
    EXPECT_FALSE(location.vocabPath.empty());
    EXPECT_FALSE(location.specialTokensPath.empty());
}

TEST_F(ModelRegistryTest, GgufModelDetection) {
    MockFileSystem mockFs;
    const std::string modelsDir = "/test/models";
    const std::string modelName = "test/model-gguf";
    const std::string modelPath = modelsDir + "/" + modelName;

    setupMockFileSystem(mockFs, modelPath, ModelFormat::GGUF);

    ModelRegistry registry(modelsDir, &mockFs);
    const ModelLocation location = registry.locateModel(modelName);

    EXPECT_TRUE(location.isValid());
    EXPECT_EQ(location.format, ModelFormat::GGUF);
    EXPECT_EQ(location.modelFiles.size(), 1);
    EXPECT_TRUE(location.modelFiles[0].ends_with("model.gguf"));
    // GGUF models shouldn't have separate config files
    EXPECT_TRUE(location.configPath.empty());
    EXPECT_TRUE(location.tokenizerConfigPath.empty());
    EXPECT_TRUE(location.vocabPath.empty());
    EXPECT_TRUE(location.specialTokensPath.empty());
}

TEST_F(ModelRegistryTest, ModelNotFound) {
    MockFileSystem mockFs;
    const std::string modelsDir = "/test/models";

    ModelRegistry registry(modelsDir, &mockFs);
    const ModelLocation location = registry.locateModel("nonexistent/model");

    EXPECT_FALSE(location.isValid());
    EXPECT_EQ(location.format, ModelFormat::UNKNOWN);
}

TEST_F(ModelRegistryTest, SafetensorsConfigFiles) {
    MockFileSystem mockFs;
    const std::string modelsDir = "/test/models";
    const std::string modelName = "test/model";
    const std::string modelPath = modelsDir + "/" + modelName;

    // Setup model with only some config files
    mockFs.existingPaths.insert(modelPath);
    mockFs.directories.insert(modelPath);
    mockFs.directoryContents[modelPath] = {"weights.safetensors", "config.json", "tokenizer.json"};

    // Add file paths
    mockFs.existingPaths.insert(modelPath + "/weights.safetensors");
    mockFs.existingPaths.insert(modelPath + "/config.json");
    mockFs.existingPaths.insert(modelPath + "/tokenizer.json");

    ModelRegistry registry(modelsDir, &mockFs);
    const ModelLocation location = registry.locateModel(modelName);

    EXPECT_TRUE(location.isValid());
    EXPECT_FALSE(location.configPath.empty());
    EXPECT_FALSE(location.vocabPath.empty());
    // These should be empty since they weren't provided
    EXPECT_TRUE(location.tokenizerConfigPath.empty());
    EXPECT_TRUE(location.specialTokensPath.empty());
}