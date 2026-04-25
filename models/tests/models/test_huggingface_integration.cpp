#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <thread>
#include <filesystem>
#include "models/api/huggingface_client.h"

using namespace models;

/**
 * Integration tests for HuggingFaceClient that make real API calls.
 * These tests require internet connectivity and may be slower.
 *
 * Run with: --gtest_filter="*Integration*"
 * Skip with: --gtest_filter="-*Integration*"
 */
class HuggingFaceIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        client = createHuggingFaceClient();

        // Set up test directory
        testDir = std::filesystem::temp_directory_path() / "neurons_test_models";
        std::filesystem::create_directories(testDir);
        client->setDownloadDirectory(testDir.string());

        // Reset callback flags
        searchComplete = false;
        modelDetailsComplete = false;
        modelFilesComplete = false;
        downloadStarted = false;
        downloadComplete = false;
        downloadFailed = false;
        errorOccurred = false;

        setupCallbacks();
    }

    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
        client.reset();
    }

    void setupCallbacks() {
        client->setSearchCallback([this](const std::vector<ModelInfo>& models, const std::string& nextPageToken, const std::string& error) {
            lastModels = models;
            lastNextPageToken = nextPageToken;
            lastError = error;
            searchComplete = true;
            if (!error.empty()) errorOccurred = true;
        });

        client->setModelDetailsCallback([this](const ModelInfo& model, const std::string& error) {
            lastModelDetails = model;
            lastError = error;
            modelDetailsComplete = true;
            if (!error.empty()) errorOccurred = true;
        });

        client->setModelFilesCallback([this](const std::string& modelId, const std::vector<FileInfo>& files, const std::string& error) {
            lastModelId = modelId;
            lastFiles = files;
            lastError = error;
            modelFilesComplete = true;
            if (!error.empty()) errorOccurred = true;
        });

        client->setDownloadStartCallback([this](const std::string& downloadId, const std::string& modelId) {
            lastDownloadId = downloadId;
            lastModelId = modelId;
            downloadStarted = true;
        });

        client->setDownloadCompleteCallback([this](const std::string& downloadId, const std::string& modelPath, int64_t totalBytes) {
            lastDownloadId = downloadId;
            lastModelPath = modelPath;
            lastTotalBytes = totalBytes;
            downloadComplete = true;
        });

        client->setDownloadFailedCallback([this](const std::string& downloadId, const std::string& error) {
            lastDownloadId = downloadId;
            lastError = error;
            downloadFailed = true;
            errorOccurred = true;
        });
    }

    // Wait for async callback with timeout
    bool waitForCallback(std::function<bool()> predicate, int timeoutMs = 10000) {
        auto start = std::chrono::steady_clock::now();
        while (!predicate()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
            if (elapsed.count() > timeoutMs) {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<HuggingFaceClient> client;
    std::filesystem::path testDir;

    // Callback state
    bool searchComplete;
    bool modelDetailsComplete;
    bool modelFilesComplete;
    bool downloadStarted;
    bool downloadComplete;
    bool downloadFailed;
    bool errorOccurred;

    // Result data
    std::vector<ModelInfo> lastModels;
    std::string lastNextPageToken;
    ModelInfo lastModelDetails;
    std::vector<FileInfo> lastFiles;
    std::string lastModelId;
    std::string lastDownloadId;
    std::string lastModelPath;
    int64_t lastTotalBytes;
    std::string lastError;
};

TEST_F(HuggingFaceIntegrationTest, SearchModelsBasic) {
    // Search for a popular model
    SearchQuery query;
    query.search = "gpt2";
    query.limit = 5;

    client->searchModels(query);

    // Wait for callback
    ASSERT_TRUE(waitForCallback([this]() { return searchComplete; }));

    // Verify results
    EXPECT_FALSE(errorOccurred) << "Error: " << lastError;
    EXPECT_FALSE(lastModels.empty()) << "Should find some GPT-2 models";

    // Check first model has basic info
    if (!lastModels.empty()) {
        const auto& model = lastModels[0];
        EXPECT_FALSE(model.id.empty());
        EXPECT_FALSE(model.name.empty());
        // GPT-2 models should be text generation
        EXPECT_EQ(model.type, ModelType::TEXT_GENERATION);
    }
}

TEST_F(HuggingFaceIntegrationTest, GetModelDetailsGPT2) {
    // Get details for the original GPT-2 model
    client->getModelDetails("gpt2");

    // Wait for callback
    ASSERT_TRUE(waitForCallback([this]() { return modelDetailsComplete; }));

    // Verify results
    EXPECT_FALSE(errorOccurred) << "Error: " << lastError;
    // HF redirects the short alias "gpt2" to "openai-community/gpt2"
    EXPECT_NE(lastModelDetails.id.find("gpt2"), std::string::npos);
    EXPECT_FALSE(lastModelDetails.name.empty());
    EXPECT_EQ(lastModelDetails.type, ModelType::TEXT_GENERATION);
    EXPECT_GT(lastModelDetails.downloads, 0); // Popular model should have downloads
}

TEST_F(HuggingFaceIntegrationTest, GetModelFilesGPT2) {
    // Get files for GPT-2 model
    client->getModelFiles("gpt2");

    // Wait for callback
    ASSERT_TRUE(waitForCallback([this]() { return modelFilesComplete; }));

    // Verify results
    EXPECT_FALSE(errorOccurred) << "Error: " << lastError;
    EXPECT_FALSE(lastFiles.empty()) << "GPT-2 should have files";

    // Check we have some expected files
    bool hasConfig = false;
    bool hasModel = false;
    for (const auto& file : lastFiles) {
        EXPECT_FALSE(file.filename.empty());
        EXPECT_FALSE(file.downloadUrl.empty());
        EXPECT_GT(file.sizeBytes, 0);

        if (file.filename.find("config.json") != std::string::npos) hasConfig = true;
        if (file.filename.find(".bin") != std::string::npos ||
            file.filename.find(".safetensors") != std::string::npos) hasModel = true;
    }

    EXPECT_TRUE(hasConfig) << "Should have config.json";
    EXPECT_TRUE(hasModel) << "Should have model files";
}

TEST_F(HuggingFaceIntegrationTest, DownloadSmallModel) {
    // First get model details to have file info
    client->getModelDetails("gpt2");
    ASSERT_TRUE(waitForCallback([this]() { return modelDetailsComplete; }));
    EXPECT_FALSE(errorOccurred) << "Error getting model details: " << lastError;

    // Then get model files
    client->getModelFiles("gpt2");
    ASSERT_TRUE(waitForCallback([this]() { return modelFilesComplete; }));
    EXPECT_FALSE(errorOccurred) << "Error getting model files: " << lastError;
    EXPECT_FALSE(lastFiles.empty());

    // Create a ModelInfo with just the config file for testing (small download)
    ModelInfo testModel;
    testModel.id = "gpt2";
    testModel.name = "gpt2";
    testModel.type = ModelType::TEXT_GENERATION;

    // Find just the config.json file for a small test download
    for (const auto& file : lastFiles) {
        if (file.filename == "config.json") {
            testModel.files.push_back(file);
            break;
        }
    }

    ASSERT_FALSE(testModel.files.empty()) << "Should find config.json file";

    // Start download
    std::string downloadId = client->downloadModel(testModel);
    EXPECT_FALSE(downloadId.empty());

    // Wait for download to start
    ASSERT_TRUE(waitForCallback([this]() { return downloadStarted; }, 5000));
    EXPECT_EQ(lastModelId, "gpt2");

    // Wait for download to complete (should be quick for just config.json)
    ASSERT_TRUE(waitForCallback([this]() { return downloadComplete || downloadFailed; }, 15000));

    if (downloadFailed) {
        ADD_FAILURE() << "Download failed: " << lastError;
    } else {
        EXPECT_TRUE(downloadComplete);
        EXPECT_FALSE(lastModelPath.empty());
        EXPECT_GT(lastTotalBytes, 0);

        // Verify file was actually downloaded
        std::filesystem::path downloadedFile = std::filesystem::path(lastModelPath) / "config.json";
        EXPECT_TRUE(std::filesystem::exists(downloadedFile))
            << "Downloaded file should exist: " << downloadedFile;

        if (std::filesystem::exists(downloadedFile)) {
            EXPECT_GT(std::filesystem::file_size(downloadedFile), 0)
                << "Downloaded file should not be empty";
        }
    }
}

TEST_F(HuggingFaceIntegrationTest, ErrorHandlingInvalidModel) {
    // Try to get details for a non-existent model
    client->getModelDetails("this-model-definitely-does-not-exist-123456789");

    // Wait for callback
    ASSERT_TRUE(waitForCallback([this]() { return modelDetailsComplete; }));

    // Should get an error
    EXPECT_TRUE(errorOccurred);
    EXPECT_FALSE(lastError.empty());
    EXPECT_TRUE(lastModelDetails.id.empty()); // Should not populate model info on error
}

// Test that verifies the HTTP client is working
TEST_F(HuggingFaceIntegrationTest, BasicConnectivity) {
    // This is a simple test to verify we can connect to HuggingFace
    SearchQuery query;
    query.limit = 1; // Just get one result

    client->searchModels(query);

    // Wait for response
    ASSERT_TRUE(waitForCallback([this]() { return searchComplete; }));

    // Should not have network errors
    EXPECT_FALSE(errorOccurred) << "Network error: " << lastError;
}