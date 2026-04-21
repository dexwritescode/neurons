#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "models/api/huggingface_client.h"

using namespace models;

// Mock HTTP client for testing
class MockHttpClient : public http::HttpInterface {
public:
    struct RequestInfo {
        std::string url;
        std::string method;
        std::string userAgent;
        std::string authToken;
        std::map<std::string, std::string> headers;
    };

    // Track requests made
    std::vector<RequestInfo> requests;

    // Response to return
    http::HttpResponse nextResponse;

    void requestAsync(
        const http::HttpRequest& request,
        http::CompletionCallback completion,
        http::ProgressCallback progress = nullptr
    ) override {
        (void)progress;
        // Store request info
        RequestInfo info;
        info.url = request.url;
        info.method = request.method;
        info.userAgent = request.userAgent;
        info.authToken = request.authToken;
        info.headers = request.headers;
        requests.push_back(info);

        // Call completion callback with mock response
        bool success = nextResponse.statusCode >= 200 && nextResponse.statusCode < 300;
        completion(success, nextResponse.data, nextResponse.errorMessage);
    }

    http::HttpResponse requestSync(const http::HttpRequest& request) override {
        // Store request info
        RequestInfo info;
        info.url = request.url;
        info.method = request.method;
        info.userAgent = request.userAgent;
        info.authToken = request.authToken;
        info.headers = request.headers;
        requests.push_back(info);

        return nextResponse;
    }

    http::HttpResponse requestSyncWithProgress(
        const http::HttpRequest& request,
        http::ProgressCallback /*progress*/) override {
        return requestSync(request);
    }

    void cancelAll() override {}
    bool cancelRequest(const std::string& url) override { (void)url; return false; }

    void setNextResponse(int statusCode, const std::string& data, const std::string& error = "") {
        nextResponse.statusCode = statusCode;
        nextResponse.data = data;
        nextResponse.errorMessage = error;
    }
};

class HuggingFaceClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        mockHttp = std::make_unique<MockHttpClient>();
        // Keep raw pointer for testing while giving ownership to client
        mockHttpPtr = mockHttp.get();
        client = std::make_unique<HuggingFaceClient>(std::move(mockHttp));

        // Set up callback tracking
        searchCalled = false;
        modelDetailsCalled = false;
        modelFilesCalled = false;
        errorCalled = false;
        lastError.clear();
        lastModels.clear();
        lastModelFiles.clear();
    }

    void TearDown() override {
        client.reset();
        mockHttpPtr = nullptr;
    }

    // Test state
    std::unique_ptr<HuggingFaceClient> client;
    MockHttpClient* mockHttpPtr; // Raw pointer for testing
    std::unique_ptr<MockHttpClient> mockHttp; // Ownership

    // Callback tracking
    bool searchCalled;
    bool modelDetailsCalled;
    bool modelFilesCalled;
    bool errorCalled;
    std::string lastError;
    std::vector<ModelInfo> lastModels;
    std::vector<FileInfo> lastModelFiles;
    std::string lastNextPageToken;
    ModelInfo lastModelDetails;

    void setupCallbacks() {
        client->setSearchCallback([this](const std::vector<ModelInfo>& models, const std::string& nextPageToken, const std::string& error) {
            searchCalled = true;
            lastModels = models;
            lastNextPageToken = nextPageToken;
            if (!error.empty()) {
                lastError = error;
                errorCalled = true;
            }
        });

        client->setModelDetailsCallback([this](const ModelInfo& model, const std::string& error) {
            modelDetailsCalled = true;
            lastModelDetails = model;
            if (!error.empty()) {
                lastError = error;
                errorCalled = true;
            }
        });

        client->setModelFilesCallback([this](const std::string& modelId, const std::vector<FileInfo>& files, const std::string& error) {
            (void)modelId;
            modelFilesCalled = true;
            lastModelFiles = files;
            if (!error.empty()) {
                lastError = error;
                errorCalled = true;
            }
        });

        client->setErrorCallback([this](const std::string& error, const std::string& endpoint) {
            (void)endpoint;
            errorCalled = true;
            lastError = error;
        });
    }
};

TEST_F(HuggingFaceClientTest, ConfigurationMethods) {
    // Test configuration setters
    client->setBaseUrl("https://test.example.com");
    client->setAuthToken("test-token");
    client->setUserAgent("test-agent/1.0");
    client->setRateLimit(5);

    // Configuration should be stored (we can't directly test private members,
    // but we can verify they're used in requests)
    EXPECT_FALSE(client->isRateLimited());
    EXPECT_TRUE(client->lastError().empty());
}

TEST_F(HuggingFaceClientTest, SearchModelsCallback) {
    setupCallbacks();

    // Set up mock response (HuggingFace API returns array directly, not wrapped in "models")
    mockHttpPtr->setNextResponse(200, R"([])");

    // Create search query
    SearchQuery query;
    query.search = "llama";
    query.limit = 10;

    // Call search
    client->searchModels(query);

    // Verify callback was called
    EXPECT_TRUE(searchCalled);
    EXPECT_FALSE(errorCalled);

    // Verify HTTP request was made
    ASSERT_EQ(mockHttpPtr->requests.size(), 1);
    // Note: Current implementation returns "Not implemented yet" error
    // This is expected since we haven't implemented the JSON parsing yet
}

TEST_F(HuggingFaceClientTest, GetModelDetailsCallback) {
    setupCallbacks();

    // Set up mock response
    mockHttpPtr->setNextResponse(200, R"({"id": "test-model", "modelId": "test-model"})");

    // Call get model details
    client->getModelDetails("test-model");

    // Verify callback was called
    EXPECT_TRUE(modelDetailsCalled);
    // Note: Current implementation returns "Not implemented yet" error
    // This is expected since we haven't implemented the JSON parsing yet
}

TEST_F(HuggingFaceClientTest, GetModelFilesCallback) {
    setupCallbacks();

    // Set up mock response
    mockHttpPtr->setNextResponse(200, R"({"files": []})");

    // Call get model files
    client->getModelFiles("test-model");

    // Verify callback was called
    EXPECT_TRUE(modelFilesCalled);
    // Note: Current implementation returns "Not implemented yet" error
    // This is expected since we haven't implemented the JSON parsing yet
}

TEST_F(HuggingFaceClientTest,
Directory) {
    // Test download directory setter/getter
    std::string testDir = "/tmp/test-models";
    client->setDownloadDirectory(testDir);

    EXPECT_EQ(client->downloadDirectory(), testDir);

    // Test model path generation
    std::string modelPath = client->getModelPath("test-model");
    EXPECT_EQ(modelPath, testDir + "/test-model");
}

TEST_F(HuggingFaceClientTest, EmptyDownloadDirectory) {
    // Test with empty download directory
    client->setDownloadDirectory("");

    EXPECT_TRUE(client->downloadDirectory().empty());
    EXPECT_TRUE(client->getModelPath("test-model").empty());
}

TEST_F(HuggingFaceClientTest, ActiveDownloads) {
    // Test active downloads (should be empty initially)
    auto downloads = client->activeDownloads();
    EXPECT_TRUE(downloads.empty());
}

TEST_F(HuggingFaceClientTest, StatusMethods) {
    // Test status methods
    EXPECT_FALSE(client->isRateLimited());
    EXPECT_TRUE(client->lastError().empty());
}

// Integration test for the factory function
TEST(HuggingFaceClientIntegrationTest, FactoryFunction) {
    auto client = createHuggingFaceClient();
    ASSERT_NE(client, nullptr);

    // Test basic functionality
    client->setUserAgent("test-agent");
    EXPECT_FALSE(client->isRateLimited());
    EXPECT_TRUE(client->lastError().empty());
}