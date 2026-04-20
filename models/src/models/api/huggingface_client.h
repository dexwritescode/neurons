#pragma once

#include <functional>
#include <memory>
#include <string>
#include <map>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

#include "../core/model_info.h"

// Forward declarations for HTTP abstraction
namespace models::http {
    class HttpInterface;
    struct HttpRequest;
    struct HttpResponse;
}

namespace models {

// HTTP abstraction for pluggable networking implementations
namespace http {
    // Progress callback: (downloaded_bytes, total_bytes, speed_bps)
    using ProgressCallback = std::function<void(int64_t, int64_t, double)>;

    // Completion callback: (success, response_data, error_message)
    using CompletionCallback = std::function<void(bool, const std::string&, const std::string&)>;

    struct HttpRequest {
        std::string url;
        std::string method = "GET";           // GET, POST, etc.
        std::string userAgent;
        std::string authToken;                // Bearer token
        std::string outputFilePath;           // For downloads, empty for in-memory
        int timeoutSeconds = 300;
        int64_t resumeFrom = 0;               // Byte offset for resume
        std::map<std::string, std::string> headers;
        std::string postData;                 // For POST requests
    };

    struct HttpResponse {
        int statusCode = 0;
        std::string data;                     // Response body (for non-download requests)
        std::string errorMessage;
        std::map<std::string, std::string> headers;
        int64_t contentLength = 0;
    };

    // Abstract HTTP interface
    class HttpInterface {
    public:
        virtual ~HttpInterface() = default;

        // Async request with callbacks
        virtual void requestAsync(
            const HttpRequest& request,
            CompletionCallback completion,
            ProgressCallback progress = nullptr
        ) = 0;

        // Sync request (blocks)
        virtual HttpResponse requestSync(const HttpRequest& request) = 0;

        // Sync request with progress — blocks and fires progress on the calling thread.
        // Safe to use from a dart:ffi background isolate (NativeCallable::isolateLocal).
        virtual HttpResponse requestSyncWithProgress(const HttpRequest& request,
                                                     ProgressCallback progress) = 0;

        // Cancel operations
        virtual void cancelAll() = 0;
        virtual bool cancelRequest(const std::string& url) = 0;
    };

    // Factory functions
    std::unique_ptr<HttpInterface> createCurlHttpClient();
}

// Download-related types (merged from ModelDownloader)
enum class DownloadState {
    QUEUED,
    DOWNLOADING,
    COMPLETED,
    FAILED,
    CANCELLED
};

struct DownloadProgress {
    std::string downloadId;
    std::string modelId;
    std::string filename;
    int64_t bytesDownloaded;
    int64_t totalBytes;
    DownloadState state;
    std::string errorMessage;
};

// Callback types for HuggingFace API responses
using SearchCallback = std::function<void(const std::vector<ModelInfo>& models, const std::string& nextPageToken, const std::string& error)>;
using ModelDetailsCallback = std::function<void(const ModelInfo& model, const std::string& error)>;
using ModelFilesCallback = std::function<void(const std::string& modelId, const std::vector<FileInfo>& files, const std::string& error)>;
using DownloadStartCallback = std::function<void(const std::string& downloadId, const std::string& modelId)>;
using ProgressCallback = std::function<void(const DownloadProgress& progress)>;
using DownloadCompleteCallback = std::function<void(const std::string& downloadId, const std::string& localPath, int64_t totalBytes)>;
using DownloadFailedCallback = std::function<void(const std::string& downloadId, const std::string& error)>;
using ErrorCallback = std::function<void(const std::string& error, const std::string& endpoint)>;
using RateLimitCallback = std::function<void(int retryAfterSeconds)>;
using AuthRequiredCallback = std::function<void(const std::string& modelId)>;

/**
 * HuggingFace Hub API client for model discovery and metadata retrieval.
 *
 * Provides access to:
 * - Model search with comprehensive filtering
 * - Model details and metadata
 * - File listings with download URLs
 * - Rate limiting and error handling
 * - Authentication for gated models
 */
class HuggingFaceClient {
public:
    explicit HuggingFaceClient(std::unique_ptr<http::HttpInterface> httpClient);
    ~HuggingFaceClient();

    // Configuration
    void setBaseUrl(const std::string& baseUrl);
    void setAuthToken(const std::string& token);
    void setRateLimit(int requestsPerSecond = 1);
    void setUserAgent(const std::string& userAgent);

    // Callback registration
    void setSearchCallback(SearchCallback callback);
    void setModelDetailsCallback(ModelDetailsCallback callback);
    void setModelFilesCallback(ModelFilesCallback callback);
    void setDownloadStartCallback(DownloadStartCallback callback);
    void setProgressCallback(ProgressCallback callback);
    void setDownloadCompleteCallback(DownloadCompleteCallback callback);
    void setDownloadFailedCallback(DownloadFailedCallback callback);
    void setErrorCallback(ErrorCallback callback);
    void setRateLimitCallback(RateLimitCallback callback);
    void setAuthRequiredCallback(AuthRequiredCallback callback);

    // Core API methods
    void searchModels(const SearchQuery& query);
    void getModelDetails(const std::string& modelId);
    void getModelFiles(const std::string& modelId);

    // Download functionality (merged from ModelDownloader)
    void setDownloadDirectory(const std::string& path);
    std::string downloadDirectory() const;
    std::string downloadModel(const ModelInfo& model);
    void cancelDownload(const std::string& downloadId);
    std::vector<DownloadProgress> activeDownloads() const;
    std::string getModelPath(const std::string& modelId) const;

    // Status
    bool isRateLimited() const;
    std::string lastError() const;

    // Synchronously download a single file on the calling thread.
    // Progress fires on the same thread — safe for NativeCallable::isolateLocal().
    struct FileSyncResult { bool success; std::string error; };
    FileSyncResult downloadFileSync(const std::string& url,
                                    const std::string& localFilePath,
                                    http::ProgressCallback progress);


private:
    // Download-related structs
    struct FileDownloadRequest {
        std::string fileId; // unique ID for this file download
        std::string modelDownloadId; // the overall model download ID
        std::string modelId;
        FileInfo fileInfo;
        std::string localFilePath;
    };

    struct DownloadTask {
        std::string fileId;
        std::string modelDownloadId;
        std::string modelId;
        FileInfo fileInfo;
        std::string localFilePath;
        DownloadState state;
        std::string errorMessage;
        int64_t bytesReceived = 0;
        int64_t bytesTotal = 0;
    };

    struct ModelDownloadSession {
        std::string downloadId;
        std::string modelId;
        std::queue<FileDownloadRequest> fileQueue;
        std::unordered_set<std::string> activeFileIds; // Currently downloading files
        std::unordered_set<std::string> completedFileIds;
        std::unordered_map<std::string, int64_t> completedFileSizes; // fileId -> bytes
        int totalFiles;
        int64_t totalBytesDownloaded;
        int64_t totalBytesExpected;
    };

    // HTTP client management
    std::unique_ptr<http::HttpInterface> m_httpClient;
    std::queue<http::HttpRequest> m_requestQueue;
    std::unordered_map<std::string, std::string> m_activeRequests; // request_id -> endpoint mapping

    // Configuration
    std::string m_baseUrl;
    std::string m_authToken;
    std::string m_userAgent;
    int m_requestsPerSecond;
    std::string m_lastError;
    bool m_rateLimited = false;

    // Download management (merged from ModelDownloader)
    std::string m_downloadDirectory;
    int m_maxConcurrentDownloads = 3;
    std::unordered_map<std::string, ModelDownloadSession> m_modelSessions; // modelDownloadId -> session
    std::unordered_map<std::string, DownloadTask> m_activeTasks; // fileId -> task

    // Callbacks
    SearchCallback m_searchCallback;
    ModelDetailsCallback m_modelDetailsCallback;
    ModelFilesCallback m_modelFilesCallback;
    DownloadStartCallback m_downloadStartCallback;
    ProgressCallback m_progressCallback;
    DownloadCompleteCallback m_downloadCompleteCallback;
    DownloadFailedCallback m_downloadFailedCallback;
    ErrorCallback m_errorCallback;
    RateLimitCallback m_rateLimitCallback;
    AuthRequiredCallback m_authRequiredCallback;

    // Request building
    http::HttpRequest buildRequest(const std::string& endpoint, const std::map<std::string, std::string>& queryParams = {});
    std::string buildSearchQueryString(const SearchQuery& query);

    // Response parsing (will use nlohmann/json or similar JSON library)
    std::vector<ModelInfo> parseSearchResponse(const std::string& jsonData);
    ModelInfo parseModelInfo(const std::string& jsonData);
    std::vector<FileInfo> parseFilesResponse(const std::string& jsonData);
    std::string extractNextPageToken(const std::map<std::string, std::string>& headers);

    // Error handling
    void handleHttpError(int statusCode, const std::string& error, const std::string& endpoint);

    // Request management
    void queueRequest(const http::HttpRequest& request);
    void processRequestQueue();

    // Download management methods (merged from ModelDownloader)
    void sortFilesBySize(std::vector<FileDownloadRequest>& files);
    void startNextDownloads(const std::string& modelDownloadId);
    void startFileDownload(const FileDownloadRequest& request);
    void finishFileDownload(const std::string& fileId, bool success, const std::string& error = {});
    void checkModelComplete(const std::string& modelDownloadId);
    std::string createModelDirectory(const std::string& modelId);
    DownloadProgress createProgressFromSession(const ModelDownloadSession& session) const;
    void calculateAndEmitProgress(const std::string& modelDownloadId);
};

// Factory functions for creating HTTP clients
std::unique_ptr<HuggingFaceClient> createHuggingFaceClient();

// CLI Synchronous Wrapper - Converts callback-based API to blocking for CLI convenience
class HuggingFaceClientSync {
public:
    explicit HuggingFaceClientSync(std::unique_ptr<HuggingFaceClient> client);

    // Blocking model download with real-time progress updates
    struct DownloadResult {
        bool success;
        std::string localPath;
        int64_t totalBytes;
        std::string error;
    };

    DownloadResult downloadModelBlocking(const ModelInfo& model);

    // Like downloadModelBlocking but executes curl on the calling thread.
    // Progress and completion callbacks fire on the same OS thread as the caller,
    // which is required when the caller is a dart:ffi background isolate using
    // NativeCallable::isolateLocal().
    DownloadResult downloadModelSync(const ModelInfo& model);

    // Blocking API methods for CLI
    struct ModelDetailsResult {
        bool success;
        ModelInfo model;
        std::string error;
    };

    struct ModelFilesResult {
        bool success;
        std::vector<FileInfo> files;
        std::string error;
    };

    ModelDetailsResult getModelDetailsBlocking(const std::string& modelId);
    ModelFilesResult getModelFilesBlocking(const std::string& modelId);

    // Allow CLI to set progress callback for real-time updates
    void setProgressCallback(std::function<void(const DownloadProgress&)> callback);

    // Delegate other methods to the async client
    HuggingFaceClient* client();

private:
    std::unique_ptr<HuggingFaceClient> m_client;
    std::function<void(const DownloadProgress&)> m_progressCallback;
};

// Factory function for CLI wrapper
std::unique_ptr<HuggingFaceClientSync> createHuggingFaceClientSync();

} // namespace models
