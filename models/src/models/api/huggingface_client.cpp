#include "huggingface_client.h"

#include <algorithm>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <curl/curl.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <random>
#include <iomanip>
#include <ctime>
#include <mutex>
#include <future>
#include <atomic>

namespace models {

// Helper function for URL encoding
static std::string urlEncode(const std::string& value) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (std::string::const_iterator i = value.begin(), n = value.end(); i != n; ++i) {
        std::string::value_type c = (*i);
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            escaped << std::uppercase;
            escaped << '%' << std::setw(2) << int((unsigned char) c);
            escaped << std::nouppercase;
        }
    }

    return escaped.str();
}

// Curl-based HTTP client implementation (Qt-free)
namespace http {

class CurlHttpClient : public HttpInterface {
public:
    CurlHttpClient() {
        // Initialize CURL globally once
        static std::once_flag curl_init_flag;
        std::call_once(curl_init_flag, []() {
            curl_global_init(CURL_GLOBAL_DEFAULT);
        });
    }

    ~CurlHttpClient() {
        // Don't call curl_global_cleanup() here as other instances or threads might still be using CURL
        // CURL cleanup should happen at program exit
        // TODO: Implement proper cleanup strategy
    }

    void requestAsync(
        const HttpRequest& request,
        CompletionCallback completion,
        ProgressCallback progress = nullptr
    ) override {
        // Launch async task (capture future to avoid nodiscard warning)
        [[maybe_unused]] auto future = std::async(std::launch::async, [request, completion, progress]() {
            HttpResponse response = performRequestWithProgress(request, progress);
            completion(response.statusCode >= 200 && response.statusCode < 300, response.data, response.errorMessage);
        });
    }

    HttpResponse requestSyncWithProgress(const HttpRequest& request,
                                          ProgressCallback progress) override {
        return performRequestWithProgress(request, progress);
    }

    HttpResponse requestSync(const HttpRequest& request) override {
        HttpResponse response;
        CURL* curl = curl_easy_init();

        if (!curl) {
            response.errorMessage = "Failed to initialize CURL";
            return response;
        }

        // Response data string
        std::string responseData;

        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());

        // Set headers
        struct curl_slist* headers = nullptr;
        if (!request.userAgent.empty()) {
            curl_easy_setopt(curl, CURLOPT_USERAGENT, request.userAgent.c_str());
        }
        if (!request.authToken.empty()) {
            std::string authHeader = "Authorization: Bearer " + request.authToken;
            headers = curl_slist_append(headers, authHeader.c_str());
        }
        for (const auto& [key, value] : request.headers) {
            std::string header = key + ": " + value;
            headers = curl_slist_append(headers, header.c_str());
        }
        if (headers) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }

        // Set method and data
        if (request.method == "POST") {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.postData.c_str());
        }

        // Set timeouts
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.timeoutSeconds);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // 10 second connection timeout

        // Follow redirects (important for HuggingFace CDN downloads)
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L); // Limit to 10 redirects

        // Write callback for response data
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        // Perform request
        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            long responseCode;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);
            response.statusCode = static_cast<int>(responseCode);
            response.data = responseData;
        } else {
            response.errorMessage = curl_easy_strerror(res);
        }

        // Cleanup
        if (headers) {
            curl_slist_free_all(headers);
        }
        curl_easy_cleanup(curl);

        return response;
    }

    void cancelAll() override {
        // TODO: Implement cancellation
    }

    bool cancelRequest(const std::string& url) override {
        (void)url;
        // TODO: Implement specific request cancellation
        return false;
    }

    // C-style callback for CURL write function
    static size_t curlWriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t realsize = size * nmemb;
        std::string* response_data = static_cast<std::string*>(userp);
        response_data->append(static_cast<char*>(contents), realsize);
        return realsize;
    }

private:
    // Progress callback structure for libcurl
    struct ProgressData {
        ProgressCallback callback;
        std::string filename;
    };

    // Static progress callback for libcurl
    static int curlProgressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
        // Fire on every tick where we have made some progress.
        // HuggingFace CDN uses chunked transfer — dltotal is 0 throughout,
        // so we must not gate on dltotal > 0.
        if (clientp && dlnow >= 0) {
            ProgressData* data = static_cast<ProgressData*>(clientp);
            if (data->callback) {
                double speed = 0.0;
                data->callback(static_cast<int64_t>(dlnow), static_cast<int64_t>(dltotal), speed);
            }
        }
        return 0; // Continue download
    }

    // Static function to perform HTTP request with progress callbacks
    static HttpResponse performRequestWithProgress(const HttpRequest& request, ProgressCallback progress) {
        HttpResponse response;
        CURL* curl = curl_easy_init();

        if (!curl) {
            response.errorMessage = "Failed to initialize CURL";
            return response;
        }

        // Response data string
        std::string responseData;
        ProgressData progressData{progress, ""};

        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());

        // Set headers
        struct curl_slist* headers = nullptr;
        if (!request.userAgent.empty()) {
            curl_easy_setopt(curl, CURLOPT_USERAGENT, request.userAgent.c_str());
        }
        if (!request.authToken.empty()) {
            std::string authHeader = "Authorization: Bearer " + request.authToken;
            headers = curl_slist_append(headers, authHeader.c_str());
        }
        for (const auto& [key, value] : request.headers) {
            std::string header = key + ": " + value;
            headers = curl_slist_append(headers, header.c_str());
        }
        if (headers) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }

        // Set method and data
        if (request.method == "POST") {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.postData.c_str());
        }

        // Set timeouts
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.timeoutSeconds);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

        // Follow redirects
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);

        // Set up progress callback if provided
        if (progress) {
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, &curlProgressCallback);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        }

        // Write callback for response data
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        // Perform request
        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            long responseCode;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);
            response.statusCode = static_cast<int>(responseCode);
            response.data = responseData;
        } else {
            response.errorMessage = curl_easy_strerror(res);
        }

        // Cleanup
        if (headers) {
            curl_slist_free_all(headers);
        }
        curl_easy_cleanup(curl);

        return response;
    }

    // Static function to perform HTTP request without object dependencies
    static HttpResponse performRequest(const HttpRequest& request) {
        HttpResponse response;
        CURL* curl = curl_easy_init();

        if (!curl) {
            response.errorMessage = "Failed to initialize CURL";
            return response;
        }

        // Response data string (direct approach for proper C callback)
        std::string responseData;

        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, request.url.c_str());

        // Set headers
        struct curl_slist* headers = nullptr;
        if (!request.userAgent.empty()) {
            curl_easy_setopt(curl, CURLOPT_USERAGENT, request.userAgent.c_str());
        }
        if (!request.authToken.empty()) {
            std::string authHeader = "Authorization: Bearer " + request.authToken;
            headers = curl_slist_append(headers, authHeader.c_str());
        }
        for (const auto& [key, value] : request.headers) {
            std::string header = key + ": " + value;
            headers = curl_slist_append(headers, header.c_str());
        }
        if (headers) {
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        }

        // Set method and data
        if (request.method == "POST") {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.postData.c_str());
        }

        // Set timeouts
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, request.timeoutSeconds);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L); // 10 second connection timeout

        // Follow redirects (important for HuggingFace CDN downloads)
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L); // Limit to 10 redirects

        // Write callback for response data (proper C function for CURL)
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &curlWriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        // Perform request
        CURLcode res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            long responseCode;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);
            response.statusCode = static_cast<int>(responseCode);
            response.data = responseData;
        } else {
            response.errorMessage = curl_easy_strerror(res);
        }

        // Cleanup
        if (headers) {
            curl_slist_free_all(headers);
        }
        curl_easy_cleanup(curl);

        return response;
    }
};

std::unique_ptr<HttpInterface> createCurlHttpClient() {
    return std::make_unique<CurlHttpClient>();
}

} // namespace http

// HuggingFaceClient implementation
HuggingFaceClient::HuggingFaceClient(std::unique_ptr<http::HttpInterface> httpClient)
    : m_httpClient(std::move(httpClient))
    , m_baseUrl("https://huggingface.co")
    , m_userAgent("neurons-models/1.0")
    , m_requestsPerSecond(1)
    , m_rateLimited(false)
    , m_maxConcurrentDownloads(3)
{
}

HuggingFaceClient::~HuggingFaceClient() = default;

void HuggingFaceClient::setBaseUrl(const std::string& baseUrl) {
    m_baseUrl = baseUrl;
}

void HuggingFaceClient::setAuthToken(const std::string& token) {
    m_authToken = token;
}

void HuggingFaceClient::setRateLimit(int requestsPerSecond) {
    m_requestsPerSecond = requestsPerSecond;
}

void HuggingFaceClient::setUserAgent(const std::string& userAgent) {
    m_userAgent = userAgent;
}

// Callback setters
void HuggingFaceClient::setSearchCallback(SearchCallback callback) {
    m_searchCallback = std::move(callback);
}

void HuggingFaceClient::setModelDetailsCallback(ModelDetailsCallback callback) {
    m_modelDetailsCallback = std::move(callback);
}

void HuggingFaceClient::setModelFilesCallback(ModelFilesCallback callback) {
    m_modelFilesCallback = std::move(callback);
}

void HuggingFaceClient::setDownloadStartCallback(DownloadStartCallback callback) {
    m_downloadStartCallback = std::move(callback);
}

void HuggingFaceClient::setProgressCallback(ProgressCallback callback) {
    m_progressCallback = std::move(callback);
}

void HuggingFaceClient::setDownloadCompleteCallback(DownloadCompleteCallback callback) {
    m_downloadCompleteCallback = std::move(callback);
}

void HuggingFaceClient::setDownloadFailedCallback(DownloadFailedCallback callback) {
    m_downloadFailedCallback = std::move(callback);
}

void HuggingFaceClient::setErrorCallback(ErrorCallback callback) {
    m_errorCallback = std::move(callback);
}

void HuggingFaceClient::setRateLimitCallback(RateLimitCallback callback) {
    m_rateLimitCallback = std::move(callback);
}

void HuggingFaceClient::setAuthRequiredCallback(AuthRequiredCallback callback) {
    m_authRequiredCallback = std::move(callback);
}

// Core API methods
void HuggingFaceClient::searchModels(const SearchQuery& query) {
    std::map<std::string, std::string> queryParams;

    // Basic search parameters
    if (!query.search.empty()) {
        queryParams["search"] = query.search;
    }
    if (!query.author.empty()) {
        queryParams["author"] = query.author;
    }
    if (!query.filter.empty()) {
        queryParams["filter"] = query.filter;
    }
    // pipeline_tag is a dedicated HF API parameter (not embedded in filter).
    // Multiple tags → send the first one; HF API only supports a single value.
    if (!query.pipelineTags.empty()) {
        queryParams["pipeline_tag"] = query.pipelineTags.front();
    }
    if (!query.sort.empty()) {
        queryParams["sort"] = query.sort;
    }
    if (!query.direction.empty()) {
        queryParams["direction"] = query.direction;
    }

    queryParams["limit"] = std::to_string(query.limit);
    queryParams["full"] = query.full ? "true" : "false";

    if (query.config) {
        queryParams["config"] = "true";
    }

    if (!query.nextPageToken.empty()) {
        queryParams["offset"] = query.nextPageToken;
    }

    auto request = buildRequest("/api/models", queryParams);

    // Execute request asynchronously
    auto searchCallback = m_searchCallback;
    m_httpClient->requestAsync(request,
        [this, searchCallback](bool success, const std::string& data, const std::string& error) {
            if (!success) {
                if (searchCallback) {
                    searchCallback({}, "", error);
                }
                return;
            }

            try {
                auto models = this->parseSearchResponse(data);
                std::string nextPageToken; // TODO: Extract from response headers
                if (searchCallback) {
                    searchCallback(models, nextPageToken, "");
                }
            } catch (const std::exception& e) {
                if (searchCallback) {
                    searchCallback({}, "", std::string("JSON parsing error: ") + e.what());
                }
            }
        });
}

void HuggingFaceClient::getModelDetails(const std::string& modelId) {
    if (modelId.empty()) {
        if (m_modelDetailsCallback) {
            ModelInfo empty;
            m_modelDetailsCallback(empty, "Model ID cannot be empty");
        }
        return;
    }

    // HF model IDs are "org/model-name" — the slash is a path separator and
    // must NOT be percent-encoded; HF returns 400 for %2F in the path.
    std::string endpoint = "/api/models/" + modelId;
    auto request = buildRequest(endpoint);

    auto modelDetailsCallback = m_modelDetailsCallback;
    m_httpClient->requestAsync(request,
        [this, modelDetailsCallback](bool success, const std::string& data, const std::string& error) {
            if (!success) {
                if (modelDetailsCallback) {
                    ModelInfo empty;
                    modelDetailsCallback(empty, error);
                }
                return;
            }

            try {
                auto model = this->parseModelInfo(data);
                if (modelDetailsCallback) {
                    modelDetailsCallback(model, "");
                }
            } catch (const std::exception& e) {
                if (modelDetailsCallback) {
                    ModelInfo empty;
                    modelDetailsCallback(empty, std::string("JSON parsing error: ") + e.what());
                }
            }
        });
}

void HuggingFaceClient::getModelFiles(const std::string& modelId) {
    if (modelId.empty()) {
        if (m_modelFilesCallback) {
            m_modelFilesCallback(modelId, {}, "Model ID cannot be empty");
        }
        return;
    }

    // HF model IDs are "org/model-name" — the slash is a path separator and
    // must NOT be percent-encoded; HF returns 400 for %2F in the path.
    std::string endpoint = "/api/models/" + modelId + "/tree/main";
    auto request = buildRequest(endpoint);

    auto modelFilesCallback = m_modelFilesCallback;
    m_httpClient->requestAsync(request,
        [this, modelFilesCallback, modelId](bool success, const std::string& data, const std::string& error) {
            if (!success) {
                if (modelFilesCallback) {
                    modelFilesCallback(modelId, {}, error);
                }
                return;
            }

            try {
                auto files = this->parseFilesResponse(data);

                // Build download URLs for each file
                for (auto& file : files) {
                    file.downloadUrl = "https://huggingface.co/" + modelId + "/resolve/main/" + file.filename;
                }

                if (modelFilesCallback) {
                    modelFilesCallback(modelId, files, "");
                }
            } catch (const std::exception& e) {
                if (modelFilesCallback) {
                    modelFilesCallback(modelId, {}, std::string("JSON parsing error: ") + e.what());
                }
            }
        });
}

// Download functionality
void HuggingFaceClient::setDownloadDirectory(const std::string& path) {
    m_downloadDirectory = path;
}

std::string HuggingFaceClient::downloadDirectory() const {
    return m_downloadDirectory;
}

std::string HuggingFaceClient::downloadModel(const ModelInfo& model) {
    if (m_downloadDirectory.empty()) {
        if (m_downloadFailedCallback) {
            m_downloadFailedCallback("", "Download directory not set");
        }
        return "";
    }

    if (model.files.empty()) {
        if (m_downloadFailedCallback) {
            m_downloadFailedCallback("", "No files to download for model: " + model.id);
        }
        return "";
    }

    // Generate unique download ID
    std::string downloadId = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());

    // Create model directory
    std::string modelDir = createModelDirectory(model.id);
    if (modelDir.empty()) {
        if (m_downloadFailedCallback) {
            m_downloadFailedCallback(downloadId, "Failed to create model directory for: " + model.id);
        }
        return "";
    }

    // Create download session
    ModelDownloadSession session;
    session.downloadId = downloadId;
    session.modelId = model.id;
    session.totalFiles = model.files.size();
    session.totalBytesDownloaded = 0;
    session.totalBytesExpected = 0;

    // Create file download requests
    for (const auto& file : model.files) {
        FileDownloadRequest request;
        request.fileId = std::to_string(std::hash<std::string>{}(file.filename + downloadId));
        request.modelDownloadId = downloadId;
        request.modelId = model.id;
        request.fileInfo = file;
        request.localFilePath = std::filesystem::path(modelDir) / file.filename;

        session.fileQueue.push(request);
        session.totalBytesExpected += file.sizeBytes;
    }

    m_modelSessions[downloadId] = session;

    if (m_downloadStartCallback) {
        m_downloadStartCallback(downloadId, model.id);
    }

    // Start downloading
    startNextDownloads(downloadId);

    return downloadId;
}

void HuggingFaceClient::cancelDownload(const std::string& downloadId) {
    (void)downloadId;
    // TODO: Implement download cancellation
}

std::vector<DownloadProgress> HuggingFaceClient::activeDownloads() const {
    // TODO: Implement active downloads listing
    return {};
}

std::string HuggingFaceClient::getModelPath(const std::string& modelId) const {
    if (m_downloadDirectory.empty()) {
        return "";
    }
    std::filesystem::path path(m_downloadDirectory);
    path /= modelId;
    return path.string();
}

// Status methods
bool HuggingFaceClient::isRateLimited() const {
    return m_rateLimited;
}

std::string HuggingFaceClient::lastError() const {
    return m_lastError;
}

// Private helper methods
http::HttpRequest HuggingFaceClient::buildRequest(const std::string& endpoint, const std::map<std::string, std::string>& queryParams) {
    http::HttpRequest request;
    request.url = m_baseUrl + endpoint;
    request.userAgent = m_userAgent;
    request.authToken = m_authToken;
    request.timeoutSeconds = 30; // Shorter timeout for better responsiveness

    // Add query parameters
    if (!queryParams.empty()) {
        request.url += "?";
        bool first = true;
        for (const auto& [key, value] : queryParams) {
            if (!first) request.url += "&";
            request.url += key + "=" + urlEncode(value);
            first = false;
        }
    }

    return request;
}

std::string HuggingFaceClient::buildSearchQueryString(const SearchQuery& query) {
    (void)query;
    // TODO: Implement query string building
    return "";
}

// Helper function to convert date string to system_clock::time_point
static std::chrono::system_clock::time_point parseISODate(const std::string& dateStr) {
    if (dateStr.empty()) return {};

    std::tm tm = {};
    std::istringstream ss(dateStr);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        return {};
    }
    std::time_t time = std::mktime(&tm);
    return std::chrono::system_clock::from_time_t(time);
}

// Helper function to parse ModelInfo from JSON object
static ModelInfo parseModelInfoFromJson(const nlohmann::json& obj) {
    ModelInfo model;

    // Core identification
    model.id = obj.value("id", "");
    model.author = obj.value("author", "");
    model.sha = obj.value("sha", "");

    // Use id as name if no explicit name provided
    model.name = model.id;
    if (model.name.find('/') != std::string::npos) {
        model.name = model.name.substr(model.name.find('/') + 1);
    }

    // Classification
    model.pipelineTag = obj.value("pipeline_tag", "");
    model.libraryName = obj.value("library_name", "");

    // Convert pipeline tag to ModelType enum
    const std::string& pipelineTag = model.pipelineTag;
    if (pipelineTag == "text-generation" || pipelineTag == "text2text-generation") {
        model.type = ModelType::TEXT_GENERATION;
    } else if (pipelineTag == "text-to-image" || pipelineTag == "image-to-image") {
        model.type = ModelType::IMAGE_GENERATION;
    } else if (pipelineTag == "automatic-speech-recognition") {
        model.type = ModelType::SPEECH_TO_TEXT;
    } else if (pipelineTag == "text-to-speech") {
        model.type = ModelType::TEXT_TO_SPEECH;
    } else if (pipelineTag == "image-classification" || pipelineTag == "object-detection") {
        model.type = ModelType::VISION;
    } else {
        model.type = ModelType::TEXT_GENERATION; // Default fallback
    }

    // Parse tags array
    if (obj.contains("tags") && obj["tags"].is_array()) {
        for (const auto& tagValue : obj["tags"]) {
            if (tagValue.is_string()) {
                model.tags.push_back(tagValue.get<std::string>());
            }
        }
    }

    // Popularity metrics
    model.downloads = obj.value("downloads", 0);
    model.likes = obj.value("likes", 0);
    model.trendingScore = obj.value("trending_score", 0.0);

    // Metadata
    model.description = obj.value("description", "");

    // Parse timestamps
    std::string createdAtStr = obj.value("created_at", "");
    std::string lastModifiedStr = obj.value("last_modified", "");

    model.createdAt = parseISODate(createdAtStr);
    model.lastModified = parseISODate(lastModifiedStr);

    // Access control
    // HF API returns gated as false (bool) for open models, or "auto"/"manual"
    // (string) for gated models — handle both to avoid type_error.302 crash.
    if (obj.contains("gated")) {
        const auto& g = obj["gated"];
        model.gated = g.is_boolean() ? g.get<bool>() : (g.is_string() && g.get<std::string>() != "false");
    }
    model.private_ = obj.value("private", false);

    // Staff pick (might be in different locations)
    model.staffPick = obj.value("staff_pick", false) || obj.value("staff_picked", false);

    // Parse files array if present (detailed model response)
    if (obj.contains("siblings") && obj["siblings"].is_array()) {
        int64_t totalSize = 0;

        for (const auto& fileValue : obj["siblings"]) {
            if (!fileValue.is_object()) continue;
            // siblings entries have no "type" field — just rfilename + size

            FileInfo fileInfo;
            fileInfo.filename = fileValue.value("rfilename", "");
            fileInfo.sizeBytes = fileValue.value("size", 0);
            fileInfo.downloadUrl = "https://huggingface.co/" + model.id + "/resolve/main/" + fileInfo.filename;

            if (!fileInfo.filename.empty()) {
                model.files.push_back(fileInfo);
                totalSize += fileInfo.sizeBytes;
            }
        }

        model.sizeBytes = totalSize;
    }

    // usedStorage is returned by the detail endpoint and by search with expand[]=usedStorage.
    // Use it when siblings didn't carry per-file sizes (search response).
    if (model.sizeBytes == 0 && obj.contains("usedStorage")) {
        model.sizeBytes = obj["usedStorage"].get<int64_t>();
    }

    // Derive formats from file extensions
    std::vector<std::string> formats;
    for (const FileInfo& file : model.files) {
        size_t dotPos = file.filename.find_last_of('.');
        if (dotPos != std::string::npos) {
            std::string extension = file.filename.substr(dotPos + 1);
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (std::find(formats.begin(), formats.end(), extension) == formats.end()) {
                formats.push_back(extension);
            }
        }
    }
    model.formats = formats;

    return model;
}

// Response parsing methods
std::vector<ModelInfo> HuggingFaceClient::parseSearchResponse(const std::string& jsonData) {
    std::vector<ModelInfo> models;

    try {
        auto json = nlohmann::json::parse(jsonData);

        if (!json.is_array()) {
            throw std::runtime_error("Expected JSON array for search response");
        }

        for (const auto& item : json) {
            if (!item.is_object()) {
                continue;
            }

            ModelInfo model = parseModelInfoFromJson(item);
            if (!model.id.empty()) {
                models.push_back(model);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
    }

    return models;
}

ModelInfo HuggingFaceClient::parseModelInfo(const std::string& jsonData) {
    try {
        auto json = nlohmann::json::parse(jsonData);

        if (!json.is_object()) {
            throw std::runtime_error("Expected JSON object for model info response");
        }

        return parseModelInfoFromJson(json);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
    }
}

std::vector<FileInfo> HuggingFaceClient::parseFilesResponse(const std::string& jsonData) {
    std::vector<FileInfo> files;

    try {
        auto json = nlohmann::json::parse(jsonData);

        if (!json.is_array()) {
            throw std::runtime_error("Expected JSON array for files response");
        }

        for (const auto& item : json) {
            if (!item.is_object()) {
                continue;
            }

            // Skip directories, only process files
            std::string type = item.value("type", "");
            if (type != "file") {
                continue;
            }

            FileInfo fileInfo;
            fileInfo.filename = item.value("path", "");
            fileInfo.sizeBytes = item.value("size", 0ULL);
            // downloadUrl will be set by the caller

            if (!fileInfo.filename.empty()) {
                files.push_back(fileInfo);
            }
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("JSON parsing error: ") + e.what());
    }

    return files;
}

std::string HuggingFaceClient::extractNextPageToken(const std::map<std::string, std::string>& headers) {
    (void)headers;
    // TODO: Implement next page token extraction from headers
    return "";
}

// Error handling
void HuggingFaceClient::handleHttpError(int statusCode, const std::string& error, const std::string& endpoint) {
    m_lastError = "HTTP " + std::to_string(statusCode) + ": " + error;
    if (m_errorCallback) {
        m_errorCallback(m_lastError, endpoint);
    }
}

// Request management
void HuggingFaceClient::queueRequest(const http::HttpRequest& request) {
    m_requestQueue.push(request);
    processRequestQueue();
}

void HuggingFaceClient::processRequestQueue() {
    // TODO: Implement rate limiting and queue processing
}

// Download management methods (placeholder implementations)
void HuggingFaceClient::sortFilesBySize(std::vector<FileDownloadRequest>& files) {
    std::sort(files.begin(), files.end(),
        [](const FileDownloadRequest& a, const FileDownloadRequest& b) {
            return a.fileInfo.sizeBytes > b.fileInfo.sizeBytes;
        });
}

void HuggingFaceClient::startNextDownloads(const std::string& modelDownloadId) {
    if (m_modelSessions.find(modelDownloadId) == m_modelSessions.end()) {
        return;
    }

    ModelDownloadSession& session = m_modelSessions[modelDownloadId];

    // Start downloads up to the concurrent limit
    while (session.activeFileIds.size() < static_cast<size_t>(m_maxConcurrentDownloads) && !session.fileQueue.empty()) {
        FileDownloadRequest request = session.fileQueue.front();
        session.fileQueue.pop();
        session.activeFileIds.insert(request.fileId);
        startFileDownload(request);
    }
}

void HuggingFaceClient::startFileDownload(const FileDownloadRequest& request) {
    // Create download task
    DownloadTask task;
    task.fileId = request.fileId;
    task.modelDownloadId = request.modelDownloadId;
    task.modelId = request.modelId;
    task.fileInfo = request.fileInfo;
    task.localFilePath = request.localFilePath;
    task.state = DownloadState::DOWNLOADING;
    task.bytesReceived = 0;
    task.bytesTotal = request.fileInfo.sizeBytes;

    m_activeTasks[request.fileId] = task;

    // Create HTTP request for file download
    http::HttpRequest httpRequest;
    httpRequest.url = request.fileInfo.downloadUrl;
    httpRequest.method = "GET";
    httpRequest.userAgent = m_userAgent;
    httpRequest.authToken = m_authToken;
    httpRequest.timeoutSeconds = 600; // Longer timeout for file downloads

    // Create progress callback that updates our task and emits progress
    auto progressCallback = [this, fileId = request.fileId](int64_t bytesDownloaded, int64_t totalBytes, double speed) {
        (void)speed; // Mark as used to avoid warning
        // Update task progress
        auto taskIt = m_activeTasks.find(fileId);
        if (taskIt != m_activeTasks.end()) {
            taskIt->second.bytesReceived = bytesDownloaded;

            // Create progress data and emit via callback
            if (m_progressCallback) {
                DownloadProgress progress;
                progress.downloadId = taskIt->second.modelDownloadId;
                progress.modelId = taskIt->second.modelId;
                progress.filename = taskIt->second.fileInfo.filename;
                progress.bytesDownloaded = bytesDownloaded;
                progress.totalBytes = totalBytes;
                progress.state = DownloadState::DOWNLOADING;

                m_progressCallback(progress);
            }
        }
    };

    // Start async download with progress updates (capture future to avoid nodiscard warning)
    [[maybe_unused]] auto downloadFuture = std::async(std::launch::async, [this, httpRequest, request, progressCallback]() {
        // Use async HTTP request with progress callback
        m_httpClient->requestAsync(httpRequest,
            [this, request](bool success, const std::string& data, const std::string& error) {
                if (success) {
                    // Write data to file
                    try {
                        std::filesystem::create_directories(std::filesystem::path(request.localFilePath).parent_path());
                        std::ofstream file(request.localFilePath, std::ios::binary);
                        if (file.is_open()) {
                            file.write(data.c_str(), data.size());
                            file.close();
                            finishFileDownload(request.fileId, true, "");
                        } else {
                            finishFileDownload(request.fileId, false, "Failed to open file for writing: " + request.localFilePath);
                        }
                    } catch (const std::exception& e) {
                        finishFileDownload(request.fileId, false, std::string("File write error: ") + e.what());
                    }
                } else {
                    finishFileDownload(request.fileId, false, error);
                }
            },
            progressCallback
        );
    });
}

void HuggingFaceClient::finishFileDownload(const std::string& fileId, bool success, const std::string& error) {
    // Find the download task
    auto taskIt = m_activeTasks.find(fileId);
    if (taskIt == m_activeTasks.end()) {
        return;
    }

    DownloadTask& task = taskIt->second;
    std::string modelDownloadId = task.modelDownloadId;

    // Update task state
    task.state = success ? DownloadState::COMPLETED : DownloadState::FAILED;
    task.errorMessage = error;

    // Find the model session
    auto sessionIt = m_modelSessions.find(modelDownloadId);
    if (sessionIt == m_modelSessions.end()) {
        return;
    }

    ModelDownloadSession& session = sessionIt->second;

    // Remove from active files
    session.activeFileIds.erase(fileId);

    if (success) {
        session.completedFileIds.insert(fileId);
        session.completedFileSizes[fileId] = task.bytesTotal;
        session.totalBytesDownloaded += task.bytesTotal;
    } else {
        // Trigger failure callback for individual file
        if (m_downloadFailedCallback) {
            m_downloadFailedCallback(modelDownloadId, "File download failed: " + task.fileInfo.filename + " - " + error);
        }

        // For now, fail the entire model download if any file fails
        // Clean up session
        m_modelSessions.erase(sessionIt);
        m_activeTasks.erase(taskIt);
        return;
    }

    // Remove completed task
    m_activeTasks.erase(taskIt);

    // Check if all files are complete
    if (static_cast<int>(session.completedFileIds.size()) == session.totalFiles) {
        // All files downloaded successfully
        if (m_downloadCompleteCallback) {
            m_downloadCompleteCallback(modelDownloadId, getModelPath(session.modelId), session.totalBytesDownloaded);
        }

        // Clean up session
        m_modelSessions.erase(sessionIt);
    } else {
        // Start next downloads if queue is not empty
        startNextDownloads(modelDownloadId);

        // Emit progress update
        calculateAndEmitProgress(modelDownloadId);
    }
}

void HuggingFaceClient::checkModelComplete(const std::string& modelDownloadId) {
    (void)modelDownloadId;
    // TODO: Implement model completion check
}

std::string HuggingFaceClient::createModelDirectory(const std::string& modelId) {
    std::string modelPath = getModelPath(modelId);
    try {
        std::filesystem::create_directories(modelPath);
        return modelPath;
    } catch (const std::exception& e) {
        return "";
    }
}

DownloadProgress HuggingFaceClient::createProgressFromSession(const ModelDownloadSession& session) const {
    DownloadProgress progress;
    progress.downloadId = session.downloadId;
    progress.modelId = session.modelId;
    progress.bytesDownloaded = session.totalBytesDownloaded;
    progress.totalBytes = session.totalBytesExpected;
    progress.state = DownloadState::DOWNLOADING;
    return progress;
}

void HuggingFaceClient::calculateAndEmitProgress(const std::string& modelDownloadId) {
    auto sessionIt = m_modelSessions.find(modelDownloadId);
    if (sessionIt == m_modelSessions.end()) {
        return;
    }

    const ModelDownloadSession& session = sessionIt->second;

    // Calculate total progress across all files
    int64_t totalDownloaded = 0;
    for (const auto& [fileId, task] : m_activeTasks) {
        if (task.modelDownloadId == modelDownloadId) {
            totalDownloaded += task.bytesReceived;
        }
    }

    // Emit overall model progress
    if (m_progressCallback) {
        DownloadProgress progress;
        progress.downloadId = modelDownloadId;
        progress.modelId = session.modelId;
        progress.filename = ""; // Empty for overall progress
        progress.bytesDownloaded = totalDownloaded;
        progress.totalBytes = session.totalBytesExpected;
        progress.state = DownloadState::DOWNLOADING;
        m_progressCallback(progress);
    }
}

HuggingFaceClient::FileSyncResult HuggingFaceClient::downloadFileSync(
    const std::string& url,
    const std::string& localFilePath,
    http::ProgressCallback progress)
{
    http::HttpRequest req;
    req.url = url;
    req.method = "GET";
    req.userAgent = m_userAgent;
    req.authToken = m_authToken;
    req.timeoutSeconds = 600;

    auto response = m_httpClient->requestSyncWithProgress(req, progress);

    if (response.statusCode < 200 || response.statusCode >= 300) {
        std::string msg = response.errorMessage.empty()
            ? ("HTTP " + std::to_string(response.statusCode))
            : response.errorMessage;
        return {false, msg};
    }

    try {
        std::filesystem::create_directories(
            std::filesystem::path(localFilePath).parent_path());
        std::ofstream f(localFilePath, std::ios::binary);
        if (!f.is_open()) {
            return {false, "Failed to open file for writing: " + localFilePath};
        }
        f.write(response.data.data(), static_cast<std::streamsize>(response.data.size()));
    } catch (const std::exception& e) {
        return {false, std::string("File write error: ") + e.what()};
    }
    return {true, ""};
}

// Factory function
std::unique_ptr<HuggingFaceClient> createHuggingFaceClient() {
    return std::make_unique<HuggingFaceClient>(http::createCurlHttpClient());
}

// HuggingFaceClientSync implementation
HuggingFaceClientSync::HuggingFaceClientSync(std::unique_ptr<HuggingFaceClient> client)
    : m_client(std::move(client)) {}

HuggingFaceClientSync::DownloadResult HuggingFaceClientSync::downloadModelBlocking(const ModelInfo& model) {
    std::promise<DownloadResult> promise;
    auto future = promise.get_future();

    // Setup completion callbacks to resolve promise
    m_client->setDownloadCompleteCallback([&promise](const std::string& downloadId, const std::string& localPath, int64_t totalBytes) {
        (void)downloadId; // Mark as used
        promise.set_value({true, localPath, totalBytes, ""});
    });

    m_client->setDownloadFailedCallback([&promise](const std::string& downloadId, const std::string& error) {
        (void)downloadId; // Mark as used
        promise.set_value({false, "", 0, error});
    });

    // Progress callback still fires in real-time while we block
    if (m_progressCallback) {
        m_client->setProgressCallback(m_progressCallback);
    }

    // Start download
    std::string downloadId = m_client->downloadModel(model);
    if (downloadId.empty()) {
        return {false, "", 0, "Failed to start download"};
    }

    // Block until complete
    return future.get();
}

HuggingFaceClientSync::DownloadResult HuggingFaceClientSync::downloadModelSync(
    const ModelInfo& model)
{
    if (model.files.empty()) {
        return {false, "", 0, "No files to download for: " + model.id};
    }

    const std::string modelDir = m_client->getModelPath(model.id);
    if (modelDir.empty()) {
        return {false, "", 0, "Download directory not set"};
    }
    try {
        std::filesystem::create_directories(modelDir);
    } catch (const std::exception& e) {
        return {false, "", 0, std::string("Failed to create model directory: ") + e.what()};
    }

    // Compute total expected bytes for overall progress reporting.
    int64_t totalExpected = 0;
    for (const auto& f : model.files) totalExpected += f.sizeBytes;

    // Stable download ID used in progress events.
    const std::string downloadId = std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    int64_t totalDone = 0;

    for (const auto& file : model.files) {
        const std::string localPath =
            (std::filesystem::path(modelDir) / file.filename).string();

        // Per-file progress shim: accumulates file bytes and forwards to m_progressCallback.
        auto fileProg = [&](int64_t dlnow, int64_t dltotal, double /*speed*/) {
            if (!m_progressCallback) return;
            DownloadProgress p;
            p.downloadId      = downloadId;
            p.modelId         = model.id;
            p.filename        = file.filename;
            p.bytesDownloaded = totalDone + dlnow;
            p.totalBytes      = totalExpected > 0 ? totalExpected
                                : (dltotal > 0 ? dltotal : file.sizeBytes);
            p.state           = DownloadState::DOWNLOADING;
            m_progressCallback(p);
        };

        auto result = m_client->downloadFileSync(file.downloadUrl, localPath, fileProg);
        if (!result.success) {
            return {false, "", 0,
                "Failed to download " + file.filename + ": " + result.error};
        }
        totalDone += file.sizeBytes;
    }

    // Emit a final COMPLETED event.
    if (m_progressCallback) {
        DownloadProgress done;
        done.downloadId      = downloadId;
        done.modelId         = model.id;
        done.bytesDownloaded = totalDone;
        done.totalBytes      = totalExpected;
        done.state           = DownloadState::COMPLETED;
        m_progressCallback(done);
    }

    return {true, modelDir, totalDone, ""};
}

void HuggingFaceClientSync::setProgressCallback(std::function<void(const DownloadProgress&)> callback) {
    m_progressCallback = std::move(callback);
}

HuggingFaceClient* HuggingFaceClientSync::client() {
    return m_client.get();
}

HuggingFaceClientSync::ModelDetailsResult HuggingFaceClientSync::getModelDetailsBlocking(const std::string& modelId) {
    std::promise<ModelDetailsResult> promise;
    auto future = promise.get_future();

    // Setup callback to resolve promise
    m_client->setModelDetailsCallback([&promise](const ModelInfo& model, const std::string& error) {
        if (error.empty()) {
            promise.set_value({true, model, ""});
        } else {
            promise.set_value({false, ModelInfo{}, error});
        }
    });

    // Start async request
    m_client->getModelDetails(modelId);

    // Block until complete
    return future.get();
}

HuggingFaceClientSync::ModelFilesResult HuggingFaceClientSync::getModelFilesBlocking(const std::string& modelId) {
    std::promise<ModelFilesResult> promise;
    auto future = promise.get_future();

    // Setup callback to resolve promise
    m_client->setModelFilesCallback([&promise](const std::string& modelId, const std::vector<FileInfo>& files, const std::string& error) {
        (void)modelId; // Mark as used
        if (error.empty()) {
            promise.set_value({true, files, ""});
        } else {
            promise.set_value({false, {}, error});
        }
    });

    // Start async request
    m_client->getModelFiles(modelId);

    // Block until complete
    return future.get();
}

// Factory function for CLI wrapper
std::unique_ptr<HuggingFaceClientSync> createHuggingFaceClientSync() {
    return std::make_unique<HuggingFaceClientSync>(createHuggingFaceClient());
}

} // namespace models