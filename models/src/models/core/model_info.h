#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

namespace models {

enum class ModelType {
    TEXT_GENERATION,
    IMAGE_GENERATION,
    SPEECH_TO_TEXT,
    TEXT_TO_SPEECH,
    VISION
};

// Search query parameters for HuggingFace API
struct SearchQuery {
    std::string search;              // Free text search (model names, authors, keywords)
    std::string author;              // Organization filter (e.g., "meta-llama", "microsoft")
    std::string filter;              // Generic tag filter (comma-separated)
    std::vector<std::string> pipelineTags; // pipeline_tag= params (e.g. "text-generation")
    std::string sort;                // "downloads", "author", "created", "updated"
    std::string direction;           // "-1" (desc), "1" (asc)
    int limit = 25;                  // Results per page (default 25, max ~100)
    bool full = true;                // Include comprehensive metadata
    bool config = false;             // Include repository configuration
    std::string nextPageToken;       // For pagination (Link header parsing)
};

// File information from HuggingFace API
struct FileInfo {
    std::string filename;            // "config.json", "model.safetensors", etc.
    uint64_t sizeBytes = 0;          // File size (0 if unknown)
    std::string downloadUrl;         // Direct download URL
};

// Extended model info supporting HuggingFace API response format
struct ModelInfo {
    // Core identification
    std::string id;                  // "meta-llama/Llama-3.1-8B-Instruct"
    std::string name;                // "Llama 3.1 8B Instruct" (display name)
    std::string author;              // "meta-llama"
    std::string sha;                 // Git commit hash

    // Classification (existing + extended)
    ModelType type;                  // TEXT_GENERATION, IMAGE_GENERATION, etc.
    std::string pipelineTag;         // "text-generation", "image-classification", etc.
    std::string libraryName;         // "transformers", "diffusers", etc.
    std::vector<std::string> tags;   // ["transformers", "llama", "conversational"]
    std::vector<std::string> formats; // ["gguf", "safetensors"] (derived from files)

    // Popularity metrics
    uint64_t downloads = 0;          // Download count
    uint64_t likes = 0;              // Likes count
    float trendingScore = 0.0f;      // Trending score (if available)

    // Metadata
    uint64_t sizeBytes = 0;          // Total model size (calculated from files)
    std::string description;         // Model description
    std::chrono::system_clock::time_point createdAt;     // Creation timestamp
    std::chrono::system_clock::time_point lastModified;  // Last modified timestamp

    // Access control
    bool gated = false;              // Requires approval/authentication
    bool private_ = false;           // Private model
    bool staffPick = false;          // Recommended by HuggingFace staff

    // Files
    std::vector<FileInfo> files;     // Complete file list with metadata
};

} // namespace models
