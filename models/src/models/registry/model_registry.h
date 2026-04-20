#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace models::registry {

enum class ModelFormat {
    SAFETENSORS,
    GGUF,
    UNKNOWN
};

// Result of locating a model on disk
struct ModelLocation {
    std::string modelName;              // e.g., "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
    std::string modelPath;              // Full path to model directory
    ModelFormat format;                 // Detected model format

    // Format-specific model files
    std::vector<std::string> modelFiles; // .safetensors or .gguf files for this format

    // Common configuration files
    std::string configPath;             // config.json path (empty if not found)
    std::string tokenizerConfigPath;    // tokenizer_config.json path (empty if not found)
    std::string vocabPath;              // tokenizer.json or vocab.txt path (empty if not found)
    std::string specialTokensPath;      // special_tokens_map.json path (empty if not found)

    bool isValid() const { return !modelPath.empty() && format != ModelFormat::UNKNOWN; }
};

// File system abstraction for dependency injection and testing
class FileSystemInterface {
public:
    virtual ~FileSystemInterface() = default;
    virtual bool exists(const std::string& path) const = 0;
    virtual bool isDir(const std::string& path) const = 0;
    virtual std::vector<std::string> entryList(const std::string& path,
                                              const std::vector<std::string>& nameFilters = {}) const = 0;
    virtual std::string absolutePath(const std::string& path) const = 0;
};

// Default file system implementation using std::filesystem
class DefaultFileSystem : public FileSystemInterface {
public:
    bool exists(const std::string& path) const override;
    bool isDir(const std::string& path) const override;
    std::vector<std::string> entryList(const std::string& path,
                                      const std::vector<std::string>& nameFilters = {}) const override;
    std::string absolutePath(const std::string& path) const override;
};

class ModelRegistry {
public:
    explicit ModelRegistry(
        const std::string& modelsDirectory,
        FileSystemInterface* fileSystem = nullptr
    );
    ~ModelRegistry() = default;

    // Core functionality - locate and identify a model
    ModelLocation locateModel(const std::string& modelName) const;

    // Enumerate all valid models in the models directory (two-level: org/model)
    std::vector<ModelLocation> listModels() const;

    // Configuration
    const std::string& modelsDirectory() const { return models_directory_; }
    void setModelsDirectory(const std::string& directory);

private:
    std::string models_directory_;
    std::unique_ptr<FileSystemInterface> owned_file_system_;
    FileSystemInterface* file_system_; // Non-owning pointer

    // Helper methods
    std::string buildModelPath(const std::string& modelName) const;
    ModelFormat detectModelFormat(const std::string& modelPath) const;
    std::vector<std::string> findModelFiles(const std::string& modelPath, ModelFormat format) const;
    std::string findConfigFile(const std::string& modelPath, const std::string& fileName) const;
};

} // namespace models::registry