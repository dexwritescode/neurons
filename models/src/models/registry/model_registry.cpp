#include "model_registry.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <string>

namespace models::registry {

// DefaultFileSystem implementation
bool DefaultFileSystem::exists(const std::string& path) const {
    return std::filesystem::exists(path);
}

bool DefaultFileSystem::isDir(const std::string& path) const {
    return std::filesystem::is_directory(path);
}

std::vector<std::string> DefaultFileSystem::entryList(const std::string& path,
                                                     const std::vector<std::string>& nameFilters) const {
    std::vector<std::string> files;

    if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path)) {
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();

            // If no filters, include all files
            if (nameFilters.empty()) {
                files.push_back(filename);
            } else {
                // Check if filename matches any filter
                for (const std::string& filter : nameFilters) {
                    if (filter.starts_with("*.")) {
                        std::string extension = filter.substr(1); // Remove *
                        if (filename.ends_with(extension)) {
                            files.push_back(filename);
                            break;
                        }
                    } else if (filename == filter) {
                        files.push_back(filename);
                        break;
                    }
                }
            }
        }
    }

    return files;
}

std::string DefaultFileSystem::absolutePath(const std::string& path) const {
    return std::filesystem::absolute(path).string();
}

// ModelRegistry implementation
ModelRegistry::ModelRegistry(const std::string& modelsDirectory, FileSystemInterface* fileSystem)
    : models_directory_(modelsDirectory)
    , file_system_(fileSystem)
{
    if (!fileSystem) {
        owned_file_system_ = std::make_unique<DefaultFileSystem>();
        file_system_ = owned_file_system_.get();
    }
}

ModelLocation ModelRegistry::locateModel(const std::string& modelName) const {
    ModelLocation location;
    location.modelName = modelName;

    // Build the expected model path - use model name directly as path
    const std::string modelPath = buildModelPath(modelName);
    location.modelPath = modelPath;

    // Check if model directory exists
    if (!file_system_->exists(modelPath) || !file_system_->isDir(modelPath)) {
        std::cerr << "Model directory not found: " << modelPath << std::endl;
        location.format = ModelFormat::UNKNOWN;
        return location;
    }

    // Detect model format
    location.format = detectModelFormat(modelPath);
    if (location.format == ModelFormat::UNKNOWN) {
        return location;  // Silently skip unsupported formats during enumeration
    }

    // Find format-specific model files
    location.modelFiles = findModelFiles(modelPath, location.format);

    // Only look for separate config files if using safetensors format
    // GGUF files are self-contained with everything embedded
    if (location.format == ModelFormat::SAFETENSORS) {
        location.configPath = findConfigFile(modelPath, "config.json");
        location.tokenizerConfigPath = findConfigFile(modelPath, "tokenizer_config.json");
        location.specialTokensPath = findConfigFile(modelPath, "special_tokens_map.json");

        // Primary tokenizer file
        location.vocabPath = findConfigFile(modelPath, "tokenizer.json");
        if (location.vocabPath.empty()) {
            location.vocabPath = findConfigFile(modelPath, "vocab.txt");
        }
    }

    return location;
}

void ModelRegistry::setModelsDirectory(const std::string& directory) {
    models_directory_ = directory;
}

std::vector<ModelLocation> ModelRegistry::listModels() const {
    std::vector<ModelLocation> results;

    if (!std::filesystem::exists(models_directory_) ||
        !std::filesystem::is_directory(models_directory_)) {
        return results;
    }

    // Two-level structure: {models_dir}/{org}/{model-name}
    for (const auto& org_entry : std::filesystem::directory_iterator(models_directory_)) {
        if (!org_entry.is_directory()) continue;

        for (const auto& model_entry : std::filesystem::directory_iterator(org_entry.path())) {
            if (!model_entry.is_directory()) continue;

            // Reconstruct the "org/model" name relative to models_directory_
            const std::string org  = org_entry.path().filename().string();
            const std::string name = model_entry.path().filename().string();
            const std::string model_name = org + "/" + name;

            ModelLocation loc = locateModel(model_name);
            if (loc.isValid()) {
                results.push_back(std::move(loc));
            }
        }
    }

    // Sort alphabetically by modelName for stable display order
    std::sort(results.begin(), results.end(),
              [](const ModelLocation& a, const ModelLocation& b) {
                  return a.modelName < b.modelName;
              });

    return results;
}

std::string ModelRegistry::buildModelPath(const std::string& modelName) const {
    // Model name is used directly as the path: "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
    // Results in: {models_directory}/mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit
    std::filesystem::path path(models_directory_);
    path /= modelName;
    return path.string();
}

ModelFormat ModelRegistry::detectModelFormat(const std::string& modelPath) const {
    const std::vector<std::string> files = file_system_->entryList(modelPath, {});

    // Check for safetensors files first (more common for MLX models)
    for (const std::string& file : files) {
        if (file.ends_with(".safetensors")) {
            return ModelFormat::SAFETENSORS;
        }
    }

    // Check for GGUF files
    for (const std::string& file : files) {
        if (file.ends_with(".gguf")) {
            return ModelFormat::GGUF;
        }
    }

    return ModelFormat::UNKNOWN;
}

std::vector<std::string> ModelRegistry::findModelFiles(const std::string& modelPath, ModelFormat format) const {
    std::vector<std::string> nameFilters;

    switch (format) {
    case ModelFormat::SAFETENSORS:
        nameFilters.push_back("*.safetensors");
        break;
    case ModelFormat::GGUF:
        nameFilters.push_back("*.gguf");
        break;
    case ModelFormat::UNKNOWN:
        return {};
    }

    const std::vector<std::string> files = file_system_->entryList(modelPath, nameFilters);

    // Return full paths
    std::vector<std::string> fullPaths;
    for (const std::string& file : files) {
        std::filesystem::path path(modelPath);
        path /= file;
        fullPaths.push_back(path.string());
    }

    return fullPaths;
}

std::string ModelRegistry::findConfigFile(const std::string& modelPath, const std::string& fileName) const {
    std::filesystem::path path(modelPath);
    path /= fileName;
    std::string fullPath = path.string();

    if (file_system_->exists(fullPath)) {
        return fullPath;
    }
    return std::string();
}

} // namespace models::registry