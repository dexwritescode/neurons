#include "model_loader.h"
#include "../core/compute_backend.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#include <algorithm>
#include <sstream>

namespace compute {

Result<std::pair<ModelConfig, std::unordered_map<std::string, Tensor>>>
ModelLoader::load_model(const std::filesystem::path& model_dir, ComputeBackend* backend) {
    if (!backend) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "Backend cannot be null"});
    }

    if (!std::filesystem::exists(model_dir)) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Model directory does not exist: " + model_dir.string()});
    }

    if (!std::filesystem::is_directory(model_dir)) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "Path is not a directory: " + model_dir.string()});
    }

    // 1. Load configuration
    auto config_result = load_config(model_dir);
    if (!config_result) {
        return std::unexpected(config_result.error());
    }

    // 2. Find safetensors files
    auto safetensors_files = find_safetensors_files(model_dir);
    if (safetensors_files.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "No .safetensors files found in directory: " + model_dir.string()});
    }

    // 3. Load all tensors
    auto tensors_result = load_all_safetensors(safetensors_files, backend);
    if (!tensors_result) {
        return std::unexpected(tensors_result.error());
    }

    return std::make_pair(*config_result, *tensors_result);
}

Result<ModelConfig> ModelLoader::load_config(const std::filesystem::path& model_dir) {
    auto config_path = model_dir / "config.json";
    return ModelConfig::from_config_file(config_path);
}

std::vector<std::filesystem::path>
ModelLoader::find_safetensors_files(const std::filesystem::path& model_dir) {
    std::vector<std::filesystem::path> safetensors_files;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                safetensors_files.push_back(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        // Return empty vector on filesystem errors
        return {};
    }

    // Sort files for consistent ordering
    std::sort(safetensors_files.begin(), safetensors_files.end());
    return safetensors_files;
}

Result<std::unordered_map<std::string, Tensor>>
ModelLoader::load_all_safetensors(const std::vector<std::filesystem::path>& safetensors_files,
                                  ComputeBackend* backend) {

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    // Check that we have an MLX backend
    if (backend->type() != BackendType::MLX) {
        return std::unexpected(Error{ErrorCode::InvalidInput,
                                   "ModelLoader requires MLX backend for safetensors loading"});
    }

    std::unordered_map<std::string, Tensor> all_tensors;

    try {
        for (const auto& file_path : safetensors_files) {
            // Use MLX to load safetensors file
            auto safetensors_result = mx::load_safetensors(file_path.string());

            // mx::load_safetensors returns a pair<tensor_map, metadata_map>
            const auto& tensor_map = safetensors_result.first;

            // Convert each MLX array to compute Tensor using wrap_native_tensor
            for (const auto& [name, mlx_array] : tensor_map) {
                // Get shape from MLX array
                auto mlx_shape = mlx_array.shape();
                std::vector<size_t> shape(mlx_shape.begin(), mlx_shape.end());

                // Create a copy of the MLX array for wrapping
                // Note: This preserves lazy evaluation - no .eval() called
                auto* array_ptr = new mx::array(mlx_array);

                // Wrap the MLX array as a Tensor
                auto tensor = backend->wrap_native_tensor(array_ptr, shape);

                // Check for duplicate tensor names across files
                if (all_tensors.find(name) != all_tensors.end()) {
                    return std::unexpected(Error{ErrorCode::InvalidInput,
                                               "Duplicate tensor name found: " + name});
                }

                all_tensors.emplace(name, std::move(tensor));
            }
        }

        return all_tensors;

    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
                                   "Failed to load safetensors: " + std::string(e.what())});
    }

#else
    return std::unexpected(Error{ErrorCode::BackendNotAvailable,
                               "MLX backend not available - cannot load safetensors"});
#endif
}

} // namespace compute