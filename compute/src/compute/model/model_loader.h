#pragma once

#include "../core/compute_types.h"
#include "../core/tensor.h"
#include "model_config.h"
#include <unordered_map>
#include <string>
#include <filesystem>
#include <vector>

namespace compute {

// Forward declaration
class ComputeBackend;

/**
 * ModelLoader handles loading transformer models from safetensors format
 * Simple, focused implementation for MLX backend with lazy evaluation preservation
 * Works generically with any MLX safetensors model
 */
class ModelLoader {
public:
    /**
     * Load complete model from directory containing config.json and .safetensors files
     * @param model_dir Path to model directory
     * @param backend ComputeBackend to use for tensor operations (must support MLX)
     * @return Result containing model config and tensor map, or error
     */
    static Result<std::pair<ModelConfig, std::unordered_map<std::string, Tensor>>>
    load_model(const std::filesystem::path& model_dir, ComputeBackend* backend);

    /**
     * Load model configuration only (no weights)
     * @param model_dir Path to model directory containing config.json
     * @return Result containing parsed model config or error
     */
    static Result<ModelConfig> load_config(const std::filesystem::path& model_dir);

private:
    /**
     * Find all .safetensors files in model directory
     * @param model_dir Path to model directory
     * @return Vector of paths to .safetensors files
     */
    static std::vector<std::filesystem::path> find_safetensors_files(
        const std::filesystem::path& model_dir);

    /**
     * Load tensors from all .safetensors files using MLX backend
     * Preserves MLX lazy evaluation - tensors are not evaluated
     * @param safetensors_files List of .safetensors file paths
     * @param backend ComputeBackend to use (must support MLX)
     * @return Result containing tensor name -> tensor map, or error
     */
    static Result<std::unordered_map<std::string, Tensor>>
    load_all_safetensors(const std::vector<std::filesystem::path>& safetensors_files,
                         ComputeBackend* backend);

    ModelLoader() = delete; // Static utility class
};

} // namespace compute