#include "model_loader.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
namespace mx = mlx::core;
#endif

#include <algorithm>

namespace compute {

Result<ModelConfig> ModelLoader::load_config(const std::filesystem::path& model_dir) {
    auto config_path = model_dir / "config.json";
    return ModelConfig::from_config_file(config_path);
}

std::vector<std::filesystem::path>
ModelLoader::find_safetensors_files(const std::filesystem::path& model_dir) {
    std::vector<std::filesystem::path> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
                files.push_back(entry.path());
        }
    } catch (const std::filesystem::filesystem_error&) {}
    std::sort(files.begin(), files.end());
    return files;
}

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)

Result<std::pair<ModelConfig, std::unordered_map<std::string, mx::array>>>
ModelLoader::load_model_mlx(const std::filesystem::path& model_dir) {
    if (!std::filesystem::exists(model_dir) || !std::filesystem::is_directory(model_dir))
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "Model directory not found: " + model_dir.string()});

    auto config_result = load_config(model_dir);
    if (!config_result) return std::unexpected(config_result.error());

    auto files = find_safetensors_files(model_dir);
    if (files.empty())
        return std::unexpected(Error{ErrorCode::InvalidInput,
            "No .safetensors files found in: " + model_dir.string()});

    std::unordered_map<std::string, mx::array> all_weights;
    try {
        for (const auto& file : files) {
            auto [tensor_map, _] = mx::load_safetensors(file.string());
            for (auto& [name, arr] : tensor_map)
                all_weights.emplace(std::move(name), std::move(arr));
        }
    } catch (const std::exception& e) {
        return std::unexpected(Error{ErrorCode::ComputeError,
            "Failed to load safetensors: " + std::string(e.what())});
    }

    return std::make_pair(std::move(*config_result), std::move(all_weights));
}

#endif // MLX_BACKEND_ENABLED

} // namespace compute
