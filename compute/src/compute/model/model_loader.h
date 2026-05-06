#pragma once

#include "../core/compute_types.h"
#include "model_config.h"
#include <filesystem>
#include <string>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include <mlx/mlx.h>
#endif

namespace compute {

class ModelLoader {
public:
    static Result<ModelConfig> load_config(const std::filesystem::path& model_dir);

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    static Result<std::pair<ModelConfig, std::unordered_map<std::string, mlx::core::array>>>
    load_model_mlx(const std::filesystem::path& model_dir);
#endif

private:
    static std::vector<std::filesystem::path> find_safetensors_files(
        const std::filesystem::path& model_dir);

    ModelLoader() = delete;
};

} // namespace compute
