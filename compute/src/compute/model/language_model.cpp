#include "language_model.h"
#include "llama_model.h"
#include "gemma_model.h"
#include "qwen3_moe_model.h"
#include "model_loader.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "qwen3_moe_model_mlx.h"
#include "gemma_model_mlx.h"
#endif

namespace compute {

// ── Factory ───────────────────────────────────────────────────────────────────
//
// Reads model_type from config.json and instantiates the right subclass.
// When a new architecture is added, add one branch here and one new subclass.

Result<std::unique_ptr<LanguageModel>> LanguageModel::load(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend,
    size_t                       context_size)
{
    // Peek at config to determine the model family without loading weights twice.
    auto config_result = ModelLoader::load_config(model_dir);
    if (!config_result) return std::unexpected(config_result.error());

    const std::string& model_type = config_result->model_type;

    if (model_type == "llama" || model_type == "mistral" ||
        model_type == "qwen2" || model_type == "qwen3") {
        auto result = LlamaModel::from_model_dir(model_dir, backend, context_size);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<LlamaModel>(std::move(*result));
    }

    if (model_type == "gemma" || model_type == "gemma2" || model_type == "gemma3_text") {
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
        auto result = GemmaModelMLX::from_model_dir(model_dir);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<GemmaModelMLX>(std::move(*result));
#else
        auto result = GemmaModel::from_model_dir(model_dir, backend);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<GemmaModel>(std::move(*result));
#endif
    }

    if (model_type == "qwen3_5_moe" || model_type == "qwen3_moe") {
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
        auto result = Qwen3MoeModelMLX::from_model_dir(model_dir, context_size);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<Qwen3MoeModelMLX>(std::move(*result));
#else
        auto result = Qwen3MoeModel::from_model_dir(model_dir, backend);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<Qwen3MoeModel>(std::move(*result));
#endif
    }

    return std::unexpected(Error{ErrorCode::InvalidModel,
        "Unsupported model type: \"" + model_type +
        "\". Supported: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3_text, qwen3_5_moe, qwen3_moe"});
}

} // namespace compute
