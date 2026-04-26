#include "language_model.h"
#include "llama_model.h"
#include "gemma_model.h"
#include "qwen3_moe_model.h"
#include "model_loader.h"
#include "sampler.h"
#include <unordered_set>

namespace compute {

// ── Default generate() ───────────────────────────────────────────────────────
//
// Shared across all model families. Subclasses override prefill() and decode().
// The rep_penalty window is limited to generated tokens only — not the prompt.

Result<std::vector<int>> LanguageModel::generate(
    const std::vector<int>&  input_ids,
    size_t                   max_new_tokens,
    SamplingParams           params,
    std::function<bool(int)> on_token)
{
    if (input_ids.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidInput, "input_ids cannot be empty"});
    }

    // Build EOS set for O(1) lookup — Llama-3 has multiple EOS token IDs.
    std::unordered_set<int> eos_set;
    if (config().eos_token_ids.has_value()) {
        for (int id : *config().eos_token_ids) eos_set.insert(id);
    } else {
        eos_set.insert(2);
    }

    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    auto logits = prefill(input_ids);
    if (!logits) return std::unexpected(logits.error());

    for (size_t step = 0; step < max_new_tokens; ++step) {
        int next_token = Sampler::sample(*logits, params, generated);
        generated.push_back(next_token);

        if (on_token && !on_token(next_token)) break;
        if (eos_set.count(next_token)) break;
        if (step + 1 == max_new_tokens) break;

        logits = decode(next_token);
        if (!logits) return std::unexpected(logits.error());
    }

    return generated;
}

// ── Factory ───────────────────────────────────────────────────────────────────
//
// Reads model_type from config.json and instantiates the right subclass.
// When a new architecture is added, add one branch here and one new subclass.

Result<std::unique_ptr<LanguageModel>> LanguageModel::load(
    const std::filesystem::path& model_dir,
    ComputeBackend*              backend)
{
    // Peek at config to determine the model family without loading weights twice.
    auto config_result = ModelLoader::load_config(model_dir);
    if (!config_result) return std::unexpected(config_result.error());

    const std::string& model_type = config_result->model_type;

    if (model_type == "llama" || model_type == "mistral" ||
        model_type == "qwen2" || model_type == "qwen3") {
        auto result = LlamaModel::from_model_dir(model_dir, backend);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<LlamaModel>(std::move(*result));
    }

    if (model_type == "gemma" || model_type == "gemma2" || model_type == "gemma3_text") {
        auto result = GemmaModel::from_model_dir(model_dir, backend);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<GemmaModel>(std::move(*result));
    }

    if (model_type == "qwen3_5_moe") {
        auto result = Qwen3MoeModel::from_model_dir(model_dir, backend);
        if (!result) return std::unexpected(result.error());
        return std::make_unique<Qwen3MoeModel>(std::move(*result));
    }

    return std::unexpected(Error{ErrorCode::InvalidModel,
        "Unsupported model type: \"" + model_type +
        "\". Supported: llama, mistral, qwen2, qwen3, gemma, gemma2, gemma3_text, qwen3_5_moe"});
}

} // namespace compute
