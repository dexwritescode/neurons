#pragma once

#include "../core/compute_types.h"
#include "model_config.h"
#include "sampler.h"
#include "simple_bpe_tokenizer.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace compute {

class ComputeBackend;

/**
 * Abstract interface for all language model implementations.
 *
 * App code (ChatEngine, CLI) and tests program against this interface only —
 * they never reference a concrete model class.
 *
 * Each model family has one concrete subclass:
 *   LlamaModel   — LlamaForCausalLM, MistralForCausalLM (identical forward pass)
 *   (future)       GemmaForCausalLM, PhiForCausalLM, ...
 *
 * New architecture = one new subclass + one line in LanguageModel::load().
 */
class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    LanguageModel(const LanguageModel&)            = delete;
    LanguageModel& operator=(const LanguageModel&) = delete;
    LanguageModel(LanguageModel&&)                 = default;
    LanguageModel& operator=(LanguageModel&&)      = default;

    // ── KV-cache inference (subclass responsibility) ─────────────────────────

    // Run full prompt, populate KV cache, return last-token logits.
    // Resets any existing cache before running.
    virtual Result<std::vector<float>> prefill(const std::vector<int>& prompt_ids) = 0;

    // Process one new token using the populated KV cache.
    // Must call prefill() first.
    virtual Result<std::vector<float>> decode(int token_id) = 0;

    // Clear KV cache (call before starting a new conversation).
    virtual void reset_cache() = 0;

    // ── Generation (shared default: prefill → loop(sample → decode)) ─────────
    //
    // on_token: called with each sampled token id. Return false to stop early.
    // Subclasses may override for architecture-specific generation strategies.
    virtual Result<std::vector<int>> generate(
        const std::vector<int>&  input_ids,
        size_t                   max_new_tokens = 200,
        SamplingParams           params         = {},
        std::function<bool(int)> on_token       = nullptr);

    // ── Model metadata (subclass responsibility) ─────────────────────────────

    virtual const ModelConfig&        config()         const = 0;
    virtual const std::string&        model_type()     const = 0;
    virtual const SimpleBpeTokenizer& tokenizer()      const = 0;
    virtual ComputeBackend*           backend()        const = 0;
    virtual size_t                    num_parameters() const = 0;

    // ── Factory ───────────────────────────────────────────────────────────────

    // Reads config.json from model_dir, dispatches to the right subclass.
    // Returns an error for unsupported model_type values.
    static Result<std::unique_ptr<LanguageModel>> load(
        const std::filesystem::path& model_dir,
        ComputeBackend*              backend);

protected:
    LanguageModel() = default;
};

} // namespace compute
