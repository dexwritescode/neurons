#pragma once

#include "../core/compute_types.h"
#include "model_config.h"
#include <functional>
#include <vector>
#include <cstddef>

namespace compute {

/**
 * Sampling parameters for text generation.
 * Applied in order: rep_penalty → temperature → top_k → top_p → min_p → softmax → sample.
 */
struct SamplingParams {
    float  temperature = 0.8f;  // 0 = greedy
    size_t top_k       = 50;    // 0 = disabled
    float  top_p       = 1.0f;  // 1.0 = disabled; nucleus sampling
    float  min_p       = 0.0f;  // 0.0 = disabled; clips tokens below min_p * max_prob
    float  rep_penalty = 1.0f;  // 1.0 = disabled; >1 penalises already-seen tokens
};

/**
 * Stateless sampler — model-agnostic, shared across all LanguageModel implementations.
 *
 * context_ids convention: pass only GENERATED tokens, NOT prompt tokens.
 * Penalising prompt tokens causes the model to avoid words from the question,
 * which produces incoherent responses on quantised models.
 */
class Sampler {
public:
    static int sample(const std::vector<float>& logits,
                      const SamplingParams&      params,
                      const std::vector<int>&    context_ids);
};

/**
 * Reusable generate loop shared by all LanguageModel subclasses.
 *
 * Accepts prefill and decode as std::function so each subclass can pass
 * lambdas that capture its own private state, without exposing those steps
 * through the LanguageModel interface.
 */
class GenerateHelper {
public:
    using PrefillFn = std::function<Result<std::vector<float>>(const std::vector<int>&)>;
    using DecodeFn  = std::function<Result<std::vector<float>>(int)>;

    static Result<std::vector<int>> run(
        const std::vector<int>&  input_ids,
        size_t                   max_new_tokens,
        SamplingParams           params,
        std::function<bool(int)> on_token,
        const ModelConfig&       config,
        PrefillFn                prefill,
        DecodeFn                 decode);
};

} // namespace compute