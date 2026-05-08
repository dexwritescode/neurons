#pragma once

#include "compute/model/language_model.h"
#include "compute/model/sampler.h"
#include "compute/core/compute_types.h"

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace compute {

// Callback types used by ToolRunner and shared across CLI/service/GUI paths.

// Receives each decoded text delta as it is generated. Return false to stop.
using TokenCb = std::function<bool(const std::string& delta)>;

// Invoked when the model emits a tool call. Return the tool result JSON, or
// nullopt to signal that the call was denied (model receives an error result).
using ToolCallCb = std::function<
    std::optional<std::string>(const LanguageModel::ToolCall&)>;

// Runs the generate-with-tools loop: a multi-turn cycle that generates tokens,
// detects tool calls, invokes the tool callback, injects the result, and
// continues generation. When tool_cb is null, runs a single-turn generation.
//
// initial_tokens: encoded prompt (with BOS). Mutated across turns — on each
//   tool turn the generated tokens and injection are appended before continuing.
// max_new_tokens: per-call limit passed to LanguageModel::generate().
// token_cb: called with each decoded text delta. Return false to cancel.
// tool_cb: null means no tool use. Called once per detected tool call.
// cancelled: set from another thread to abort the loop.
//
// Returns the total number of generated tokens across all turns, or an error.
class ToolRunner {
public:
    Result<uint32_t> run(
        LanguageModel&           model,
        std::vector<int>         initial_tokens,
        size_t                   max_new_tokens,
        const SamplingParams&    params,
        TokenCb                  token_cb,
        ToolCallCb               tool_cb,
        const std::atomic<bool>& cancelled);

    static constexpr int kMaxToolTurns = 5;
};

} // namespace compute
