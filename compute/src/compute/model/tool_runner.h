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

// Returned by ToolRunner::run().
// When tool_cb is null and the model emits a tool call, execution stops and
// pending_tool is populated — the caller is responsible for executing the tool
// and sending the result back in a subsequent request (HTTP client-side model).
struct ToolRunResult {
    uint32_t                               gen_tokens = 0;
    std::optional<LanguageModel::ToolCall> pending_tool;
};

// Runs the generate-with-tools loop: a multi-turn cycle that generates tokens,
// detects tool calls, invokes the tool callback, injects the result, and
// continues generation.
//
// initial_tokens: encoded prompt (with BOS). Mutated across turns — on each
//   tool turn the generated tokens and injection are appended before continuing.
// max_new_tokens: per-call limit passed to LanguageModel::generate().
// token_cb: called with each decoded text delta. Return false to cancel.
// tool_cb: when non-null, tool calls are executed server-side and generation
//   continues (gRPC/CLI path). When null, the first detected tool call is
//   surfaced in ToolRunResult::pending_tool and the loop stops — client-side
//   execution (HTTP path).
// cancelled: set from another thread to abort the loop.
class ToolRunner {
public:
    Result<ToolRunResult> run(
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
