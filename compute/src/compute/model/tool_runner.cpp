#include "compute/model/tool_runner.h"

#include <string>
#include <vector>

namespace compute {

Result<ToolRunResult> ToolRunner::run(
        LanguageModel&           model,
        std::vector<int>         all_tokens,
        size_t                   max_new_tokens,
        const SamplingParams&    params,
        TokenCb                  token_cb,
        ToolCallCb               tool_cb,
        const std::atomic<bool>& cancelled) {

    const auto& tok       = model.tokenizer();
    const bool can_detect = model.supports_tool_use();
    const bool can_execute = can_detect && (tool_cb != nullptr);
    ToolRunResult out{};

    for (int turn = 0; turn <= kMaxToolTurns; ++turn) {
        std::vector<int> gen_so_far;
        std::string decoded_so_far;
        std::string accumulated;
        std::optional<LanguageModel::ToolCall> pending_tool;

        auto result = model.generate(all_tokens, max_new_tokens, params,
            [&](int tok_id) -> bool {
                if (cancelled.load(std::memory_order_relaxed)) return false;
                if (model.config().is_eos_token(tok_id)) return false;

                gen_so_far.push_back(tok_id);
                const std::string new_decoded = tok.decode(gen_so_far);
                const std::string delta = new_decoded.substr(decoded_so_far.size());
                decoded_so_far = new_decoded;
                accumulated += delta;

                if (can_detect) {
                    auto tc = model.detect_tool_call(accumulated);
                    if (tc.has_value()) {
                        pending_tool = std::move(tc);
                        return false;
                    }
                    // Suppress tokens once the tool-call open tag is in the
                    // stream — the block is internal protocol, not user output.
                    if (accumulated.find("<tool_call>") != std::string::npos)
                        return true;
                }
                return token_cb(delta);
            });

        out.gen_tokens += static_cast<uint32_t>(gen_so_far.size());

        if (!result.has_value())
            return std::unexpected(result.error());

        if (!pending_tool || turn == kMaxToolTurns) break;

        // Client-side execution: surface the tool call and stop.
        if (!can_execute) {
            out.pending_tool = std::move(pending_tool);
            break;
        }

        // Server-side execution: invoke the tool, inject the result, continue.
        auto tool_result = tool_cb(*pending_tool);
        const std::string result_json = tool_result.value_or(
            R"({"error":"Tool call denied"})");
        const std::string injection = model.format_tool_result(
            pending_tool->name, result_json);

        // Append generated tokens + injection before the next turn. No BOS on
        // continuations.
        all_tokens.insert(all_tokens.end(), gen_so_far.begin(), gen_so_far.end());
        const auto inj_tokens = tok.encode(injection, /*add_special_tokens=*/false);
        all_tokens.insert(all_tokens.end(), inj_tokens.begin(), inj_tokens.end());
    }

    return out;
}

} // namespace compute
