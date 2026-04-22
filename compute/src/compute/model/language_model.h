#pragma once

#include "../core/compute_types.h"
#include "model_config.h"
#include "sampler.h"
#include "simple_bpe_tokenizer.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
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

    // ── Tool-use capability ───────────────────────────────────────────────────

    // A parsed tool call emitted by the model during generation.
    struct ToolCall {
        std::string name;            // function name
        std::string arguments_json;  // JSON object string (OpenAI-style)
    };

    // True if this model supports structured tool/function calling.
    // Detected at load time via tokenizer vocab probe + model-name heuristics.
    virtual bool supports_tool_use() const { return false; }

    // Given a JSON array of tool definitions (OpenAI format), returns the
    // text to prepend to the system message for this model family.
    // Returns empty string for models that don't support tool use.
    virtual std::string format_tool_system_prompt(const std::string& tools_json) const { return ""; }

    // Scan accumulated generated text for a complete tool-call block.
    // Returns the parsed ToolCall when a complete block is detected.
    // The caller should stop generation and invoke the tool when this fires.
    // Returns nullopt while the block is incomplete or absent.
    virtual std::optional<ToolCall> detect_tool_call(const std::string& text) const { return std::nullopt; }

    // Build the text to inject after a tool call so the model can continue.
    // result_json: the tool's output (or an error string if denied).
    // Returns the full injection string including any assistant-turn prefix
    // needed to resume generation for this model family.
    virtual std::string format_tool_result(const std::string& tool_name,
                                           const std::string& result_json) const { return ""; }

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
