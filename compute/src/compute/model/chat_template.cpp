#include "chat_template.h"
#include "simple_bpe_tokenizer.h"

#include <minja/minja.hpp>
#include <nlohmann/json.hpp>

namespace compute {

// ── Helpers ───────────────────────────────────────────────────────────────────

static nlohmann::ordered_json to_json(const std::vector<ChatMessage>& messages) {
    auto arr = nlohmann::ordered_json::array();
    for (const auto& m : messages)
        arr.push_back(nlohmann::ordered_json{{"role", m.role}, {"content", m.content}});
    return arr;
}

// Hardcoded family dispatch: used when chat_template is absent or rendering throws.
// Family is detected from the tokenizer vocab rather than a model_type string.
static std::string fallback(
    const SimpleBpeTokenizer&       tok,
    const std::string&              system_prompt,
    const std::vector<ChatMessage>& messages)
{
    if (tok.find_token_id("<|start_header_id|>") != -1) {
        std::string out = "<|begin_of_text|>";
        if (!system_prompt.empty())
            out += "<|start_header_id|>system<|end_header_id|>\n\n"
                 + system_prompt + "<|eot_id|>\n";
        for (const auto& m : messages)
            out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
                 + m.content + "<|eot_id|>\n";
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }
    if (tok.find_token_id("<|im_start|>") != -1) {
        std::string out;
        if (!system_prompt.empty())
            out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        for (const auto& m : messages)
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return out;
    }
    if (tok.find_token_id("[INST]") != -1) {
        std::string out;
        for (const auto& m : messages) {
            if (m.role == "user")           out += "[INST] " + m.content + " [/INST]";
            else if (m.role == "assistant") out += m.content + "</s>";
        }
        return out;
    }
    if (tok.find_token_id("<start_of_turn>") != -1) {
        std::string out;
        for (const auto& m : messages)
            out += "<start_of_turn>" + m.role + "\n" + m.content + "<end_of_turn>\n";
        out += "<start_of_turn>model\n";
        return out;
    }
    // Default: TinyLlama / Llama-2
    std::string out;
    if (!system_prompt.empty())
        out += "<|system|>\n" + system_prompt + "</s>\n";
    for (const auto& m : messages) {
        if (m.role == "user")           out += "<|user|>\n" + m.content + "</s>\n<|assistant|>\n";
        else if (m.role == "assistant") out += m.content + "</s>\n";
    }
    return out;
}

// ── Public ────────────────────────────────────────────────────────────────────

std::string render_chat_template(
    const std::string&              template_str,
    const std::string&              bos_token,
    const std::string&              eos_token,
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt)
{
    auto tmpl = minja::Parser::parse(template_str, {});
    nlohmann::ordered_json ctx_json = {
        {"messages",              to_json(messages)},
        {"bos_token",             bos_token},
        {"eos_token",             eos_token},
        {"add_generation_prompt", add_generation_prompt},
    };
    return tmpl->render(minja::Context::make(minja::Value(ctx_json)));
}

std::string apply_chat_template(
    const SimpleBpeTokenizer&       tok,
    const std::string&              system_prompt,
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt)
{
    const auto& tmpl_str = tok.config().chat_template;
    if (tmpl_str.empty())
        return fallback(tok, system_prompt, messages);

    const auto& cfg = tok.config();

    // Try with system message prepended.
    std::vector<ChatMessage> with_sys;
    if (!system_prompt.empty())
        with_sys.push_back({"system", system_prompt});
    with_sys.insert(with_sys.end(), messages.begin(), messages.end());

    try {
        return render_chat_template(tmpl_str, cfg.bos_token, cfg.eos_token,
                                    with_sys, add_generation_prompt);
    } catch (const std::exception&) {}

    // Some templates raise_exception on the system role — retry without it.
    if (!system_prompt.empty()) {
        try {
            return render_chat_template(tmpl_str, cfg.bos_token, cfg.eos_token,
                                        messages, add_generation_prompt);
        } catch (const std::exception&) {}
    }

    return fallback(tok, system_prompt, messages);
}

} // namespace compute
