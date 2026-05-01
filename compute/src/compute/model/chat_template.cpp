#include "chat_template.h"

namespace compute {

std::string apply_chat_template(
    const std::string&             model_type,
    bool                           is_llama3,
    const std::string&             system_prompt,
    const std::vector<ChatMessage>& messages)
{
    if (is_llama3) {
        // <|begin_of_text|> is part of the template; the tokenizer does not add BOS.
        std::string out = "<|begin_of_text|>";
        if (!system_prompt.empty()) {
            out += "<|start_header_id|>system<|end_header_id|>\n\n"
                 + system_prompt + "<|eot_id|>\n";
        }
        for (const auto& m : messages) {
            out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n"
                 + m.content + "<|eot_id|>\n";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    if (model_type == "qwen2" || model_type == "qwen3" || model_type == "qwen3_5_moe") {
        std::string out;
        if (!system_prompt.empty()) {
            out += "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        }
        for (const auto& m : messages) {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
        out += "<|im_start|>assistant\n";
        return out;
    }

    if (model_type == "mistral") {
        // Mistral v0.3 was not trained with system prompts; system_prompt is ignored.
        // BOS is prepended by the tokenizer (add_special_tokens=true).
        std::string out;
        for (const auto& m : messages) {
            if (m.role == "user")
                out += "[INST] " + m.content + " [/INST]";
            else if (m.role == "assistant")
                out += m.content + "</s>";
        }
        return out;
    }

    if (model_type == "gemma" || model_type == "gemma2" || model_type == "gemma3_text") {
        // Gemma standard template has no system block; system_prompt is ignored.
        // BOS is prepended by the tokenizer (add_special_tokens=true).
        std::string out;
        for (const auto& m : messages) {
            out += "<start_of_turn>" + m.role + "\n" + m.content + "<end_of_turn>\n";
        }
        out += "<start_of_turn>model\n";
        return out;
    }

    // Default: TinyLlama / Llama-2 template
    std::string out;
    if (!system_prompt.empty()) {
        out += "<|system|>\n" + system_prompt + "</s>\n";
    }
    for (const auto& m : messages) {
        if (m.role == "user")
            out += "<|user|>\n" + m.content + "</s>\n<|assistant|>\n";
        else if (m.role == "assistant")
            out += m.content + "</s>\n";
    }
    return out;
}

} // namespace compute
