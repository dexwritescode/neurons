#pragma once

#include <string>
#include <vector>

namespace compute {

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

// Render a chat prompt for the given model family.
//
// is_llama3:     true when model_type=="llama" AND the tokenizer contains the
//                Llama-3 header tokens (<|start_header_id|> etc.).
// system_prompt: passed empty to suppress the system block (Mistral, Gemma
//                don't have one in their standard templates).
// messages:      full conversation history; the last entry must have role "user".
//                The function appends the appropriate assistant-turn opener.
//
// Single-turn callers pass messages = {{"user", prompt}}.
// Multi-turn callers pass the full history ending with the current user turn.
std::string apply_chat_template(
    const std::string&            model_type,
    bool                          is_llama3,
    const std::string&            system_prompt,
    const std::vector<ChatMessage>& messages
);

} // namespace compute
