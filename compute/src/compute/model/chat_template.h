#pragma once

#include <string>
#include <vector>

namespace compute {

class HFTokenizer;

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

// Low-level: render a Jinja template string directly.
// messages: full message list — include a {"system", ...} entry if the template
//           supports it; the caller controls whether to prepend one.
// Exposed so unit tests can exercise the renderer without a real tokenizer.
std::string render_chat_template(
    const std::string&              template_str,
    const std::string&              bos_token,
    const std::string&              eos_token,
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt = true
);

// High-level: read chat_template from tokenizer.config(), prepend system_prompt
// as a system message, and render via minja.
// Falls back to hardcoded family dispatch if the template is absent or fails.
std::string apply_chat_template(
    const HFTokenizer&              tokenizer,
    const std::string&              system_prompt,
    const std::vector<ChatMessage>& messages,
    bool                            add_generation_prompt = true
);

} // namespace compute
