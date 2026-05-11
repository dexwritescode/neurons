#include <gtest/gtest.h>
#include "compute/model/chat_template.h"

using compute::ChatMessage;
using compute::render_chat_template;

// ── Jinja template strings used across tests ──────────────────────────────────
//
// These are simplified but semantically correct templates for each model family,
// matching what the models' tokenizer_config.json files actually contain.

static const char* kLlama3Template = R"(
{{- bos_token }}
{%- for message in messages %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n' + message.content + '<|eot_id|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
)";

static const char* kChatMLTemplate = R"(
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
)";

static const char* kMistralTemplate = R"(
{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '[INST] ' + message.content + ' [/INST]' }}
{%- elif message.role == 'assistant' %}
{{- message.content + '</s>' }}
{%- endif %}
{%- endfor %}
)";

static const char* kGemmaTemplate = R"(
{{- bos_token }}
{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '<start_of_turn>user\n' + message.content + '<end_of_turn>\n' }}
{%- elif message.role == 'assistant' %}
{{- '<start_of_turn>model\n' + message.content + '<end_of_turn>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<start_of_turn>model\n' }}
{%- endif %}
)";

static const char* kLlama2Template = R"(
{%- if messages[0].role == 'system' %}
{{- '<|system|>\n' + messages[0].content + '</s>\n' }}
{%- set messages = messages[1:] %}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' %}
{{- '<|user|>\n' + message.content + '</s>\n<|assistant|>\n' }}
{%- elif message.role == 'assistant' %}
{{- message.content + '</s>\n' }}
{%- endif %}
{%- endfor %}
)";

// ── Single-turn tests ─────────────────────────────────────────────────────────

TEST(RenderChatTemplate, Llama3SingleTurn) {
    std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user",   "Hello"},
    };
    auto out = render_chat_template(kLlama3Template, "<|begin_of_text|>", "<|eot_id|>", msgs);
    EXPECT_EQ(out.substr(0, 17), "<|begin_of_text|>");
    EXPECT_NE(out.find("<|start_header_id|>system<|end_header_id|>"), std::string::npos);
    EXPECT_NE(out.find("You are a helpful assistant.<|eot_id|>"),     std::string::npos);
    EXPECT_NE(out.find("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"), std::string::npos);
    EXPECT_NE(out.find("<|start_header_id|>assistant<|end_header_id|>"), std::string::npos);
}

TEST(RenderChatTemplate, QwenSingleTurn) {
    std::vector<ChatMessage> msgs = {
        {"system", "You are a helpful assistant."},
        {"user",   "Hello"},
    };
    auto out = render_chat_template(kChatMLTemplate, "<s>", "</s>", msgs);
    EXPECT_NE(out.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"), std::string::npos);
    EXPECT_NE(out.find("<|im_start|>user\nHello<|im_end|>"),  std::string::npos);
    EXPECT_NE(out.find("<|im_start|>assistant\n"),             std::string::npos);
}

TEST(RenderChatTemplate, MistralSingleTurn) {
    std::vector<ChatMessage> msgs = {{"user", "Hello"}};
    auto out = render_chat_template(kMistralTemplate, "<s>", "</s>", msgs);
    EXPECT_EQ(out.find("[INST] Hello [/INST]"), std::string::npos == out.find("[INST] Hello [/INST]") ? std::string::npos : out.find("[INST] Hello [/INST]"));
    EXPECT_NE(out.find("[INST] Hello [/INST]"), std::string::npos);
}

TEST(RenderChatTemplate, GemmaSingleTurn) {
    std::vector<ChatMessage> msgs = {{"user", "Hello"}};
    auto out = render_chat_template(kGemmaTemplate, "<bos>", "<eos>", msgs);
    EXPECT_EQ(out.substr(0, 5), "<bos>");
    EXPECT_NE(out.find("<start_of_turn>user\nHello<end_of_turn>"), std::string::npos);
    EXPECT_NE(out.find("<start_of_turn>model\n"),                   std::string::npos);
    EXPECT_EQ(out.find("system"),                                    std::string::npos);
}

// ── Multi-turn tests ──────────────────────────────────────────────────────────

TEST(RenderChatTemplate, Llama3MultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"system",    "sys"},
        {"user",      "What is 2+2?"},
        {"assistant", "4."},
        {"user",      "And 3+3?"},
    };
    auto out = render_chat_template(kLlama3Template, "<|begin_of_text|>", "<|eot_id|>", msgs);
    auto pos_first = out.find("What is 2+2?");
    auto pos_asst  = out.find("4.");
    auto pos_last  = out.find("And 3+3?");
    EXPECT_NE(pos_first, std::string::npos);
    EXPECT_NE(pos_asst,  std::string::npos);
    EXPECT_NE(pos_last,  std::string::npos);
    EXPECT_LT(pos_first, pos_asst);
    EXPECT_LT(pos_asst,  pos_last);
    EXPECT_GT(out.rfind("<|start_header_id|>assistant<|end_header_id|>"), pos_last);
}

TEST(RenderChatTemplate, QwenMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = render_chat_template(kChatMLTemplate, "<s>", "</s>", msgs);
    EXPECT_NE(out.find("<|im_start|>user\nHi<|im_end|>"),           std::string::npos);
    EXPECT_NE(out.find("<|im_start|>assistant\nHello!<|im_end|>"),  std::string::npos);
    EXPECT_NE(out.find("<|im_start|>user\nBye<|im_end|>"),          std::string::npos);
    EXPECT_EQ(out.substr(out.size() - 22), "<|im_start|>assistant\n");
}

TEST(RenderChatTemplate, MistralMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = render_chat_template(kMistralTemplate, "<s>", "</s>", msgs);
    EXPECT_NE(out.find("[INST] Hi [/INST]"),   std::string::npos);
    EXPECT_NE(out.find("Hello!</s>"),           std::string::npos);
    EXPECT_NE(out.find("[INST] Bye [/INST]"),  std::string::npos);
}

TEST(RenderChatTemplate, GemmaMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = render_chat_template(kGemmaTemplate, "<bos>", "<eos>", msgs);
    EXPECT_NE(out.find("<start_of_turn>user\nHi<end_of_turn>"),           std::string::npos);
    EXPECT_NE(out.find("<start_of_turn>model\nHello!<end_of_turn>"),       std::string::npos);
    EXPECT_NE(out.find("<start_of_turn>user\nBye<end_of_turn>"),           std::string::npos);
    EXPECT_EQ(out.substr(out.size() - 21), "<start_of_turn>model\n");
}

// ── System prompt handling ─────────────────────────────────────────────────────

TEST(RenderChatTemplate, MistralIgnoresSystemRole) {
    // Mistral template skips any role that isn't user/assistant.
    std::vector<ChatMessage> msgs_sys = {{"system", "ignore me"}, {"user", "Hi"}};
    std::vector<ChatMessage> msgs_no  = {{"user", "Hi"}};
    auto with_sys    = render_chat_template(kMistralTemplate, "<s>", "</s>", msgs_sys);
    auto without_sys = render_chat_template(kMistralTemplate, "<s>", "</s>", msgs_no);
    EXPECT_EQ(with_sys, without_sys);
    EXPECT_EQ(with_sys.find("ignore me"), std::string::npos);
}

TEST(RenderChatTemplate, ChatMLNoSystemBlock) {
    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto out = render_chat_template(kChatMLTemplate, "<s>", "</s>", msgs);
    EXPECT_EQ(out.find("<|im_start|>system"), std::string::npos);
    EXPECT_NE(out.find("<|im_start|>user"),   std::string::npos);
}

TEST(RenderChatTemplate, Llama2SystemBlock) {
    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user",   "Hi"},
    };
    auto out = render_chat_template(kLlama2Template, "<s>", "</s>", msgs);
    EXPECT_NE(out.find("<|system|>\nYou are helpful.</s>"), std::string::npos);
    EXPECT_NE(out.find("<|user|>\nHi</s>\n<|assistant|>"),  std::string::npos);
}

// ── BOS token passthrough ─────────────────────────────────────────────────────

TEST(RenderChatTemplate, BosTokenExpanded) {
    std::vector<ChatMessage> msgs = {{"user", "Hi"}};
    auto out = render_chat_template(kLlama3Template, "<MY_BOS>", "<eos>", msgs);
    EXPECT_EQ(out.substr(0, 8), "<MY_BOS>");
}
