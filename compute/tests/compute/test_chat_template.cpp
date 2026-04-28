#include <gtest/gtest.h>
#include "compute/model/chat_template.h"

using compute::ChatMessage;
using compute::apply_chat_template;

// ── Single-turn (single-turn is multi-turn with one user message) ─────────────

TEST(ChatTemplate, Llama2SingleTurn) {
    auto out = apply_chat_template("llama", /*is_llama3=*/false,
        "You are a helpful assistant.", {{"user", "Hello"}});
    EXPECT_NE(out.find("<|system|>"), std::string::npos);
    EXPECT_NE(out.find("You are a helpful assistant."), std::string::npos);
    EXPECT_NE(out.find("<|user|>\nHello</s>"), std::string::npos);
    EXPECT_NE(out.find("<|assistant|>"), std::string::npos);
}

TEST(ChatTemplate, Llama3SingleTurn) {
    auto out = apply_chat_template("llama", /*is_llama3=*/true,
        "You are a helpful assistant.", {{"user", "Hello"}});
    EXPECT_EQ(out.substr(0, 17), "<|begin_of_text|>");
    EXPECT_NE(out.find("<|start_header_id|>system<|end_header_id|>"), std::string::npos);
    EXPECT_NE(out.find("You are a helpful assistant.<|eot_id|>"), std::string::npos);
    EXPECT_NE(out.find("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"), std::string::npos);
    EXPECT_NE(out.find("<|start_header_id|>assistant<|end_header_id|>"), std::string::npos);
}

TEST(ChatTemplate, QwenSingleTurn) {
    for (const auto& mt : {"qwen2", "qwen3", "qwen3_5_moe"}) {
        auto out = apply_chat_template(mt, false,
            "You are a helpful assistant.", {{"user", "Hello"}});
        EXPECT_NE(out.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"), std::string::npos) << mt;
        EXPECT_NE(out.find("<|im_start|>user\nHello<|im_end|>"), std::string::npos) << mt;
        EXPECT_NE(out.find("<|im_start|>assistant\n"), std::string::npos) << mt;
    }
}

TEST(ChatTemplate, MistralSingleTurn) {
    auto out = apply_chat_template("mistral", false, "ignored", {{"user", "Hello"}});
    EXPECT_EQ(out, "[INST] Hello [/INST]");
}

TEST(ChatTemplate, GemmaSingleTurn) {
    for (const auto& mt : {"gemma", "gemma2", "gemma3_text"}) {
        auto out = apply_chat_template(mt, false, "ignored", {{"user", "Hello"}});
        EXPECT_NE(out.find("<start_of_turn>user\nHello<end_of_turn>"), std::string::npos) << mt;
        EXPECT_NE(out.find("<start_of_turn>model\n"), std::string::npos) << mt;
        EXPECT_EQ(out.find("system"), std::string::npos) << mt;
    }
}

// ── Multi-turn: history turns appear between system and final user turn ───────

TEST(ChatTemplate, Llama3MultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "What is 2+2?"},
        {"assistant", "4."},
        {"user",      "And 3+3?"},
    };
    auto out = apply_chat_template("llama", true, "sys", msgs);
    // History pair appears before final user turn
    auto pos_first_user = out.find("What is 2+2?");
    auto pos_asst       = out.find("4.");
    auto pos_last_user  = out.find("And 3+3?");
    EXPECT_NE(pos_first_user, std::string::npos);
    EXPECT_NE(pos_asst,       std::string::npos);
    EXPECT_NE(pos_last_user,  std::string::npos);
    EXPECT_LT(pos_first_user, pos_asst);
    EXPECT_LT(pos_asst,       pos_last_user);
    // Assistant opener at the end
    auto pos_asst_opener = out.rfind("<|start_header_id|>assistant<|end_header_id|>");
    EXPECT_GT(pos_asst_opener, pos_last_user);
}

TEST(ChatTemplate, QwenMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = apply_chat_template("qwen2", false, "sys", msgs);
    EXPECT_NE(out.find("<|im_start|>user\nHi<|im_end|>"),           std::string::npos);
    EXPECT_NE(out.find("<|im_start|>assistant\nHello!<|im_end|>"),  std::string::npos);
    EXPECT_NE(out.find("<|im_start|>user\nBye<|im_end|>"),          std::string::npos);
    // Ends with assistant opener
    EXPECT_EQ(out.substr(out.size() - 22), "<|im_start|>assistant\n");
}

TEST(ChatTemplate, MistralMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = apply_chat_template("mistral", false, "ignored", msgs);
    EXPECT_NE(out.find("[INST] Hi [/INST]"),   std::string::npos);
    EXPECT_NE(out.find("Hello!</s>"),           std::string::npos);
    EXPECT_NE(out.find("[INST] Bye [/INST]"),  std::string::npos);
}

TEST(ChatTemplate, GemmaMultiTurn) {
    std::vector<ChatMessage> msgs = {
        {"user",      "Hi"},
        {"assistant", "Hello!"},
        {"user",      "Bye"},
    };
    auto out = apply_chat_template("gemma", false, "ignored", msgs);
    EXPECT_NE(out.find("<start_of_turn>user\nHi<end_of_turn>"),          std::string::npos);
    EXPECT_NE(out.find("<start_of_turn>assistant\nHello!<end_of_turn>"), std::string::npos);
    EXPECT_NE(out.find("<start_of_turn>user\nBye<end_of_turn>"),         std::string::npos);
    EXPECT_EQ(out.substr(out.size() - 21), "<start_of_turn>model\n");
}

// ── System prompt suppression ─────────────────────────────────────────────────

TEST(ChatTemplate, MistralIgnoresSystemPrompt) {
    auto with_sys    = apply_chat_template("mistral", false, "sys", {{"user", "Hi"}});
    auto without_sys = apply_chat_template("mistral", false, "",    {{"user", "Hi"}});
    EXPECT_EQ(with_sys, without_sys);
    EXPECT_EQ(with_sys.find("sys"), std::string::npos);
}

TEST(ChatTemplate, GemmaIgnoresSystemPrompt) {
    auto with_sys    = apply_chat_template("gemma", false, "sys", {{"user", "Hi"}});
    auto without_sys = apply_chat_template("gemma", false, "",    {{"user", "Hi"}});
    EXPECT_EQ(with_sys, without_sys);
    EXPECT_EQ(with_sys.find("sys"), std::string::npos);
}

TEST(ChatTemplate, EmptySystemPromptSuppressesBlock) {
    // Qwen with no system prompt should not emit a system block
    auto out = apply_chat_template("qwen2", false, "", {{"user", "Hi"}});
    EXPECT_EQ(out.find("<|im_start|>system"), std::string::npos);
    EXPECT_NE(out.find("<|im_start|>user"), std::string::npos);
}
