#pragma once

#include <memory>
#include <string>
#include <vector>
#include "base_command.h"
#include "models/registry/model_registry.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class ChatCommand : public BaseCommand {
public:
    explicit ChatCommand(NeuronsConfig* config);
    ~ChatCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    struct Turn {
        std::string role; // "user" or "assistant"
        std::string content;
    };

    std::string model_name_;
    std::string system_prompt_ = "You are a helpful assistant.";
    int   max_tokens_  = 4096;
    float temperature_ = 0.7f;
    int   top_k_       = 40;
    float top_p_       = 0.9f;
    float rep_penalty_ = 1.1f;

    NeuronsConfig* config_;
    std::unique_ptr<models::registry::ModelRegistry> model_registry_;

    int run_repl(const std::string& model_path);
    std::string buildPrompt(const std::string& model_type,
                            const std::vector<Turn>& history,
                            const std::string& user_input,
                            bool has_llama3_tokens) const;
    void printHelp() const;
};

} // namespace neurons::cli
