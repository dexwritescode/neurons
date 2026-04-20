#include "token_command.h"
#include <iostream>

namespace neurons::cli {

TokenCommand::TokenCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "token";
    description_  = "Manage HuggingFace API token";
}

void TokenCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);

    auto* set = cmd->add_subcommand("set", "Store a HuggingFace token");
    set->add_option("token", token_value_, "HuggingFace token (hf_...)")->required();
    set->callback([this]() { subcommand_ = "set"; });

    auto* clear = cmd->add_subcommand("clear", "Remove the stored token");
    clear->callback([this]() { subcommand_ = "clear"; });

    auto* status = cmd->add_subcommand("status", "Show token status");
    status->callback([this]() { subcommand_ = "status"; });

    cmd->require_subcommand(1);
    cmd->callback([this]() { std::exit(execute()); });
    (void)clear; (void)status;
}

int TokenCommand::execute() {
    if (subcommand_ == "set") {
        if (token_value_.empty()) {
            std::cerr << "Token cannot be empty\n";
            return 1;
        }
        config_->setHfToken(token_value_);
        if (!config_->save()) return 1;
        std::cout << "HuggingFace token saved.\n";
        return 0;
    }
    if (subcommand_ == "clear") {
        config_->clearHfToken();
        if (!config_->save()) return 1;
        std::cout << "HuggingFace token cleared.\n";
        return 0;
    }
    if (subcommand_ == "status") {
        const auto& tok = config_->hfToken();
        if (tok.empty()) {
            std::cout << "No token set. Use: neurons token set hf_...\n";
        } else {
            std::cout << "Token: " << tok.substr(0, 8) << "...\n";
        }
        return 0;
    }
    return 1;
}

} // namespace neurons::cli
