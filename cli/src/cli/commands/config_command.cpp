#include "config_command.h"
#include <iostream>

namespace neurons::cli {

ConfigCommand::ConfigCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "config";
    description_  = "Show or change Neurons configuration";
}

void ConfigCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);

    auto* show = cmd->add_subcommand("show", "Print current configuration");
    show->callback([this]() { do_show_ = true; });

    auto* set = cmd->add_subcommand("set", "Set a configuration value");
    set->add_option("key", set_key_, "Configuration key (e.g. dir)")->required();
    set->add_option("value", set_value_, "New value")->required();
    set->callback([this]() { do_show_ = false; });

    cmd->require_subcommand(1);
    cmd->callback([this]() { std::exit(execute()); });
}

int ConfigCommand::execute() {
    if (do_show_) return cmd_show();
    return cmd_set(set_key_, set_value_);
}

int ConfigCommand::cmd_show() {
    std::cout << "dir         : " << config_->neuronsDir().string() << "\n";
    std::cout << "models_dir  : " << config_->modelsDirectory().string() << "\n";
    std::cout << "chats_dir   : " << config_->chatsDirectory().string() << "\n";
    const auto& token = config_->hfToken();
    if (token.empty()) {
        std::cout << "hf_token    : (not set)\n";
    } else {
        std::cout << "hf_token    : " << token.substr(0, 8) << "...\n";
    }
    const auto& nodeId = config_->activeNodeId();
    std::cout << "active_node : " << (nodeId.empty() ? "(local)" : nodeId) << "\n";
    std::cout << "nodes       : " << config_->nodes().size() << " configured\n";
    return 0;
}

int ConfigCommand::cmd_set(const std::string& key, const std::string& value) {
    if (key == "dir") {
        config_->setNeuronsDir(value);
        if (!config_->save()) return 1;
        std::cout << "Set dir to: " << value << "\n";
        return 0;
    }
    std::cerr << "Unknown config key: " << key << "\n";
    std::cerr << "Available keys: dir\n";
    return 1;
}

} // namespace neurons::cli
