#pragma once

#include <memory>
#include <string>
#include "base_command.h"
#include "models/registry/model_registry.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class LoadCommand : public BaseCommand {
public:
    explicit LoadCommand(NeuronsConfig* config);
    ~LoadCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    std::string model_name_;
    std::string prompt_;
    bool verbose_;
    bool dry_run_;
    // Defaults mirror ChatEngine.h so the CLI faithfully reproduces GUI behavior.
    int   max_tokens_  = 200;
    float temperature_ = 0.7f;
    int   top_k_       = 40;
    float top_p_       = 0.9f;
    float rep_penalty_ = 1.1f;

    NeuronsConfig* config_;
    std::unique_ptr<models::registry::ModelRegistry> model_registry_;

    int run_inference(const std::string& model_path);
    void print_model_info(const models::registry::ModelLocation& location);
    void print_usage_help();
};

} // namespace neurons::cli