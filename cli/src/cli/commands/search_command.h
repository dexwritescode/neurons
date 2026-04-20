#pragma once

#include <string>
#include <vector>
#include "base_command.h"
#include "models/api/huggingface_client.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class SearchCommand : public BaseCommand {
public:
    explicit SearchCommand(NeuronsConfig* config);
    ~SearchCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    std::string query_;
    int limit_ = 20;
    std::string sort_;           // downloads | likes | trending | lastModified
    std::vector<std::string> pipeline_tags_;
    std::string author_;
    bool show_gated_ = false;

    NeuronsConfig* config_;

    void print_usage_help();
};

} // namespace neurons::cli
