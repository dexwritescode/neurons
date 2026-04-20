#pragma once

#include "base_command.h"
#include "cli/utils/table_formatter.h"
#include "cli/config/neurons_config.h"
#include <filesystem>
#include <cstdint>

namespace neurons::cli {

class ListCommand : public BaseCommand {
public:
    explicit ListCommand(NeuronsConfig* config);
    ~ListCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    bool show_details_;
    bool show_running_;
    std::string filter_type_;

    NeuronsConfig* config_;

    void list_downloaded_models();
    void list_running_models();
    void print_usage_help();
    int64_t calculateDirectorySize(const std::filesystem::path& dir);
};

}
