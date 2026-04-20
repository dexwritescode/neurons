#pragma once

#include "base_command.h"
#include "cli/config/neurons_config.h"
#include <string>

namespace neurons::cli {

class ConfigCommand : public BaseCommand {
public:
    explicit ConfigCommand(NeuronsConfig* config);
    ~ConfigCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    NeuronsConfig* config_;

    // Sub-action state
    bool do_show_ = false;
    std::string set_key_;
    std::string set_value_;

    int cmd_show();
    int cmd_set(const std::string& key, const std::string& value);
};

} // namespace neurons::cli
