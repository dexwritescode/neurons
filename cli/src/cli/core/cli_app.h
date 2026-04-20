#pragma once

#include <CLI/CLI.hpp>
#include <memory>
#include <vector>
#include "cli/commands/base_command.h"

namespace neurons::cli {

class CliApp {
public:
    CliApp();
    ~CliApp() = default;

    void setup();
    int run(int argc, char** argv);

    void add_command(std::unique_ptr<BaseCommand> command);

private:
    CLI::App app_;
    std::vector<std::unique_ptr<BaseCommand>> commands_;
};

}