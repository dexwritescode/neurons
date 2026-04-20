#pragma once

#include <CLI/CLI.hpp>
#include <string>
#include <vector>

namespace neurons::cli {

class BaseCommand {
public:
    virtual ~BaseCommand() = default;

    virtual void setup_command(CLI::App& app) = 0;
    virtual int execute() = 0;

protected:
    std::string command_name_;
    std::string description_;
};

}