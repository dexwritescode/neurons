#include "cli_app.h"
#include <iostream>

namespace neurons::cli {

CliApp::CliApp()
    : app_{"Neurons CLI", "Command line interface for Neurons"} {
    app_.require_subcommand(1);
}

void CliApp::setup() {
    for (auto& command : commands_) {
        command->setup_command(app_);
    }
}

int CliApp::run(int argc, char** argv) {
    try {
        app_.parse(argc, argv);
        return 0;
    } catch (const CLI::ParseError& e) {
        return app_.exit(e);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

void CliApp::add_command(std::unique_ptr<BaseCommand> command) {
    commands_.push_back(std::move(command));
}

}
