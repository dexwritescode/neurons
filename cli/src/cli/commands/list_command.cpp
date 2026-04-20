#include "list_command.h"
#include <iostream>
#include <filesystem>
#include "cli/utils/format_utils.h"

namespace neurons::cli {

ListCommand::ListCommand(NeuronsConfig* config)
    : show_details_(false)
    , show_running_(false)
    , config_(config) {
}

void ListCommand::setup_command(CLI::App& app) {
    auto* list_cmd = app.add_subcommand("list", "List downloaded and running models");

    list_cmd->add_flag("-d,--details", show_details_, "Show detailed information");
    list_cmd->add_flag("-r,--running", show_running_, "Show only running models");
    list_cmd->add_option("-t,--type", filter_type_, "Filter by model type");

    list_cmd->callback([this]() {
        execute();
    });
}

int ListCommand::execute() {
    if (show_running_) {
        list_running_models();
    } else {
        list_downloaded_models();
    }
    return 0;
}

void ListCommand::list_downloaded_models() {
    std::filesystem::path dirPath = config_->modelsDirectory();

    if (!std::filesystem::exists(dirPath)) {
        std::cout << "No models directory found. Download some models first!" << std::endl;
        return;
    }

    // Get organization directories (e.g., "mlx-community", "Qwen", "TinyLlama")
    std::vector<std::string> orgDirs;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
            if (entry.is_directory()) {
                orgDirs.push_back(entry.path().filename().string());
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Error reading models directory: " << e.what() << std::endl;
        return;
    }

    if (orgDirs.empty()) {
        std::cout << "No models found in " << dirPath << std::endl;
        return;
    }

    TableFormatter table;

    if (show_details_) {
        table.set_headers({"Model ID", "Size", "Status", "Path"});
    } else {
        table.set_headers({"Model ID", "Size", "Status"});
    }

    bool foundModels = false;

    // Iterate through organization directories
    for (const std::string& orgName : orgDirs) {
        std::filesystem::path orgDir = dirPath / orgName;

        // Get model directories within the organization
        std::vector<std::string> modelDirs;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(orgDir)) {
                if (entry.is_directory()) {
                    modelDirs.push_back(entry.path().filename().string());
                }
            }
        } catch (const std::exception& e) {
            continue; // Skip this org directory if we can't read it
        }

        for (const std::string& modelName : modelDirs) {
            foundModels = true;
            std::filesystem::path modelDir = orgDir / modelName;
            std::string modelId = orgName + "/" + modelName;

            // Calculate total size recursively
            int64_t totalSize = calculateDirectorySize(modelDir);

            std::string sizeStr = utils::formatFileSize(totalSize);
            std::string status = "Downloaded";

            // Check if this matches the filter
            if (!filter_type_.empty()) {
                // For now, we don't have type information stored, so we'll skip filtering
                // In a real implementation, this would read metadata from the model directory
            }

            if (show_details_) {
                table.add_model_row(modelId, sizeStr, status, modelDir.string());
            } else {
                table.add_model_row(modelId, sizeStr, status);
            }
        }
    }

    if (!foundModels) {
        std::cout << "No models found in organization directories in " << dirPath << std::endl;
        return;
    }

    std::cout << "Downloaded Models:" << std::endl;
    table.print();
}

void ListCommand::list_running_models() {
    // Placeholder for future compute module integration
    TableFormatter table;
    table.set_headers({"Model ID", "Status", "Memory", "Compute Device"});

    // This will be implemented when the compute module is integrated
    std::cout << "Running Models:" << std::endl;
    std::cout << "No models currently running. (Compute module not yet integrated)" << std::endl;
}

int64_t ListCommand::calculateDirectorySize(const std::filesystem::path& dir) {
    int64_t totalSize = 0;

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                totalSize += entry.file_size();
            }
        }
    } catch (const std::exception& e) {
        // If we can't read the directory, return 0
        return 0;
    }

    return totalSize;
}


void ListCommand::print_usage_help() {
    std::cout << "Usage: neurons list [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  neurons list                    # List all downloaded models" << std::endl;
    std::cout << "  neurons list --details          # Show detailed information" << std::endl;
    std::cout << "  neurons list --running          # Show only running models" << std::endl;
    std::cout << "  neurons list --type text-gen    # Filter by model type" << std::endl;
}

}

