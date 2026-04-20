#include "download_command.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <thread>
#include <chrono>
#include <indicators/cursor_control.hpp>

namespace neurons::cli {

DownloadCommand::DownloadCommand(NeuronsConfig* config)
    : force_download_(false)
    , list_files_(false)
    , config_(config)
    , exit_code_(0)
    , operation_complete_(false)
    , have_model_details_(false)
    , have_model_files_(false)
    , total_model_bytes_(0)
    , total_downloaded_bytes_(0)
    , last_total_downloaded_(-1) {

    // Create reactive HuggingFace client with CLI wrapper
    hf_client_sync_ = models::createHuggingFaceClientSync();

    progress_bar_ = std::make_unique<ProgressBar>("Downloading");

    // Set progress callback for the sync wrapper
    hf_client_sync_->setProgressCallback([this](const models::DownloadProgress& progress) {
        on_progress_updated(progress);
    });

    // Set download directory from config
    hf_client_sync_->client()->setDownloadDirectory(config_->modelsDirectory().string());
}

void DownloadCommand::setup_command(CLI::App& app) {
    auto* download_cmd = app.add_subcommand("download", "Download models from HuggingFace");

    download_cmd->add_option("model", model_id_, "Model ID to download (e.g., microsoft/DialoGPT-small)")
        ->required();

    download_cmd->add_option("-o,--output", output_dir_, "Output directory for downloaded models");

    download_cmd->add_flag("-f,--force", force_download_, "Force re-download even if model exists");

    download_cmd->add_flag("-l,--list-files", list_files_, "List model files without downloading");

    download_cmd->callback([this]() {
        exit_code_ = execute();
    });
}

int DownloadCommand::execute() {
    if (model_id_.empty()) {
        std::cerr << "Error: Model ID is required" << std::endl;
        print_usage_help();
        return 1;
    }

    // Set custom output directory if provided
    if (!output_dir_.empty()) {
        try {
            std::filesystem::create_directories(output_dir_);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return 1;
        }
        hf_client_sync_->client()->setDownloadDirectory(output_dir_);
    }

    // Check if model already exists
    std::string modelPath = hf_client_sync_->client()->getModelPath(model_id_);
    if (std::filesystem::exists(modelPath) && !force_download_) {
        std::cout << "Model already exists at: " << modelPath << std::endl;
        std::cout << "Use --force to re-download" << std::endl;
        return 0;
    }

    // Reset state for new download
    total_model_bytes_ = 0;
    total_downloaded_bytes_ = 0;
    last_total_downloaded_ = -1;
    file_progress_.clear();
    file_totals_.clear();

    auto detailsResult = hf_client_sync_->getModelDetailsBlocking(model_id_);
    if (!detailsResult.success) {
        std::cerr << "Failed to fetch model details: " << detailsResult.error << std::endl;
        return 1;
    }

    auto filesResult = hf_client_sync_->getModelFilesBlocking(model_id_);
    if (!filesResult.success) {
        std::cerr << "Failed to fetch model files: " << filesResult.error << std::endl;
        return 1;
    }

    // Combine the results
    pending_model_info_ = detailsResult.model;
    pending_model_info_.files = filesResult.files;

    // Process the complete model info
    process_complete_model_info();

    return exit_code_;
}

void DownloadCommand::on_download_started(const std::string& downloadId, const std::string& modelId) {
    (void)downloadId; // Unused parameter
    std::cout << "Downloading: " << modelId << std::endl;
    progress_bar_->start();
}

void DownloadCommand::on_progress_updated(const models::DownloadProgress& progress) {
    // If filename is empty, this is overall model progress from calculateAndEmitProgress
    if (progress.filename.empty()) {
        // Use overall progress directly, but only if it's different from last update
        if (progress.bytesDownloaded != last_total_downloaded_) {
            progress_bar_->update_progress(progress.bytesDownloaded, progress.totalBytes);
            last_total_downloaded_ = progress.bytesDownloaded;
        }
    } else {
        // Individual file progress - use real file sizes from download
        file_progress_[progress.filename] = progress.bytesDownloaded;

        // Update file total sizes dynamically from actual downloads
        if (progress.totalBytes > 0) {
            file_totals_[progress.filename] = progress.totalBytes;
        }

        // Calculate total downloaded and total size from real download data
        int64_t total_downloaded = 0;

        for (const auto& [filename, bytes] : file_progress_) {
            total_downloaded += bytes;
        }

        // Show overall model progress using real download totals, but only if changed
        if (total_downloaded != last_total_downloaded_) {
            progress_bar_->update_progress(total_downloaded, total_model_bytes_);
            last_total_downloaded_ = total_downloaded;
        }
    }
}

void DownloadCommand::on_download_completed(const std::string& downloadId, const std::string& localPath, int64_t totalBytes) {
    (void)downloadId; // Unused parameter
    (void)totalBytes; // Unused parameter
    // Just restore cursor and add newline, don't call finish() which seems to cause extra output
    if (progress_bar_) {
        indicators::show_console_cursor(true);
    }

    std::cout << "Done! Model saved to: " << localPath << std::endl;
    exit_code_ = 0;
    operation_complete_ = true;
}

void DownloadCommand::on_download_failed(const std::string& downloadId, const std::string& error) {
    (void)downloadId; // Unused parameter
    progress_bar_->finish();
    std::cerr << "Download failed: " << error << std::endl;
    exit_code_ = 1;
    operation_complete_ = true;
}

void DownloadCommand::on_model_details_ready(const models::ModelInfo& model) {
    pending_model_info_ = model;
    have_model_details_ = true;

    // Check if we have both pieces
    if (have_model_files_) {
        process_complete_model_info();
    }
}

void DownloadCommand::on_model_files_ready(const std::string& modelId, const std::vector<models::FileInfo>& files) {
    (void)modelId; // Unused parameter
    pending_model_info_.files = files;
    have_model_files_ = true;

    // Check if we have both pieces
    if (have_model_details_) {
        process_complete_model_info();
    }
}

void DownloadCommand::on_hf_error_occurred(const std::string& error, const std::string& endpoint) {
    std::cerr << "HuggingFace API error [" << endpoint << "]: " << error << std::endl;
    exit_code_ = 1;
    operation_complete_ = true;
}

void DownloadCommand::process_complete_model_info() {
    // Calculate total model size from all files
    total_model_bytes_ = 0;
    for (const auto& file : pending_model_info_.files) {
        total_model_bytes_ += file.sizeBytes;
    }

    if (list_files_) {
        std::cout << "Model files would be downloaded:" << std::endl;
        for (const auto& file : pending_model_info_.files) {
            std::cout << "  - " << file.filename;
            if (file.sizeBytes > 0) {
                const double sizeGB = static_cast<double>(file.sizeBytes) / (1024.0 * 1024.0 * 1024.0);
                std::cout << " (" << std::fixed << std::setprecision(2) << sizeGB << " GB)";
            }
            std::cout << std::endl;
        }

        // Show total model size
        if (total_model_bytes_ > 0) {
            const double totalSizeGB = static_cast<double>(total_model_bytes_) / (1024.0 * 1024.0 * 1024.0);
            std::cout << "Total model size: " << std::fixed << std::setprecision(2) << totalSizeGB << " GB" << std::endl;
        }

        exit_code_ = 0;
        return;
    }

    // Initialize progress tracking
    total_downloaded_bytes_ = 0;
    file_progress_.clear();

    // Start progress bar manually since we're using the blocking method
    std::cout << "Downloading: " << pending_model_info_.id << std::endl;
    progress_bar_->start();

    // Use the sync wrapper's blocking download method which properly handles progress callbacks
    auto result = hf_client_sync_->downloadModelBlocking(pending_model_info_);

    // Restore cursor when done
    if (progress_bar_) {
        indicators::show_console_cursor(true);
    }

    if (!result.success) {
        std::cerr << "Download failed: " << result.error << std::endl;
        exit_code_ = 1;
    } else {
        std::cout << "Done! Model saved to: " << result.localPath << std::endl;
        exit_code_ = 0;
    }
}


void DownloadCommand::print_usage_help() {
    std::cout << "Usage: neurons download <model_id> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  neurons download microsoft/DialoGPT-small" << std::endl;
    std::cout << "  neurons download microsoft/DialoGPT-small -o /path/to/models" << std::endl;
    std::cout << "  neurons download microsoft/DialoGPT-small --list-files" << std::endl;
}

}

