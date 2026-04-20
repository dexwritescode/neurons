#pragma once

#include <memory>
#include <string>
#include <map>
#include "base_command.h"
#include "models/core/model_info.h"
#include "models/api/huggingface_client.h"
#include "cli/utils/progress_bar.h"
#include "cli/config/neurons_config.h"

namespace neurons::cli {

class DownloadCommand : public BaseCommand {
public:
    explicit DownloadCommand(NeuronsConfig* config);
    ~DownloadCommand() override = default;

    void setup_command(CLI::App& app) override;
    int execute() override;

private:
    // Callback methods (replacing Qt slots)
    void on_download_started(const std::string& downloadId, const std::string& modelId);
    void on_progress_updated(const models::DownloadProgress& progress);
    void on_download_completed(const std::string& downloadId, const std::string& localPath, int64_t totalBytes);
    void on_download_failed(const std::string& downloadId, const std::string& error);
    void on_model_details_ready(const models::ModelInfo& model);
    void on_model_files_ready(const std::string& modelId, const std::vector<models::FileInfo>& files);
    void on_hf_error_occurred(const std::string& error, const std::string& endpoint);

    std::string model_id_;
    std::string output_dir_;
    bool force_download_;
    bool list_files_;

    NeuronsConfig* config_;
    std::unique_ptr<models::HuggingFaceClientSync> hf_client_sync_;
    std::unique_ptr<ProgressBar> progress_bar_;

    // Simple state tracking (no need for atomic since CLI is single-threaded)
    int exit_code_;
    bool operation_complete_;

    // State for combining model details and files
    models::ModelInfo pending_model_info_;
    bool have_model_details_;
    bool have_model_files_;

    // Track overall download progress across all files
    uint64_t total_model_bytes_;
    uint64_t total_downloaded_bytes_;
    int64_t last_total_downloaded_;  // Track last update to prevent duplicate progress bar displays
    std::map<std::string, uint64_t> file_progress_; // filename -> bytes downloaded
    std::map<std::string, uint64_t> file_totals_;   // filename -> total bytes (from actual downloads)

    void print_usage_help();
    void process_complete_model_info();
};

}
