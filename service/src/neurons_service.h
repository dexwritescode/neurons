#pragma once

#include "neurons.grpc.pb.h"
#include "logger.h"
#include "compute/core/compute_backend.h"
#include "compute/model/language_model.h"
#include "models/api/huggingface_client.h"

#include <grpcpp/grpcpp.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace neurons_service {

class NeuronsServiceImpl final : public neurons::NeuronsInference::Service {
public:
    // backend must be created and initialized on the main thread before construction.
    // The service takes ownership and uses it for all model loads.
    NeuronsServiceImpl(std::string models_dir,
                       std::unique_ptr<compute::ComputeBackend> backend);

    grpc::Status LoadModel(grpc::ServerContext* ctx,
                           const neurons::LoadModelRequest* req,
                           neurons::LoadModelResponse* resp) override;

    grpc::Status UnloadModel(grpc::ServerContext* ctx,
                             const neurons::UnloadModelRequest* req,
                             neurons::UnloadModelResponse* resp) override;

    grpc::Status Generate(grpc::ServerContext* ctx,
                          const neurons::GenerateRequest* req,
                          grpc::ServerWriter<neurons::GenerateResponse>* writer) override;

    grpc::Status GetStatus(grpc::ServerContext* ctx,
                           const neurons::StatusRequest* req,
                           neurons::StatusResponse* resp) override;

    grpc::Status ListModels(grpc::ServerContext* ctx,
                            const neurons::ListModelsRequest* req,
                            neurons::ListModelsResponse* resp) override;

    grpc::Status SearchModels(grpc::ServerContext* ctx,
                              const neurons::SearchModelsRequest* req,
                              neurons::SearchModelsResponse* resp) override;

    grpc::Status GetModelInfo(grpc::ServerContext* ctx,
                              const neurons::GetModelInfoRequest* req,
                              neurons::GetModelInfoResponse* resp) override;

    grpc::Status DownloadModel(grpc::ServerContext* ctx,
                               const neurons::DownloadModelRequest* req,
                               grpc::ServerWriter<neurons::DownloadProgressResponse>* writer) override;

    grpc::Status CancelDownload(grpc::ServerContext* ctx,
                                const neurons::CancelDownloadRequest* req,
                                neurons::CancelDownloadResponse* resp) override;

    grpc::Status SetHfToken(grpc::ServerContext* ctx,
                            const neurons::SetHfTokenRequest* req,
                            neurons::SetHfTokenResponse* resp) override;

    grpc::Status StreamLogs(grpc::ServerContext* ctx,
                            const neurons::StreamLogsRequest* req,
                            grpc::ServerWriter<neurons::LogEntry>* writer) override;

    // Load a model directly without going through the gRPC call path.
    // Must be called from the main thread (MLX requires main-thread model loading).
    bool load_model_internal(const std::string& path, std::string& error_out);

    // Set the backend after construction. Must be called from the main thread
    // before any LoadModel calls, and before server->Wait().
    void set_backend(std::unique_ptr<compute::ComputeBackend> backend);

    // Set the HuggingFace Bearer token used for gated model downloads/searches.
    // Pass an empty string to clear. Thread-safe.
    void set_hf_token(const std::string& token);

    // ── Internal methods for FFI / non-gRPC callers ──────────────────────────

    // Callback receives decoded token text. Return false to stop generation.
    using GenerateTokenCb = std::function<bool(const std::string& token)>;

    // Run inference without going through gRPC. Builds the model-specific chat
    // prompt from req (same logic as the gRPC Generate RPC). Blocks until done
    // or cancelled.  cancelled — set from another thread to abort.
    // Returns true on success.
    bool generate_internal(const neurons::GenerateRequest& req,
                           const std::atomic<bool>&        cancelled,
                           GenerateTokenCb                 cb,
                           std::string&                    error_out,
                           uint32_t*                       prompt_tokens_out = nullptr,
                           uint32_t*                       gen_tokens_out    = nullptr);

    // Callback receives (bytes_done, bytes_total, speed_bps, current_file).
    // Return false to cancel.
    using DownloadProgressCb =
        std::function<bool(int64_t, int64_t, double, const std::string&)>;

    // Download a HuggingFace model without going through gRPC. Blocks until done.
    bool download_internal(const std::string&  repo_id,
                           DownloadProgressCb  cb,
                           std::string&        error_out);

    // Called by the server binary to report the HTTP port for GetStatus.
    void set_http_port(int port) { http_port_ = port; }

private:
    std::string models_dir_;
    int         http_port_{0};

    // HuggingFace client for model search and download
    std::unique_ptr<models::HuggingFaceClientSync> hf_client_;

    // Model state — guarded by mutex (one model loaded at a time)
    mutable std::mutex model_mutex_;
    std::unique_ptr<compute::ComputeBackend> backend_;
    std::unique_ptr<compute::LanguageModel>  model_;
    std::string                              model_path_;
    float                                    last_tok_per_sec_{0.0f};

    // Build the formatted prompt from history + new user message.
    // token_budget > 0: trim history so prompt fits within that many tokens
    // (caller computes as context_window - max_tokens). 0 = char-based fallback.
    std::string build_prompt(const compute::LanguageModel& model,
                             const neurons::GenerateRequest& req,
                             int token_budget = 0) const;
};

} // namespace neurons_service
