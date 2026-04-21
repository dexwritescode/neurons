#include "neurons_service.h"

#include "compute/core/compute_backend.h"
#include "compute/model/language_model.h"
#include "models/registry/model_registry.h"

#include <grpcpp/grpcpp.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <numeric>
#include <sstream>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
static uint64_t system_memory_bytes() {
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
    return mem;
}
#else
static uint64_t system_memory_bytes() { return 0; }
#endif

namespace neurons_service {

namespace fs = std::filesystem;

// ── Constructor ──────────────────────────────────────────────────────────────

NeuronsServiceImpl::NeuronsServiceImpl(std::string models_dir,
                                       std::unique_ptr<compute::ComputeBackend> backend)
    : models_dir_(std::move(models_dir))
    , backend_(std::move(backend))
    , hf_client_(models::createHuggingFaceClientSync()) {
    hf_client_->client()->setDownloadDirectory(models_dir_);
}

// ── set_backend ──────────────────────────────────────────────────────────────

void NeuronsServiceImpl::set_backend(std::unique_ptr<compute::ComputeBackend> backend) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    backend_ = std::move(backend);
}

// ── set_hf_token ─────────────────────────────────────────────────────────────

void NeuronsServiceImpl::set_hf_token(const std::string& token) {
    hf_client_->client()->setAuthToken(token);
}

// ── load_model_internal ──────────────────────────────────────────────────────

bool NeuronsServiceImpl::load_model_internal(const std::string& path, std::string& error_out) {
    neurons::LoadModelRequest  req;
    neurons::LoadModelResponse resp;
    req.set_model_path(path);
    LoadModel(nullptr, &req, &resp);
    if (!resp.success()) { error_out = resp.error(); return false; }
    return true;
}

// ── LoadModel ────────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::LoadModel(grpc::ServerContext* /*ctx*/,
                                           const neurons::LoadModelRequest* req,
                                           neurons::LoadModelResponse* resp) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    const fs::path model_path(req->model_path());
    if (!fs::exists(model_path)) {
        resp->set_success(false);
        resp->set_error("Model path does not exist: " + req->model_path());
        return grpc::Status::OK;
    }

    // Backend is pre-created on the main thread at startup (MLX requires main thread init)
    if (!backend_) {
        resp->set_success(false);
        resp->set_error("No backend — service requires a backend passed at construction");
        return grpc::Status::OK;
    }

    // Tear down any existing model (backend stays alive)
    model_.reset();
    model_path_.clear();

    // Load model using the pre-created backend
    auto model_result = compute::LanguageModel::load(model_path, backend_.get());
    if (!model_result.has_value()) {
        resp->set_success(false);
        resp->set_error("Model load failed: " + model_result.error().message);
        return grpc::Status::OK;
    }

    model_     = std::move(*model_result);
    model_path_ = req->model_path();

    resp->set_success(true);
    resp->set_model_type(model_->model_type());
    resp->set_vocab_size(static_cast<uint32_t>(model_->config().vocab_size));
    resp->set_num_layers(static_cast<uint32_t>(model_->config().num_hidden_layers));
    resp->set_max_position_embeddings(static_cast<uint32_t>(model_->config().max_position_embeddings));
    resp->set_supports_tool_use(model_->supports_tool_use());
    return grpc::Status::OK;
}

// ── UnloadModel ──────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::UnloadModel(grpc::ServerContext* /*ctx*/,
                                              const neurons::UnloadModelRequest* /*req*/,
                                              neurons::UnloadModelResponse* resp) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    model_.reset();
    model_path_.clear();
    resp->set_success(true);
    return grpc::Status::OK;
}

// ── GetStatus ────────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::GetStatus(grpc::ServerContext* /*ctx*/,
                                            const neurons::StatusRequest* /*req*/,
                                            neurons::StatusResponse* resp) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    const bool loaded = (model_ != nullptr);
    resp->set_model_loaded(loaded);

    // Determine backend name
    std::string backend_name = "cpu";
    std::string gpu_name     = "CPU";
    if (backend_) {
        switch (backend_->type()) {
            case compute::BackendType::MLX:
                backend_name = "mlx";
                gpu_name     = "Apple Silicon";
                break;
            case compute::BackendType::SimdNeon:
                backend_name = "neon";
                gpu_name     = "ARM NEON";
                break;
            default: break;
        }
    }

    if (loaded) {
        resp->set_model_path(model_path_);
        resp->set_model_type(model_->model_type());
        resp->set_vocab_size(static_cast<uint32_t>(model_->config().vocab_size));
        resp->set_num_layers(static_cast<uint32_t>(model_->config().num_hidden_layers));
        resp->set_max_position_embeddings(static_cast<uint32_t>(model_->config().max_position_embeddings));
        resp->set_backend(backend_name);
        resp->set_supports_tool_use(model_->supports_tool_use());
    }

    // Always populate one GpuSlot (the single backend device on this node)
    auto* slot = resp->add_gpus();
    slot->set_gpu_id("0");
    slot->set_gpu_name(gpu_name);
    slot->set_vram_total_bytes(system_memory_bytes()); // unified on Apple Silicon
    slot->set_vram_used_bytes(0);                      // not measurable via MLX API
    if (loaded) {
        slot->set_loaded_model(model_path_);
        slot->set_model_type(model_->model_type());
        slot->set_tok_per_sec(last_tok_per_sec_);
    }

    resp->set_http_port(static_cast<uint32_t>(http_port_));

    return grpc::Status::OK;
}

// ── ListModels ───────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::ListModels(grpc::ServerContext* /*ctx*/,
                                             const neurons::ListModelsRequest* req,
                                             neurons::ListModelsResponse* resp) {
    const std::string dir = req->models_dir().empty() ? models_dir_ : req->models_dir();

    models::registry::ModelRegistry registry(dir);
    for (const auto& loc : registry.listModels()) {
        if (!loc.isValid()) continue;

        // Sum size of all model files
        uint64_t total = 0;
        for (const auto& f : loc.modelFiles) {
            std::error_code ec;
            total += fs::file_size(f, ec);
        }

        auto* info = resp->add_models();
        info->set_name(loc.modelName);
        info->set_path(loc.modelPath);
        info->set_size_bytes(total);
    }
    return grpc::Status::OK;
}

// ── Generate ─────────────────────────────────────────────────────────────────

// Helper: token-count a string using the model tokenizer (no special tokens).
static int count_tokens(const compute::SimpleBpeTokenizer& tok, const std::string& text) {
    return static_cast<int>(tok.encode(text, /*add_special_tokens=*/false).size());
}

// Trim `blocks` (oldest-first) so that their total token count fits within `budget`.
// Returns the start index into `blocks` (inclusive) from which blocks should be included.
static int trim_blocks_to_budget(const compute::SimpleBpeTokenizer& tok,
                                  const std::vector<std::string>& blocks,
                                  int budget) {
    // Count tokens per block (once).
    std::vector<int> costs;
    costs.reserve(blocks.size());
    for (const auto& b : blocks) costs.push_back(count_tokens(tok, b));

    // Greedily include blocks from newest to oldest until budget is exhausted.
    int used = 0;
    int start = static_cast<int>(blocks.size());
    for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
        if (used + costs[i] <= budget) {
            used += costs[i];
            start = i;
        } else break;
    }
    return start;
}

std::string NeuronsServiceImpl::build_prompt(const compute::LanguageModel& mdl,
                                              const neurons::GenerateRequest& req,
                                              int token_budget) const {
    const std::string& type = mdl.model_type();
    const auto& tok = mdl.tokenizer();
    const int max_chars = 6000;  // char-based fallback when token_budget == 0

    if (type == "llama" && mdl.tokenizer().find_token_id("<|start_header_id|>") != -1) {
        // Llama-3 chat template (identified by presence of header tokens in vocab):
        //   <|begin_of_text|>  — BOS is in the template; add_bos_token is null/false
        //   <|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
        //   <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
        //   <|start_header_id|>assistant<|end_header_id|>\n\n
        const std::string hdr_open  = "<|start_header_id|>";
        const std::string hdr_close = "<|end_header_id|>\n\n";
        const std::string eot       = "<|eot_id|>";

        const std::string sys_block =
            "<|begin_of_text|>" +
            hdr_open + "system" + hdr_close +
            "You are a helpful assistant." + eot + "\n";
        const std::string latest_block =
            hdr_open + "user" + hdr_close + req.prompt() + eot + "\n" +
            hdr_open + "assistant" + hdr_close;
        const int overhead = static_cast<int>(sys_block.size() + latest_block.size());

        std::vector<std::string> blocks;
        const int history_count = req.history_size();
        for (int i = 0; i + 1 < history_count; i += 2) {
            const auto& u = req.history(i);
            const auto& a = req.history(i + 1);
            if (u.role() == "user" && a.role() == "assistant") {
                blocks.push_back(
                    hdr_open + "user" + hdr_close + u.content() + eot + "\n" +
                    hdr_open + "assistant" + hdr_close + a.content() + eot + "\n");
            }
        }

        int start;
        if (token_budget > 0) {
            const int fixed_tokens = count_tokens(tok, sys_block + latest_block);
            start = trim_blocks_to_budget(tok, blocks, token_budget - fixed_tokens);
        } else {
            int budget = max_chars - overhead;
            start = static_cast<int>(blocks.size());
            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
                if (budget >= static_cast<int>(blocks[i].size())) {
                    budget -= static_cast<int>(blocks[i].size());
                    start = i;
                } else break;
            }
        }

        std::string result = sys_block;
        for (int i = start; i < static_cast<int>(blocks.size()); ++i) result += blocks[i];
        result += latest_block;
        return result;
    }

    if (type == "qwen2" || type == "qwen3") {
        // Qwen2/Qwen3 ChatML template:
        //   <|im_start|>system\n{system}<|im_end|>\n
        //   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>\n
        //   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
        const std::string im_start = "<|im_start|>";
        const std::string im_end   = "<|im_end|>";

        const std::string sys_block =
            im_start + "system\n" + "You are a helpful assistant." + im_end + "\n";
        const std::string latest_block =
            im_start + "user\n" + req.prompt() + im_end + "\n" +
            im_start + "assistant\n";
        const int overhead = static_cast<int>(sys_block.size() + latest_block.size());

        std::vector<std::string> blocks;
        const int history_count = req.history_size();
        for (int i = 0; i + 1 < history_count; i += 2) {
            const auto& u = req.history(i);
            const auto& a = req.history(i + 1);
            if (u.role() == "user" && a.role() == "assistant") {
                blocks.push_back(
                    im_start + "user\n" + u.content() + im_end + "\n" +
                    im_start + "assistant\n" + a.content() + im_end + "\n");
            }
        }

        int start;
        if (token_budget > 0) {
            const int fixed_tokens = count_tokens(tok, sys_block + latest_block);
            start = trim_blocks_to_budget(tok, blocks, token_budget - fixed_tokens);
        } else {
            int budget = max_chars - overhead;
            start = static_cast<int>(blocks.size());
            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
                if (budget >= static_cast<int>(blocks[i].size())) {
                    budget -= static_cast<int>(blocks[i].size());
                    start = i;
                } else break;
            }
        }

        std::string result = sys_block;
        for (int i = start; i < static_cast<int>(blocks.size()); ++i) result += blocks[i];
        result += latest_block;
        return result;
    }

    if (type == "gemma" || type == "gemma2" || type == "gemma3_text") {
        // Gemma chat template: <start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n
        // BOS is prepended by tokenizer. No system prompt in Gemma's standard template.
        std::vector<std::string> blocks;
        const int history_count = req.history_size();
        for (int i = 0; i + 1 < history_count; i += 2) {
            const auto& u = req.history(i);
            const auto& a = req.history(i + 1);
            if (u.role() == "user" && a.role() == "assistant") {
                blocks.push_back(
                    "<start_of_turn>user\n" + u.content() + "<end_of_turn>\n" +
                    "<start_of_turn>model\n" + a.content() + "<end_of_turn>\n");
            }
        }
        const std::string latest = "<start_of_turn>user\n" + req.prompt() +
                                   "<end_of_turn>\n<start_of_turn>model\n";

        int start;
        if (token_budget > 0) {
            const int fixed_tokens = count_tokens(tok, latest);
            start = trim_blocks_to_budget(tok, blocks, token_budget - fixed_tokens);
        } else {
            int budget = max_chars - static_cast<int>(latest.size());
            start = static_cast<int>(blocks.size());
            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
                if (budget >= static_cast<int>(blocks[i].size())) {
                    budget -= static_cast<int>(blocks[i].size());
                    start = i;
                } else break;
            }
        }

        std::string result;
        for (int i = start; i < static_cast<int>(blocks.size()); ++i) result += blocks[i];
        result += latest;
        return result;
    }

    if (type == "mistral") {
        // Mistral v0.3: <s>[INST] {user} [/INST]{assistant}</s>[INST] {user} [/INST]
        // BOS prepended by tokenizer (add_special_tokens=true), not included here.
        std::vector<std::string> blocks;
        const int history_count = req.history_size();
        for (int i = 0; i + 1 < history_count; i += 2) {
            const auto& u = req.history(i);
            const auto& a = req.history(i + 1);
            if (u.role() == "user" && a.role() == "assistant") {
                blocks.push_back("[INST] " + u.content() + " [/INST]" + a.content() + "</s>");
            }
        }
        const std::string latest = "[INST] " + req.prompt() + " [/INST]";

        int start;
        if (token_budget > 0) {
            const int fixed_tokens = count_tokens(tok, latest);
            start = trim_blocks_to_budget(tok, blocks, token_budget - fixed_tokens);
        } else {
            int budget = max_chars - static_cast<int>(latest.size());
            start = static_cast<int>(blocks.size());
            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
                if (budget >= static_cast<int>(blocks[i].size())) {
                    budget -= static_cast<int>(blocks[i].size());
                    start = i;
                } else break;
            }
        }

        std::string result;
        for (int i = start; i < static_cast<int>(blocks.size()); ++i) result += blocks[i];
        result += latest;
        return result;
    }

    // Default: TinyLlama / Llama-2 chat template
    const std::string system_block = "<|system|>\nYou are a helpful assistant.</s>\n";
    const std::string latest_block = "<|user|>\n" + req.prompt() + "</s>\n<|assistant|>\n";
    const int overhead = static_cast<int>(system_block.size() + latest_block.size());

    std::vector<std::string> blocks;
    const int history_count = req.history_size();
    for (int i = 0; i + 1 < history_count; i += 2) {
        const auto& u = req.history(i);
        const auto& a = req.history(i + 1);
        if (u.role() == "user" && a.role() == "assistant") {
            blocks.push_back("<|user|>\n" + u.content() + "</s>\n" +
                             "<|assistant|>\n" + a.content() + "</s>\n");
        }
    }

    int start;
    if (token_budget > 0) {
        const int fixed_tokens = count_tokens(tok, system_block + latest_block);
        start = trim_blocks_to_budget(tok, blocks, token_budget - fixed_tokens);
    } else {
        int budget = max_chars - overhead;
        start = static_cast<int>(blocks.size());
        for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
            if (budget >= static_cast<int>(blocks[i].size())) {
                budget -= static_cast<int>(blocks[i].size());
                start = i;
            } else break;
        }
    }

    std::string result = system_block;
    for (int i = start; i < static_cast<int>(blocks.size()); ++i) result += blocks[i];
    result += latest_block;
    return result;
}

grpc::Status NeuronsServiceImpl::Generate(grpc::ServerContext* ctx,
                                           const neurons::GenerateRequest* req,
                                           grpc::ServerWriter<neurons::GenerateResponse>* writer) {
    // Snapshot model pointer under lock — generation itself is lock-free
    compute::LanguageModel* mdl = nullptr;
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        mdl = model_.get();
    }

    if (!mdl) {
        neurons::GenerateResponse resp;
        resp.set_done(true);
        resp.set_error("No model loaded. Call LoadModel first.");
        writer->Write(resp);
        return grpc::Status::OK;
    }

    // Delegate to generate_internal — this also activates MCP tool use when
    // connected servers are available.
    const std::atomic<bool> not_cancelled{false};
    uint32_t prompt_token_count = 0;
    uint32_t gen_token_count    = 0;
    std::string gen_error;
    const auto gen_start = std::chrono::steady_clock::now();

    generate_internal(*req, not_cancelled,
        [&](const std::string& delta) -> bool {
            if (ctx->IsCancelled()) return false;
            neurons::GenerateResponse resp;
            resp.set_token(delta);
            resp.set_done(false);
            writer->Write(resp);
            return true;
        },
        gen_error,
        &prompt_token_count,
        &gen_token_count);

    // Record tok/s for GetStatus
    if (gen_token_count > 0) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gen_start).count();
        std::lock_guard<std::mutex> lk(model_mutex_);
        last_tok_per_sec_ = elapsed > 0.0
            ? static_cast<float>(gen_token_count / elapsed) : 0.0f;
    }

    // Final message with stats
    neurons::GenerateResponse final_resp;
    final_resp.set_done(true);
    final_resp.set_prompt_tokens(prompt_token_count);
    final_resp.set_gen_tokens(gen_token_count);
    if (!gen_error.empty()) {
        final_resp.set_error(gen_error);
    }
    writer->Write(final_resp);

    return grpc::Status::OK;
}

// ── SearchModels ──────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::SearchModels(grpc::ServerContext* /*ctx*/,
                                               const neurons::SearchModelsRequest* req,
                                               neurons::SearchModelsResponse* resp) {
    models::SearchQuery q;
    q.search = req->query();
    q.sort   = req->sort().empty() ? "downloads" : req->sort();
    q.limit  = req->limit() > 0 ? req->limit() : 30;
    q.full   = true;
    if (!req->author().empty()) q.author = req->author();
    for (int i = 0; i < req->pipeline_tags_size(); ++i) {
        q.pipelineTags.push_back(req->pipeline_tags(i));
    }

    std::promise<std::pair<std::vector<models::ModelInfo>, std::string>> promise;
    auto future = promise.get_future();

    hf_client_->client()->setSearchCallback(
        [&promise](const std::vector<models::ModelInfo>& models,
                   const std::string& /*nextPage*/,
                   const std::string& error) {
            promise.set_value({models, error});
        });
    hf_client_->client()->searchModels(q);

    auto [results, error] = future.get();
    if (!error.empty()) {
        resp->set_error(error);
        return grpc::Status::OK;
    }
    for (const auto& m : results) {
        auto* r = resp->add_results();
        r->set_model_id(m.id);
        r->set_downloads(static_cast<int64_t>(m.downloads));
        r->set_gated(m.gated);
    }
    return grpc::Status::OK;
}

// ── GetModelInfo ──────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::GetModelInfo(grpc::ServerContext* /*ctx*/,
                                               const neurons::GetModelInfoRequest* req,
                                               neurons::GetModelInfoResponse* resp) {
    const std::string& model_id = req->model_id();

    // Fetch file list
    auto files_result = hf_client_->getModelFilesBlocking(model_id);
    if (!files_result.success) {
        resp->set_error(files_result.error);
        return grpc::Status::OK;
    }

    int64_t total = 0;
    for (const auto& f : files_result.files) {
        auto* fi = resp->add_files();
        fi->set_filename(f.filename);
        fi->set_size_bytes(static_cast<int64_t>(f.sizeBytes));
        total += static_cast<int64_t>(f.sizeBytes);
    }
    resp->set_model_id(model_id);
    resp->set_total_size_bytes(total);

    // Fetch README via HTTP
    auto http = models::http::createCurlHttpClient();
    models::http::HttpRequest http_req;
    http_req.url = "https://huggingface.co/" + model_id + "/resolve/main/README.md";
    http_req.timeoutSeconds = 15;
    auto http_resp = http->requestSync(http_req);
    if (http_resp.statusCode == 200) {
        resp->set_readme(http_resp.data);
    }
    // README not found is non-fatal — leave field empty

    return grpc::Status::OK;
}

// ── DownloadModel ─────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::DownloadModel(grpc::ServerContext* ctx,
                                                const neurons::DownloadModelRequest* req,
                                                grpc::ServerWriter<neurons::DownloadProgressResponse>* writer) {
    const std::string& model_id = req->model_id();

    // Need full ModelInfo (with files) to start download
    auto details = hf_client_->getModelDetailsBlocking(model_id);
    if (!details.success) {
        neurons::DownloadProgressResponse err;
        err.set_error(details.error);
        err.set_done(true);
        writer->Write(err);
        return grpc::Status::OK;
    }

    std::mutex write_mutex;
    std::string active_download_id;

    hf_client_->setProgressCallback([&](const models::DownloadProgress& p) {
        if (ctx->IsCancelled()) return;

        std::lock_guard<std::mutex> lk(write_mutex);
        active_download_id = p.downloadId;

        neurons::DownloadProgressResponse frame;
        frame.set_download_id(p.downloadId);
        frame.set_current_file(p.filename);
        frame.set_bytes_downloaded(p.bytesDownloaded);
        frame.set_total_bytes(p.totalBytes);

        const bool done      = (p.state == models::DownloadState::COMPLETED);
        const bool cancelled = (p.state == models::DownloadState::CANCELLED);
        const bool failed    = (p.state == models::DownloadState::FAILED);
        frame.set_done(done);
        frame.set_cancelled(cancelled);
        if (failed) frame.set_error(p.errorMessage);

        writer->Write(frame);
    });

    auto result = hf_client_->downloadModelBlocking(details.model);
    // If client cancelled via CancelDownload(), the blocking call returns here.
    if (!result.success && !result.error.empty()) {
        // Only write a final error frame if we haven't already sent done/cancelled.
        std::lock_guard<std::mutex> lk(write_mutex);
        neurons::DownloadProgressResponse err;
        err.set_error(result.error);
        err.set_done(true);
        writer->Write(err);
    }

    return grpc::Status::OK;
}

// ── CancelDownload ────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::CancelDownload(grpc::ServerContext* /*ctx*/,
                                                  const neurons::CancelDownloadRequest* req,
                                                  neurons::CancelDownloadResponse* resp) {
    hf_client_->client()->cancelDownload(req->download_id());
    resp->set_success(true);
    return grpc::Status::OK;
}

// ── SetHfToken ────────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::SetHfToken(grpc::ServerContext* /*ctx*/,
                                              const neurons::SetHfTokenRequest* req,
                                              neurons::SetHfTokenResponse* /*resp*/) {
    set_hf_token(req->token());
    return grpc::Status::OK;
}

// ── generate_internal ─────────────────────────────────────────────────────────

bool NeuronsServiceImpl::generate_internal(const neurons::GenerateRequest& req,
                                            const std::atomic<bool>&        cancelled,
                                            GenerateTokenCb                 cb,
                                            std::string&                    error_out,
                                            uint32_t*                       prompt_tokens_out,
                                            uint32_t*                       gen_tokens_out,
                                            ToolCallCb                      tool_cb) {
    compute::LanguageModel* mdl = nullptr;
    {
        std::lock_guard<std::mutex> lock(model_mutex_);
        mdl = model_.get();
    }
    if (!mdl) { error_out = "No model loaded"; return false; }

    const std::string session_id = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());

    const int n_max = (req.has_params() && req.params().max_tokens() > 0)
                      ? req.params().max_tokens() : 200;
    const int ctx_win    = req.has_params() ? req.params().context_window() : 0;
    const int tok_budget = (ctx_win > 0) ? ctx_win - n_max : 0;

    const std::string base_prompt = build_prompt(*mdl, req, tok_budget);
    const auto& tok = mdl->tokenizer();

    compute::SamplingParams params;
    if (req.has_params()) {
        const auto& p = req.params();
        params.temperature = p.temperature() > 0.0f ? p.temperature() : 0.7f;
        params.top_p       = p.top_p()       > 0.0f ? p.top_p()       : 0.9f;
        params.top_k       = p.top_k()       > 0    ? p.top_k()       : 40;
        params.rep_penalty = p.rep_penalty()  > 0.0f ? p.rep_penalty() : 1.1f;
    }

    // Encode the initial prompt with BOS; tool injections are appended without BOS.
    std::vector<int> all_tokens = tok.encode(base_prompt, /*add_special_tokens=*/true);
    if (prompt_tokens_out) *prompt_tokens_out = static_cast<uint32_t>(all_tokens.size());

    // If no explicit callback provided, use the McpManager if tools are available.
    if (!tool_cb && mcp_manager_.has_active_tools() && mdl->supports_tool_use()) {
        tool_cb = mcp_manager_.make_tool_call_cb(session_id);
    }
    const bool can_use_tools = (tool_cb != nullptr) && mdl->supports_tool_use();
    static constexpr int kMaxToolTurns = 5;
    uint32_t total_gen = 0;

    for (int turn = 0; turn <= kMaxToolTurns; ++turn) {
        std::vector<int> gen_so_far;
        std::string decoded_so_far;
        std::string accumulated;
        std::optional<compute::LanguageModel::ToolCall> pending_tool;

        auto result = mdl->generate(all_tokens, static_cast<size_t>(n_max), params,
            [&](int tok_id) -> bool {
                if (cancelled.load(std::memory_order_relaxed)) return false;
                if (mdl->config().is_eos_token(tok_id)) return false;

                gen_so_far.push_back(tok_id);
                const std::string new_decoded = tok.decode(gen_so_far);
                const std::string delta = new_decoded.substr(decoded_so_far.size());
                decoded_so_far = new_decoded;
                accumulated += delta;

                if (can_use_tools) {
                    auto tc = mdl->detect_tool_call(accumulated);
                    if (tc.has_value()) {
                        pending_tool = std::move(tc);
                        return false;
                    }
                }
                return cb(delta);
            });

        total_gen += static_cast<uint32_t>(gen_so_far.size());

        if (!result.has_value()) {
            error_out = "Generation failed: " + result.error().message;
            return false;
        }

        if (!pending_tool || turn == kMaxToolTurns) break;

        // Invoke the tool callback — nullopt means denied.
        auto tool_result = tool_cb(*pending_tool);
        const std::string result_json = tool_result.value_or(
            R"({"error":"Tool call denied"})");
        const std::string injection = mdl->format_tool_result(
            pending_tool->name, result_json);

        // Extend the token list with what was generated + the injected result.
        // No BOS on continuations.
        all_tokens.insert(all_tokens.end(), gen_so_far.begin(), gen_so_far.end());
        const std::vector<int> inj_tokens = tok.encode(injection, /*add_special_tokens=*/false);
        all_tokens.insert(all_tokens.end(), inj_tokens.begin(), inj_tokens.end());
    }

    if (gen_tokens_out) *gen_tokens_out = total_gen;
    return true;
}

// ── download_internal ─────────────────────────────────────────────────────────

bool NeuronsServiceImpl::download_internal(const std::string& repo_id,
                                            DownloadProgressCb cb,
                                            std::string&       error_out) {
    auto details = hf_client_->getModelDetailsBlocking(repo_id);
    if (!details.success) { error_out = details.error; return false; }

    // getModelDetails gives metadata only; file listing is a separate API call.
    auto filesResult = hf_client_->getModelFilesBlocking(repo_id);
    if (!filesResult.success) { error_out = filesResult.error; return false; }
    details.model.files = filesResult.files;

    // Use downloadModelSync so that libcurl progress callbacks fire on the calling
    // thread.  download_internal is invoked from the dart:ffi background isolate;
    // NativeCallable::isolateLocal() requires callbacks on that same OS thread.
    hf_client_->setProgressCallback([&](const models::DownloadProgress& p) {
        if (p.state == models::DownloadState::FAILED) {
            error_out = p.errorMessage;
            return;
        }
        if (p.state == models::DownloadState::COMPLETED ||
            p.state == models::DownloadState::CANCELLED) {
            return;
        }
        cb(p.bytesDownloaded, p.totalBytes, 0.0, p.filename);
    });

    auto result = hf_client_->downloadModelSync(details.model);
    if (!result.success && !result.error.empty()) {
        error_out = result.error;
        return false;
    }
    return true;
}

// ── StreamLogs ───────────────────────────────────────────────────────────────

grpc::Status NeuronsServiceImpl::StreamLogs(grpc::ServerContext* ctx,
                                             const neurons::StreamLogsRequest* req,
                                             grpc::ServerWriter<neurons::LogEntry>* writer) {
    // Determine minimum level (default INFO)
    const std::string min = req->min_level().empty() ? "INFO" : req->min_level();
    auto levelNum = [](const std::string& l) {
        if (l == "ERROR") return 2;
        if (l == "WARN")  return 1;
        return 0; // INFO and anything else
    };
    const int minNum = levelNum(min);

    // First flush recent history
    for (const auto& e : Logger::global().recent(200)) {
        if (levelNum(e.level) < minNum) continue;
        neurons::LogEntry entry;
        entry.set_timestamp_ms(e.timestamp_ms);
        entry.set_level(e.level);
        entry.set_message(e.message);
        if (!writer->Write(entry)) return grpc::Status::OK;
    }

    // Then stream live entries
    std::mutex sub_mutex;
    std::deque<neurons::LogEntry> pending;
    std::condition_variable sub_cv;

    auto sub_id = Logger::global().subscribe([&](const LogEntry& e) {
        if (levelNum(e.level) < minNum) return;
        neurons::LogEntry entry;
        entry.set_timestamp_ms(e.timestamp_ms);
        entry.set_level(e.level);
        entry.set_message(e.message);
        {
            std::lock_guard lock(sub_mutex);
            pending.push_back(entry);
        }
        sub_cv.notify_one();
    });

    while (!ctx->IsCancelled()) {
        std::unique_lock<std::mutex> lock(sub_mutex);
        sub_cv.wait_for(lock, std::chrono::milliseconds(500),
                        [&] { return !pending.empty() || ctx->IsCancelled(); });
        while (!pending.empty()) {
            if (!writer->Write(pending.front())) {
                lock.unlock();
                Logger::global().unsubscribe(sub_id);
                return grpc::Status::OK;
            }
            pending.pop_front();
        }
    }

    Logger::global().unsubscribe(sub_id);
    return grpc::Status::OK;
}

} // namespace neurons_service
