#include "neurons_ffi.h"
#include "neurons_service.h"
#include "compute/core/compute_backend.h"

#include <google/protobuf/util/json_util.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

// ── NeuronsCore ───────────────────────────────────────────────────────────────

struct NeuronsCore {
    std::string models_dir;
    std::unique_ptr<neurons_service::NeuronsServiceImpl> service;
    std::atomic<bool> cancel_generate{false};
    std::atomic<bool> cancel_download{false};
};

// ── Helpers ───────────────────────────────────────────────────────────────────

static void write_err(const std::string& msg, char* buf, int len) {
    if (!buf || len <= 0) return;
    std::strncpy(buf, msg.c_str(), static_cast<std::size_t>(len - 1));
    buf[len - 1] = '\0';
}

static char* heap_str(const std::string& s) {
    char* p = new char[s.size() + 1];
    std::memcpy(p, s.data(), s.size() + 1);
    return p;
}

// Simple JSON serialization for proto messages via protobuf's JSON util.
static char* proto_to_json(const google::protobuf::Message& msg) {
    std::string out;
    google::protobuf::util::JsonPrintOptions opts;
    opts.always_print_fields_with_no_presence = true;
    auto status = google::protobuf::util::MessageToJsonString(msg, &out, opts);
    if (!status.ok()) return heap_str("{}");
    return heap_str(out);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

NeuronsCore* neurons_create(const char* models_dir) {
    auto* h = new (std::nothrow) NeuronsCore();
    if (!h) return nullptr;
    h->models_dir = models_dir ? models_dir : "";
    // Service is constructed without a backend; backend is set after init.
    h->service = std::make_unique<neurons_service::NeuronsServiceImpl>(
        h->models_dir, nullptr);
    return h;
}

void neurons_destroy(NeuronsCore* h) {
    delete h;
}

int neurons_init_backend(NeuronsCore* h, char* err, int err_len) {
    if (!h) { write_err("null handle", err, err_len); return -1; }

    auto result = compute::BackendFactory::create(compute::BackendType::MLX);
    if (!result.has_value()) {
        write_err("Backend create failed: " + result.error().message, err, err_len);
        return -1;
    }
    auto init = (*result)->initialize();
    if (!init.has_value()) {
        write_err("Backend init failed: " + init.error().message, err, err_len);
        return -1;
    }
    // Transfer ownership of the backend into the service (same pattern as main.cpp).
    h->service->set_backend(std::move(*result));
    return 0;
}

// ── HuggingFace auth ──────────────────────────────────────────────────────────

void neurons_set_hf_token(NeuronsCore* h, const char* token) {
    if (!h) return;
    h->service->set_hf_token(token ? token : "");
}

// ── Model management ──────────────────────────────────────────────────────────

int neurons_load_model(NeuronsCore* h, const char* path, char* err, int err_len) {
    if (!h || !path) { write_err("null argument", err, err_len); return -1; }
    std::string error;
    if (!h->service->load_model_internal(path, error)) {
        write_err(error, err, err_len);
        return -1;
    }
    return 0;
}

void neurons_unload_model(NeuronsCore* h) {
    if (!h) return;
    neurons::UnloadModelRequest  req;
    neurons::UnloadModelResponse resp;
    h->service->UnloadModel(nullptr, &req, &resp);
}

int neurons_delete_model(NeuronsCore* h, const char* path, char* err, int err_len) {
    if (!h || !path) { write_err("null argument", err, err_len); return -1; }

    neurons::StatusRequest  sreq;
    neurons::StatusResponse sresp;
    h->service->GetStatus(nullptr, &sreq, &sresp);
    if (sresp.model_loaded() && sresp.model_path() == path) {
        write_err("Model is currently loaded — eject it first", err, err_len);
        return -1;
    }

    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        write_err("Delete failed: " + ec.message(), err, err_len);
        return -1;
    }
    return 0;
}

char* neurons_get_status(NeuronsCore* h) {
    if (!h) return heap_str("{}");
    neurons::StatusRequest  req;
    neurons::StatusResponse resp;
    h->service->GetStatus(nullptr, &req, &resp);
    return proto_to_json(resp);
}

char* neurons_list_models(NeuronsCore* h) {
    if (!h) return heap_str("[]");
    neurons::ListModelsRequest  req;
    neurons::ListModelsResponse resp;
    h->service->ListModels(nullptr, &req, &resp);
    return proto_to_json(resp);
}

// ── Generation ────────────────────────────────────────────────────────────────

int neurons_generate(NeuronsCore*   h,
                     const char*    user_prompt,
                     const char*    history_json,
                     int            max_tokens,
                     int            context_window,
                     float          temperature,
                     float          top_p,
                     int            top_k,
                     float          rep_penalty,
                     NeuronsTokenCb cb,
                     void*          userdata,
                     char* err, int err_len) {
    if (!h || !user_prompt || !cb) { write_err("null argument", err, err_len); return -1; }

    h->cancel_generate.store(false, std::memory_order_relaxed);

    // Build a GenerateRequest so we can reuse NeuronsServiceImpl::build_prompt()
    // for consistent chat-template application across FFI and gRPC paths.
    neurons::GenerateRequest req;
    req.set_prompt(user_prompt);

    if (history_json && *history_json) {
        try {
            auto arr = nlohmann::json::parse(history_json);
            for (const auto& turn : arr) {
                auto* msg = req.add_history();
                msg->set_role(turn.value("role", ""));
                msg->set_content(turn.value("content", ""));
            }
        } catch (const std::exception& e) {
            write_err(std::string("history_json parse error: ") + e.what(), err, err_len);
            return -1;
        }
    }

    // build_prompt() is private; access it via a thin friend or replicate the
    // proto call. Use load_model_internal's approach: construct a minimal
    // request and call the service's internal method.
    // Since build_prompt is private, call generate_internal with a pre-built
    // prompt via the service's generate path using a null gRPC context.
    // We replicate just the prompt-build call here via the proto request.
    auto* p = req.mutable_params();
    if (max_tokens > 0)      p->set_max_tokens(max_tokens);
    if (context_window > 0)  p->set_context_window(context_window);
    if (temperature > 0)     p->set_temperature(temperature);
    if (top_p > 0)           p->set_top_p(top_p);
    if (top_k > 0)           p->set_top_k(top_k);
    if (rep_penalty > 0)     p->set_rep_penalty(rep_penalty);

    std::string error;
    bool ok = h->service->generate_internal(req, h->cancel_generate,
        [cb, userdata](const std::string& token) -> bool {
            return cb(token.c_str(), userdata) == 0;
        },
        error);

    if (!ok && !error.empty()) {
        write_err(error, err, err_len);
        return -1;
    }
    return 0;
}

void neurons_cancel(NeuronsCore* h) {
    if (h) h->cancel_generate.store(true, std::memory_order_relaxed);
}

// ── Model browser / download ──────────────────────────────────────────────────

char* neurons_search_models(NeuronsCore* h, const char* query, int limit,
                             const char* sort,
                             const char* pipeline_tags_json,
                             const char* author,
                             char* err, int err_len) {
    if (!h || !query) { write_err("null argument", err, err_len); return nullptr; }
    neurons::SearchModelsRequest  req;
    neurons::SearchModelsResponse resp;
    req.set_query(query);
    req.set_limit(limit > 0 ? limit : 30);
    if (sort && *sort) req.set_sort(sort);
    if (author && *author) req.set_author(author);
    // Parse pipeline_tags_json: JSON array of strings e.g. ["text-generation"]
    if (pipeline_tags_json && *pipeline_tags_json) {
        try {
            auto arr = nlohmann::json::parse(pipeline_tags_json);
            if (arr.is_array()) {
                for (const auto& tag : arr) {
                    if (tag.is_string()) req.add_pipeline_tags(tag.get<std::string>());
                }
            }
        } catch (...) {}
    }
    h->service->SearchModels(nullptr, &req, &resp);
    if (!resp.error().empty()) {
        write_err(resp.error(), err, err_len);
        return nullptr;
    }
    return proto_to_json(resp);
}

char* neurons_get_model_info(NeuronsCore* h, const char* repo_id,
                              char* err, int err_len) {
    if (!h || !repo_id) { write_err("null argument", err, err_len); return nullptr; }
    neurons::GetModelInfoRequest  req;
    neurons::GetModelInfoResponse resp;
    req.set_model_id(repo_id);
    h->service->GetModelInfo(nullptr, &req, &resp);
    if (!resp.error().empty()) {
        write_err(resp.error(), err, err_len);
        return nullptr;
    }
    return proto_to_json(resp);
}

int neurons_download_model(NeuronsCore*      h,
                            const char*       repo_id,
                            NeuronsDownloadCb cb,
                            void*             userdata,
                            char* err, int err_len) {
    if (!h || !repo_id || !cb) { write_err("null argument", err, err_len); return -1; }

    h->cancel_download.store(false, std::memory_order_relaxed);

    std::string error;
    bool ok = h->service->download_internal(
        repo_id,
        [cb, userdata](int64_t done, int64_t total, double speed,
                        const std::string& file) -> bool {
            return cb(done, total, speed, file.c_str(), userdata) == 0;
        },
        error);

    if (!ok && !error.empty()) {
        write_err(error, err, err_len);
        return -1;
    }
    return 0;
}

void neurons_cancel_download(NeuronsCore* h) {
    if (!h) return;
    h->cancel_download.store(true, std::memory_order_relaxed);
    // Also signal through the HF client (handled inside download_internal callback).
}

// ── Utilities ─────────────────────────────────────────────────────────────────

void neurons_free_string(char* s) {
    delete[] s;
}
