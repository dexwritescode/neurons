#include "http_server.h"
#include "neurons_service.h"
#include "neurons.pb.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace neurons_service {

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string generateId() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream ss;
    ss << "chatcmpl-" << std::hex << rng();
    return ss.str();
}

static int64_t unixSec() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static void addCors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

// ── Construction ─────────────────────────────────────────────────────────────

OpenAiHttpServer::OpenAiHttpServer(NeuronsServiceImpl& service, Logger& logger)
    : service_(service), logger_(logger)
    , svr_(std::make_unique<httplib::Server>())
{
    setupRoutes();
}

OpenAiHttpServer::~OpenAiHttpServer() {
    stop();
}

bool OpenAiHttpServer::start(int port) {
    if (!svr_->bind_to_port("0.0.0.0", port)) {
        logger_.error("HTTP server: failed to bind to port " + std::to_string(port));
        return false;
    }
    port_ = port;
    running_ = true;
    thread_ = std::thread([this]() {
        svr_->listen_after_bind();
        running_ = false;
    });
    logger_.info("OpenAI HTTP server listening on port " + std::to_string(port));
    return true;
}

void OpenAiHttpServer::stop() {
    if (running_) {
        svr_->stop();
        if (thread_.joinable()) thread_.join();
        running_ = false;
    }
}

// ── Route setup ──────────────────────────────────────────────────────────────

void OpenAiHttpServer::setupRoutes() {
    // CORS preflight
    svr_->Options(".*", [](const httplib::Request&, httplib::Response& res) {
        addCors(res);
        res.status = 204;
    });

    svr_->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handleModels(req, res);
    });

    svr_->Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handleChatCompletions(req, res);
    });

    // Health check
    svr_->Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
}

// ── GET /v1/models ────────────────────────────────────────────────────────────

void OpenAiHttpServer::handleModels(const httplib::Request&, httplib::Response& res) {
    neurons::StatusRequest sreq;
    neurons::StatusResponse sresp;
    service_.GetStatus(nullptr, &sreq, &sresp);

    json data = json::array();
    if (sresp.model_loaded()) {
        data.push_back({
            {"id",       sresp.model_path()},
            {"object",   "model"},
            {"created",  unixSec()},
            {"owned_by", "neurons"}
        });
    }

    json resp = {{"object", "list"}, {"data", data}};
    addCors(res);
    res.set_content(resp.dump(), "application/json");
}

// ── POST /v1/chat/completions ─────────────────────────────────────────────────

void OpenAiHttpServer::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        addCors(res);
        res.status = 400;
        res.set_content("{\"error\":{\"message\":\"Invalid JSON\",\"type\":\"invalid_request_error\"}}", "application/json");
        return;
    }

    const bool stream = body.value("stream", false);

    // Build GenerateRequest from OpenAI messages
    neurons::GenerateRequest greq;

    const auto& messages = body.value("messages", json::array());
    std::string last_user_content;
    for (const auto& m : messages) {
        const std::string role    = m.value("role", "");
        const std::string content = m.value("content", "");
        if (role == "user") {
            // The last user message becomes the prompt; earlier ones go to history
            if (!last_user_content.empty()) {
                auto* h = greq.add_history();
                h->set_role("user");
                h->set_content(last_user_content);
            }
            last_user_content = content;
        } else if (role == "assistant") {
            auto* h = greq.add_history();
            h->set_role("assistant");
            h->set_content(content);
        } else if (role == "system") {
            // system message: prepend to first user turn as [role: system]
            // For now inject into history so build_prompt can pick it up
            auto* h = greq.add_history();
            h->set_role("system");
            h->set_content(content);
        }
    }
    greq.set_prompt(last_user_content);

    // Sampling params
    auto* p = greq.mutable_params();
    if (body.contains("temperature")) p->set_temperature(body["temperature"].get<float>());
    if (body.contains("top_p"))       p->set_top_p(body["top_p"].get<float>());
    if (body.contains("max_tokens"))  p->set_max_tokens(body["max_tokens"].get<int>());

    const std::string id     = generateId();
    const std::string model  = body.value("model", "neurons");

    if (!stream) {
        // Non-streaming: collect full response then return
        std::string full_content;
        std::string error_out;
        std::atomic<bool> cancelled{false};

        bool ok = service_.generate_internal(greq, cancelled,
            [&](const std::string& tok) {
                full_content += tok;
                return true;
            }, error_out);

        addCors(res);
        if (!ok) {
            json err = {{"error", {{"message", error_out}, {"type", "server_error"}}}};
            res.status = 500;
            res.set_content(err.dump(), "application/json");
            return;
        }

        json resp = {
            {"id",      id},
            {"object",  "chat.completion"},
            {"created", unixSec()},
            {"model",   model},
            {"choices", json::array({
                {
                    {"index",         0},
                    {"message",       {{"role", "assistant"}, {"content", full_content}}},
                    {"finish_reason", "stop"}
                }
            })},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}
        };
        res.set_content(resp.dump(), "application/json");
        return;
    }

    // Streaming SSE
    addCors(res);
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    std::string error_out;
    std::atomic<bool> cancelled{false};

    res.set_chunked_content_provider("text/event-stream",
        [this, greq, id, model, &cancelled, &error_out](size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
            // Role delta (first chunk)
            json first = {
                {"id",      id},
                {"object",  "chat.completion.chunk"},
                {"created", unixSec()},
                {"model",   model},
                {"choices", json::array({{
                    {"index",         0},
                    {"delta",         {{"role", "assistant"}}},
                    {"finish_reason", nullptr}
                }})}
            };
            std::string first_ev = "data: " + first.dump() + "\n\n";
            if (!sink.write(first_ev.c_str(), first_ev.size())) { cancelled = true; return false; }

            bool ok = service_.generate_internal(greq, cancelled,
                [&](const std::string& tok) -> bool {
                    if (!sink.is_writable()) { cancelled = true; return false; }
                    json chunk = {
                        {"id",      id},
                        {"object",  "chat.completion.chunk"},
                        {"created", unixSec()},
                        {"model",   model},
                        {"choices", json::array({{
                            {"index",         0},
                            {"delta",         {{"content", tok}}},
                            {"finish_reason", nullptr}
                        }})}
                    };
                    std::string ev = "data: " + chunk.dump() + "\n\n";
                    return sink.write(ev.c_str(), ev.size());
                }, error_out);

            (void)ok;

            // Final stop chunk
            json stop_chunk = {
                {"id",      id},
                {"object",  "chat.completion.chunk"},
                {"created", unixSec()},
                {"model",   model},
                {"choices", json::array({{
                    {"index",         0},
                    {"delta",         json::object()},
                    {"finish_reason", "stop"}
                }})}
            };
            std::string stop_ev = "data: " + stop_chunk.dump() + "\n\n";
            sink.write(stop_ev.c_str(), stop_ev.size());

            std::string done_ev = "data: [DONE]\n\n";
            sink.write(done_ev.c_str(), done_ev.size());
            sink.done();
            return true;
        });
}

} // namespace neurons_service
