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

static std::string generateCallId() {
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream ss;
    ss << "call_" << std::hex << rng();
    return ss.str();
}

void OpenAiHttpServer::handleChatCompletions(const httplib::Request& req, httplib::Response& res) {
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
    const std::string id    = generateId();
    const std::string model = body.value("model", "neurons");

    // Detect whether this is a tool-use request.
    const json& tools_arr   = body.contains("tools") ? body["tools"] : json(nullptr);
    const std::string tool_choice = body.value("tool_choice", "auto");
    const bool tools_requested =
        !tools_arr.is_null() && tools_arr.is_array() &&
        !tools_arr.empty()   && tool_choice != "none";

    const json& messages = body.contains("messages") ? body["messages"] : json::array();

    // Sampling params proto (shared between both paths).
    neurons::SamplingParams params_proto;
    if (body.contains("temperature")) params_proto.set_temperature(body["temperature"].get<float>());
    if (body.contains("top_p"))       params_proto.set_top_p(body["top_p"].get<float>());
    const int max_tokens = body.value("max_tokens", 0);

    // ── Tool-use path (Option B: client-side execution) ───────────────────────
    if (tools_requested) {
        if (!stream) {
            std::string full_content;
            std::string error_out;
            NeuronsServiceImpl::HttpCompletionResult result;

            bool ok = service_.generate_http_completion(
                messages, tools_arr, params_proto, max_tokens,
                [&](const std::string& delta) -> bool {
                    full_content += delta;
                    return true;
                },
                result, error_out);

            addCors(res);
            if (!ok) {
                json err = {{"error", {{"message", error_out}, {"type", "server_error"}}}};
                res.status = 500;
                res.set_content(err.dump(), "application/json");
                return;
            }

            json choice;
            if (result.tool_call) {
                const std::string call_id = generateCallId();
                choice = {
                    {"index", 0},
                    {"message", {
                        {"role",       "assistant"},
                        {"content",    nullptr},
                        {"tool_calls", json::array({{
                            {"id",   call_id},
                            {"type", "function"},
                            {"function", {
                                {"name",      result.tool_call->name},
                                {"arguments", result.tool_call->arguments_json}
                            }}
                        }})}
                    }},
                    {"finish_reason", "tool_calls"}
                };
            } else {
                choice = {
                    {"index",         0},
                    {"message",       {{"role", "assistant"}, {"content", full_content}}},
                    {"finish_reason", "stop"}
                };
            }

            json resp = {
                {"id",      id},
                {"object",  "chat.completion"},
                {"created", unixSec()},
                {"model",   model},
                {"choices", json::array({choice})},
                {"usage",   {
                    {"prompt_tokens",     static_cast<int>(result.prompt_tokens)},
                    {"completion_tokens", static_cast<int>(result.gen_tokens)},
                    {"total_tokens",      static_cast<int>(result.prompt_tokens + result.gen_tokens)}
                }}
            };
            res.set_content(resp.dump(), "application/json");
            return;
        }

        // Streaming + tools
        addCors(res);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");

        res.set_chunked_content_provider("text/event-stream",
            [this, messages, tools_arr, params_proto, max_tokens, id, model]
            (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {

                // Role announce chunk
                json first = {
                    {"id", id}, {"object", "chat.completion.chunk"},
                    {"created", unixSec()}, {"model", model},
                    {"choices", json::array({{
                        {"index", 0},
                        {"delta", {{"role", "assistant"}}},
                        {"finish_reason", nullptr}
                    }})}
                };
                std::string ev = "data: " + first.dump() + "\n\n";
                if (!sink.write(ev.c_str(), ev.size())) return false;

                std::string error_out;
                NeuronsServiceImpl::HttpCompletionResult result;

                service_.generate_http_completion(
                    messages, tools_arr, params_proto, max_tokens,
                    [&](const std::string& delta) -> bool {
                        if (!sink.is_writable()) return false;
                        json chunk = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", unixSec()}, {"model", model},
                            {"choices", json::array({{
                                {"index", 0},
                                {"delta", {{"content", delta}}},
                                {"finish_reason", nullptr}
                            }})}
                        };
                        std::string cev = "data: " + chunk.dump() + "\n\n";
                        return sink.write(cev.c_str(), cev.size());
                    },
                    result, error_out);

                // Final chunk — tool call or stop
                json final_chunk;
                if (result.tool_call) {
                    const std::string call_id = generateCallId();
                    final_chunk = {
                        {"id", id}, {"object", "chat.completion.chunk"},
                        {"created", unixSec()}, {"model", model},
                        {"choices", json::array({{
                            {"index", 0},
                            {"delta", {
                                {"tool_calls", json::array({{
                                    {"index", 0},
                                    {"id",   call_id},
                                    {"type", "function"},
                                    {"function", {
                                        {"name",      result.tool_call->name},
                                        {"arguments", result.tool_call->arguments_json}
                                    }}
                                }})}
                            }},
                            {"finish_reason", "tool_calls"}
                        }})}
                    };
                } else {
                    final_chunk = {
                        {"id", id}, {"object", "chat.completion.chunk"},
                        {"created", unixSec()}, {"model", model},
                        {"choices", json::array({{
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", "stop"}
                        }})}
                    };
                }
                ev = "data: " + final_chunk.dump() + "\n\n";
                sink.write(ev.c_str(), ev.size());
                sink.write("data: [DONE]\n\n", 14);
                sink.done();
                return true;
            });
        return;
    }

    // ── Non-tool path (existing behaviour) ───────────────────────────────────
    neurons::GenerateRequest greq;

    std::string last_user_content;
    for (const auto& m : messages) {
        const std::string role    = m.value("role", "");
        const std::string content = m.value("content", "");
        if (role == "user") {
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
            auto* h = greq.add_history();
            h->set_role("system");
            h->set_content(content);
        } else if (role == "tool") {
            auto* tr = greq.add_tool_results();
            tr->set_name(m.value("name", ""));
            tr->set_result_json(content);
        }
    }
    greq.set_prompt(last_user_content);

    auto* p = greq.mutable_params();
    *p = params_proto;
    if (max_tokens > 0) p->set_max_tokens(max_tokens);

    if (!stream) {
        std::string full_content;
        std::string error_out;
        std::atomic<bool> cancelled{false};

        bool ok = service_.generate_internal(greq, cancelled,
            [&](const std::string& tok) { full_content += tok; return true; },
            error_out);

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
            {"choices", json::array({{
                {"index",         0},
                {"message",       {{"role", "assistant"}, {"content", full_content}}},
                {"finish_reason", "stop"}
            }})},
            {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}
        };
        res.set_content(resp.dump(), "application/json");
        return;
    }

    // Streaming, no tools
    addCors(res);
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    std::string error_out;
    std::atomic<bool> cancelled{false};

    res.set_chunked_content_provider("text/event-stream",
        [this, greq, id, model, &cancelled, &error_out]
        (size_t /*offset*/, httplib::DataSink& sink) mutable -> bool {
            json first = {
                {"id", id}, {"object", "chat.completion.chunk"},
                {"created", unixSec()}, {"model", model},
                {"choices", json::array({{
                    {"index", 0},
                    {"delta", {{"role", "assistant"}}},
                    {"finish_reason", nullptr}
                }})}
            };
            std::string ev = "data: " + first.dump() + "\n\n";
            if (!sink.write(ev.c_str(), ev.size())) { cancelled = true; return false; }

            service_.generate_internal(greq, cancelled,
                [&](const std::string& tok) -> bool {
                    if (!sink.is_writable()) { cancelled = true; return false; }
                    json chunk = {
                        {"id", id}, {"object", "chat.completion.chunk"},
                        {"created", unixSec()}, {"model", model},
                        {"choices", json::array({{
                            {"index", 0},
                            {"delta", {{"content", tok}}},
                            {"finish_reason", nullptr}
                        }})}
                    };
                    std::string cev = "data: " + chunk.dump() + "\n\n";
                    return sink.write(cev.c_str(), cev.size());
                }, error_out);

            json stop_chunk = {
                {"id", id}, {"object", "chat.completion.chunk"},
                {"created", unixSec()}, {"model", model},
                {"choices", json::array({{
                    {"index", 0},
                    {"delta", json::object()},
                    {"finish_reason", "stop"}
                }})}
            };
            ev = "data: " + stop_chunk.dump() + "\n\n";
            sink.write(ev.c_str(), ev.size());
            sink.write("data: [DONE]\n\n", 14);
            sink.done();
            return true;
        });
}

} // namespace neurons_service
