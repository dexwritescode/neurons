#include "remote_node_runner.h"

#include <iostream>

#if defined(NEURONS_REMOTE_ENABLED)

#include <chrono>
#include <iomanip>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>

#include "neurons.grpc.pb.h"
#include "neurons.pb.h"

namespace neurons::cli {

RemoteNodeRunner::RemoteNodeRunner(Options opts)
    : opts_(std::move(opts))
{
    channel_ = grpc::CreateChannel(opts_.endpoint,
                                   grpc::InsecureChannelCredentials());
    stub_ = neurons::NeuronsInference::NewStub(channel_);
}

void RemoteNodeRunner::printHelp() const {
    std::cout << "  /exit    — quit\n";
    std::cout << "  /clear   — clear conversation history\n";
    std::cout << "  /history — show conversation history\n";
    std::cout << "  /help    — show this help\n";
}

int RemoteNodeRunner::run_repl() {
    std::cout << "Connected to " << opts_.endpoint << " | Type /help for commands\n\n";

    struct Turn { std::string role, content; };
    std::vector<Turn> history;
    std::string line;

    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;

        if (line == "/clear") {
            history.clear();
            std::cout << "[Conversation cleared]\n\n";
            continue;
        }
        if (line == "/history") {
            if (history.empty()) {
                std::cout << "[No history yet]\n\n";
            } else {
                for (const auto& t : history)
                    std::cout << (t.role == "user" ? "You: " : "Assistant: ")
                              << t.content << "\n\n";
            }
            continue;
        }
        if (line == "/help") { printHelp(); std::cout << "\n"; continue; }

        neurons::GenerateRequest req;
        for (const auto& t : history) {
            auto* msg = req.add_history();
            msg->set_role(t.role);
            msg->set_content(t.content);
        }
        req.set_prompt(line);
        req.set_session_id("cli-remote-session");

        auto* p = req.mutable_params();
        p->set_temperature(opts_.temperature);
        p->set_top_k(opts_.top_k);
        p->set_top_p(opts_.top_p);
        p->set_rep_penalty(opts_.rep_penalty);
        if (opts_.max_tokens > 0) p->set_max_tokens(opts_.max_tokens);

        req.set_tool_use_enabled(opts_.tools_enabled);
        req.set_allow_shell_fallback(opts_.allow_shell_fallback);
        for (const auto& s : opts_.tool_servers)
            req.add_active_mcp_servers(s);

        grpc::ClientContext ctx;
        auto stream = stub_->Generate(&ctx, req);

        std::cout << "Assistant: " << std::flush;

        std::string full_reply;
        uint32_t gen_tokens = 0;
        auto t_start = std::chrono::steady_clock::now();
        bool had_error = false;

        neurons::GenerateResponse resp;
        while (stream->Read(&resp)) {
            if (!resp.error().empty()) {
                std::cerr << "\nError: " << resp.error() << "\n";
                had_error = true;
                break;
            }

            switch (resp.event_case()) {
                case neurons::GenerateResponse::kToken: {
                    const auto& delta = resp.token();
                    full_reply += delta;
                    std::cout << delta << std::flush;
                    break;
                }
                case neurons::GenerateResponse::kToolCall:
                case neurons::GenerateResponse::kToolResult:
                    break;
                case neurons::GenerateResponse::kApprovalRequest: {
                    const auto& ar = resp.approval_request();
                    bool allowed = resolve_tool_approval(
                        ar.tool(), ar.server(), ar.args_json(), opts_.tool_policy);

                    neurons::ToolApprovalResponse approval;
                    approval.set_approval_id(ar.approval_id());
                    approval.set_approved(allowed);

                    grpc::ClientContext approval_ctx;
                    neurons::ToolApprovalResult result;
                    stub_->RespondToolApproval(&approval_ctx, approval, &result);
                    break;
                }
                default:
                    break;
            }

            if (resp.done()) {
                gen_tokens = resp.gen_tokens();
                break;
            }
        }

        if (const auto status = stream->Finish(); !status.ok() && !had_error) {
            std::cerr << "\nRPC error: " << status.error_message() << "\n";
            return 1;
        }

        std::cout << "\n";

        if (!had_error) {
            auto t_end = std::chrono::steady_clock::now();
            double secs = std::chrono::duration<double>(t_end - t_start).count();
            double tps  = gen_tokens > 0 ? gen_tokens / secs : 0.0;
            std::cout << "[" << gen_tokens << " tokens | "
                      << std::fixed << std::setprecision(1) << tps << " tok/s]\n\n";

            history.push_back({"user",      line});
            history.push_back({"assistant", full_reply});
        }
    }

    std::cout << "\nGoodbye.\n";
    return 0;
}

} // namespace neurons::cli

#else  // NEURONS_REMOTE_ENABLED not defined

namespace neurons::cli {

RemoteNodeRunner::RemoteNodeRunner(Options opts) : opts_(std::move(opts)) {}

void RemoteNodeRunner::printHelp() const {}

int RemoteNodeRunner::run_repl() {
    std::cerr << "Error: remote inference requires gRPC support.\n";
    std::cerr << "Rebuild with gRPC installed (brew install grpc).\n";
    return 1;
}

} // namespace neurons::cli

#endif  // NEURONS_REMOTE_ENABLED
