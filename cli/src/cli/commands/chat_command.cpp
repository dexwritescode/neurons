#include "chat_command.h"
#include <iostream>
#include <atomic>
#include <chrono>
#include <future>
#include <iomanip>
#include <sstream>
#include "cli/utils/format_utils.h"
#include <nlohmann/json.hpp>

#include "compute/core/compute_backend.h"
#include "compute/model/chat_template.h"
#include "compute/model/language_model.h"
#include "compute/model/tool_runner.h"
#include "compute/core/compute_types.h"
#include "mcp/mcp_manager.h"
#include "mcp/mcp_types.h"
#include "cli/utils/tool_policy.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#endif

namespace neurons::cli {

ChatCommand::ChatCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "chat";
    description_  = "Interactive multi-turn chat with a local model";

    model_registry_ = std::make_unique<models::registry::ModelRegistry>(
        config_->modelsDirectory().string(),
        nullptr
    );
}

void ChatCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);

    cmd->add_option("model", model_name_,
        "Model name for local inference (e.g., mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit); "
        "omit when using --node");
    cmd->add_option("--node,-n", node_endpoint_,
        "Remote node name (from 'neurons node list') or gRPC URL (host:port or grpc://host:port)");
    cmd->add_option("--system,-s", system_prompt_, "System prompt");
    cmd->add_option("--max-tokens", max_tokens_, "Max tokens per turn (default: 4096)");
    cmd->add_option("--temperature", temperature_, "Sampling temperature (default: 0.7)");
    cmd->add_option("--top-k", top_k_, "Top-k sampling (default: 40)");
    cmd->add_option("--top-p", top_p_, "Top-p sampling (default: 0.9)");
    cmd->add_option("--rep-penalty", rep_penalty_, "Repetition penalty (default: 1.1)");
    cmd->add_flag("--tools,-t", tools_enabled_, "Enable MCP tool use");
    cmd->add_flag("--allow-commands", allow_shell_fallback_,
                  "Allow the model to invoke arbitrary shell commands not registered "
                  "as MCP tools (requires --tools; always prompts for approval)");
    cmd->add_option("--tool-servers", tool_servers_,
                    "Restrict tool use to specific MCP servers (default: all)");
    cmd->add_option("--tool-policy", tool_policy_,
                    "Tool approval policy: allow (auto-approve), deny (auto-deny), "
                    "ask (y/N prompt; default)");

    cmd->callback([this]() { std::exit(execute()); });
}

int ChatCommand::execute() {
    ToolPolicy policy = ToolPolicy::Ask;
    try {
        policy = parse_tool_policy(tool_policy_);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (!node_endpoint_.empty()) {
        // Resolve node name → host:port, or strip grpc:// scheme from a URL.
        std::string endpoint = node_endpoint_;
        if (endpoint.rfind("grpc://", 0) == 0) {
            endpoint = endpoint.substr(7);
        } else if (endpoint.find(':') == std::string::npos) {
            // Looks like a node name — look it up in config.
            auto node = config_->findNode(endpoint);
            if (!node) {
                std::cerr << "Error: Node '" << endpoint << "' not found.\n";
                std::cerr << "Run 'neurons node list' to see available nodes.\n";
                return 1;
            }
            endpoint = node->host + ":" + std::to_string(node->port);
        }

        RemoteNodeRunner::Options opts;
        opts.endpoint      = endpoint;
        opts.system_prompt = system_prompt_;
        opts.max_tokens    = max_tokens_;
        opts.temperature   = temperature_;
        opts.top_k         = top_k_;
        opts.top_p         = top_p_;
        opts.rep_penalty   = rep_penalty_;
        opts.tools_enabled        = tools_enabled_;
        opts.allow_shell_fallback = allow_shell_fallback_;
        opts.tool_policy          = policy;
        opts.tool_servers         = tool_servers_;
        return RemoteNodeRunner(std::move(opts)).run_repl();
    }

    // Local inference — model argument is required.
    if (model_name_.empty()) {
        std::cerr << "Error: 'model' argument is required for local inference.\n";
        std::cerr << "Pass --node to connect to a remote service instead.\n";
        return 1;
    }

    const auto location = model_registry_->locateModel(model_name_);
    if (!location.isValid()) {
        std::cerr << "Error: Model '" << model_name_ << "' not found in "
                  << config_->modelsDirectory().string() << "\n";
        std::cerr << "Run 'neurons list' to see available models\n";
        return 1;
    }
    return run_repl(location.modelPath, policy);
}

std::string ChatCommand::buildPrompt(const std::string& model_type,
                                     const std::vector<Turn>& history,
                                     const std::string& user_input,
                                     bool has_llama3_tokens) const {
    std::vector<compute::ChatMessage> messages;
    messages.reserve(history.size() + 1);
    for (const auto& t : history)
        messages.push_back({t.role, t.content});
    messages.push_back({"user", user_input});
    return compute::apply_chat_template(
        model_type, has_llama3_tokens, system_prompt_, messages);
}

void ChatCommand::printHelp() const {
    std::cout << "  /exit    — quit\n";
    std::cout << "  /clear   — clear conversation history\n";
    std::cout << "  /history — show conversation history\n";
    std::cout << "  /help    — show this help\n";
}

int ChatCommand::run_repl(const std::string& model_path, ToolPolicy policy) {
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    std::cout << "Loading model..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();

    auto backend_result = compute::BackendFactory::create(compute::BackendType::MLX);
    if (!backend_result) {
        std::cerr << "\nError creating backend: " << backend_result.error().message << "\n";
        return 1;
    }
    auto backend = std::move(*backend_result);

    if (auto init = backend->initialize(); !init) {
        std::cerr << "\nError initializing backend: " << init.error().message << "\n";
        return 1;
    }

    auto inf_result = compute::LanguageModel::load(model_path, backend.get());
    if (!inf_result) {
        std::cerr << "\nError loading model: " << inf_result.error().message << "\n";
        return 1;
    }
    auto inference = std::move(*inf_result);

    auto t1 = std::chrono::steady_clock::now();
    double load_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << load_secs << "s)\n";

    const std::string model_type = inference->model_type();
    const bool has_llama3_tokens = (model_type == "llama") &&
        (inference->tokenizer().find_token_id("<|start_header_id|>") != -1);

    std::cout << "Model: " << model_type << " | Type /help for commands\n";

    // ── Tool-use setup ────────────────────────────────────────────────────────
    std::unique_ptr<neurons_service::McpManager> mcp_mgr;
    neurons_service::ToolCallCb tool_cb;

    if (tools_enabled_) {
        if (!inference->supports_tool_use()) {
            std::cout << "[tools] Warning: model does not support tool use — --tools ignored\n";
        } else {
            mcp_mgr = std::make_unique<neurons_service::McpManager>();
            mcp_mgr->load_config();
            mcp_mgr->load_permissions();
            mcp_mgr->connect_enabled();


            // Approval callback: behaviour driven by --tool-policy flag.
            neurons_service::ApprovalCb approval_cb =
                [policy](const neurons_service::ToolApprovalRequest& req)
                    -> std::future<bool>
            {
                std::promise<bool> p;
                p.set_value(resolve_tool_approval(
                    req.tool, req.server, req.args_json, policy));
                return p.get_future();
            };

            tool_cb = mcp_mgr->make_tool_call_cb(
                "cli-session", "cli-chat", approval_cb, tool_servers_,
                allow_shell_fallback_);

            // Inject tool definitions into the system prompt.
            if (mcp_mgr->has_active_tools(tool_servers_)) {
                std::string tool_json = mcp_mgr->tools_json(tool_servers_);
                system_prompt_ = inference->format_tool_system_prompt(tool_json);
                auto n = mcp_mgr->list_tools(tool_servers_).size();
                std::cout << "[tools] " << n << " tool(s) available\n";
            } else {
                std::cout << "[tools] Warning: no tools available from configured servers\n";
            }
        }
    }

    std::cout << "\n";

    std::vector<Turn> history;
    std::string line;

    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line)) break; // EOF / Ctrl-D

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
                for (const auto& t : history) {
                    std::cout << (t.role == "user" ? "You: " : "Assistant: ") << t.content << "\n\n";
                }
            }
            continue;
        }

        if (line == "/help") {
            printHelp();
            std::cout << "\n";
            continue;
        }

        std::string prompt = buildPrompt(model_type, history, line, has_llama3_tokens);
        auto token_ids = inference->tokenizer().encode(prompt, /*add_special_tokens=*/true);

        std::cout << "Assistant: " << std::flush;

        auto& tok = inference->tokenizer();
        const auto& inf_cfg = inference->config();
        int token_count = 0;
        std::vector<int> generated_so_far;
        std::string decoded_so_far;

        // F.14.3: thinking block streaming for reasoning models (e.g. Qwen3).
        const bool is_reasoning = inference->is_reasoning_model();
        bool in_think = false;
        size_t print_pos = 0;

        auto flush_decoded = [&]() {
            if (!is_reasoning) {
                if (decoded_so_far.size() > print_pos) {
                    std::cout << decoded_so_far.substr(print_pos) << std::flush;
                    print_pos = decoded_so_far.size();
                }
                return;
            }
            static const std::string OPEN  = "<think>";
            static const std::string CLOSE = "</think>";
            static const size_t HOLD = CLOSE.size() - 1;

            while (print_pos < decoded_so_far.size()) {
                if (!in_think) {
                    size_t pos = decoded_so_far.find(OPEN, print_pos);
                    if (pos == std::string::npos) {
                        size_t safe = decoded_so_far.size() > HOLD ? decoded_so_far.size() - HOLD : print_pos;
                        if (safe > print_pos) {
                            std::cout << decoded_so_far.substr(print_pos, safe - print_pos) << std::flush;
                            print_pos = safe;
                        }
                        break;
                    }
                    if (pos > print_pos)
                        std::cout << decoded_so_far.substr(print_pos, pos - print_pos) << std::flush;
                    print_pos = pos + OPEN.size();
                    in_think = true;
                } else {
                    size_t pos = decoded_so_far.find(CLOSE, print_pos);
                    if (pos == std::string::npos) {
                        size_t safe = decoded_so_far.size() > HOLD ? decoded_so_far.size() - HOLD : print_pos;
                        print_pos = safe;
                        break;
                    }
                    print_pos = pos + CLOSE.size();
                    in_think = false;
                }
            }
        };

        auto t_gen_start = std::chrono::steady_clock::now();

        compute::SamplingParams params;
        params.temperature = temperature_;
        params.top_k       = static_cast<size_t>(top_k_);
        params.top_p       = top_p_;
        params.rep_penalty = rep_penalty_;

        uint32_t total_tokens = 0;

        if (tool_cb) {
            // ── Tool-use path: ToolRunner handles multi-turn generate/call/inject.
            std::atomic<bool> cancelled{false};
            auto tool_result = compute::ToolRunner{}.run(
                *inference, token_ids,
                static_cast<size_t>(max_tokens_),
                params,
                [&](const std::string& delta) -> bool {
                    ++token_count;
                    decoded_so_far += delta;
                    flush_decoded();
                    return true;
                },
                tool_cb,
                cancelled);

            if (!tool_result) {
                std::cerr << "Error during generation: " << tool_result.error().message << "\n";
                return 1;
            }
            total_tokens = tool_result.value();
        } else {
            // ── Standard path: single-turn generate with full-sequence decode.
            auto gen_result = inference->generate(
                token_ids,
                static_cast<size_t>(max_tokens_),
                params,
                [&](int token_id) -> bool {
                    ++token_count;
                    if (inf_cfg.is_eos_token(token_id)) return false;
                    generated_so_far.push_back(token_id);
                    decoded_so_far = tok.decode(generated_so_far, /*skip_special_tokens=*/true);
                    flush_decoded();
                    return true;
                }
            );

            if (!gen_result) {
                std::cerr << "Error during generation: " << gen_result.error().message << "\n";
                return 1;
            }
            total_tokens = static_cast<uint32_t>(token_count);
        }

        if (!in_think && print_pos < decoded_so_far.size())
            std::cout << decoded_so_far.substr(print_pos) << std::flush;

        std::cout << "\n";

        auto t_gen_end = std::chrono::steady_clock::now();
        double gen_secs = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
        double toks_per_sec = total_tokens > 0 ? total_tokens / gen_secs : 0.0;
        std::cout << "[" << total_tokens << " tokens | "
                  << std::fixed << std::setprecision(1) << toks_per_sec << " tok/s]\n\n";

        // Strip <think>...</think> from history so the model doesn't see its own reasoning
        // on the next turn (the Qwen3 template handles it via </think> splitting, but
        // since we build prompts manually we exclude it entirely).
        std::string reply_for_history = decoded_so_far;
        if (is_reasoning) {
            auto close_pos = reply_for_history.find("</think>");
            if (close_pos != std::string::npos) {
                reply_for_history = reply_for_history.substr(close_pos + 8);
                size_t first = reply_for_history.find_first_not_of("\n ");
                if (first != std::string::npos) reply_for_history = reply_for_history.substr(first);
            }
        }
        history.push_back({"user", line});
        history.push_back({"assistant", reply_for_history});
    }

    backend->cleanup();
    std::cout << "\nGoodbye.\n";
    return 0;

#else
    (void)model_path;
    std::cerr << "Error: Inference requires Apple Silicon with MLX backend\n";
    return 1;
#endif
}

} // namespace neurons::cli
