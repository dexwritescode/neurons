#include "chat_command.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "cli/utils/format_utils.h"

#include "compute/core/compute_backend.h"
#include "compute/model/language_model.h"
#include "compute/core/compute_types.h"

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

    cmd->add_option("model", model_name_, "Model name (e.g., mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit)")
        ->required();
    cmd->add_option("--system,-s", system_prompt_, "System prompt");
    cmd->add_option("--max-tokens", max_tokens_, "Max tokens per turn (default: 512)");
    cmd->add_option("--temperature", temperature_, "Sampling temperature (default: 0.7)");
    cmd->add_option("--top-k", top_k_, "Top-k sampling (default: 40)");
    cmd->add_option("--top-p", top_p_, "Top-p sampling (default: 0.9)");
    cmd->add_option("--rep-penalty", rep_penalty_, "Repetition penalty (default: 1.1)");

    cmd->callback([this]() { std::exit(execute()); });
}

int ChatCommand::execute() {
    const auto location = model_registry_->locateModel(model_name_);
    if (!location.isValid()) {
        std::cerr << "Error: Model '" << model_name_ << "' not found in "
                  << config_->modelsDirectory().string() << "\n";
        std::cerr << "Run 'neurons list' to see available models\n";
        return 1;
    }
    return run_repl(location.modelPath);
}

std::string ChatCommand::buildPrompt(const std::string& model_type,
                                     const std::vector<Turn>& history,
                                     const std::string& user_input,
                                     bool has_llama3_tokens) const {
    const bool is_llama3 = (model_type == "llama") && has_llama3_tokens;

    if (is_llama3) {
        std::string out = "<|begin_of_text|>";
        out += "<|start_header_id|>system<|end_header_id|>\n\n" + system_prompt_ + "<|eot_id|>\n";
        for (const auto& t : history) {
            out += "<|start_header_id|>" + t.role + "<|end_header_id|>\n\n" + t.content + "<|eot_id|>\n";
        }
        out += "<|start_header_id|>user<|end_header_id|>\n\n" + user_input + "<|eot_id|>\n";
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    if (model_type == "qwen2") {
        std::string out = "<|im_start|>system\n" + system_prompt_ + "<|im_end|>\n";
        for (const auto& t : history) {
            out += "<|im_start|>" + t.role + "\n" + t.content + "<|im_end|>\n";
        }
        out += "<|im_start|>user\n" + user_input + "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return out;
    }

    if (model_type == "mistral") {
        // Mistral uses [INST]/[/INST] without system prompt support
        std::string out;
        for (const auto& t : history) {
            if (t.role == "user")      out += "[INST] " + t.content + " [/INST]";
            else if (t.role == "assistant") out += t.content + "</s>";
        }
        out += "[INST] " + user_input + " [/INST]";
        return out;
    }

    if (model_type == "gemma" || model_type == "gemma2" || model_type == "gemma3_text") {
        std::string out;
        for (const auto& t : history) {
            out += "<start_of_turn>" + t.role + "\n" + t.content + "<end_of_turn>\n";
        }
        out += "<start_of_turn>user\n" + user_input + "<end_of_turn>\n";
        out += "<start_of_turn>model\n";
        return out;
    }

    // TinyLlama / Llama-2 template
    std::string out = "<|system|>\n" + system_prompt_ + "</s>\n";
    for (const auto& t : history) {
        if (t.role == "user")      out += "<|user|>\n" + t.content + "</s>\n<|assistant|>\n";
        else if (t.role == "assistant") out += t.content + "</s>\n";
    }
    out += "<|user|>\n" + user_input + "</s>\n<|assistant|>\n";
    return out;
}

void ChatCommand::printHelp() const {
    std::cout << "  /exit    — quit\n";
    std::cout << "  /clear   — clear conversation history\n";
    std::cout << "  /history — show conversation history\n";
    std::cout << "  /help    — show this help\n";
}

int ChatCommand::run_repl(const std::string& model_path) {
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

    std::cout << "Model: " << model_type << " | Type /help for commands\n\n";

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
        std::string assistant_reply;

        auto t_gen_start = std::chrono::steady_clock::now();

        compute::SamplingParams params;
        params.temperature = temperature_;
        params.top_k       = static_cast<size_t>(top_k_);
        params.top_p       = top_p_;
        params.rep_penalty = rep_penalty_;

        auto gen_result = inference->generate(
            token_ids,
            static_cast<size_t>(max_tokens_),
            params,
            [&](int token_id) -> bool {
                ++token_count;
                if (inf_cfg.is_eos_token(token_id)) return false;
                generated_so_far.push_back(token_id);
                std::string new_decoded = tok.decode(generated_so_far, /*skip_special_tokens=*/true);
                if (new_decoded.size() > decoded_so_far.size()) {
                    std::string chunk = new_decoded.substr(decoded_so_far.size());
                    std::cout << chunk << std::flush;
                    assistant_reply += chunk;
                }
                decoded_so_far = std::move(new_decoded);
                return true;
            }
        );

        std::cout << "\n";

        if (!gen_result) {
            std::cerr << "Error during generation: " << gen_result.error().message << "\n";
            return 1;
        }

        auto t_gen_end = std::chrono::steady_clock::now();
        double gen_secs = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
        double toks_per_sec = token_count > 0 ? token_count / gen_secs : 0.0;
        std::cout << "[" << token_count << " tokens | "
                  << std::fixed << std::setprecision(1) << toks_per_sec << " tok/s]\n\n";

        // Add turn to history
        history.push_back({"user", line});
        history.push_back({"assistant", assistant_reply});
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
