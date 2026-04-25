#include "load_command.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "cli/utils/format_utils.h"

#include "compute/core/compute_backend.h"
#include "compute/model/language_model.h"
#include "compute/core/compute_types.h"

#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
#include "compute/backends/mlx/mlx_backend.h"
#endif

namespace neurons::cli {

LoadCommand::LoadCommand(NeuronsConfig* config)
    : verbose_(false)
    , dry_run_(false)
    , config_(config)
{
    command_name_ = "load";
    description_  = "Load a model and run inference";

    model_registry_ = std::make_unique<models::registry::ModelRegistry>(
        config_->modelsDirectory().string(),
        nullptr
    );
}

void LoadCommand::setup_command(CLI::App& app) {
    auto* load_cmd = app.add_subcommand(command_name_, description_);

    load_cmd->add_option("model", model_name_, "Model name (e.g., mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit)")
        ->required();

    load_cmd->add_option("-p,--prompt", prompt_, "Prompt to run inference on");
    load_cmd->add_option("--max-tokens", max_tokens_, "Maximum new tokens to generate (default: 4096)");
    load_cmd->add_option("--temperature", temperature_, "Sampling temperature (default: 0.8)");
    load_cmd->add_option("--top-k", top_k_, "Top-k sampling (default: 50, 0=disabled)");
    load_cmd->add_option("--top-p", top_p_, "Top-p nucleus sampling threshold (default: 1.0=disabled)");
    load_cmd->add_option("--rep-penalty", rep_penalty_, "Repetition penalty >1 reduces repeated tokens (default: 1.0=disabled)");
    load_cmd->add_flag("-v,--verbose", verbose_, "Show detailed model information");
    load_cmd->add_flag("--dry-run", dry_run_, "Show what would be loaded without running inference");

    load_cmd->callback([this]() {
        std::exit(execute());
    });
}

int LoadCommand::execute() {
    if (model_name_.empty()) {
        std::cerr << "Error: Model name is required\n";
        print_usage_help();
        return 1;
    }

    const auto location = model_registry_->locateModel(model_name_);
    if (!location.isValid()) {
        std::cerr << "Error: Model '" << model_name_ << "' not found in "
                  << config_->modelsDirectory() << "\n";
        std::cerr << "Run 'neurons list' to see available models\n";
        return 1;
    }

    if (verbose_) {
        print_model_info(location);
    }

    if (dry_run_) {
        if (!verbose_) print_model_info(location);
        std::cout << "\n[DRY RUN] --dry-run specified, skipping inference\n";
        return 0;
    }

    if (prompt_.empty()) {
        // No prompt — just print model info
        print_model_info(location);
        std::cout << "\nReady. Use --prompt \"<text>\" to run inference.\n";
        return 0;
    }

    return run_inference(location.modelPath);
}

int LoadCommand::run_inference(const std::string& model_path) {
#if defined(__APPLE__) && defined(__aarch64__) && defined(MLX_BACKEND_ENABLED)
    // Create and initialize backend
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

    // Load model — auto-dispatches to LlamaModel / MistralModel / … based on config.json.
    auto inf_result = compute::LanguageModel::load(model_path, backend.get());
    if (!inf_result) {
        std::cerr << "\nError loading model: " << inf_result.error().message << "\n";
        return 1;
    }
    auto inference = std::move(*inf_result);

    auto t1 = std::chrono::steady_clock::now();
    double load_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << load_secs << "s)\n";

    // Chat template selection must mirror ChatEngine::buildPrompt() so CLI
    // output faithfully reproduces what the GUI user will see. If you change
    // this, change ChatEngine::buildPrompt() too (or factor both into a
    // shared helper).
    const std::string model_type = inference->model_type();
    const bool is_llama3 = (model_type == "llama") &&
        (inference->tokenizer().find_token_id("<|start_header_id|>") != -1);
    std::string formatted;
    if (is_llama3) {
        // Llama-3 chat template (identified by presence of header tokens in vocab).
        // BOS (<|begin_of_text|>) is embedded in the template — add_bos_token is null
        // in tokenizer_config.json, so BOS is not auto-prepended by the tokenizer.
        formatted =
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n" + prompt_ + "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (model_type == "qwen2" || model_type == "qwen3") {
        // Qwen2/Qwen3 ChatML template (<|im_start|>/<|im_end|> control tokens).
        formatted =
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + prompt_ + "<|im_end|>\n"
            "<|im_start|>assistant\n";
    } else if (model_type == "mistral") {
        // Mistral v0.3 was not trained with system prompts.
        formatted = "[INST] " + prompt_ + " [/INST]";
    } else if (model_type == "gemma" || model_type == "gemma2" || model_type == "gemma3_text") {
        // Gemma chat template: <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
        // BOS (<bos>) is prepended by the tokenizer (add_special_tokens=true).
        formatted =
            "<start_of_turn>user\n" + prompt_ + "<end_of_turn>\n"
            "<start_of_turn>model\n";
    } else {
        // TinyLlama / Llama-2 chat template
        formatted =
            "<|system|>\nYou are a helpful assistant.</s>\n"
            "<|user|>\n" + prompt_ + "</s>\n"
            "<|assistant|>\n";
    }
    std::cout << "Model type: " << model_type << "\n";

    auto token_ids = inference->tokenizer().encode(formatted, /*add_special_tokens=*/true);
    std::cout << "Prompt tokens: " << token_ids.size() << "\n\n";

    // Stream generation: decode the cumulative sequence each step and print
    // the new suffix. This correctly handles ▁ (SentencePiece space marker)
    // across token boundaries without needing per-token context.
    auto& tok = inference->tokenizer();
    const auto& inf_cfg = inference->config();
    int token_count = 0;
    std::vector<int> generated_so_far;
    std::string decoded_so_far;

    // F.14.3: thinking block state for reasoning models (e.g. Qwen3).
    // Generated text begins with <think>...reasoning...</think>\n\nanswer.
    // We stream reasoning content dimmed and print the answer normally.
    const bool is_reasoning = inference->is_reasoning_model();
    bool in_think = false;
    size_t print_pos = 0;  // offset into decoded_so_far already sent to stdout

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
        static const size_t HOLD = CLOSE.size() - 1;  // hold back to avoid split tag at boundary

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
                std::cout << "\033[2m" << std::flush;  // dim on — start of thinking block
                print_pos = pos + OPEN.size();
                in_think = true;
            } else {
                size_t pos = decoded_so_far.find(CLOSE, print_pos);
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
                std::cout << "\033[0m\n" << std::flush;  // dim off + separator after thinking
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

    auto gen_result = inference->generate(
        token_ids,
        static_cast<size_t>(max_tokens_),
        params,
        [&](int token_id) -> bool {
            ++token_count;
            if (inf_cfg.is_eos_token(token_id)) return false;  // stop — don't print EOS
            generated_so_far.push_back(token_id);
            decoded_so_far = tok.decode(generated_so_far, /*skip_special_tokens=*/true);
            flush_decoded();
            return true;
        }
    );

    // Flush any held-back characters (partial tag guard) and reset ANSI state.
    print_pos = decoded_so_far.size() > 0 ? print_pos : 0;
    if (print_pos < decoded_so_far.size())
        std::cout << decoded_so_far.substr(print_pos) << std::flush;
    if (in_think)
        std::cout << "\033[0m" << std::flush;  // ensure dim is off if generation stopped mid-think

    std::cout << "\n";

    if (!gen_result) {
        std::cerr << "Error during generation: " << gen_result.error().message << "\n";
        return 1;
    }

    auto t_gen_end = std::chrono::steady_clock::now();
    double gen_secs = std::chrono::duration<double>(t_gen_end - t_gen_start).count();
    double toks_per_sec = token_count > 0 ? token_count / gen_secs : 0.0;

    std::cout << "\n[" << token_count << " tokens | "
              << std::fixed << std::setprecision(1) << toks_per_sec << " tok/s]\n";

    backend->cleanup();
    return 0;

#else
    (void)model_path;
    std::cerr << "Error: Inference requires Apple Silicon with MLX backend\n";
    return 1;
#endif
}

void LoadCommand::print_model_info(const models::registry::ModelLocation& location) {
    std::cout << "\n=== Model Information ===\n";
    std::cout << "Name: " << location.modelName << "\n";
    std::cout << "Path: " << location.modelPath << "\n";

    std::string formatStr;
    switch (location.format) {
    case models::registry::ModelFormat::SAFETENSORS: formatStr = "SafeTensors"; break;
    case models::registry::ModelFormat::GGUF:        formatStr = "GGUF"; break;
    case models::registry::ModelFormat::UNKNOWN:     formatStr = "Unknown"; break;
    }
    std::cout << "Format: " << formatStr << "\n";

    std::cout << "\nModel Files (" << location.modelFiles.size() << "):\n";
    int64_t totalSize = 0;
    for (const std::string& file : location.modelFiles) {
        std::filesystem::path fp(file);
        int64_t size = 0;
        try { size = std::filesystem::file_size(fp); } catch (...) {}
        totalSize += size;
        std::cout << "  - " << fp.filename().string();
        if (verbose_) std::cout << " (" << utils::formatFileSize(size) << ")";
        std::cout << "\n";
    }
    if (location.modelFiles.size() > 1 || verbose_) {
        std::cout << "  Total size: " << utils::formatFileSize(totalSize) << "\n";
    }

    if (location.format == models::registry::ModelFormat::SAFETENSORS) {
        std::cout << "\nConfiguration Files:\n";
        if (!location.configPath.empty())
            std::cout << "  - config.json\n";
        if (!location.tokenizerConfigPath.empty())
            std::cout << "  - tokenizer_config.json\n";
        if (!location.vocabPath.empty())
            std::cout << "  - " << std::filesystem::path(location.vocabPath).filename().string() << "\n";
    }
}

void LoadCommand::print_usage_help() {
    std::cout << "\nUsage: neurons load <model_name> [options]\n\n";
    std::cout << "Examples:\n";
    std::cout << "  neurons load mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit\n";
    std::cout << "  neurons load mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit --prompt \"What is the capital of France?\"\n";
    std::cout << "  neurons load mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit --prompt \"Hello\" --temperature 0.5 --max-tokens 100\n";
}

} // namespace neurons::cli