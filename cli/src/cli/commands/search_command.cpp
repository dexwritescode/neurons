#include "search_command.h"
#include <iostream>
#include <iomanip>
#include <future>
#include <sstream>

namespace neurons::cli {

static std::string fmt_downloads(uint64_t n) {
    std::ostringstream ss;
    if (n >= 1'000'000) ss << std::fixed << std::setprecision(1) << n / 1'000'000.0 << "M";
    else if (n >= 1'000) ss << std::fixed << std::setprecision(0) << n / 1'000.0 << "K";
    else ss << n;
    return ss.str();
}

SearchCommand::SearchCommand(NeuronsConfig* config) : config_(config) {}

void SearchCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand("search", "Search HuggingFace for models");

    cmd->add_option("query", query_, "Search terms (model name, author, keywords)")
       ->required();

    cmd->add_option("-n,--limit", limit_,
        "Number of results (default: 20, max ~100)")
       ->default_val(20);

    cmd->add_option("-s,--sort", sort_,
        "Sort order: downloads (default), likes, trending, lastModified")
       ->default_val("downloads");

    cmd->add_option("-p,--pipeline", pipeline_tags_,
        "Filter by pipeline tag (repeatable). e.g. --pipeline text-generation\n"
        "  Common values: text-generation, text2text-generation,\n"
        "                 question-answering, summarization, translation")
       ->allow_extra_args(false);

    cmd->add_option("-a,--author", author_,
        "Filter by author/organisation (e.g. mlx-community, google, meta-llama)");

    cmd->add_flag("-g,--gated", show_gated_,
        "Show gated models (requires HF token to download)");

    cmd->callback([this]() { execute(); });
}

int SearchCommand::execute() {
    auto hf = models::createHuggingFaceClientSync();
    hf->client()->setDownloadDirectory(config_->modelsDirectory().string());

    models::SearchQuery q;
    q.search    = query_;
    q.sort      = sort_.empty() ? "downloads" : sort_;
    q.limit     = limit_;
    q.full      = true;
    q.author    = author_;

    for (const auto& tag : pipeline_tags_) {
        q.pipelineTags.push_back(tag);
    }

    std::promise<std::pair<std::vector<models::ModelInfo>, std::string>> promise;
    auto future = promise.get_future();

    hf->client()->setSearchCallback(
        [&promise](const std::vector<models::ModelInfo>& models,
                   const std::string& /*nextPage*/,
                   const std::string& err) {
            promise.set_value({models, err});
        });
    hf->client()->searchModels(q);

    auto [results, error] = future.get();

    if (!error.empty()) {
        std::cerr << "Search error: " << error << std::endl;
        return 1;
    }

    if (results.empty()) {
        std::cout << "No results for \"" << query_ << "\"." << std::endl;
        return 0;
    }

    // Header
    std::cout << "\nSearch results for \"" << query_ << "\""
              << " (sorted by " << q.sort << "):\n\n";

    // Column widths
    const int idW     = 55;
    const int dlW     = 10;
    const int gatedW  = 7;

    std::cout << std::left
              << std::setw(idW)    << "Model ID"
              << std::setw(dlW)    << "Downloads"
              << std::setw(gatedW) << "Gated"
              << "\n"
              << std::string(idW + dlW + gatedW, '-') << "\n";

    int shown = 0;
    for (const auto& m : results) {
        if (m.gated && !show_gated_) continue;

        std::string id = m.id;
        if (static_cast<int>(id.size()) > idW - 2)
            id = id.substr(0, idW - 5) + "...";

        std::cout << std::left
                  << std::setw(idW)    << id
                  << std::setw(dlW)    << fmt_downloads(m.downloads)
                  << std::setw(gatedW) << (m.gated ? "yes" : "")
                  << "\n";
        ++shown;
    }

    if (shown == 0 && !results.empty()) {
        std::cout << "(all " << results.size()
                  << " results are gated — re-run with --gated to show them)\n";
    }

    std::cout << "\n" << shown << " result(s). "
              << "Download with: neurons download <model-id>\n";
    return 0;
}

void SearchCommand::print_usage_help() {
    std::cout << "Usage: neurons search <query> [options]\n\n"
              << "Examples:\n"
              << "  neurons search llama                        # keyword search\n"
              << "  neurons search \"\" -a mlx-community          # all mlx-community models\n"
              << "  neurons search mistral -p text-generation   # text-gen only\n"
              << "  neurons search qwen -s trending -n 10       # top 10 trending\n";
}

} // namespace neurons::cli
