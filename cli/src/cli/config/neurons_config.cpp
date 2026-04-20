#include "neurons_config.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace neurons::cli {

NeuronsConfig::NeuronsConfig() {
    neurons_dir_ = defaultNeuronsDir();
    load();
    ensureDirectoriesExist();
}

std::filesystem::path NeuronsConfig::defaultNeuronsDir() {
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (home) return std::filesystem::path(home) / ".neurons";
    return std::filesystem::path(".neurons");
}

std::filesystem::path NeuronsConfig::configFilePath() {
    return defaultNeuronsDir() / "config.json";
}

std::filesystem::path NeuronsConfig::neuronsDir() const {
    return neurons_dir_;
}

void NeuronsConfig::setNeuronsDir(const std::filesystem::path& dir) {
    neurons_dir_ = dir;
    ensureDirectoriesExist();
}

std::filesystem::path NeuronsConfig::modelsDirectory() const {
    // NEURONS_MODELS env var overrides
    const char* env = std::getenv("NEURONS_MODELS");
    if (env && env[0] != '\0') return std::filesystem::path(env);
    return neurons_dir_ / "models";
}

std::filesystem::path NeuronsConfig::chatsDirectory() const {
    return neurons_dir_ / "chats";
}

std::string NeuronsConfig::hfToken() const {
    return hf_token_;
}

void NeuronsConfig::setHfToken(const std::string& token) {
    hf_token_ = token;
}

void NeuronsConfig::clearHfToken() {
    hf_token_.clear();
}

std::string NeuronsConfig::activeNodeId() const {
    return active_node_id_;
}

void NeuronsConfig::setActiveNodeId(const std::string& id) {
    active_node_id_ = id;
}

const std::vector<NodeConfig>& NeuronsConfig::nodes() const {
    return nodes_;
}

std::optional<NodeConfig> NeuronsConfig::findNode(const std::string& id) const {
    for (const auto& n : nodes_) {
        if (n.id == id) return n;
    }
    return std::nullopt;
}

void NeuronsConfig::addNode(const NodeConfig& node) {
    for (auto& n : nodes_) {
        if (n.id == node.id) { n = node; return; }
    }
    nodes_.push_back(node);
}

bool NeuronsConfig::removeNode(const std::string& id) {
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
                           [&](const NodeConfig& n) { return n.id == id; });
    if (it == nodes_.end()) return false;
    nodes_.erase(it);
    if (active_node_id_ == id) active_node_id_.clear();
    return true;
}

void NeuronsConfig::updateNode(const NodeConfig& node) {
    addNode(node);
}

bool NeuronsConfig::load() {
    auto path = configFilePath();
    if (!std::filesystem::exists(path)) return true; // not an error

    try {
        std::ifstream f(path);
        auto j = json::parse(f);

        if (j.contains("dir") && !j["dir"].get<std::string>().empty()) {
            neurons_dir_ = std::filesystem::path(j["dir"].get<std::string>());
        }
        if (j.contains("hf_token")) hf_token_ = j["hf_token"].get<std::string>();
        if (j.contains("active_node_id")) active_node_id_ = j["active_node_id"].get<std::string>();

        if (j.contains("nodes") && j["nodes"].is_array()) {
            nodes_.clear();
            for (const auto& jn : j["nodes"]) {
                NodeConfig n;
                n.id   = jn.value("id", "");
                n.name = jn.value("name", "");
                n.host = jn.value("host", "localhost");
                n.port = jn.value("port", 50051);
                n.hf_token = jn.value("hf_token", "");
                if (!n.id.empty()) nodes_.push_back(n);
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not read config: " << e.what() << "\n";
        return false;
    }
}

bool NeuronsConfig::save() const {
    auto path = configFilePath();
    try {
        std::filesystem::create_directories(path.parent_path());

        json j;
        j["dir"] = neurons_dir_.string();
        j["hf_token"] = hf_token_;
        j["active_node_id"] = active_node_id_;

        json jnodes = json::array();
        for (const auto& n : nodes_) {
            json jn;
            jn["id"]   = n.id;
            jn["name"] = n.name;
            jn["host"] = n.host;
            jn["port"] = n.port;
            if (!n.hf_token.empty()) jn["hf_token"] = n.hf_token;
            jnodes.push_back(jn);
        }
        j["nodes"] = jnodes;

        std::ofstream f(path);
        f << j.dump(2) << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << "\n";
        return false;
    }
}

void NeuronsConfig::ensureDirectoriesExist() const {
    try {
        std::filesystem::create_directories(neurons_dir_);
        std::filesystem::create_directories(modelsDirectory());
        std::filesystem::create_directories(chatsDirectory());
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not create directories: " << e.what() << "\n";
    }
}

} // namespace neurons::cli
