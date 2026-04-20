#pragma once

#include <string>
#include <vector>
#include <optional>
#include <filesystem>

namespace neurons::cli {

struct NodeConfig {
    std::string id;
    std::string name;
    std::string host;
    int port = 50051;
    std::string hf_token; // optional per-node override
};

class NeuronsConfig {
public:
    NeuronsConfig();
    ~NeuronsConfig() = default;

    // Root directory (~/.neurons by default)
    std::filesystem::path neuronsDir() const;
    void setNeuronsDir(const std::filesystem::path& dir);

    // Derived directories
    std::filesystem::path modelsDirectory() const;
    std::filesystem::path chatsDirectory() const;

    // HuggingFace token
    std::string hfToken() const;
    void setHfToken(const std::string& token);
    void clearHfToken();

    // Active node
    std::string activeNodeId() const;
    void setActiveNodeId(const std::string& id);

    // Node management
    const std::vector<NodeConfig>& nodes() const;
    std::optional<NodeConfig> findNode(const std::string& id) const;
    void addNode(const NodeConfig& node);
    bool removeNode(const std::string& id);
    void updateNode(const NodeConfig& node);

    // Persistence
    bool load();
    bool save() const;

    static std::filesystem::path defaultNeuronsDir();
    static std::filesystem::path configFilePath();

private:
    std::filesystem::path neurons_dir_;
    std::string hf_token_;
    std::string active_node_id_;
    std::vector<NodeConfig> nodes_;

    void ensureDirectoriesExist() const;
};

} // namespace neurons::cli
