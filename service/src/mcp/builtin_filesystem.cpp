#include "builtin_filesystem.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <fnmatch.h>
#include <sstream>

namespace fs = std::filesystem;

namespace neurons_service {
namespace BuiltinFilesystem {

static constexpr std::size_t kMaxReadBytes = 1024 * 1024; // 1 MB

// ── Tool definitions ──────────────────────────────────────────────────────────

std::vector<ToolDef> tool_defs() {
    return {
        {
            "neurons-filesystem",
            "read_file",
            "Read the complete contents of a file at the given path.",
            R"({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"}
                },
                "required": ["path"]
            })"
        },
        {
            "neurons-filesystem",
            "write_file",
            "Write content to a file, creating it and any parent directories if needed.",
            R"({
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Text content to write"}
                },
                "required": ["path", "content"]
            })"
        },
        {
            "neurons-filesystem",
            "list_dir",
            "List the entries of a directory. Returns name, type (file/dir), and size for each entry.",
            R"({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"]
            })"
        },
        {
            "neurons-filesystem",
            "search_files",
            "Recursively search for files whose names match a glob pattern within a directory.",
            R"SCHEMA({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.cpp'"},
                    "root":    {"type": "string", "description": "Directory to search (default: current dir)"}
                },
                "required": ["pattern"]
            })SCHEMA"
        },
    };
}

// ── Tool implementations ──────────────────────────────────────────────────────

static std::string read_file(const nlohmann::json& args) {
    const auto path = args.value("path", "");
    if (path.empty()) return "";  // error returned via result

    std::error_code ec;
    if (!fs::exists(path, ec) || ec)
        return R"({"error":"File not found: )" + path + R"("})";
    if (!fs::is_regular_file(path, ec) || ec)
        return R"({"error":"Not a regular file: )" + path + R"("})";

    const auto size = fs::file_size(path, ec);
    if (ec) return R"({"error":"Cannot stat file"})";
    if (size > kMaxReadBytes)
        return R"ERR({"error":"File too large (> 1 MB)"})ERR";

    std::ifstream f(path, std::ios::binary);
    if (!f) return R"({"error":"Cannot open file"})";

    std::ostringstream buf;
    buf << f.rdbuf();
    nlohmann::json out;
    out["content"] = buf.str();
    out["path"]    = path;
    out["size"]    = size;
    return out.dump();
}

static std::string write_file(const nlohmann::json& args) {
    const auto path    = args.value("path", "");
    const auto content = args.value("content", "");
    if (path.empty()) return R"({"error":"'path' is required"})";

    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    // Ignore ec — parent may already exist.

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return R"({"error":"Cannot write file: )" + path + R"("})";
    f << content;
    if (!f) return R"({"error":"Write failed"})";

    nlohmann::json out;
    out["written"] = content.size();
    out["path"]    = path;
    return out.dump();
}

static std::string list_dir(const nlohmann::json& args) {
    const auto path = args.value("path", ".");
    std::error_code ec;
    if (!fs::exists(path, ec) || ec)
        return R"({"error":"Directory not found: )" + path + R"("})";
    if (!fs::is_directory(path, ec) || ec)
        return R"({"error":"Not a directory: )" + path + R"("})";

    nlohmann::json entries = nlohmann::json::array();
    for (const auto& entry : fs::directory_iterator(path, ec)) {
        if (ec) break;
        nlohmann::json e;
        e["name"] = entry.path().filename().string();
        e["type"] = entry.is_directory(ec) ? "dir" : "file";
        if (!entry.is_directory(ec)) {
            std::error_code se;
            e["size"] = entry.file_size(se);
        }
        entries.push_back(std::move(e));
    }
    nlohmann::json out;
    out["path"]    = path;
    out["entries"] = std::move(entries);
    return out.dump();
}

static std::string search_files(const nlohmann::json& args) {
    const auto pattern = args.value("pattern", "");
    const auto root    = args.value("root", ".");
    if (pattern.empty()) return R"({"error":"'pattern' is required"})";

    std::error_code ec;
    if (!fs::exists(root, ec) || ec)
        return R"({"error":"Root directory not found: )" + root + R"("})";

    nlohmann::json matches = nlohmann::json::array();
    for (const auto& entry : fs::recursive_directory_iterator(root,
             fs::directory_options::skip_permission_denied, ec)) {
        if (ec) { ec.clear(); continue; }
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (fnmatch(pattern.c_str(), name.c_str(), 0) == 0)
            matches.push_back(entry.path().string());
        if (matches.size() >= 500) break;  // safety cap
    }
    nlohmann::json out;
    out["pattern"] = pattern;
    out["root"]    = root;
    out["matches"] = std::move(matches);
    return out.dump();
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

std::string handle(const std::string& tool,
                   const std::string& args_json,
                   std::string&       result_out) {
    nlohmann::json args;
    try {
        args = args_json.empty() ? nlohmann::json::object()
                                 : nlohmann::json::parse(args_json);
    } catch (const nlohmann::json::exception& e) {
        return std::string("Invalid args JSON: ") + e.what();
    }

    if      (tool == "read_file")    result_out = read_file(args);
    else if (tool == "write_file")   result_out = write_file(args);
    else if (tool == "list_dir")     result_out = list_dir(args);
    else if (tool == "search_files") result_out = search_files(args);
    else return "Unknown tool: " + tool;

    return "";
}

} // namespace BuiltinFilesystem
} // namespace neurons_service
