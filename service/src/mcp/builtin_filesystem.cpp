#include "builtin_filesystem.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace neurons_service {
namespace BuiltinFilesystem {

static constexpr std::size_t kMaxReadBytes = 512 * 1024; // 512 KB

// ── Tool definitions ──────────────────────────────────────────────────────────

std::vector<ToolDef> tool_defs() {
    return {
        {
            "neurons-filesystem",
            "read_file",
            "Read the contents of a file. Returns the file text.",
            R"SCHEMA({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file"
                    }
                },
                "required": ["path"]
            })SCHEMA"
        },
        {
            "neurons-filesystem",
            "write_file",
            "Write text content to a file, creating or overwriting it.",
            R"SCHEMA({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write"
                    }
                },
                "required": ["path", "content"]
            })SCHEMA"
        },
    };
}

// ── Tool implementations ──────────────────────────────────────────────────────

static std::string read_file(const nlohmann::json& args) {
    const auto path_str = args.value("path", "");
    if (path_str.empty())
        return R"({"error":"'path' is required"})";

    const fs::path path(path_str);
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec)
        return nlohmann::json{{"error", "Cannot access file: " + ec.message()}}.dump();

    std::ifstream f(path, std::ios::binary);
    if (!f)
        return nlohmann::json{{"error", "Cannot open file: " + path_str}}.dump();

    const auto read_size = std::min(static_cast<std::size_t>(size), kMaxReadBytes);
    std::string content(read_size, '\0');
    f.read(content.data(), static_cast<std::streamsize>(read_size));

    const bool truncated = size > kMaxReadBytes;
    nlohmann::json out;
    out["content"]   = content;
    out["truncated"] = truncated;
    if (truncated)
        out["truncation_note"] = "File exceeded 512 KB and was truncated.";
    return out.dump();
}

static std::string write_file(const nlohmann::json& args) {
    const auto path_str = args.value("path", "");
    const auto content  = args.value("content", "");
    if (path_str.empty())
        return R"({"error":"'path' is required"})";

    const fs::path path(path_str);
    std::error_code ec;
    if (path.has_parent_path())
        fs::create_directories(path.parent_path(), ec);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return nlohmann::json{{"error", "Cannot open file for writing: " + path_str}}.dump();

    f << content;
    if (!f)
        return nlohmann::json{{"error", "Write failed: " + path_str}}.dump();

    return nlohmann::json{{"success", true}, {"bytes_written", content.size()}}.dump();
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

    if      (tool == "read_file")  result_out = read_file(args);
    else if (tool == "write_file") result_out = write_file(args);
    else return "Unknown tool: " + tool;

    return "";
}

} // namespace BuiltinFilesystem
} // namespace neurons_service
