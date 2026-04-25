#include "builtin_shell.h"

#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace neurons_service {
namespace BuiltinShell {

static constexpr std::size_t kMaxOutputBytes = 8 * 1024; // 8 KB

// ── Tool definitions ──────────────────────────────────────────────────────────

std::vector<ToolDef> tool_defs() {
    return {{
        "neurons-shell",
        "run_command",
        "Run a shell command and return its output (stdout + stderr combined). "
        "Output is capped at 8 KB. Use with care — this executes on the host system.",
        R"SCHEMA({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (default: cwd)"
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30, max: 120)"
                }
            },
            "required": ["command"]
        })SCHEMA"
    }};
}

// ── Tool implementation ───────────────────────────────────────────────────────

static std::string run_command(const nlohmann::json& args) {
    const auto command   = args.value("command", "");
    const auto cwd       = args.value("cwd", "");
    const int  timeout_s = std::min(args.value("timeout_s", 30), 120);

    if (command.empty())
        return R"({"error":"'command' is required"})";

    // Change to requested working directory for this command.
    std::string full_cmd = command;
    if (!cwd.empty()) {
        // Prefix cd so the command runs in the right directory.
        // Single-shell execution: errors in cd surface in output.
        full_cmd = "cd " + cwd + " && " + command;
    }

    // Add timeout wrapper and merge stderr into stdout.
    const std::string timed = "timeout " + std::to_string(timeout_s)
                            + " sh -c " + nlohmann::json(full_cmd).dump()
                            + " 2>&1";

    FILE* pipe = popen(timed.c_str(), "r");
    if (!pipe)
        return R"({"error":"Failed to start command"})";

    std::string output;
    output.reserve(kMaxOutputBytes);
    std::array<char, 512> buf{};
    bool truncated = false;

    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        const std::size_t remaining = kMaxOutputBytes - output.size();
        if (remaining == 0) { truncated = true; break; }
        const std::string_view chunk(buf.data());
        output.append(chunk.substr(0, std::min(chunk.size(), remaining)));
    }
    // Drain pipe to avoid SIGPIPE if we stopped early.
    if (truncated) while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {}

    const int status = pclose(pipe);
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    nlohmann::json out;
    out["output"]    = output;
    out["exit_code"] = exit_code;
    out["truncated"] = truncated;
    if (truncated) out["truncation_note"] = "Output exceeded 8 KB and was truncated.";
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

    if (tool == "run_command") result_out = run_command(args);
    else return "Unknown tool: " + tool;

    return "";
}

} // namespace BuiltinShell
} // namespace neurons_service
