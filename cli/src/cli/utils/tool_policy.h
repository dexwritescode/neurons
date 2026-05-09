#pragma once

#include <string>

namespace neurons::cli {

enum class ToolPolicy { Allow, Deny, Ask };

// Parse from a CLI string value. Throws std::invalid_argument on unknown input.
ToolPolicy parse_tool_policy(const std::string& s);

// Evaluate a pending tool call against the policy.
// ToolPolicy::Ask: prints a one-line y/N prompt to stdout and reads from stdin.
// Returns true if the call should proceed.
bool resolve_tool_approval(const std::string& tool,
                           const std::string& server,
                           const std::string& args_json,
                           ToolPolicy policy);

} // namespace neurons::cli
