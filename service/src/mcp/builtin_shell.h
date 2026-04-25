#pragma once

#include "mcp_types.h"
#include <string>
#include <vector>

namespace neurons_service {

// Built-in shell MCP server.
// Tools: run_command.
// Default permission: always_ask (always shown with destructive warning).
namespace BuiltinShell {

std::vector<ToolDef> tool_defs();

std::string handle(const std::string& tool,
                   const std::string& args_json,
                   std::string&       result_out);

} // namespace BuiltinShell
} // namespace neurons_service
