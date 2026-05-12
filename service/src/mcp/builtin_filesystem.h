#pragma once

#include "mcp_types.h"
#include <string>
#include <vector>

namespace neurons_service {

// Built-in filesystem MCP server.
// Tools: read_file (always_allow), write_file (always_ask).
namespace BuiltinFilesystem {

std::vector<ToolDef> tool_defs();

std::string handle(const std::string& tool,
                   const std::string& args_json,
                   std::string&       result_out);

} // namespace BuiltinFilesystem
} // namespace neurons_service
