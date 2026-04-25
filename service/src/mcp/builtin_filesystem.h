#pragma once

#include "mcp_types.h"
#include <string>
#include <vector>

namespace neurons_service {

// Built-in filesystem MCP server.
// Tools: read_file, write_file, list_dir, search_files.
// Default permissions: allow_session for reads, always_ask for writes.
namespace BuiltinFilesystem {

std::vector<ToolDef> tool_defs();

// Dispatch a tool call. Returns "" on success (result_out filled),
// or an error message on failure.
std::string handle(const std::string& tool,
                   const std::string& args_json,
                   std::string&       result_out);

} // namespace BuiltinFilesystem
} // namespace neurons_service
