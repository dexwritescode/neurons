#pragma once

#include <string>

namespace neurons::cli {

// Show an interactive FTXUI approval menu for a tool call.
// Returns: 0 = Allow once, 1 = Always allow, 2 = Deny once, 3 = Always deny.
int show_tool_approval_ui(const std::string& tool,
                          const std::string& server,
                          const std::string& args_json);

} // namespace neurons::cli
