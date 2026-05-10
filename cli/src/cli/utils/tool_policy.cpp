#include "tool_policy.h"

#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace neurons::cli {

ToolPolicy parse_tool_policy(const std::string& s) {
    if (s == "allow") return ToolPolicy::Allow;
    if (s == "deny")  return ToolPolicy::Deny;
    if (s == "ask")   return ToolPolicy::Ask;
    throw std::invalid_argument(
        "Unknown tool policy '" + s + "'. Valid values: allow, deny, ask");
}

bool resolve_tool_approval(const std::string& tool,
                           const std::string& server,
                           const std::string& args_json,
                           ToolPolicy policy) {
    switch (policy) {
        case ToolPolicy::Allow: return true;
        case ToolPolicy::Deny:  return false;
        case ToolPolicy::Ask: {
            std::string preview = args_json;
            try {
                auto args = nlohmann::json::parse(args_json);
                for (auto& [k, v] : args.items())
                    if (v.is_string()) { preview = v.get<std::string>(); break; }
            } catch (...) {}
            std::cout << "\n[tool] " << server << "/" << tool
                      << " \"" << preview << "\"  Allow? [y/N]: " << std::flush;
            std::string ans;
            std::getline(std::cin, ans);
            return (!ans.empty() && (ans[0] == 'y' || ans[0] == 'Y'));
        }
    }
    return false;
}

} // namespace neurons::cli
