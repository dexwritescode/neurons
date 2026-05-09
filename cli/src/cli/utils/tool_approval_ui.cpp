#include "tool_approval_ui.h"

#include <iostream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <nlohmann/json.hpp>

namespace neurons::cli {

int show_tool_approval_ui(const std::string& tool,
                          const std::string& server,
                          const std::string& args_json) {
    using namespace ftxui;

    // Build a readable one-line command summary from the args JSON.
    std::string cmd = tool;
    try {
        auto args = nlohmann::json::parse(args_json);
        for (auto& [k, v] : args.items())
            if (v.is_string()) cmd += " " + v.get<std::string>();
    } catch (...) {
        cmd += " " + args_json;
    }

    std::vector<std::string> entries = {
        "Allow once", "Always allow", "Deny once", "Always deny"};
    int selected = 0;

    auto menu = Menu(&entries, &selected);
    auto screen = ScreenInteractive::TerminalOutput();

    // CatchEvent wraps menu directly so arrow-key events reach it without
    // passing through an intermediate Renderer that has no OnEvent override.
    auto event_handler = CatchEvent(menu, [&](Event event) {
        if (event == Event::Return) {
            screen.ExitLoopClosure()();
            return true;
        }
        return false;
    });

    auto ui = Renderer(event_handler, [&] {
        return vbox({
            window(
                text(" Tool Approval ") | bold,
                vbox({
                    hbox({text(" Tool:   ") | dim, text(tool)   | bold}),
                    hbox({text(" Server: ") | dim, text(server) | dim}),
                    separator(),
                    hbox({text(" $ ")      | dim, paragraph(cmd) | color(Color::Yellow)}),
                })
            ),
            menu->Render() | border,
            text(" ↑/↓  navigate   Enter  confirm ") | dim | center,
        });
    });

    // Redirect STDIN_FILENO to /dev/tty before entering FTXUI.
    //
    // After line-based REPL input (std::getline / std::cin), the C++ iostream
    // layer may have buffered state that confuses FTXUI's select()+read() loop.
    // Opening /dev/tty directly gives FTXUI a clean terminal file description
    // to set raw mode on and read keystrokes from, regardless of what std::cin
    // has done to the inherited stdin fd.
    //
    // On macOS /dev/tty is the process's controlling terminal — the same
    // physical device as interactive stdin — so tcgetattr/tcsetattr operate on
    // the correct terminal and terminal-mode save/restore is sound.
#if !defined(_WIN32)
    int saved_stdin = dup(STDIN_FILENO);
    int tty_fd = open("/dev/tty", O_RDWR | O_CLOEXEC);
    if (tty_fd != -1) {
        dup2(tty_fd, STDIN_FILENO);
        close(tty_fd);
    }
#endif

    std::cout << std::flush;
    screen.Loop(ui);

#if !defined(_WIN32)
    if (saved_stdin != -1) {
        dup2(saved_stdin, STDIN_FILENO);
        close(saved_stdin);
    }
#endif

    return selected;
}

} // namespace neurons::cli
