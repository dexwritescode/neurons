#include "progress_bar.h"
#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <iomanip>

namespace neurons::cli {

std::string formatBytes(int64_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(unitIndex == 0 ? 0 : 1) << size << " " << units[unitIndex];
    return oss.str();
}

ProgressBar::ProgressBar(const std::string& description)
    : description_(description)
    , is_active_(false)
    , max_progress_set_(false) {

    // Force stdout to be treated as a TTY for progress bar updates
    bool isATty = ::isatty(fileno(stdout));
    if (!isATty) {
        // If you're running under Qt Creator's "Application Output" pane or similar,
        // this will help the indicators library work properly
        std::cout.flush();
    }
}

void ProgressBar::set_description(const std::string& description) {
    description_ = description;
}

void ProgressBar::start() {
    if (is_active_) return;

    // Hide cursor and enable in-place updates
    indicators::show_console_cursor(false);

    progress_bar_ = std::make_unique<indicators::ProgressBar>(
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"█"},
        indicators::option::Lead{"█"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PrefixText{description_},
        indicators::option::ForegroundColor{indicators::Color::green},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{false},
        indicators::option::FontStyles{std::vector{indicators::FontStyle::bold}},
        indicators::option::ShowPercentage{true},
        indicators::option::MaxProgress{100}
    );

    is_active_ = true;
}

void ProgressBar::finish() {
    if (!is_active_ || !progress_bar_) return;

    progress_bar_->set_progress(100);
    progress_bar_->mark_as_completed();

    // Restore cursor and add final newline
    indicators::show_console_cursor(true);
    std::cout << std::endl;

    is_active_ = false;
}

void ProgressBar::update_progress(int percentage) {
    if (!is_active_ || !progress_bar_) return;

    progress_bar_->set_progress(static_cast<size_t>(std::max(0, std::min(100, percentage))));
}

void ProgressBar::update_progress(const int64_t bytesReceived, const int64_t bytesTotal) {
    if (!is_active_ || !progress_bar_ || bytesTotal <= 0) return;

    // For the first update, set the max progress to total bytes
    if (!max_progress_set_ && bytesTotal > 0) {
        progress_bar_->set_option(indicators::option::MaxProgress{static_cast<size_t>(bytesTotal)});
        max_progress_set_ = true;
    }

    // Update progress with actual bytes for better time estimation
    const size_t progressValue = bytesReceived;

    // Add human-readable byte display FIRST, before setting progress
    const std::string bytesText = formatBytes(bytesReceived) + "/" + formatBytes(bytesTotal);
    progress_bar_->set_option(indicators::option::PostfixText{" " + bytesText});

    progress_bar_->set_progress(progressValue);
}

void ProgressBar::update_byte_display(const int64_t bytesReceived, const int64_t bytesTotal) {
    if (!is_active_ || !progress_bar_ || bytesTotal <= 0) return;

    // Update the byte display text and trigger refresh
    const std::string bytesText = formatBytes(bytesReceived) + "/" + formatBytes(bytesTotal);
    progress_bar_->set_option(indicators::option::PostfixText{" " + bytesText});
}

void ProgressBar::set_status(const std::string& status) {
    if (!is_active_ || !progress_bar_) return;

    progress_bar_->set_option(indicators::option::PostfixText{status});
}

}
