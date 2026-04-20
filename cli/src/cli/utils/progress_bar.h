#pragma once

#include <indicators/progress_bar.hpp>
#include <memory>
#include <string>
#include <cstdint>

namespace neurons::cli {

class ProgressBar {
public:
    explicit ProgressBar(const std::string& description = "");
    ~ProgressBar() = default;

    void set_description(const std::string& description);
    void start();
    void finish();

    void update_progress(int percentage);
    void update_progress(int64_t bytesReceived, int64_t bytesTotal);
    void update_byte_display(int64_t bytesReceived, int64_t bytesTotal);
    void set_status(const std::string& status);

private:
    std::unique_ptr<indicators::ProgressBar> progress_bar_;
    std::string description_;
    bool is_active_;
    bool max_progress_set_;
};

}