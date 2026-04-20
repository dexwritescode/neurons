#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace neurons_service {

struct LogEntry {
    int64_t     timestamp_ms;
    std::string level;   // "INFO" | "WARN" | "ERROR"
    std::string message;
};

class Logger {
public:
    static Logger& global();

    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);

    // Snapshot of the most recent `n` entries (thread-safe).
    std::vector<LogEntry> recent(size_t n = 200) const;

    // Subscribe: callback is invoked on the thread that calls info/warn/error.
    // Returns a subscriber id usable with unsubscribe().
    using Subscriber = std::function<void(const LogEntry&)>;
    uint64_t subscribe(Subscriber cb);
    void unsubscribe(uint64_t id);

private:
    void append(const std::string& level, const std::string& msg);

    mutable std::mutex               mutex_;
    std::deque<LogEntry>             ring_;
    static constexpr size_t          kRingMax = 2000;
    std::map<uint64_t, Subscriber>   subscribers_;
    uint64_t                         next_id_{1};
};

} // namespace neurons_service
