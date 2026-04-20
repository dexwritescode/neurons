#include "logger.h"

#include <chrono>
#include <iostream>
#include <vector>

namespace neurons_service {

static int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

Logger& Logger::global() {
    static Logger instance;
    return instance;
}

void Logger::info(const std::string& msg)  { append("INFO",  msg); }
void Logger::warn(const std::string& msg)  { append("WARN",  msg); }
void Logger::error(const std::string& msg) { append("ERROR", msg); }

void Logger::append(const std::string& level, const std::string& msg) {
    LogEntry e{nowMs(), level, msg};

    // Print to stderr so the service is usable without a subscriber.
    std::cerr << "[" << level << "] " << msg << "\n";

    std::vector<Subscriber> subs;
    {
        std::lock_guard lock(mutex_);
        ring_.push_back(e);
        if (ring_.size() > kRingMax) ring_.pop_front();
        for (const auto& [id, cb] : subscribers_) subs.push_back(cb);
    }
    for (const auto& cb : subs) cb(e);
}

std::vector<LogEntry> Logger::recent(size_t n) const {
    std::lock_guard lock(mutex_);
    size_t start = ring_.size() > n ? ring_.size() - n : 0;
    return {ring_.begin() + static_cast<ptrdiff_t>(start), ring_.end()};
}

uint64_t Logger::subscribe(Subscriber cb) {
    std::lock_guard lock(mutex_);
    uint64_t id = next_id_++;
    subscribers_[id] = std::move(cb);
    return id;
}

void Logger::unsubscribe(uint64_t id) {
    std::lock_guard lock(mutex_);
    subscribers_.erase(id);
}

} // namespace neurons_service
