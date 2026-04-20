#pragma once

#include "logger.h"

#include <httplib.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace neurons_service {

class NeuronsServiceImpl;

class OpenAiHttpServer {
public:
    OpenAiHttpServer(NeuronsServiceImpl& service, Logger& logger);
    ~OpenAiHttpServer();

    // Starts the HTTP server on the given port in a background thread.
    // Returns false if the port is already in use.
    bool start(int port);

    // Stops the server (blocking until the background thread exits).
    void stop();

    bool is_running() const { return running_.load(); }
    int  port()       const { return port_; }

private:
    NeuronsServiceImpl&          service_;
    Logger&                      logger_;
    std::unique_ptr<httplib::Server> svr_;
    std::thread                  thread_;
    std::atomic<bool>            running_{false};
    int                          port_{0};

    void setupRoutes();
    void handleModels(const httplib::Request& req, httplib::Response& res);
    void handleChatCompletions(const httplib::Request& req, httplib::Response& res);
};

} // namespace neurons_service
