#pragma once

#include <string>
#include <vector>

#if defined(NEURONS_REMOTE_ENABLED)
#include <memory>
#include <grpcpp/grpcpp.h>
#include "neurons.grpc.pb.h"
#endif

namespace neurons::cli {

// Runs an interactive chat REPL connected to a remote neurons-service over gRPC.
// Mirrors ChatCommand::run_repl() but routes inference through the Generate RPC.
// Only compiled when NEURONS_REMOTE_ENABLED is defined (requires gRPC).
class RemoteNodeRunner {
public:
    struct Options {
        std::string endpoint;       // host:port (no scheme)
        std::string system_prompt;
        int   max_tokens   = 4096;
        float temperature  = 0.7f;
        int   top_k        = 40;
        float top_p        = 0.9f;
        float rep_penalty  = 1.1f;
        bool  tools_enabled        = false;
        bool  allow_shell_fallback = false;
        std::vector<std::string> tool_servers;
    };

    explicit RemoteNodeRunner(Options opts);

    int run_repl();

private:
    Options opts_;
#if defined(NEURONS_REMOTE_ENABLED)
    std::shared_ptr<grpc::Channel>                     channel_;
    std::unique_ptr<neurons::NeuronsInference::Stub>   stub_;
#endif
    void printHelp() const;
};

} // namespace neurons::cli
