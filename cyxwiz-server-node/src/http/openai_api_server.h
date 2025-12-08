// openai_api_server.h - OpenAI-compatible HTTP API server (stub)
#pragma once

#include <string>
#include <memory>

namespace cyxwiz::servernode {

class DeploymentManager;

// Stub implementation for HTTP API server
class OpenAIAPIServer {
public:
    explicit OpenAIAPIServer(int port)
        : port_(port), deployment_manager_(nullptr) {}

    OpenAIAPIServer(int port, DeploymentManager* deployment_manager)
        : port_(port), deployment_manager_(deployment_manager) {}

    ~OpenAIAPIServer() = default;

    bool Start() { return false; }  // TODO: Implement
    void Stop() {}
    bool IsRunning() const { return false; }

private:
    int port_;
    DeploymentManager* deployment_manager_;
};

} // namespace cyxwiz::servernode
