// openai_api_server.h - HTTP REST API server for model inference
#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>

// Forward declare httplib types to avoid including in header
namespace httplib {
class Server;
}

namespace cyxwiz::servernode {

class DeploymentManager;

// HTTP REST API server providing inference endpoints
class OpenAIAPIServer {
public:
    explicit OpenAIAPIServer(int port);
    OpenAIAPIServer(int port, DeploymentManager* deployment_manager);
    ~OpenAIAPIServer();

    // Start the HTTP server (non-blocking, runs in a thread)
    bool Start();

    // Stop the HTTP server
    void Stop();

    // Check if server is running
    bool IsRunning() const;

    // Get the port
    int GetPort() const { return port_; }

    // Set deployment manager (can be set after construction)
    void SetDeploymentManager(DeploymentManager* manager);

private:
    // Register all routes
    void RegisterRoutes();

    // Health check endpoint
    void HandleHealth();

    // List deployed models
    void HandleModels();

    // Run inference prediction
    void HandlePredict();

private:
    int port_;
    DeploymentManager* deployment_manager_;
    std::unique_ptr<httplib::Server> server_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
};

} // namespace cyxwiz::servernode
