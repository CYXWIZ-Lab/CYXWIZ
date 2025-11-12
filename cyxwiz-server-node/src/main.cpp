// Fix Windows min/max macro conflicts with C++ standard library
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>
#include <thread>
#include <csignal>
#include <atomic>
#include "deployment_handler.h"
#include "deployment_manager.h"
#include "terminal_handler.h"
#include "node_client.h"

// Global flag for shutdown
std::atomic<bool> g_shutdown{false};

void SignalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("Received shutdown signal");
        g_shutdown = true;
    }
}

int main(int argc, char** argv) {
    spdlog::info("CyxWiz Server Node v{}", cyxwiz::GetVersionString());
    spdlog::info("========================================");

    // Install signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // Initialize backend
    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize backend");
        return 1;
    }

    // TODO: Parse command line arguments for node_id, listen addresses, etc.
    std::string node_id = "node_" + std::to_string(std::time(nullptr));
    std::string deployment_address = "0.0.0.0:50052";  // Different from Central Server's 50051
    std::string terminal_address = "0.0.0.0:50053";
    std::string central_server = "localhost:50051";    // Central Server address

    spdlog::info("Node ID: {}", node_id);
    spdlog::info("Deployment service: {}", deployment_address);
    spdlog::info("Terminal service: {}", terminal_address);

    try {
        // Create deployment manager
        auto deployment_manager = std::make_shared<cyxwiz::servernode::DeploymentManager>(node_id);

        // Create and start deployment handler
        cyxwiz::servernode::DeploymentHandler deployment_handler(
            deployment_address,
            deployment_manager
        );

        if (!deployment_handler.Start()) {
            spdlog::error("Failed to start deployment handler");
            cyxwiz::Shutdown();
            return 1;
        }

        // Create and start terminal handler
        cyxwiz::servernode::TerminalHandler terminal_handler(terminal_address);

        if (!terminal_handler.Start()) {
            spdlog::error("Failed to start terminal handler");
            deployment_handler.Stop();
            cyxwiz::Shutdown();
            return 1;
        }

        // Register with Central Server
        spdlog::info("Connecting to Central Server at {}...", central_server);
        auto node_client = std::make_unique<cyxwiz::servernode::NodeClient>(central_server, node_id);

        if (!node_client->Register()) {
            spdlog::error("Failed to register with Central Server");
            spdlog::warn("Server Node will run in standalone mode");
        } else {
            spdlog::info("Successfully registered with Central Server");

            // Start heartbeat loop
            if (!node_client->StartHeartbeat(10)) {  // 10 second interval
                spdlog::error("Failed to start heartbeat");
            }
        }

        spdlog::info("========================================");
        spdlog::info("Server Node is ready!");
        spdlog::info("  Deployment endpoint: {}", deployment_address);
        spdlog::info("  Terminal endpoint:   {}", terminal_address);
        spdlog::info("  Active deployments:  0");
        spdlog::info("========================================");
        spdlog::info("Press Ctrl+C to shutdown");

        // Main loop - wait for shutdown signal
        // Heartbeat is running in background thread
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        spdlog::info("Shutting down gracefully...");

        // Stop services
        terminal_handler.Stop();
        deployment_handler.Stop();

        spdlog::info("Server Node shutdown complete");

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        cyxwiz::Shutdown();
        return 1;
    }

    cyxwiz::Shutdown();
    return 0;
}
