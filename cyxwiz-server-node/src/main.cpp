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
#include <grpcpp/grpcpp.h>
#include "deployment_handler.h"
#include "deployment_manager.h"
#include "terminal_handler.h"
#include "node_client.h"
#include "job_executor.h"
#include "node_service.h"

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
    std::string node_service_address = "0.0.0.0:50054";  // NodeService for job assignment
    std::string central_server = "localhost:50051";    // Central Server address

    spdlog::info("Node ID: {}", node_id);
    spdlog::info("Deployment service: {}", deployment_address);
    spdlog::info("Terminal service: {}", terminal_address);
    spdlog::info("Node service: {}", node_service_address);

    try {
        // Get current device for job execution (or nullptr to let JobExecutor choose)
        cyxwiz::Device* device = cyxwiz::Device::GetCurrentDevice();

        // Create JobExecutor for ML training jobs
        auto job_executor = std::make_unique<cyxwiz::servernode::JobExecutor>(node_id, device);

        // NodeClient will be set up after registration
        cyxwiz::servernode::NodeClient* node_client_ptr = nullptr;

        // Set up progress callback
        job_executor->SetProgressCallback([&node_id](
            const std::string& job_id,
            double progress,
            const cyxwiz::servernode::TrainingMetrics& metrics)
        {
            spdlog::info("Job {} progress: {:.1f}% - Epoch {}/{}, Loss: {:.4f}",
                job_id, progress * 100.0,
                metrics.current_epoch, metrics.total_epochs,
                metrics.loss);
            // TODO: Report progress to Central Server via NodeClient
        });

        // Set up completion callback
        job_executor->SetCompletionCallback([&node_id](
            const std::string& job_id,
            bool success,
            const std::string& error_msg)
        {
            if (success) {
                spdlog::info("Job {} completed successfully", job_id);
            } else {
                spdlog::error("Job {} failed: {}", job_id, error_msg);
            }
            // TODO: Report completion to Central Server via NodeClient
        });

        // Create NodeServiceImpl to handle job assignments from Central Server
        auto node_service = std::make_unique<cyxwiz::servernode::NodeServiceImpl>(
            job_executor.get(),
            node_id
        );

        // Create gRPC server for NodeService
        grpc::ServerBuilder node_service_builder;
        node_service_builder.AddListeningPort(
            node_service_address,
            grpc::InsecureServerCredentials()
        );
        node_service_builder.RegisterService(node_service.get());
        std::unique_ptr<grpc::Server> node_grpc_server = node_service_builder.BuildAndStart();

        if (!node_grpc_server) {
            spdlog::error("Failed to start NodeService gRPC server");
            cyxwiz::Shutdown();
            return 1;
        }

        spdlog::info("NodeService started on {}", node_service_address);

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
        spdlog::info("  Deployment endpoint:  {}", deployment_address);
        spdlog::info("  Terminal endpoint:    {}", terminal_address);
        spdlog::info("  Node service:         {}", node_service_address);
        spdlog::info("  Active jobs:          {}", job_executor->GetActiveJobCount());
        spdlog::info("========================================");
        spdlog::info("Press Ctrl+C to shutdown");

        // Main loop - wait for shutdown signal
        // Heartbeat is running in background thread
        while (!g_shutdown) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        spdlog::info("Shutting down gracefully...");

        // Stop services
        spdlog::info("Stopping NodeService gRPC server...");
        node_grpc_server->Shutdown();

        spdlog::info("Stopping terminal and deployment handlers...");
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
