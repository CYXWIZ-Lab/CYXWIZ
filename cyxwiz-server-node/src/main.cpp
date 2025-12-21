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
#include <cstring>
#include <grpcpp/grpcpp.h>
#include "deployment_handler.h"
#include "deployment_manager.h"
#include "terminal_handler.h"
#include "node_client.h"
#include "job_executor.h"
#include "node_service.h"
#include "job_execution_service.h"
#include "core/backend_manager.h"

#ifdef CYXWIZ_HAS_GUI
#include "gui/server_application.h"
#endif

#ifdef CYXWIZ_HAS_TUI
#include "tui/tui_application.h"
#endif

// Interface mode selection
enum class InterfaceMode {
    Headless,  // No UI, background service
    GUI,       // ImGui-based graphical interface
    TUI        // FTXUI-based terminal interface
};

// Detect interface mode from command-line arguments
InterfaceMode DetectMode(int argc, char** argv) {
    // Check for explicit --mode argument
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            const char* mode = argv[i] + 7;
            if (std::strcmp(mode, "gui") == 0) {
#ifdef CYXWIZ_HAS_GUI
                return InterfaceMode::GUI;
#else
                spdlog::warn("GUI mode requested but not compiled in, falling back to headless");
                return InterfaceMode::Headless;
#endif
            } else if (std::strcmp(mode, "tui") == 0) {
#ifdef CYXWIZ_HAS_TUI
                return InterfaceMode::TUI;
#else
                spdlog::warn("TUI mode requested but not compiled in, falling back to headless");
                return InterfaceMode::Headless;
#endif
            } else if (std::strcmp(mode, "headless") == 0) {
                return InterfaceMode::Headless;
            } else {
                spdlog::warn("Unknown mode '{}', falling back to headless", mode);
                return InterfaceMode::Headless;
            }
        }
        // Short forms
        if (std::strcmp(argv[i], "--gui") == 0) {
#ifdef CYXWIZ_HAS_GUI
            return InterfaceMode::GUI;
#else
            spdlog::warn("GUI mode requested but not compiled in, falling back to headless");
            return InterfaceMode::Headless;
#endif
        }
        if (std::strcmp(argv[i], "--tui") == 0) {
#ifdef CYXWIZ_HAS_TUI
            return InterfaceMode::TUI;
#else
            spdlog::warn("TUI mode requested but not compiled in, falling back to headless");
            return InterfaceMode::Headless;
#endif
        }
    }

    // Auto-detect based on environment
#ifdef _WIN32
    // On Windows, default to GUI if available and not in a console-only environment
    #ifdef CYXWIZ_HAS_GUI
    // Check if running in Windows Terminal or PowerShell (prefer TUI) or double-clicked (prefer GUI)
    // For now, default to TUI for better server experience
    #ifdef CYXWIZ_HAS_TUI
    return InterfaceMode::TUI;
    #else
    return InterfaceMode::GUI;
    #endif
    #endif
#else
    // On Linux/macOS, check for display
    const char* display = std::getenv("DISPLAY");
    const char* wayland = std::getenv("WAYLAND_DISPLAY");

    if (display || wayland) {
        // Display available
        #ifdef CYXWIZ_HAS_TUI
        // Default to TUI for server experience even with display
        return InterfaceMode::TUI;
        #elif defined(CYXWIZ_HAS_GUI)
        return InterfaceMode::GUI;
        #endif
    }
#endif

    // Default to headless
    return InterfaceMode::Headless;
}

// Global flag for shutdown
std::atomic<bool> g_shutdown{false};

void SignalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        spdlog::info("Received shutdown signal");
        g_shutdown = true;
    }
}

// Forward declaration for headless mode
int RunHeadlessMode(int argc, char** argv);

int main(int argc, char** argv) {
    spdlog::info("CyxWiz Server Node v{}", cyxwiz::GetVersionString());
    spdlog::info("========================================");

    // Detect interface mode
    InterfaceMode mode = DetectMode(argc, argv);

    // Print mode
    const char* mode_name = (mode == InterfaceMode::GUI) ? "GUI" :
                           (mode == InterfaceMode::TUI) ? "TUI" : "Headless";
    spdlog::info("Interface mode: {}", mode_name);

    // Initialize backend
    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize backend");
        return 1;
    }

    // Initialize BackendManager for GUI/TUI modes
    auto& backend = cyxwiz::servernode::core::BackendManager::Instance();

    // Initialize BackendManager with default config for TUI/GUI modes (metrics collection)
    if (mode == InterfaceMode::TUI || mode == InterfaceMode::GUI) {
        cyxwiz::servernode::core::NodeConfig config;
        config.node_id = "node_" + std::to_string(std::time(nullptr));
        config.deployment_enabled = false;  // Daemon handles deployment in these modes
        if (!backend.Initialize(config)) {
            spdlog::warn("Failed to initialize BackendManager - metrics may not be available");
        }
    }

    // Run in selected mode
    int result = 0;

    switch (mode) {
#ifdef CYXWIZ_HAS_GUI
        case InterfaceMode::GUI: {
            spdlog::info("Starting GUI mode...");
            cyxwiz::servernode::gui::ServerApplication app(argc, argv);
            app.Run();
            break;
        }
#endif

#ifdef CYXWIZ_HAS_TUI
        case InterfaceMode::TUI: {
            spdlog::info("Starting TUI mode...");
            cyxwiz::servernode::tui::TUIApplication app(argc, argv);
            app.Run();
            break;
        }
#endif

        case InterfaceMode::Headless:
        default:
            result = RunHeadlessMode(argc, argv);
            break;
    }

    cyxwiz::Shutdown();
    return result;
}

// Original headless mode implementation
int RunHeadlessMode(int argc, char** argv) {
    // Install signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    // TODO: Parse command line arguments for node_id, listen addresses, etc.
    std::string node_id = "node_" + std::to_string(std::time(nullptr));
    std::string p2p_service_address = "0.0.0.0:50052";  // P2P service for Engine connections
    std::string terminal_address = "0.0.0.0:50053";
    std::string node_service_address = "0.0.0.0:50054";  // NodeService for job assignment
    std::string deployment_address = "0.0.0.0:50055";  // Deployment service
    std::string central_server = "localhost:50051";    // Central Server address

    spdlog::info("Node ID: {}", node_id);
    spdlog::info("P2P service (Engine): {}", p2p_service_address);
    spdlog::info("Terminal service: {}", terminal_address);
    spdlog::info("Node service: {}", node_service_address);
    spdlog::info("Deployment service: {}", deployment_address);

    try {
        // Get current device for job execution (or nullptr to let JobExecutor choose)
        cyxwiz::Device* device = cyxwiz::Device::GetCurrentDevice();

        // Create JobExecutor for ML training jobs (shared_ptr for multiple service access)
        auto job_executor = std::make_shared<cyxwiz::servernode::JobExecutor>(node_id, device);

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

        // Create and start P2P JobExecutionService for direct Engine connections
        auto p2p_service = std::make_unique<cyxwiz::server_node::JobExecutionServiceImpl>();
        // P2P JWT secret - must match Central Server's jwt.secret config
        // Development default matches Central Server's config.toml
        const std::string p2p_secret = "your-super-secret-jwt-key-change-in-production";
        p2p_service->Initialize(job_executor, central_server, node_id, p2p_secret);

        if (!p2p_service->StartServer(p2p_service_address)) {
            spdlog::error("Failed to start P2P JobExecutionService");
            node_grpc_server->Shutdown();
            cyxwiz::Shutdown();
            return 1;
        }

        spdlog::info("P2P JobExecutionService started on {}", p2p_service_address);

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
        spdlog::info("  P2P service (Engine): {}", p2p_service_address);
        spdlog::info("  Node service:         {}", node_service_address);
        spdlog::info("  Deployment endpoint:  {}", deployment_address);
        spdlog::info("  Terminal endpoint:    {}", terminal_address);
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
        spdlog::info("Stopping P2P JobExecutionService...");
        p2p_service->StopServer();

        spdlog::info("Stopping NodeService gRPC server...");
        node_grpc_server->Shutdown();

        spdlog::info("Stopping terminal and deployment handlers...");
        terminal_handler.Stop();
        deployment_handler.Stop();

        spdlog::info("Server Node shutdown complete");

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
