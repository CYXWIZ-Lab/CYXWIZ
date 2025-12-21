/**
 * Standalone P2P Server - For manual testing of P2P service
 *
 * Usage: standalone_p2p_server [port]
 * Example: standalone_p2p_server 50052
 */

#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include "../src/job_execution_service.h"

using namespace cyxwiz::server_node;

std::unique_ptr<JobExecutionServiceImpl> g_service;
bool g_running = true;

void SignalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n[!] Shutting down P2P server..." << std::endl;
        g_running = false;
        if (g_service) {
            g_service->StopServer();
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       CyxWiz Server Node - P2P Service (Standalone)     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Parse port from arguments
    std::string port = (argc > 1) ? argv[1] : "50052";
    std::string listen_address = "0.0.0.0:" + port;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Listen Address: " << listen_address << std::endl;
    std::cout << "  Central Server: localhost:50051 (for notifications)" << std::endl;
    std::cout << std::endl;

    // Setup signal handlers
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    try {
        // Create and initialize P2P service
        g_service = std::make_unique<JobExecutionServiceImpl>();
        g_service->Initialize(nullptr, "localhost:50051", "standalone_test_node", "test_p2p_secret");

        // Start the P2P server
        if (!g_service->StartServer(listen_address)) {
            std::cerr << "[ERROR] Failed to start P2P server on " << listen_address << std::endl;
            return 1;
        }

        std::cout << "[OK] P2P server listening on " << listen_address << std::endl;
        std::cout << "[**] Ready to accept connections from Engine clients" << std::endl;
        std::cout << "   Press Ctrl+C to stop" << std::endl;
        std::cout << std::endl;

        // Keep the server running
        while (g_running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "[OK] Server stopped cleanly" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }
}
