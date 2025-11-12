#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <iostream>

int main(int argc, char** argv) {
    spdlog::info("CyxWiz Server Node v{}", cyxwiz::GetVersionString());

    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize backend");
        return 1;
    }

    // TODO: Parse command line arguments
    // TODO: Initialize gRPC server
    // TODO: Register with Central Server
    // TODO: Start job execution loop
    // TODO: Implement btop-style TUI for monitoring

    spdlog::info("Server node running...");
    spdlog::info("Press Ctrl+C to exit");

    // Keep running
    std::cin.get();

    cyxwiz::Shutdown();
    return 0;
}
