// gui_main.cpp - Entry point for cyxwiz-server-gui
// GUI/TUI client that connects to the daemon via gRPC IPC

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <spdlog/spdlog.h>
#include <iostream>
#include <cstring>

#include "ipc/daemon_client.h"
#include "core/backend_manager.h"

#ifdef CYXWIZ_HAS_GUI
#include "gui/server_application.h"
#endif

#ifdef CYXWIZ_HAS_TUI
#include "tui/tui_application.h"
#endif

// Interface mode selection
enum class InterfaceMode {
    GUI,       // ImGui-based graphical interface (default)
    TUI        // FTXUI-based terminal interface
};

void PrintUsage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "\nOptions:\n"
              << "  --daemon=ADDR    Daemon address to connect to (default: localhost:50054)\n"
              << "  --mode=gui|tui   Interface mode (default: gui)\n"
              << "  --gui            Use GUI mode\n"
              << "  --tui            Use TUI mode\n"
              << "  --help           Show this help message\n"
              << "\nThis client connects to a running cyxwiz-server-daemon to manage:\n"
              << "  - Training jobs\n"
              << "  - Model deployments\n"
              << "  - API keys\n"
              << "  - Node configuration\n"
              << std::endl;
}

struct GUIConfig {
    std::string daemon_address = "localhost:50054";
    InterfaceMode mode = InterfaceMode::GUI;
};

GUIConfig ParseArgs(int argc, char** argv) {
    GUIConfig config;

    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--daemon=", 9) == 0) {
            config.daemon_address = argv[i] + 9;
        } else if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            const char* mode = argv[i] + 7;
            if (std::strcmp(mode, "gui") == 0) {
                config.mode = InterfaceMode::GUI;
            } else if (std::strcmp(mode, "tui") == 0) {
                config.mode = InterfaceMode::TUI;
            } else {
                spdlog::warn("Unknown mode '{}', using GUI", mode);
            }
        } else if (std::strcmp(argv[i], "--gui") == 0) {
            config.mode = InterfaceMode::GUI;
        } else if (std::strcmp(argv[i], "--tui") == 0) {
            config.mode = InterfaceMode::TUI;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage(argv[0]);
            std::exit(0);
        }
    }

    // Check compile-time availability
#ifndef CYXWIZ_HAS_GUI
    if (config.mode == InterfaceMode::GUI) {
        spdlog::warn("GUI mode not compiled in, falling back to TUI");
        config.mode = InterfaceMode::TUI;
    }
#endif

#ifndef CYXWIZ_HAS_TUI
    if (config.mode == InterfaceMode::TUI) {
        spdlog::warn("TUI mode not compiled in, falling back to GUI");
        config.mode = InterfaceMode::GUI;
    }
#endif

    return config;
}

int main(int argc, char** argv) {
    spdlog::info("CyxWiz Server GUI v0.3.0");

    // Parse arguments
    GUIConfig config = ParseArgs(argc, argv);

    const char* mode_name = (config.mode == InterfaceMode::GUI) ? "GUI" : "TUI";
    spdlog::info("Interface mode: {}", mode_name);

    // Create daemon client (connection will happen in background)
    auto daemon_client = std::make_shared<cyxwiz::servernode::ipc::DaemonClient>();
    daemon_client->SetTargetAddress(config.daemon_address);

    // Start GUI immediately - daemon connection happens asynchronously
    int result = 0;

    switch (config.mode) {
#ifdef CYXWIZ_HAS_GUI
        case InterfaceMode::GUI: {
            spdlog::info("Starting GUI...");
            cyxwiz::servernode::gui::ServerApplication app(argc, argv, daemon_client);
            app.Run();
            break;
        }
#endif

#ifdef CYXWIZ_HAS_TUI
        case InterfaceMode::TUI: {
            spdlog::info("Starting TUI...");
            cyxwiz::servernode::tui::TUIApplication app(argc, argv, daemon_client);
            app.Run();
            break;
        }
#endif

        default:
            spdlog::error("No interface mode available");
            result = 1;
            break;
    }

    // Disconnect from daemon
    daemon_client->Disconnect();

    return result;
}
