#include "application.h"
#include "plugin/plugin_manager.h"
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    // Setup logging with file output
    try {
        // Get executable directory for log file
        std::filesystem::path exe_path = std::filesystem::current_path();
        std::filesystem::path log_path = exe_path / "engine_log.txt";

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(), true);
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto logger = std::make_shared<spdlog::logger>("cyxwiz", spdlog::sinks_init_list{file_sink, console_sink});
        logger->set_level(spdlog::level::info);
        spdlog::set_default_logger(logger);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }

#ifdef CYXWIZ_DEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    spdlog::info("Starting CyxWiz Engine v{}", cyxwiz::GetVersionString());

    // Initialize backend
    if (!cyxwiz::Initialize()) {
        spdlog::error("Failed to initialize CyxWiz backend");
        return 1;
    }

    // List available devices
    auto devices = cyxwiz::Device::GetAvailableDevices();
    spdlog::info("Available compute devices:");
    for (const auto& device : devices) {
        spdlog::info("  - {} ({})", device.name, static_cast<int>(device.type));
    }

    // Create and run application
    try {
        CyxWizApp app(argc, argv);

        // Initialize plugin system
        {
            auto& pm = cyxwiz::plugin::PluginManager::Instance();
            std::filesystem::path exe_dir = std::filesystem::current_path();
            std::filesystem::path user_dir;
#ifdef _WIN32
            if (auto* appdata = std::getenv("APPDATA"))
                user_dir = std::filesystem::path(appdata) / "cyxwiz" / "plugins";
#else
            if (auto* home = std::getenv("HOME"))
                user_dir = std::filesystem::path(home) / ".cyxwiz" / "plugins";
#endif
            std::vector<std::filesystem::path> search_paths;
            search_paths.push_back(exe_dir / "plugins");
            if (!user_dir.empty()) search_paths.push_back(user_dir);
            pm.SetSearchPaths(search_paths);
            pm.LoadAllFromSearchPaths();
            pm.InitializeAll();
            spdlog::info("Plugin system: {} plugins loaded", pm.GetPluginCount());
        }

        int result = app.Run();

        // Cleanup plugins before backend
        cyxwiz::plugin::PluginManager::Instance().ShutdownAll();
        cyxwiz::plugin::PluginManager::Instance().UnloadAll();

        cyxwiz::Shutdown();

        return result;
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        cyxwiz::Shutdown();
        return 1;
    }
}
