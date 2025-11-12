#include "application.h"
#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>
#include <iostream>

int main(int argc, char** argv) {
    // Setup logging
    spdlog::set_level(spdlog::level::info);

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

        // Log device info to GUI console after app is initialized
        // Note: This will be displayed when the first frame renders

        int result = app.Run();

        // Cleanup
        cyxwiz::Shutdown();

        return result;
    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        cyxwiz::Shutdown();
        return 1;
    }
}
