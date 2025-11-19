#include "cyxwiz/cyxwiz.h"
#include "cyxwiz/device.h"
#include <iostream>

int main() {
    std::cout << "Testing CyxWiz GPU Detection..." << std::endl;

    if (cyxwiz::Initialize()) {
        std::cout << "CyxWiz Backend initialized successfully!" << std::endl;
        std::cout << "Version: " << cyxwiz::GetVersionString() << std::endl;

        auto devices = cyxwiz::Device::GetAvailableDevices();
        std::cout << "\nAvailable devices: " << devices.size() << std::endl;

        for (size_t i = 0; i < devices.size(); i++) {
            const auto& dev = devices[i];
            std::cout << "\nDevice " << i << ":" << std::endl;
            std::cout << "  Name: " << dev.name << std::endl;
            std::cout << "  Type: " << static_cast<int>(dev.type) << std::endl;
            std::cout << "  Memory: " << (dev.memory_total / (1024*1024)) << " MB" << std::endl;
            std::cout << "  FP64: " << (dev.supports_fp64 ? "Yes" : "No") << std::endl;
        }

        cyxwiz::Shutdown();
        return 0;
    } else {
        std::cout << "Failed to initialize CyxWiz Backend!" << std::endl;
        return 1;
    }
}
