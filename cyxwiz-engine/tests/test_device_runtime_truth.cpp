#include <cyxwiz/cyxwiz.h>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    try {
        const bool initialized_here = !cyxwiz::IsInitialized();
        if (initialized_here) {
            Check(cyxwiz::Initialize(), "backend initialization should succeed");
        }

        auto* initialized_device = cyxwiz::Device::GetCurrentDevice();
        Check(initialized_device != nullptr,
              "initialized ArrayFire runtime should expose an active device");

        const auto available = cyxwiz::Device::GetAvailableDevices();
        bool active_is_available = false;
        for (const auto& device : available) {
            if (device.type == initialized_device->GetType() &&
                device.device_id == initialized_device->GetDeviceId()) {
                active_is_available = true;
                break;
            }
        }
        Check(active_is_available,
              "active runtime device should appear in device discovery");

        {
            cyxwiz::Device temporary_cpu(cyxwiz::DeviceType::CPU, 0);
            temporary_cpu.SetActive();
            Check(temporary_cpu.IsActive(),
                  "value-equivalent CPU device should report active");
        }

        auto* active_after_temporary = cyxwiz::Device::GetCurrentDevice();
        Check(active_after_temporary != nullptr,
              "destroying the selection object must not erase runtime truth");
        Check(active_after_temporary->GetType() == cyxwiz::DeviceType::CPU &&
                  active_after_temporary->GetDeviceId() == 0,
              "active runtime should remain CPU after temporary object destruction");

        cyxwiz::Device equivalent_cpu(cyxwiz::DeviceType::CPU, 0);
        Check(equivalent_cpu.IsActive(),
              "IsActive should compare backend/device identity, not object address");

        if (initialized_here) {
            cyxwiz::Shutdown();
        }
        std::cout << "Device runtime truth contract passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << "\n";
        return 1;
    }
}
