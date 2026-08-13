#include <cyxwiz/device.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

const char* BackendName(cyxwiz::DeviceType type) {
    switch (type) {
        case cyxwiz::DeviceType::CPU: return "arrayfire_cpu";
        case cyxwiz::DeviceType::CUDA: return "arrayfire_cuda";
        case cyxwiz::DeviceType::OPENCL: return "arrayfire_opencl";
        case cyxwiz::DeviceType::ONEAPI: return "arrayfire_oneapi";
        case cyxwiz::DeviceType::METAL: return "unsupported_metal";
        case cyxwiz::DeviceType::VULKAN: return "unsupported_vulkan";
        default: return "arrayfire_unknown";
    }
}

std::string Quoted(const std::string& value) {
    std::ostringstream out;
    out << std::quoted(value);
    return out.str();
}

std::string PciLocation(const cyxwiz::DeviceInfo& info) {
    if (!info.pci_location_known) return "unknown";
    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(4) << info.pci_domain << ':'
        << std::setw(2) << info.pci_bus << ':'
        << std::setw(2) << info.pci_device << '.'
        << info.pci_function;
    return out.str();
}

template <typename Value>
std::string KnownValue(bool known, const Value& value) {
    if (!known) return "unknown";
    std::ostringstream out;
    out << value;
    return out.str();
}

} // namespace

int main() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    std::cout << "inventory_version=1\n";
    std::cout << "device_count=" << devices.size() << '\n';
    for (const auto& device : devices) {
        std::cout
            << "route"
            << " backend=" << BackendName(device.type)
            << " device_id=" << device.device_id
            << " name=" << Quoted(device.name)
            << " kind=" << cyxwiz::DeviceKindName(device.kind)
            << " identity_confidence="
            << cyxwiz::DeviceIdentityConfidenceName(
                   device.identity_confidence)
            << " provider="
            << Quoted(device.provider_known ? device.provider : "unknown")
            << " driver="
            << Quoted(device.driver_version_known
                          ? device.driver_version
                          : "unknown")
            << " vendor_id="
            << KnownValue(device.hardware_vendor_id_known,
                          device.hardware_vendor_id)
            << " hardware_device_id="
            << KnownValue(device.hardware_device_id_known,
                          device.hardware_device_id)
            << " pci=" << PciLocation(device)
            << " uuid="
            << KnownValue(device.hardware_uuid_known, device.hardware_uuid)
            << " luid="
            << KnownValue(device.hardware_luid_known, device.hardware_luid)
            << " fingerprint="
            << KnownValue(device.physical_fingerprint_known,
                          device.physical_fingerprint)
            << " metadata="
            << cyxwiz::DeviceMetadataStatusName(device.metadata_status)
            << " selectable="
            << (device.device_selectable ? "true" : "false")
            << '\n';
    }
    return 0;
}
