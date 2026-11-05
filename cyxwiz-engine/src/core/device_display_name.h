#pragma once

// A device name as screens show it: ArrayFire reports names like
// "NVIDIA_GeForce_GTX_1050_Ti"; they read with spaces, trimmed. Header-only
// so the Engine and the Installer share it without extra sources.

#include <algorithm>
#include <string>

namespace cyxwiz {

inline std::string DeviceDisplayName(std::string name) {
    std::replace(name.begin(), name.end(), '_', ' ');
    const auto first = name.find_first_not_of(' ');
    if (first == std::string::npos) return {};
    const auto last = name.find_last_not_of(' ');
    return name.substr(first, last - first + 1);
}

}  // namespace cyxwiz
