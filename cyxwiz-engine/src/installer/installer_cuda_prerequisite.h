#pragma once

#include <string>

namespace cyxwiz::installer {

struct InstallerCudaDriverProbe {
    bool driver_library_found = false;
    bool required_api_found = false;
    int initialization_status = -1;
    int device_count = 0;
    int driver_api_version = 0;
};

struct InstallerCudaPrerequisiteState {
    bool driver_detected = false;
    bool device_available = false;
    int device_count = 0;
    std::string driver_api_version;
    std::string message;
};

InstallerCudaPrerequisiteState EvaluateInstallerCudaDriverProbe(
    const InstallerCudaDriverProbe& probe);

InstallerCudaPrerequisiteState DetectInstallerCudaPrerequisite();

}  // namespace cyxwiz::installer
