#pragma once

#include <filesystem>
#include <string>

namespace cyxwiz::runtime {

bool LaunchInstallerAndWait(
    const std::filesystem::path& installer_path,
    int& exit_code,
    std::string& error);

}  // namespace cyxwiz::runtime
