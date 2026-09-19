#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::installer {

bool InstallerPathWithin(const std::filesystem::path &path,
                         const std::filesystem::path &directory);

// Only the small manager dependency closure is copied, never the
// engine/runtime.
bool StageExternalInstallerSession(
    const std::filesystem::path &executable_directory,
    const std::filesystem::path &metadata_root,
    const std::vector<std::filesystem::path> &loaded_modules,
    const std::filesystem::path &destination, std::string &error);

std::vector<std::filesystem::path> LoadedInstallerModules(std::string &error);
bool RelaunchExternalInstallerSession(const std::filesystem::path &directory,
                                      const std::vector<std::string> &arguments,
                                      std::string &error);

// Must run before creating the window. An installed manager relocates itself so
// no executable, library, or working directory pins the installation during
// removal.
enum class ExternalInstallerSession { Ready, Relaunched, Failed };
ExternalInstallerSession PrepareExternalInstallerSession(
    const std::filesystem::path &executable_directory,
    const std::filesystem::path &runtime_root,
    const std::filesystem::path &metadata_root,
    const std::vector<std::string> &arguments, std::string &error);

} // namespace cyxwiz::installer
