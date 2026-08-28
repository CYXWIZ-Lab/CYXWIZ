#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace cyxwiz::installer {

struct InstallerHelperProgress {
  std::string stage;
  std::uint64_t completed_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::size_t component_index = 0;
  std::size_t component_count = 0;
  std::string message;
};

std::string CreateInstallerProgressToken();
bool IsInstallerProgressToken(const std::string &token);
std::filesystem::path InstallerProgressPath(
    const std::filesystem::path &runtime_root, const std::string &token);
bool PublishInstallerProgress(const std::filesystem::path &path,
                              const InstallerHelperProgress &progress);
std::optional<InstallerHelperProgress>
ReadInstallerProgress(const std::filesystem::path &path);

} // namespace cyxwiz::installer
