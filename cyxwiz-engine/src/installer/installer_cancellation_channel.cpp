#include "installer_cancellation_channel.h"

#include "installer_progress_channel.h"

#include <fstream>
#include <system_error>

namespace cyxwiz::installer {

std::filesystem::path InstallerCancellationPath(const std::string &token) {
  if (!IsInstallerProgressToken(token))
    return {};
  std::error_code error;
  const auto temporary_root = std::filesystem::temp_directory_path(error);
  if (error || !temporary_root.is_absolute())
    return {};
  return temporary_root / "CyxWiz" / "installer-control" /
         (token + ".cancel");
}

bool RequestInstallerCancellation(const std::string &token) {
  const auto path = InstallerCancellationPath(token);
  if (path.empty())
    return false;
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return false;
  const auto next = path.string() + ".next";
  {
    std::ofstream output(next, std::ios::binary | std::ios::trunc);
    if (!output)
      return false;
    output << "cancel\n";
    if (!output)
      return false;
  }
#ifdef _WIN32
  std::filesystem::remove(path, error);
  error.clear();
#endif
  std::filesystem::rename(next, path, error);
  if (error) {
    std::filesystem::remove(next, error);
    return false;
  }
  return true;
}

bool IsInstallerCancellationRequested(const std::string &token) {
  const auto path = InstallerCancellationPath(token);
  if (path.empty())
    return false;
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) && !error;
}

void ClearInstallerCancellation(const std::string &token) {
  const auto path = InstallerCancellationPath(token);
  if (path.empty())
    return;
  std::error_code error;
  std::filesystem::remove(path, error);
  std::filesystem::remove(path.string() + ".next", error);
}

} // namespace cyxwiz::installer
