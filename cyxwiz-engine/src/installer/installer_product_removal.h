#pragma once

#include "product_installation_receipt.h"

#include <filesystem>
#include <string>

namespace cyxwiz::installer {

struct InstallerProductRemovalState {
  bool installed = false;
  bool available = false;
  bool requires_stable_host = false;
  std::filesystem::path install_root;
  runtime::ProductInstallScope scope =
      runtime::ProductInstallScope::CurrentUser;
  std::string message;
};

InstallerProductRemovalState
InspectInstallerProductRemoval(const std::filesystem::path &runtime_root,
                               bool external_session);

struct InstallerProductRemovalResult {
  bool succeeded = false;
  std::string message;
};

InstallerProductRemovalResult
RemoveInstallerProduct(const InstallerProductRemovalState &state,
                       const std::filesystem::path &executable_directory);

} // namespace cyxwiz::installer
