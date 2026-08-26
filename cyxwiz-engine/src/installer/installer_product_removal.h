#pragma once

#include "product_installation_receipt.h"

#include <filesystem>
#include <string>

namespace cyxwiz::installer {

struct InstallerProductRemovalState {
  bool installed = false;
  bool available = false;
  std::filesystem::path install_root;
  runtime::ProductInstallScope scope =
      runtime::ProductInstallScope::CurrentUser;
  std::string message;
};

InstallerProductRemovalState InspectInstallerProductRemoval(
    const std::filesystem::path &runtime_root,
    bool stable_bootstrapper_host);

bool QueueInstallerProductRemoval(
    const InstallerProductRemovalState &state,
    std::string &message);

} // namespace cyxwiz::installer
