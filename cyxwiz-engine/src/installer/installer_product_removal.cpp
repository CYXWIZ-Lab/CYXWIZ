#include "installer_product_removal.h"

#include "backend_pack_platform.h"
#include "product_removal_authorization.h"
#include "product_removal_request.h"

#include <system_error>

namespace cyxwiz::installer {

InstallerProductRemovalState InspectInstallerProductRemoval(
    const std::filesystem::path &runtime_root,
    bool stable_bootstrapper_host) {
  InstallerProductRemovalState state;
  if (!runtime_root.is_absolute() ||
      runtime_root != runtime_root.lexically_normal() ||
      runtime_root.filename() != "runtime") {
    state.message = "Full product removal requires an exact installed runtime";
    return state;
  }
  state.install_root = runtime_root.parent_path();

  runtime::ProductInstallationReceipt receipt;
  std::string error;
  if (!runtime::LoadProductInstallationReceipt(
          state.install_root, receipt, error)) {
    state.message = "Full product removal is unavailable: " + error;
    return state;
  }
  state.installed = true;
  state.scope = receipt.scope;

  runtime::ProductRemovalAuthorization authorization;
  if (!runtime::CaptureProductRemovalAuthorization(
          state.install_root, state.scope, authorization, error)) {
    state.message = "Full product removal is unavailable: " + error;
    return state;
  }
  const auto finalizer = state.install_root /
      std::string(runtime::CurrentProductRemovalFinalizerExecutableName());
  std::error_code filesystem_error;
  if (std::filesystem::symlink_status(finalizer, filesystem_error).type() !=
          std::filesystem::file_type::regular ||
      filesystem_error) {
    state.message =
        "Full product removal is unavailable: the verified finalizer is missing";
    return state;
  }
  if (!stable_bootstrapper_host) {
    state.requires_stable_host = true;
    state.message =
        "Open this installation through CyxWiz Installer to remove the product";
    return state;
  }

  state.available = true;
  state.message =
      "Removes this CyxWiz installation and all installed backend packs";
  return state;
}

bool QueueInstallerProductRemoval(
    const InstallerProductRemovalState &state,
    std::string &message) {
  if (!state.installed || !state.available ||
      state.install_root.empty()) {
    message = state.message.empty()
        ? "Full product removal is not available"
        : state.message;
    return false;
  }
  runtime::ProductRemovalAuthorization authorization;
  if (!runtime::QueueProductRemovalRequest(
          state.install_root, state.scope, authorization, message)) {
    message = "Cannot queue full product removal: " + message;
    return false;
  }
  if (authorization.install_root != state.install_root ||
      authorization.scope != state.scope) {
    message = "The queued product removal identity changed unexpectedly";
    return false;
  }
  message =
      "Product removal is queued and will begin after the installer closes";
  return true;
}

} // namespace cyxwiz::installer
