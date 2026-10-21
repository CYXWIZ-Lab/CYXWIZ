#include "installer_product_removal.h"

#include "backend_pack_platform.h"
#include "installer_external_session.h"
#include "product_removal_authorization.h"
#include "product_removal_transaction.h"

#include <system_error>

namespace cyxwiz::installer {

InstallerProductRemovalState
InspectInstallerProductRemoval(const std::filesystem::path &runtime_root,
                               bool external_session) {
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
  if (!runtime::LoadProductInstallationReceipt(state.install_root, receipt,
                                               error)) {
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
  const auto finalizer =
      state.install_root /
      std::string(runtime::CurrentProductRemovalFinalizerExecutableName());
  std::error_code filesystem_error;
  if (std::filesystem::symlink_status(finalizer, filesystem_error).type() !=
          std::filesystem::file_type::regular ||
      filesystem_error) {
    state.message = "Full product removal is unavailable: the verified "
                    "finalizer is missing";
    return state;
  }
  if (!external_session) {
    state.requires_stable_host = true;
    state.message = "Full uninstall requires an external maintenance session";
    return state;
  }

  state.available = true;
  state.message =
      "Removes this CyxWiz installation and all installed backend packs";
  return state;
}

InstallerProductRemovalResult
RemoveInstallerProduct(const InstallerProductRemovalState &state,
                       const std::filesystem::path &executable_directory) {
  InstallerProductRemovalResult result;
  try {
    if (!state.installed || !state.available || state.install_root.empty() ||
        InstallerPathWithin(executable_directory, state.install_root) ||
        InstallerPathWithin(std::filesystem::current_path(),
                            state.install_root)) {
      result.message =
          "Uninstall requires a manager running outside the installation";
      return result;
    }
    const auto modules = LoadedInstallerModules(result.message);
    if (!result.message.empty())
      return result;
    for (const auto &module : modules) {
      if (InstallerPathWithin(module, state.install_root)) {
        result.message = "Uninstall is blocked by a loaded product library: " +
                         module.string();
        return result;
      }
    }
    runtime::ProductRemovalAuthorization authorization;
    if (!runtime::CaptureProductRemovalAuthorization(
            state.install_root, state.scope, authorization, result.message))
      return result;
    runtime::ProductRemovalTransactionResult transaction;
    result.succeeded = runtime::ExecuteProductRemovalTransaction(
        authorization, transaction, result.message);
    if (result.succeeded)
      result.message = "CyxWiz was uninstalled successfully. You can install "
                       "it again or close this window.";
    else
      result.message = "Uninstall did not complete: " + result.message;
  } catch (const std::exception &exception) {
    result.message =
        "Uninstall did not complete: " + std::string(exception.what());
  }
  return result;
}

} // namespace cyxwiz::installer
