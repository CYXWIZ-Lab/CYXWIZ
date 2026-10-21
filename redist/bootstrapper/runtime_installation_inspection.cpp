#include "runtime_installation_inspection.h"

#include "backend_pack_platform.h"
#include "product_installation_receipt.h"

#include <system_error>

namespace cyxwiz::runtime {
namespace {
std::filesystem::file_status InspectPath(const std::filesystem::path &path,
                                         std::error_code &error) {
  auto status = std::filesystem::symlink_status(path, error);
  if (error == std::errc::no_such_file_or_directory)
    error.clear();
  return status;
}

RuntimeInstallationInspection Recovery(const std::string &reason) {
  return {RuntimeInstallationCondition::RecoveryRequired,
          {},
          "Recovery required: " + reason +
              ". Installation is blocked to preserve existing files. Automatic "
              "repair is not available; "
              "keep this folder for recovery, or select a different empty "
              "installation location."};
}
} // namespace

RuntimeInstallationInspection
InspectRuntimeInstallation(const std::filesystem::path &root) {
  namespace fs = std::filesystem;
  if (!root.is_absolute() || root.lexically_normal() == root.root_path())
    return Recovery("an absolute, non-root runtime location is required");
  std::error_code error;
  const auto root_status = InspectPath(root, error);
  if (error || (root_status.type() != fs::file_type::not_found &&
                root_status.type() != fs::file_type::directory))
    return Recovery("the runtime directory is unreadable or redirected");

  const auto active_path = root / "active-runtime.json";
  const auto active_status = InspectPath(active_path, error);
  if (error)
    return Recovery("the activation state cannot be inspected: " +
                    error.message());
  if (active_status.type() != fs::file_type::not_found) {
    if (active_status.type() != fs::file_type::regular)
      return Recovery("the activation state is not a regular file");
    RuntimeInstallationInspection result;
    std::string reason;
    ActiveRuntime resolved;
    if (!LoadActiveRuntimeState(active_path, result.active, reason) ||
        !ResolveRuntimeState(root, result.active, resolved, reason))
      return Recovery(reason);
    result.condition = RuntimeInstallationCondition::Active;
    result.message = "Existing installation found";
    return result;
  }

  // Download cache, trusted catalogs and private staging alone do not mean an
  // installed product exists: interrupted downloads must remain retryable.
  // Inspect only fixed product markers, never recursively scan package files.
  for (const auto *name :
       {"base", "packs", "installed-metadata", "rollback", "repair-backups"}) {
    const auto marker = root / name;
    const auto status = InspectPath(marker, error);
    if (error)
      return Recovery("cannot inspect " + std::string(name) + ": " +
                      error.message());
    if (status.type() == fs::file_type::not_found)
      continue;
    if (status.type() != fs::file_type::directory ||
        !fs::is_empty(marker, error) || error)
      return Recovery("activation state is missing but " + std::string(name) +
                      " remains");
  }
  for (const auto &marker :
       {ProductInstallationReceiptPath(root.parent_path()),
        root.parent_path() / CurrentRuntimeBootstrapperExecutableName(),
        root.parent_path() / CurrentProductRemovalFinalizerExecutableName()}) {
    const auto status = InspectPath(marker, error);
    if (error || status.type() != fs::file_type::not_found)
      return Recovery(
          "activation state is missing but installed product markers remain");
  }
  return {
      RuntimeInstallationCondition::Fresh, {}, "No installed runtime found"};
}
} // namespace cyxwiz::runtime
