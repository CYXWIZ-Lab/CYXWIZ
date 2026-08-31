#pragma once

#include "backend_pack_installer_platform.h"
#include "installer_operation.h"
#include "installer_removal_view.h"
#include "installer_theme.h"

#include <array>
#include <filesystem>
#include <set>
#include <string>

namespace cyxwiz::installer::gui {

enum class InstallerViewActionKind {
  None,
  RefreshCatalog,
  UseInstallLocation,
  ApplyPlan,
  CancelOperation,
  CancelAndClose,
  LaunchEngine,
  OpenInstalledManager,
  RemoveProduct,
  Close
};

struct InstallerViewAction {
  InstallerViewActionKind kind = InstallerViewActionKind::None;
  std::filesystem::path install_root;
  CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser;
  BackendPackInstallerPlan plan;
  bool launch_after_install = true;
};

struct InstallerViewState {
  BackendPackInstallChoice choice = BackendPackInstallChoice::Recommended;
  std::set<std::string> custom_selection;
  std::array<char, 2048> install_path_text{};
  CyxWizInstallScope install_scope = CyxWizInstallScope::CurrentUser;
  bool install_location_dirty = false;
  bool review_requested = false;
  bool launch_after_install = true;
  bool install_completed = false;
  bool engine_launched = false;
  bool cancellation_requested = false;
  bool close_confirmation_requested = false;
  bool pack_removal_confirmation_requested = false;
  std::string pending_pack_removal_id;
  std::string install_location_message;
  BackendPackInstallerPlan pending_plan;
  InstallerRemovalViewState removal;
  int page = 0;
};

InstallerViewAction RenderInstallerView(
    InstallerViewState &state, const InstallerCatalogState &catalog,
    const CyxWizInstallLocation &install_location,
    const InstallerProductRemovalState &product_removal,
    const std::string &platform_name, bool operation_running,
    bool operation_cancellable,
    const std::string &operation_message,
    const InstallerPlanExecutionProgress &operation_progress,
    const InstallerVisualAssets &assets);

} // namespace cyxwiz::installer::gui
