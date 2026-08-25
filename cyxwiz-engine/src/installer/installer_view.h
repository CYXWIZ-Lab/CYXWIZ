#pragma once

#include "backend_pack_installer_platform.h"
#include "installer_operation.h"
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
  Close
};

struct InstallerViewAction {
  InstallerViewActionKind kind = InstallerViewActionKind::None;
  std::filesystem::path install_root;
  CyxWizInstallScope scope = CyxWizInstallScope::CurrentUser;
  BackendPackInstallerPlan plan;
};

struct InstallerViewState {
  BackendPackInstallChoice choice = BackendPackInstallChoice::Recommended;
  std::set<std::string> custom_selection;
  std::array<char, 2048> install_path_text{};
  CyxWizInstallScope install_scope = CyxWizInstallScope::CurrentUser;
  bool install_location_dirty = false;
  std::string install_location_message;
  BackendPackInstallerPlan pending_plan;
  int page = 0;
};

InstallerViewAction RenderInstallerView(
    InstallerViewState &state, const InstallerCatalogState &catalog,
    const CyxWizInstallLocation &install_location,
    const std::string &platform_name, bool operation_running,
    const std::string &operation_message,
    const InstallerPlanExecutionProgress &operation_progress,
    const InstallerVisualAssets &assets);

} // namespace cyxwiz::installer::gui
