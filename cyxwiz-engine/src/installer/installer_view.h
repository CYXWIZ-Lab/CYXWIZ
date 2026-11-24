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
  bool uninstall_completed = false;
  // Why the last uninstall was refused (it restored everything); shown
  // above the buttons until the next action, so Launch CyxWiz never reads
  // as a finished uninstall.
  std::string uninstall_refused;
  bool engine_launched = false;
  bool cancellation_requested = false;
  bool close_confirmation_requested = false;
  bool pack_removal_confirmation_requested = false;
  std::string pending_pack_removal_id;
  std::string install_location_message;
  BackendPackInstallerPlan pending_plan;
  InstallerRemovalViewState removal;
  int page = 0;
  // A tab to open on the next frame (1 = Individual components), when the
  // Engine opens the installer for one component; -1 = none.
  int requested_page = -1;
  // The component (pack id, or an add-on's plugin id) the page scrolls to
  // once it is drawn; the
  // scroll is applied for a few frames, while auto-sized cards measure.
  std::string focus_pack_id;
  int focus_frames = 0;
  // The summary footer's height (buttons and notes) as drawn last frame: the
  // summary details take the rest of the column. 0 before the first frame.
  float summary_footer_height = 0.0f;
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
