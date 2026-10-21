#pragma once

#include "installer_product_removal.h"

namespace cyxwiz::installer::gui {

struct InstallerRemovalViewState {
  bool acknowledged = false;
  bool open_requested = false;
};

enum class InstallerRemovalViewAction {
  None,
  OpenInstalledManager,
  RemoveProduct,
};

InstallerRemovalViewAction RenderInstallerRemovalControl(
    InstallerRemovalViewState &view_state,
    const InstallerProductRemovalState &removal,
    bool operation_running);

} // namespace cyxwiz::installer::gui
