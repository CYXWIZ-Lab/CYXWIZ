#pragma once

#include "installer_product_removal.h"

namespace cyxwiz::installer::gui {

struct InstallerRemovalViewState {
  bool acknowledged = false;
};

bool RenderInstallerRemovalControl(
    InstallerRemovalViewState &view_state,
    const InstallerProductRemovalState &removal,
    bool operation_running);

} // namespace cyxwiz::installer::gui
