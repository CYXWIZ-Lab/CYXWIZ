#pragma once

// The layer capture choices (When, every N epochs, What, On), drawn once for
// Preferences > Studio Debugger and the Studio Debugger's capture control
// (TOFIX178 6.2d). Edits `settings` in place; true when something changed.

#include "../../core/engine_config.h"

namespace cyxwiz {

bool RenderLayerCaptureControls(core::StudioDebuggerConfig& settings);

}  // namespace cyxwiz
