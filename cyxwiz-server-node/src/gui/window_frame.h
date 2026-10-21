// window_frame.h - the OS window frame follows the screen (TOFIX139 P7).
#pragma once

#include <imgui.h>

namespace cyxwiz::servernode::gui {

// Colours the window's title bar and border (Windows 11; dark mode stays on).
// Other platforms: nothing to do. Cheap to call every frame: it only talks to
// the OS when the colour changes.
void SetWindowFrameColor(const ImVec4& color);

}  // namespace cyxwiz::servernode::gui
