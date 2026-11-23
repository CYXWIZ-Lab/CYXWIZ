#pragma once

// Small drawing helpers the MuJoCo panels share, next to the Engine's widgets
// (PluginContext::GetUi, plugin/plugin_ui.h).

#include "plugin/plugin_ui.h"

#include <imgui.h>

namespace cyxwiz::plugin::mujoco {

inline ImVec4 ToImVec4(const UiColor& c) { return ImVec4(c.r, c.g, c.b, c.a); }

inline ImU32 ToU32(const UiColor& c, float alpha = 1.0f) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(c.r, c.g, c.b, c.a * alpha));
}

// Moves to the same line when `width` still fits, else starts a new line, so
// nothing hides off the right edge of a narrow panel.
inline void SameLineIfFits(float width) {
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < width) ImGui::NewLine();
}

}  // namespace cyxwiz::plugin::mujoco
