// One Signal Scope's trace as the Scope's Properties preview and the
// Simulation Monitor draw it (TOFIX195 S2.4).
#pragma once

#include <imgui.h>
#include <vector>

namespace gui {

// Time runs along X and follows the newest sample (at least 2 s shown);
// auto_scale fits Y to the samples; series_label names the line in the
// legend. Draws nothing for an empty trace.
void DrawScopeTrace(const char* id, const char* series_label, const std::vector<float>& times,
                    const std::vector<float>& values, const ImVec2& size, bool auto_scale, const ImVec4& colour);

}  // namespace gui
