#pragma once

// Studio Debugger colours (TOFIX128): every colour comes from the active
// theme through the shared Console palette, so the debugger follows dark and
// light themes and uses the same status colours as the rest of the Engine.

#include "../../core/studio_debugger_presentation.h"
#include "../console_palette.h"

#include <imgui.h>

#include <string>

namespace cyxwiz {

inline const ::gui::ConsolePalette& DebuggerPalette() {
    return ::gui::CurrentConsolePalette();
}

inline ImVec4 DebuggerToneColor(DebuggerTone tone) {
    const auto& p = DebuggerPalette();
    switch (tone) {
        case DebuggerTone::Success: return p.success;
        case DebuggerTone::Warning: return p.warning;
        case DebuggerTone::Danger: return p.error;
        case DebuggerTone::Info: return p.info;
        case DebuggerTone::Muted: return p.muted;
        case DebuggerTone::Neutral: return p.text;
    }
    return p.text;
}

inline ImVec4 DebuggerWithAlpha(ImVec4 color, float alpha) {
    color.w = alpha;
    return color;
}

inline ImU32 DebuggerU32(const ImVec4& color, float alpha = 1.0f) {
    return ImGui::ColorConvertFloat4ToU32(DebuggerWithAlpha(color, color.w * alpha));
}

inline ImVec4 DebuggerTraceStatusColor(const std::string& status) {
    return DebuggerToneColor(TraceStatusTone(status));
}

inline ImVec4 DebuggerText() { return DebuggerPalette().text; }
inline ImVec4 DebuggerMuted() { return DebuggerPalette().muted; }
inline ImVec4 DebuggerFaint() { return DebuggerPalette().faint; }
inline ImVec4 DebuggerBright() { return DebuggerPalette().bright; }
inline ImVec4 DebuggerSuccess() { return DebuggerPalette().success; }
inline ImVec4 DebuggerWarning() { return DebuggerPalette().warning; }
inline ImVec4 DebuggerDanger() { return DebuggerPalette().error; }
inline ImVec4 DebuggerInfo() { return DebuggerPalette().info; }
inline ImVec4 DebuggerAccent() { return DebuggerPalette().accent; }
inline ImVec4 DebuggerAccentText() { return DebuggerPalette().accent_text; }
inline ImVec4 DebuggerPanelBg() { return DebuggerPalette().panel; }
inline ImVec4 DebuggerInputBg() { return DebuggerPalette().input; }
inline ImVec4 DebuggerBorder() { return DebuggerPalette().border; }

// A small rounded status pill: dot + label on a tinted background.
inline void DebuggerStatusPill(const char* label, DebuggerTone tone) {
    const ImVec4 color = DebuggerToneColor(tone);
    const ImVec2 text = ImGui::CalcTextSize(label);
    const float height = ImGui::GetTextLineHeight() + 4.0f;
    const float dot = 7.0f;
    const ImVec2 size(text.x + dot + 20.0f, height);
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, ImVec2(min.x + size.x, min.y + size.y),
                        DebuggerU32(color, 0.16f), height * 0.5f);
    draw->AddCircleFilled(ImVec2(min.x + 9.0f + dot * 0.5f, min.y + height * 0.5f),
                          dot * 0.5f, DebuggerU32(color));
    draw->AddText(ImVec2(min.x + 14.0f + dot, min.y + 2.0f), DebuggerU32(color), label);
    ImGui::Dummy(size);
}

} // namespace cyxwiz
