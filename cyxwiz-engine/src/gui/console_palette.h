#pragma once

// Colours for the redesigned Console views (REPL, Logs, Commands) derived
// from the active ImGui theme, with dark and light variants of the status
// and syntax colours (tofix121). Neutrals follow the theme; status and
// syntax colours are tuned per background so they stay readable.

#include <imgui.h>

namespace gui {

struct ConsolePalette {
    ImVec4 text, muted, faint, bright;
    ImVec4 panel, input, bar, border, inner_border;
    ImVec4 accent, accent_text, continuation;
    ImVec4 success, warning, error, info, critical;
    ImVec4 keyword, builtin, string, number, decorator;
    ImVec4 error_card, selection, hover;
    bool light = false;
};

// Rebuilt at most once per frame.
const ConsolePalette &CurrentConsolePalette();

}  // namespace gui
