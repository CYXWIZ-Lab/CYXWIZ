#include "console_palette.h"
#include "ui_tokens.h"

#include "../core/appearance_options.h"
#include "theme.h"

namespace gui {

namespace {

ImVec4 Mix(const ImVec4 &a, const ImVec4 &b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t,
                  a.w + (b.w - a.w) * t);
}

ImVec4 Opaque(ImVec4 c) {
    c.w = 1.0f;
    return c;
}

}  // namespace

const ConsolePalette &CurrentConsolePalette() {
    static ConsolePalette palette;
    static int built_frame = -1;
    static int built_preset = -1;
    const int frame = ImGui::GetFrameCount();
    const int preset = static_cast<int>(GetTheme().GetCurrentPreset());
    if (frame == built_frame && preset == built_preset)
        return palette;
    built_frame = frame;
    built_preset = preset;

    const ImGuiStyle &style = ImGui::GetStyle();
    const ImVec4 window = Opaque(style.Colors[ImGuiCol_WindowBg]);
    const cyxwiz::ui::Tokens &tokens = cyxwiz::ui::CurrentTokens();
    const bool light = tokens.light;
    const ImVec4 text = Opaque(style.Colors[ImGuiCol_Text]);
    const ImVec4 muted = Opaque(style.Colors[ImGuiCol_TextDisabled]);
    const ImVec4 ink = light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);

    ConsolePalette p;
    p.light = light;
    p.text = text;
    p.muted = muted;
    p.faint = Mix(muted, window, 0.30f);
    p.bright = Mix(text, ink, 0.45f);
    p.panel = Opaque(style.Colors[ImGuiCol_PopupBg]);
    p.input = Mix(window, light ? ImVec4(1, 1, 1, 1) : ImVec4(0, 0, 0, 1), light ? 0.6f : 0.28f);
    p.bar = Mix(window, ink, 0.03f);
    p.border = Mix(window, ink, light ? 0.18f : 0.16f);
    p.inner_border = Mix(window, ink, light ? 0.10f : 0.07f);
    p.continuation = p.faint;

    // Brand accent and status colours are the shared tokens.
    p.accent = tokens.accent;
    p.accent_text = tokens.accent_text;
    p.success = tokens.success;
    p.warning = tokens.warning;
    p.error = tokens.error;
    p.critical = tokens.critical;
    p.info = tokens.info;

    if (light) {
        p.keyword = ImVec4(0.70f, 0.10f, 0.45f, 1.0f);
        p.builtin = ImVec4(0.05f, 0.38f, 0.68f, 1.0f);
        p.string = ImVec4(0.12f, 0.46f, 0.14f, 1.0f);
        p.number = ImVec4(0.62f, 0.38f, 0.00f, 1.0f);
        p.decorator = ImVec4(0.42f, 0.28f, 0.78f, 1.0f);
    } else {
        p.keyword = ImVec4(1.00f, 0.48f, 0.70f, 1.0f);
        p.builtin = ImVec4(0.47f, 0.78f, 1.00f, 1.0f);
        p.string = ImVec4(0.65f, 0.84f, 0.65f, 1.0f);
        p.number = ImVec4(0.96f, 0.77f, 0.42f, 1.0f);
        p.decorator = ImVec4(0.81f, 0.78f, 1.00f, 1.0f);
    }
    if (preset == static_cast<int>(ThemePreset::CyxWizDark)) {
        // The approved Console design's exact neutrals (tofix121 mockups).
        p.faint = ImVec4(0.42f, 0.45f, 0.54f, 1.0f);
        p.bright = ImVec4(0.95f, 0.96f, 0.97f, 1.0f);
        p.panel = ImVec4(0.067f, 0.086f, 0.118f, 1.0f);
        p.input = ImVec4(0.043f, 0.055f, 0.075f, 1.0f);
        p.bar = ImVec4(0.051f, 0.067f, 0.094f, 1.0f);
        p.border = ImVec4(0.141f, 0.196f, 0.322f, 1.0f);
        p.inner_border = ImVec4(0.10f, 0.13f, 0.19f, 1.0f);
        p.continuation = ImVec4(0.29f, 0.33f, 0.41f, 1.0f);
    }
    p.error_card = ImVec4(p.error.x, p.error.y, p.error.z, light ? 0.08f : 0.07f);
    p.selection = ImVec4(p.accent.x, p.accent.y, p.accent.z, light ? 0.14f : 0.12f);
    p.hover = light ? ImVec4(0, 0, 0, 0.035f) : ImVec4(1, 1, 1, 0.025f);
    palette = p;
    return palette;
}

}  // namespace gui
