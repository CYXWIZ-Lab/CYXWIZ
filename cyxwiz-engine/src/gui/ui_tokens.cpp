#include "ui_tokens.h"

#include "icons.h"
#include "../core/appearance_options.h"

namespace cyxwiz::ui {

ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

ImVec4 WithAlpha(ImVec4 colour, float alpha) {
    colour.w = alpha;
    return colour;
}

unsigned int ToU32(const ImVec4& colour) {
    return ImGui::ColorConvertFloat4ToU32(colour);
}

Tokens BuildTokens(const ImGuiStyle& style) {
    Tokens t;
    t.bg_window = WithAlpha(style.Colors[ImGuiCol_WindowBg], 1.0f);
    t.light = appearance::IsLightBackground(t.bg_window.x, t.bg_window.y, t.bg_window.z);
    const ImVec4 ink = t.light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);

    // Neutrals follow the theme so every preset keeps its own character.
    t.text = WithAlpha(style.Colors[ImGuiCol_Text], 1.0f);
    t.text_dim = WithAlpha(style.Colors[ImGuiCol_TextDisabled], 1.0f);
    t.text_faint = Mix(t.text_dim, t.bg_window, 0.30f);
    t.text_bright = Mix(t.text, ink, 0.45f);
    t.bg_panel = WithAlpha(style.Colors[ImGuiCol_PopupBg], 1.0f);
    t.bg_bar = WithAlpha(style.Colors[ImGuiCol_MenuBarBg], 1.0f);
    t.bg_raised = WithAlpha(style.Colors[ImGuiCol_Button], 1.0f);
    t.bg_input = WithAlpha(style.Colors[ImGuiCol_FrameBg], 1.0f);
    t.hover = t.light ? ImVec4(0, 0, 0, 0.035f) : ImVec4(1, 1, 1, 0.025f);
    t.border = WithAlpha(style.Colors[ImGuiCol_Border], 1.0f);
    t.border_soft = WithAlpha(style.Colors[ImGuiCol_Separator], 1.0f);

    // Brand purple is the same everywhere; its text tint depends on the
    // background (approved in tofix119 C and tofix121).
    t.accent = ImVec4(0.357f, 0.239f, 0.961f, 1.0f);
    t.accent_hover = ImVec4(0.439f, 0.333f, 0.980f, 1.0f);
    t.accent_active = ImVec4(0.290f, 0.192f, 0.820f, 1.0f);
    t.accent_text = t.light ? ImVec4(0.290f, 0.192f, 0.820f, 1.0f) : ImVec4(0.702f, 0.651f, 1.0f, 1.0f);
    t.selection = WithAlpha(t.accent, t.light ? 0.14f : 0.12f);

    // Status colours: the dark set is the approved vocabulary; the light
    // set keeps at least 4.5:1 against a near-white window.
    if (t.light) {
        t.success = ImVec4(0.06f, 0.47f, 0.26f, 1.0f);
        t.warning = ImVec4(0.60f, 0.38f, 0.00f, 1.0f);
        t.caution = ImVec4(0.70f, 0.30f, 0.05f, 1.0f);
        t.error = ImVec4(0.78f, 0.17f, 0.15f, 1.0f);
        t.critical = ImVec4(0.72f, 0.05f, 0.25f, 1.0f);
        t.info = ImVec4(0.22f, 0.33f, 0.82f, 1.0f);
        t.pending = ImVec4(0.38f, 0.42f, 0.50f, 1.0f);
        t.running = ImVec4(0.35f, 0.26f, 0.78f, 1.0f);
    } else {
        t.success = ImVec4(0.24f, 0.84f, 0.55f, 1.0f);
        t.warning = ImVec4(0.90f, 0.75f, 0.29f, 1.0f);
        t.caution = ImVec4(0.96f, 0.63f, 0.29f, 1.0f);
        t.error = ImVec4(1.00f, 0.48f, 0.45f, 1.0f);
        t.critical = ImVec4(1.00f, 0.30f, 0.43f, 1.0f);
        t.info = ImVec4(0.60f, 0.65f, 1.00f, 1.0f);
        t.pending = ImVec4(0.64f, 0.69f, 0.78f, 1.0f);
        t.running = ImVec4(0.70f, 0.65f, 1.00f, 1.0f);
    }
    return t;
}

const Tokens& CurrentTokens() {
    static Tokens tokens;
    static int built_frame = -1;
    const int frame = ImGui::GetFrameCount();
    if (frame != built_frame) {
        tokens = BuildTokens(ImGui::GetStyle());
        built_frame = frame;
    }
    return tokens;
}

StatusStyle StatusStyleFor(Status status, const Tokens& t) {
    switch (status) {
        case Status::Verified: return {t.success, ICON_FA_CIRCLE_CHECK, "Verified"};
        case Status::NotVerifiedYet: return {t.pending, ICON_FA_CLOCK, "Not verified yet"};
        case Status::Failed: return {t.caution, ICON_FA_TRIANGLE_EXCLAMATION, "Failed"};
        case Status::NotSupported: return {t.error, ICON_FA_CIRCLE_XMARK, "Not supported on this device"};
        case Status::NeedsDriverUpdate: return {t.warning, ICON_FA_WRENCH, "Needs driver update"};
        case Status::Verifying: return {t.running, ICON_FA_SPINNER, "Verifying..."};
        case Status::Recommended: return {t.accent_text, ICON_FA_STAR, "Recommended"};
    }
    return {t.text_dim, "", ""};
}

}  // namespace cyxwiz::ui
