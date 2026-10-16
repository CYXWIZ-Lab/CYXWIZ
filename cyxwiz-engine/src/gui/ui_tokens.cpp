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

namespace {

ImVec4 ScaleAt(const ImVec4* stops, int count, float v) {
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    const float at = v * static_cast<float>(count - 1);
    const int lo = at >= static_cast<float>(count - 1) ? count - 2 : static_cast<int>(at);
    return Mix(stops[lo], stops[lo + 1], at - static_cast<float>(lo));
}

}  // namespace

ImVec4 NodeColour(const Tokens& t, NodeHue hue) {
    const int i = static_cast<int>(hue);
    return t.node[i >= 0 && i < Tokens::kNodeHueCount ? i : static_cast<int>(NodeHue::Neutral)];
}

ImVec4 SeriesColour(const Tokens& t, size_t index) {
    const ImVec4 base = t.series[index % Tokens::kSeriesCount];
    const size_t round = (index / Tokens::kSeriesCount) % 3;
    return round == 0 ? base : WithAlpha(Mix(base, t.bg_window, 0.3f * static_cast<float>(round)), 1.0f);
}

ImVec4 SequentialAt(const Tokens& t, float v) {
    return ScaleAt(t.scale_sequential, 4, v);
}

ImVec4 DivergingAt(const Tokens& t, float v) {
    return ScaleAt(t.scale_diverging, 3, (v + 1.0f) * 0.5f);
}

ImVec4 ReadableOn(const Tokens& t, const ImVec4& background) {
    // The theme's bright text or the window colour, whichever contrasts
    // more with the fill (mid-tone fills fail a plain light/dark split).
    const auto contrast = [&](const ImVec4& c) {
        const float a = appearance::Luminance(c.x, c.y, c.z);
        const float b = appearance::Luminance(background.x, background.y, background.z);
        return (a > b ? a + 0.05f : b + 0.05f) / (a > b ? b + 0.05f : a + 0.05f);
    };
    return contrast(t.text_bright) >= contrast(t.bg_window) ? t.text_bright : t.bg_window;
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

    // Plot series: violet, sky, amber, green, coral, pink. Each reads on the
    // window (3:1 or more) and differs from the others; the light set is
    // darker so it reads on a near-white window.
    if (t.light) {
        const ImVec4 series[] = {{0.357f, 0.239f, 0.961f, 1.0f}, {0.122f, 0.471f, 0.706f, 1.0f},
                                 {0.659f, 0.420f, 0.000f, 1.0f}, {0.055f, 0.502f, 0.282f, 1.0f},
                                 {0.776f, 0.184f, 0.157f, 1.0f}, {0.690f, 0.180f, 0.520f, 1.0f}};
        for (int i = 0; i < Tokens::kSeriesCount; ++i) t.series[i] = series[i];
    } else {
        const ImVec4 series[] = {{0.616f, 0.561f, 1.000f, 1.0f}, {0.498f, 0.784f, 0.910f, 1.0f},
                                 {0.902f, 0.749f, 0.290f, 1.0f}, {0.239f, 0.839f, 0.549f, 1.0f},
                                 {1.000f, 0.478f, 0.451f, 1.0f}, {0.949f, 0.482f, 0.769f, 1.0f}};
        for (int i = 0; i < Tokens::kSeriesCount; ++i) t.series[i] = series[i];
    }
    // Node families: each one's most-used colour from the old per-type node
    // table, so the canvas keeps its look (Data Input blue, Dense green, ...).
    static_assert(static_cast<int>(NodeHue::Count) == Tokens::kNodeHueCount);
    const ImVec4 node[] = {
        {0.118f, 0.533f, 0.898f, 1.0f}  /* DataSources */,
        {1.000f, 0.596f, 0.000f, 1.0f}  /* DataTransform */,
        {0.612f, 0.153f, 0.690f, 1.0f}  /* Analytics */,
        {0.153f, 0.682f, 0.376f, 1.0f}  /* Layers */,
        {0.953f, 0.612f, 0.071f, 1.0f}  /* Activation */,
        {0.608f, 0.349f, 0.714f, 1.0f}  /* Pooling */,
        {0.925f, 0.439f, 0.388f, 1.0f}  /* Normalization */,
        {0.404f, 0.227f, 0.718f, 1.0f}  /* Attention */,
        {0.247f, 0.318f, 0.710f, 1.0f}  /* Recurrent */,
        {0.102f, 0.737f, 0.612f, 1.0f}  /* ShapeOps */,
        {0.545f, 0.765f, 0.290f, 1.0f}  /* MergeOps */,
        {0.361f, 0.420f, 0.753f, 1.0f}  /* Upsampling */,
        {0.753f, 0.224f, 0.169f, 1.0f}  /* Training */,
        {0.914f, 0.118f, 0.388f, 1.0f}  /* Regularization */,
        {0.506f, 0.780f, 0.518f, 1.0f}  /* Preprocessing */,
        {0.000f, 0.675f, 0.757f, 1.0f}  /* DataPipeline */,
        {0.000f, 0.588f, 0.533f, 1.0f}  /* TextProcessing */,
        {1.000f, 0.702f, 0.251f, 1.0f}  /* TimeSeries */,
        {0.494f, 0.341f, 0.761f, 1.0f}  /* Audio */,
        {0.161f, 0.384f, 1.000f, 1.0f}  /* DNN */,
        {0.957f, 0.443f, 0.424f, 1.0f}  /* RL */,
        {0.188f, 0.247f, 0.624f, 1.0f}  /* Signal */,
        {0.267f, 0.533f, 0.667f, 1.0f}  /* Plugin */,
        {0.376f, 0.490f, 0.545f, 1.0f}  /* Neutral */};
    for (int i = 0; i < Tokens::kNodeHueCount; ++i) t.node[i] = node[i];

    // The plot area is the window colour, a shade deeper (lighter on light).
    t.plot_bg = Mix(t.bg_window, t.light ? ImVec4(1, 1, 1, 1) : ImVec4(0, 0, 0, 1), t.light ? 0.5f : 0.08f);
    t.plot_grid = WithAlpha(t.text_dim, t.light ? 0.18f : 0.14f);
    if (t.light) {
        t.scale_sequential[0] = ImVec4(0.93f, 0.91f, 1.00f, 1.0f);
        t.scale_sequential[1] = ImVec4(0.70f, 0.65f, 1.00f, 1.0f);
        t.scale_sequential[2] = t.accent;
        t.scale_sequential[3] = ImVec4(0.20f, 0.12f, 0.60f, 1.0f);
    } else {
        t.scale_sequential[0] = Mix(t.plot_bg, t.accent, 0.25f);
        t.scale_sequential[1] = t.accent;
        t.scale_sequential[2] = ImVec4(0.702f, 0.651f, 1.0f, 1.0f);
        t.scale_sequential[3] = ImVec4(0.92f, 0.90f, 1.00f, 1.0f);
    }
    t.scale_diverging[0] = ImVec4(0.29f, 0.56f, 0.85f, 1.0f);
    t.scale_diverging[1] = t.plot_bg;
    t.scale_diverging[2] = ImVec4(0.88f, 0.44f, 0.31f, 1.0f);
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

void PushListRowColours(const Tokens& t) {
    ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
}

void PopListRowColours() {
    ImGui::PopStyleColor(3);
}

}  // namespace cyxwiz::ui
