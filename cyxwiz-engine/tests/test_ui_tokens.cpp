// Design tokens (TOFIX129 step 0.3): the status vocabulary and the light and
// dark colour sets, checked for the words, icons and contrast the
// cyxwiz-engine-frontend skill requires.
#include "../src/gui/ui_tokens.h"
#include "../src/core/appearance_options.h"

#include <imgui.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::ui;

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool Near(const ImVec4& a, float r, float g, float b) {
    return std::fabs(a.x - r) < 0.005f && std::fabs(a.y - g) < 0.005f && std::fabs(a.z - b) < 0.005f;
}

float Contrast(const ImVec4& fg, const ImVec4& bg) {
    const float l1 = cyxwiz::appearance::Luminance(fg.x, fg.y, fg.z);
    const float l2 = cyxwiz::appearance::Luminance(bg.x, bg.y, bg.z);
    const float hi = std::max(l1, l2), lo = std::min(l1, l2);
    return (hi + 0.05f) / (lo + 0.05f);
}

ImGuiStyle StyleWithWindow(float r, float g, float b) {
    ImGuiStyle style;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(r, g, b, 1.0f);
    style.Colors[ImGuiCol_Text] = r > 0.5f ? ImVec4(0.12f, 0.13f, 0.16f, 1.0f) : ImVec4(0.839f, 0.859f, 0.902f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = r > 0.5f ? ImVec4(0.40f, 0.42f, 0.48f, 1.0f) : ImVec4(0.545f, 0.580f, 0.655f, 1.0f);
    return style;
}

}  // namespace

int main() {
    // CyxWiz Dark window background.
    const Tokens dark = BuildTokens(StyleWithWindow(0.059f, 0.075f, 0.098f));
    Check(!dark.light, "navy window is dark");
    // The approved status vocabulary (skill table), exact values.
    Check(Near(dark.success, 0.24f, 0.84f, 0.55f), "dark success");
    Check(Near(dark.pending, 0.64f, 0.69f, 0.78f), "dark pending");
    Check(Near(dark.caution, 0.96f, 0.63f, 0.29f), "dark failed (caution)");
    Check(Near(dark.error, 1.00f, 0.48f, 0.45f), "dark not supported (error)");
    Check(Near(dark.warning, 0.90f, 0.75f, 0.29f), "dark needs driver update (warning)");
    Check(Near(dark.running, 0.70f, 0.65f, 1.00f), "dark verifying (running)");
    Check(Near(dark.accent_text, 0.702f, 0.651f, 1.0f), "dark badge (accent text)");
    Check(Near(dark.accent, 0.357f, 0.239f, 0.961f), "brand purple");

    // CyxWiz Light window background: every status colour must read.
    const Tokens light = BuildTokens(StyleWithWindow(0.96f, 0.96f, 0.97f));
    Check(light.light, "near-white window is light");
    Check(Near(light.accent, 0.357f, 0.239f, 0.961f), "brand purple is the same on light");
    const ImVec4 window(0.96f, 0.96f, 0.97f, 1.0f);
    const struct { const char* name; ImVec4 colour; } light_colours[] = {
        {"success", light.success}, {"warning", light.warning}, {"caution", light.caution},
        {"error", light.error}, {"critical", light.critical}, {"info", light.info},
        {"pending", light.pending}, {"running", light.running}, {"accent_text", light.accent_text}};
    for (const auto& c : light_colours) {
        const float ratio = Contrast(c.colour, window);
        Check(ratio >= 4.5f, std::string("light ") + c.name + " contrast >= 4.5 (got " + std::to_string(ratio) + ")");
    }
    // And the dark set against the CyxWiz Dark window.
    const ImVec4 navy(0.059f, 0.075f, 0.098f, 1.0f);
    const struct { const char* name; ImVec4 colour; } dark_colours[] = {
        {"success", dark.success}, {"warning", dark.warning}, {"caution", dark.caution},
        {"error", dark.error}, {"info", dark.info}, {"pending", dark.pending},
        {"running", dark.running}, {"accent_text", dark.accent_text}};
    for (const auto& c : dark_colours) {
        const float ratio = Contrast(c.colour, navy);
        Check(ratio >= 4.5f, std::string("dark ") + c.name + " contrast >= 4.5 (got " + std::to_string(ratio) + ")");
    }

    // Plot series (TOFIX134 P1): six colours that read on the window (3:1,
    // the graphics minimum) on dark, Unreal grey and light windows, and
    // that differ from each other.
    const Tokens grey = BuildTokens(StyleWithWindow(0.161f, 0.161f, 0.161f));
    const struct { const char* name; const Tokens* tokens; ImVec4 window; } sets[] = {
        {"dark", &dark, navy}, {"grey", &grey, ImVec4(0.161f, 0.161f, 0.161f, 1.0f)}, {"light", &light, window}};
    for (const auto& set : sets) {
        for (int i = 0; i < Tokens::kSeriesCount; ++i) {
            const float ratio = Contrast(set.tokens->series[i], set.window);
            Check(ratio >= 3.0f, std::string(set.name) + " series " + std::to_string(i) + " contrast >= 3 (got " +
                                     std::to_string(ratio) + ")");
            for (int j = i + 1; j < Tokens::kSeriesCount; ++j) {
                const ImVec4 a = set.tokens->series[i], b = set.tokens->series[j];
                const float d = std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
                Check(d >= 0.25f, std::string(set.name) + " series " + std::to_string(i) + " and " + std::to_string(j) +
                                      " differ (distance " + std::to_string(d) + ")");
            }
        }
        // The plot area follows the window colour (one surface rule).
        const ImVec4 bg = set.tokens->plot_bg;
        Check(std::fabs(bg.x - set.window.x) < 0.06f && std::fabs(bg.z - set.window.z) < 0.06f,
              std::string(set.name) + " plot area is the window colour, a shade off");
    }

    // Status vocabulary: the words users see, with an icon each.
    const struct { Status status; const char* label; } words[] = {
        {Status::Verified, "Verified"}, {Status::NotVerifiedYet, "Not verified yet"}, {Status::Failed, "Failed"},
        {Status::NotSupported, "Not supported on this device"}, {Status::NeedsDriverUpdate, "Needs driver update"},
        {Status::Verifying, "Verifying..."}, {Status::Recommended, "Recommended"}};
    for (const auto& w : words) {
        const StatusStyle s = StatusStyleFor(w.status, dark);
        Check(std::string(s.label) == w.label, std::string("status word: ") + w.label);
        Check(s.icon && s.icon[0], std::string("status icon: ") + w.label);
    }
    Check(Near(StatusStyleFor(Status::Failed, dark).colour, 0.96f, 0.63f, 0.29f), "Failed uses caution orange");
    Check(Near(StatusStyleFor(Status::NotSupported, dark).colour, 1.0f, 0.48f, 0.45f), "Not supported uses error red");

    // Spacing and rounding scale.
    Check(dark.space_xs < dark.space_sm && dark.space_sm < dark.space_md && dark.space_md < dark.space_lg &&
          dark.space_lg < dark.space_xl, "spacing scale ascends");
    Check(dark.rounding_button == 6.0f && dark.rounding_card == 8.0f, "button 6, card 8");
    Check(dark.button_padding.x == 16.0f && dark.button_padding.y == 8.0f, "regular button padding 16x8");
    Check(dark.button_padding_small.x == 10.0f && dark.button_padding_small.y == 4.0f, "small button padding 10x4");

    std::cout << "ui tokens: dark and light sets, 7 status words, contrast >= 4.5. OK\n";
    return 0;
}
