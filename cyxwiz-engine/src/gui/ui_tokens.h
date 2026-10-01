#pragma once

// Design tokens for every CyxWiz screen (Engine and Installer): colours,
// spacing and rounding in one place, with light and dark values, derived
// from the active ImGui theme once per frame (TOFIX129 step 0.3).
//
// Rules:
//  - Screens read CurrentTokens(); they do not type colour literals.
//  - The light/dark decision is made here once (appearance::IsLightBackground).
//  - Status words, colours and icons are the shared vocabulary of the
//    cyxwiz-engine-frontend skill; StatusStyleFor() is the only source.
// No backend calls: the installer compiles this file too.

#include <imgui.h>

namespace cyxwiz::ui {

struct Tokens {
    bool light = false;

    // Text
    ImVec4 text;          // body text
    ImVec4 text_dim;      // secondary text, labels, hints
    ImVec4 text_faint;    // placeholders, separators in text
    ImVec4 text_bright;   // emphasised text, headings

    // Surfaces
    ImVec4 bg_window;     // the window itself
    ImVec4 bg_panel;      // popups, cards on a window
    ImVec4 bg_bar;        // headers, toolbars, status bar
    ImVec4 bg_raised;     // buttons, chips, table header
    ImVec4 bg_input;      // text fields
    ImVec4 hover;         // translucent hover wash
    ImVec4 selection;     // translucent selection wash
    ImVec4 border;        // fields, buttons, cards
    ImVec4 border_soft;   // separators, inner lines

    // Brand
    ImVec4 accent;        // filled primary button, selected marker
    ImVec4 accent_hover;
    ImVec4 accent_active;
    ImVec4 accent_text;   // links, badges, accent text on this background

    // Status
    ImVec4 success;
    ImVec4 warning;       // amber: needs attention, driver update
    ImVec4 caution;       // orange: failed, timed out, wrong result
    ImVec4 error;         // red: not supported, destructive
    ImVec4 critical;
    ImVec4 info;
    ImVec4 pending;       // not yet, unknown, idle
    ImVec4 running;       // in progress

    // Spacing (px at 1x density)
    float space_xs = 4.0f;
    float space_sm = 6.0f;
    float space_md = 8.0f;
    float space_lg = 12.0f;
    float space_xl = 16.0f;

    // Rounding
    float rounding_button = 6.0f;
    float rounding_field = 6.0f;
    float rounding_card = 8.0f;
    float rounding_chip = 10.0f;

    // Padding
    ImVec2 button_padding = ImVec2(16.0f, 8.0f);
    ImVec2 button_padding_small = ImVec2(10.0f, 4.0f);
    ImVec2 card_padding = ImVec2(14.0f, 10.0f);
};

// Rebuilt at most once per frame from the active ImGui style.
const Tokens& CurrentTokens();
// The same derivation for any style (tests, previews).
Tokens BuildTokens(const ImGuiStyle& style);

// Colour helpers shared by every screen.
ImVec4 Mix(const ImVec4& a, const ImVec4& b, float t);
ImVec4 WithAlpha(ImVec4 colour, float alpha);
unsigned int ToU32(const ImVec4& colour);

// Shared status vocabulary (words, colours and icons used by Engine and
// Installer alike).
enum class Status {
    Verified,          // done, passed, ready
    NotVerifiedYet,    // not run yet
    Failed,            // ran and failed, timed out, wrong result
    NotSupported,      // cannot work on this device, crashed
    NeedsDriverUpdate, // missing driver or provider
    Verifying,         // in progress
    Recommended        // badge
};

struct StatusStyle {
    ImVec4 colour;
    const char* icon;   // FontAwesome glyph from icons.h
    const char* label;  // the word users see
};

StatusStyle StatusStyleFor(Status status, const Tokens& tokens = CurrentTokens());

}  // namespace cyxwiz::ui
