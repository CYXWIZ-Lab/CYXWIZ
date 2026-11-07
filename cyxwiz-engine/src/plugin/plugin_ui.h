// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

#include <cstddef>
#include <string>

namespace cyxwiz::plugin {

// ============================================================================
// PluginUi - the Engine's widgets and theme colours for plugin panels
// (plugin API 1.4).
//
// A plugin draws its panels in the Engine's ImGui context
// (IPlugin::SetImGuiContext). Plain ImGui widgets there use the theme's ImGui
// colours, but not the Engine's button styles or its brand and status
// colours. These calls draw exactly what the Engine's own screens draw and
// follow the user's theme, so a plugin window looks like the rest of the
// Engine. Get it with PluginContext::GetUi(); call it only on the UI thread,
// inside IPanelProvider::RenderPanel.
// ============================================================================

// A colour in 0..1 floats (ImVec4{r, g, b, a} in ImGui).
struct UiColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;
};

// The theme's colours. Read them every frame: the user can switch themes.
struct UiColors {
    UiColor text;          // body text
    UiColor text_dim;      // secondary text, labels, hints
    UiColor text_bright;   // headings, emphasised text
    UiColor bg_window;     // the window
    UiColor bg_panel;      // cards and popups on the window
    UiColor bg_raised;     // buttons, chips
    UiColor bg_input;      // text fields
    UiColor accent;        // the brand colour (filled primary button)
    UiColor accent_text;   // links and accent text
    UiColor success;       // done, loaded, verified
    UiColor warning;       // needs attention
    UiColor caution;       // failed, timed out
    UiColor error;         // not possible, destructive
    UiColor pending;       // idle, not started
    UiColor running;       // in progress
};

enum class UiButtonSize { Small, Regular };

class PluginUi {
public:
    virtual ~PluginUi() = default;

    virtual UiColors Colors() const = 0;

    // Buttons. Each returns true when clicked while enabled; a disabled button
    // stays visible and shows `disabled_reason` on hover. `width` 0 fits the
    // label. Use one PrimaryButton per area (its main action), SecondaryButton
    // for the others, LinkButton for navigation, DangerButton for removal.
    virtual bool PrimaryButton(const char* label, bool enabled = true, const char* disabled_reason = nullptr,
                               UiButtonSize size = UiButtonSize::Regular, float width = 0.0f) = 0;
    virtual bool SecondaryButton(const char* label, bool enabled = true, const char* disabled_reason = nullptr,
                                 UiButtonSize size = UiButtonSize::Regular, float width = 0.0f) = 0;
    virtual bool LinkButton(const char* label, bool enabled = true) = 0;
    virtual bool DangerButton(const char* label, bool enabled = true, const char* disabled_reason = nullptr,
                              UiButtonSize size = UiButtonSize::Regular) = 0;
    // Width and height a button takes, to right-align it, size a column or
    // centre a row of mixed items.
    virtual float ButtonWidth(const char* label, UiButtonSize size) const = 0;
    virtual float ButtonHeight(UiButtonSize size) const = 0;

    // One of several options side by side. True when `selected` changed.
    virtual bool SegmentedControl(const char* id, const char* const* labels, int count, int* selected) = 0;
    virtual float SegmentedControlWidth(const char* const* labels, int count) const = 0;

    // Rounded status pill with a dot in `dot` (Running, Paused...). True when
    // clicked.
    virtual bool StatusPill(const char* id, const char* text, const UiColor& dot) = 0;
    virtual float StatusPillWidth(const char* text) const = 0;

    // A number the user types (no slider or drag): the draft applies on Enter
    // or when the field is left, clamped to min..max (shown on hover). `step`
    // > 0 adds - / + buttons. `width` is the field without the buttons. True
    // when the value changed.
    virtual bool NumberField(const char* id, double* value, double min, double max, double step = 0.0,
                             const char* format = "%.2f", float width = 80.0f, const char* hint = nullptr) = 0;

    // Search field with an icon and a hint; `width` 0 fills the line. True
    // when the text changed.
    virtual bool SearchField(const char* id, char* buffer, std::size_t size, const char* hint = "Search",
                             float width = 0.0f) = 0;

    // Centred empty state: title and a muted hint. Follow it with the actions.
    virtual void EmptyState(const char* title, const char* hint = nullptr) = 0;

    // The system's open-file dialog. `filter_name` and `extensions` ("xml" or
    // "png,jpg") limit the choice. Returns the chosen path, or empty when the
    // user cancels. Blocks until the dialog closes.
    virtual std::string OpenFileDialog(const std::string& title, const std::string& filter_name,
                                       const std::string& extensions) = 0;

protected:
    PluginUi() = default;
    PluginUi(const PluginUi&) = delete;
    PluginUi& operator=(const PluginUi&) = delete;
};

} // namespace cyxwiz::plugin
