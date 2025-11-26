#pragma once

#include <imgui.h>
#include <imnodes.h>
#include <string>
#include <vector>
#include <functional>

namespace gui {

/**
 * Available theme presets for the CyxWiz Engine
 */
enum class ThemePreset {
    CyxWizDark,      // Custom dark theme with CyxWiz branding
    CyxWizLight,     // Light theme variant
    VSCodeDark,      // Visual Studio Code inspired dark theme
    UnrealEngine,    // Unreal Engine inspired theme
    ModernDark,      // Clean modern dark theme
    HighContrast,    // High contrast for accessibility
    COUNT
};

/**
 * Theme configuration structure
 */
struct ThemeConfig {
    // Window styling
    float window_rounding = 4.0f;
    float frame_rounding = 2.0f;
    float popup_rounding = 4.0f;
    float scrollbar_rounding = 4.0f;
    float grab_rounding = 2.0f;
    float tab_rounding = 4.0f;

    // Borders
    float window_border_size = 1.0f;
    float frame_border_size = 0.0f;
    float popup_border_size = 1.0f;

    // Padding and spacing
    ImVec2 window_padding = ImVec2(8.0f, 8.0f);
    ImVec2 frame_padding = ImVec2(6.0f, 4.0f);
    ImVec2 item_spacing = ImVec2(8.0f, 4.0f);
    ImVec2 item_inner_spacing = ImVec2(4.0f, 4.0f);

    // Sizes
    float scrollbar_size = 14.0f;
    float grab_min_size = 12.0f;
    float indent_spacing = 20.0f;
};

/**
 * Theme system for CyxWiz Engine
 * Provides professional-looking themes and easy customization
 */
class Theme {
public:
    Theme();
    ~Theme() = default;

    // Apply a preset theme
    void ApplyPreset(ThemePreset preset);

    // Get current preset
    ThemePreset GetCurrentPreset() const { return current_preset_; }

    // Get preset name for display
    static const char* GetPresetName(ThemePreset preset);

    // Get all available presets
    static std::vector<ThemePreset> GetAvailablePresets();

    // Apply custom config
    void ApplyConfig(const ThemeConfig& config);

    // Get current config
    const ThemeConfig& GetConfig() const { return config_; }

    // Render theme selector UI (returns true if theme changed)
    bool RenderThemeSelector();

    // Color customization
    void SetAccentColor(const ImVec4& color);
    ImVec4 GetAccentColor() const { return accent_color_; }

private:
    // Theme application methods
    void ApplyCyxWizDark();
    void ApplyCyxWizLight();
    void ApplyVSCodeDark();
    void ApplyUnrealEngine();
    void ApplyModernDark();
    void ApplyHighContrast();

    // Apply common style settings
    void ApplyStyleConfig();

    // Apply ImNodes styling based on current theme
    void ApplyImNodesStyle();

    ThemePreset current_preset_ = ThemePreset::CyxWizDark;
    ThemeConfig config_;
    ImVec4 accent_color_ = ImVec4(0.26f, 0.59f, 0.98f, 1.0f);  // Default blue accent
};

// Global theme instance
Theme& GetTheme();

} // namespace gui
