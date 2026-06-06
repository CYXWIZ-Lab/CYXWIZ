#include "theme.h"
#include <imgui.h>

namespace gui {

// Global theme instance
static Theme g_theme;

Theme& GetTheme() {
    return g_theme;
}

Theme::Theme() {
    // Don't apply preset here - contexts don't exist yet during static init
    // The application will call ApplyPreset() after ImGui/ImNodes contexts are created
}

const char* Theme::GetPresetName(ThemePreset preset) {
    switch (preset) {
        // CyxWiz branded
        case ThemePreset::CyxWizDark:      return "CyxWiz Dark";
        case ThemePreset::CyxWizLight:     return "CyxWiz Light";
        // Classic IDE
        case ThemePreset::VSCodeDark:      return "VS Code Dark";
        case ThemePreset::UnrealEngine:    return "Unreal Engine";
        case ThemePreset::ModernDark:      return "Modern Dark";
        case ThemePreset::HighContrast:    return "High Contrast";
        // Vibrant themes
        case ThemePreset::Dracula:         return "Dracula";
        case ThemePreset::OneDarkPro:      return "One Dark Pro";
        case ThemePreset::Nord:            return "Nord";
        case ThemePreset::CatppuccinMocha: return "Catppuccin Mocha";
        // CyxOS Platform themes
        case ThemePreset::CyxOSAqua:       return "CyxOS Aqua";
        case ThemePreset::CyxOSFluent:     return "CyxOS Fluent";
        case ThemePreset::CyxOSCoder:      return "CyxOS Coder";
        case ThemePreset::CyxOSOffice:     return "CyxOS Office";
        // CyxOS Retro TUI themes
        case ThemePreset::CyxOSTuiClassic: return "CyxOS TUI Classic";
        case ThemePreset::CyxOSTuiMatrix:  return "CyxOS TUI Matrix";
        case ThemePreset::CyxOSTuiAmber:   return "CyxOS TUI Amber";
        default:                           return "Unknown";
    }
}

std::vector<ThemePreset> Theme::GetAvailablePresets() {
    return {
        // CyxWiz branded
        ThemePreset::CyxWizDark,
        ThemePreset::CyxWizLight,
        // Classic IDE
        ThemePreset::VSCodeDark,
        ThemePreset::UnrealEngine,
        ThemePreset::ModernDark,
        ThemePreset::HighContrast,
        // Vibrant themes
        ThemePreset::Dracula,
        ThemePreset::OneDarkPro,
        ThemePreset::Nord,
        ThemePreset::CatppuccinMocha,
        // CyxOS Platform themes
        ThemePreset::CyxOSAqua,
        ThemePreset::CyxOSFluent,
        ThemePreset::CyxOSCoder,
        ThemePreset::CyxOSOffice,
        // CyxOS Retro TUI themes
        ThemePreset::CyxOSTuiClassic,
        ThemePreset::CyxOSTuiMatrix,
        ThemePreset::CyxOSTuiAmber
    };
}

void Theme::ApplyPreset(ThemePreset preset) {
    current_preset_ = preset;

    switch (preset) {
        // CyxWiz branded
        case ThemePreset::CyxWizDark:      ApplyCyxWizDark(); break;
        case ThemePreset::CyxWizLight:     ApplyCyxWizLight(); break;
        // Classic IDE
        case ThemePreset::VSCodeDark:      ApplyVSCodeDark(); break;
        case ThemePreset::UnrealEngine:    ApplyUnrealEngine(); break;
        case ThemePreset::ModernDark:      ApplyModernDark(); break;
        case ThemePreset::HighContrast:    ApplyHighContrast(); break;
        // Vibrant themes
        case ThemePreset::Dracula:         ApplyDracula(); break;
        case ThemePreset::OneDarkPro:      ApplyOneDarkPro(); break;
        case ThemePreset::Nord:            ApplyNord(); break;
        case ThemePreset::CatppuccinMocha: ApplyCatppuccinMocha(); break;
        // CyxOS Platform themes
        case ThemePreset::CyxOSAqua:       ApplyCyxOSAqua(); break;
        case ThemePreset::CyxOSFluent:     ApplyCyxOSFluent(); break;
        case ThemePreset::CyxOSCoder:      ApplyCyxOSCoder(); break;
        case ThemePreset::CyxOSOffice:     ApplyCyxOSOffice(); break;
        // CyxOS Retro TUI themes
        case ThemePreset::CyxOSTuiClassic: ApplyCyxOSTuiClassic(); break;
        case ThemePreset::CyxOSTuiMatrix:  ApplyCyxOSTuiMatrix(); break;
        case ThemePreset::CyxOSTuiAmber:   ApplyCyxOSTuiAmber(); break;
        default:                           ApplyCyxWizDark(); break;
    }

    ApplyStyleConfig();
    ApplyImNodesStyle();  // Apply matching node editor styling
    ApplyDockStyle();     // Apply matching dock tab styling
}

void Theme::ApplyConfig(const ThemeConfig& config) {
    config_ = config;
    ApplyStyleConfig();
}

void Theme::ApplyStyleConfig() {
    ImGuiStyle& style = ImGui::GetStyle();

    // Rounding
    style.WindowRounding = config_.window_rounding;
    style.FrameRounding = config_.frame_rounding;
    style.PopupRounding = config_.popup_rounding;
    style.ScrollbarRounding = config_.scrollbar_rounding;
    style.GrabRounding = config_.grab_rounding;
    style.TabRounding = config_.tab_rounding;

    // Borders
    style.WindowBorderSize = config_.window_border_size;
    style.FrameBorderSize = config_.frame_border_size;
    style.PopupBorderSize = config_.popup_border_size;

    // Padding and spacing
    style.WindowPadding = config_.window_padding;
    style.FramePadding = config_.frame_padding;
    style.ItemSpacing = config_.item_spacing;
    style.ItemInnerSpacing = config_.item_inner_spacing;

    // Sizes
    style.ScrollbarSize = config_.scrollbar_size;
    style.GrabMinSize = config_.grab_min_size;
    style.IndentSpacing = config_.indent_spacing;
}

void Theme::SetAccentColor(const ImVec4& color) {
    accent_color_ = color;
    // Re-apply current preset with new accent
    ApplyPreset(current_preset_);
}

bool Theme::RenderThemeSelector() {
    bool changed = false;

    if (ImGui::BeginCombo("Theme", GetPresetName(current_preset_))) {
        for (auto preset : GetAvailablePresets()) {
            bool is_selected = (current_preset_ == preset);
            if (ImGui::Selectable(GetPresetName(preset), is_selected)) {
                ApplyPreset(preset);
                changed = true;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    return changed;
}

} // namespace gui
