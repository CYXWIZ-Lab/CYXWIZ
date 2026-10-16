#include "theme.h"
#include "dock_style.h"
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
        case ThemePreset::CyxWizLaunch:    return "CyxWiz Launch";
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
        ThemePreset::CyxWizLaunch,
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

const char* Theme::GetPresetGroup(ThemePreset preset) {
    switch (preset) {
        case ThemePreset::CyxWizDark:
        case ThemePreset::CyxWizLight:
        case ThemePreset::CyxWizLaunch:    return "CyxWiz";
        case ThemePreset::VSCodeDark:
        case ThemePreset::UnrealEngine:
        case ThemePreset::ModernDark:
        case ThemePreset::HighContrast:    return "IDE";
        case ThemePreset::Dracula:
        case ThemePreset::OneDarkPro:
        case ThemePreset::Nord:
        case ThemePreset::CatppuccinMocha: return "Vibrant";
        case ThemePreset::CyxOSAqua:
        case ThemePreset::CyxOSFluent:
        case ThemePreset::CyxOSCoder:
        case ThemePreset::CyxOSOffice:     return "CyxOS";
        case ThemePreset::CyxOSTuiClassic:
        case ThemePreset::CyxOSTuiMatrix:
        case ThemePreset::CyxOSTuiAmber:   return "Retro terminal";
        default:                           return "Other";
    }
}

Theme::PresetSwatch Theme::GetPresetSwatch(ThemePreset preset) {
    // Colours taken from each preset's definition (theme_presets.cpp).
    const auto c = [](float r, float g, float b) { return ImVec4(r, g, b, 1.0f); };
    switch (preset) {
        case ThemePreset::CyxWizDark:      return {c(0.059f, 0.075f, 0.098f), c(0.067f, 0.086f, 0.118f), c(0.357f, 0.239f, 0.961f)};
        case ThemePreset::CyxWizLight:     return {c(0.96f, 0.96f, 0.97f), c(0.88f, 0.88f, 0.90f), c(0.20f, 0.50f, 0.80f)};
        case ThemePreset::CyxWizLaunch:    return {c(0.04f, 0.07f, 0.13f), c(0.06f, 0.10f, 0.17f), c(0.02f, 0.36f, 0.92f)};
        case ThemePreset::VSCodeDark:      return {c(0.118f, 0.118f, 0.118f), c(0.153f, 0.153f, 0.153f), c(0.075f, 0.463f, 0.788f)};
        case ThemePreset::UnrealEngine:    return {c(0.12f, 0.12f, 0.12f), c(0.16f, 0.16f, 0.16f), c(0.13f, 0.59f, 0.95f)};
        case ThemePreset::ModernDark:      return {c(0.10f, 0.10f, 0.12f), c(0.14f, 0.14f, 0.16f), c(0.40f, 0.55f, 0.80f)};
        case ThemePreset::HighContrast:    return {c(0.00f, 0.00f, 0.00f), c(0.10f, 0.10f, 0.10f), c(0.00f, 0.80f, 1.00f)};
        case ThemePreset::Dracula:         return {c(0.16f, 0.16f, 0.21f), c(0.22f, 0.22f, 0.28f), c(0.74f, 0.58f, 0.98f)};
        case ThemePreset::OneDarkPro:      return {c(0.16f, 0.17f, 0.20f), c(0.21f, 0.22f, 0.26f), c(0.38f, 0.69f, 0.94f)};
        case ThemePreset::Nord:            return {c(0.18f, 0.20f, 0.25f), c(0.26f, 0.30f, 0.37f), c(0.53f, 0.75f, 0.82f)};
        case ThemePreset::CatppuccinMocha: return {c(0.12f, 0.12f, 0.18f), c(0.19f, 0.20f, 0.27f), c(0.80f, 0.65f, 0.97f)};
        case ThemePreset::CyxOSAqua:       return {c(0.11f, 0.11f, 0.12f), c(0.18f, 0.18f, 0.18f), c(0.00f, 0.48f, 1.00f)};
        case ThemePreset::CyxOSFluent:     return {c(0.13f, 0.13f, 0.13f), c(0.18f, 0.18f, 0.18f), c(0.00f, 0.47f, 0.83f)};
        case ThemePreset::CyxOSCoder:      return {c(0.12f, 0.12f, 0.18f), c(0.20f, 0.20f, 0.26f), c(0.54f, 0.71f, 0.98f)};
        case ThemePreset::CyxOSOffice:     return {c(0.12f, 0.16f, 0.22f), c(0.22f, 0.25f, 0.32f), c(0.15f, 0.39f, 0.92f)};
        case ThemePreset::CyxOSTuiClassic: return {c(0.04f, 0.04f, 0.04f), c(0.05f, 0.05f, 0.05f), c(0.20f, 1.00f, 0.20f)};
        case ThemePreset::CyxOSTuiMatrix:  return {c(0.00f, 0.00f, 0.00f), c(0.00f, 0.07f, 0.00f), c(0.00f, 1.00f, 0.25f)};
        case ThemePreset::CyxOSTuiAmber:   return {c(0.05f, 0.04f, 0.02f), c(0.10f, 0.07f, 0.04f), c(1.00f, 0.69f, 0.00f)};
        default:                           return {c(0.059f, 0.075f, 0.098f), c(0.067f, 0.086f, 0.118f), c(0.357f, 0.239f, 0.961f)};
    }
}

void Theme::ApplyPreset(ThemePreset preset) {
    current_preset_ = preset;
    // Start from the defaults: a preset that does not set its own padding
    // or spacing must not inherit the previous preset's (TOFIX129 0.4).
    config_ = ThemeConfig{};
    ImGui::GetStyle().TabBarOverlineSize = 2.0f;

    switch (preset) {
        // CyxWiz branded
        case ThemePreset::CyxWizDark:      ApplyCyxWizDark(); break;
        case ThemePreset::CyxWizLight:     ApplyCyxWizLight(); break;
        case ThemePreset::CyxWizLaunch:    ApplyCyxWizLaunch(); break;
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
    HarmonizeSurfaces();  // last: the dock style set its own tab colours
}

// Owner 2026-10-02: every surface follows the window background. Presets
// (and the dock style after them) gave table headers, scrollbars, title
// bars, dock tabs and the empty dock area their own darker or lighter greys
// (Unreal Engine: window 41, table header 31, title bar 20/31, tabs 31, an
// orange selected tab, scrollbar track 20 with a 77 thumb), so tables, docked
// windows and title bars looked pasted on. These are derived from the window
// colour now, in every theme; each preset keeps its own window, text and
// accent colours, and the selected tab keeps the preset's accent overline.
void Theme::HarmonizeSurfaces() {
    ImVec4* c = ImGui::GetStyle().Colors;
    const ImVec4 bg = c[ImGuiCol_WindowBg];
    const ImVec4 text = c[ImGuiCol_Text];
    auto toward_text = [&](float amount) {
        return ImVec4(bg.x + (text.x - bg.x) * amount, bg.y + (text.y - bg.y) * amount, bg.z + (text.z - bg.z) * amount, 1.0f);
    };
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TableHeaderBg] = ImVec4(bg.x, bg.y, bg.z, 1.0f);
    c[ImGuiCol_TableBorderStrong] = toward_text(0.10f);
    c[ImGuiCol_TableBorderLight] = toward_text(0.06f);
    c[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(text.x, text.y, text.z, 0.025f);

    // Scrollbar thumbs: a step toward the text, not a separate grey.
    c[ImGuiCol_ScrollbarGrab] = toward_text(0.14f);
    c[ImGuiCol_ScrollbarGrabHovered] = toward_text(0.22f);
    c[ImGuiCol_ScrollbarGrabActive] = toward_text(0.30f);

    // Lines (window border, separators) in the surface tone.
    c[ImGuiCol_Border] = toward_text(0.10f);
    c[ImGuiCol_Separator] = toward_text(0.10f);

    // Window title bars, menu bars and the dock area: the window colour.
    const ImVec4 solid(bg.x, bg.y, bg.z, 1.0f);
    c[ImGuiCol_TitleBg] = solid;
    c[ImGuiCol_TitleBgActive] = solid;
    c[ImGuiCol_TitleBgCollapsed] = solid;
    c[ImGuiCol_DockingEmptyBg] = solid;
    c[ImGuiCol_MenuBarBg] = solid;  // the main menu and a window's own menu row

    // Dock and window tabs: the window colour; the selected tab is marked by
    // the preset's accent overline, a hovered one by a light wash.
    DockStyle& dock = GetDockStyle();
    DockTabStyle tabs = dock.GetStyle();
    tabs.tab_bg = solid;
    tabs.tab_bg_active = solid;
    tabs.tab_bg_unfocused = solid;
    tabs.tab_bg_hovered = toward_text(0.08f);
    tabs.tab_text = ImVec4(text.x + (bg.x - text.x) * 0.35f, text.y + (bg.y - text.y) * 0.35f, text.z + (bg.z - text.z) * 0.35f, 1.0f);
    tabs.tab_text_active = text;
    tabs.tab_separator_color = toward_text(0.10f);
    tabs.dock_bg = solid;
    tabs.dock_border = toward_text(0.10f);
    dock.SetStyle(tabs);  // writes the ImGui tab, dock and border colours
    const ImVec4 mark = tabs.active_indicator_color;
    c[ImGuiCol_TabSelectedOverline] = mark;
    c[ImGuiCol_TabDimmedSelectedOverline] = ImVec4(mark.x, mark.y, mark.z, mark.w * 0.5f);
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
    style.ChildBorderSize = config_.child_border_size;

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
