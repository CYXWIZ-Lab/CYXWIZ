#include "theme.h"
#include "dock_style.h"
#include <imgui.h>
#include <imnodes.h>

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
        case ThemePreset::CyxWizDark:      return "CyxWiz Dark";
        case ThemePreset::CyxWizLight:     return "CyxWiz Light";
        case ThemePreset::VSCodeDark:      return "VS Code Dark";
        case ThemePreset::UnrealEngine:    return "Unreal Engine";
        case ThemePreset::ModernDark:      return "Modern Dark";
        case ThemePreset::HighContrast:    return "High Contrast";
        // New vibrant themes
        case ThemePreset::Dracula:         return "Dracula";
        case ThemePreset::OneDarkPro:      return "One Dark Pro";
        case ThemePreset::Nord:            return "Nord";
        case ThemePreset::CatppuccinMocha: return "Catppuccin Mocha";
        default:                           return "Unknown";
    }
}

std::vector<ThemePreset> Theme::GetAvailablePresets() {
    return {
        ThemePreset::CyxWizDark,
        ThemePreset::CyxWizLight,
        ThemePreset::VSCodeDark,
        ThemePreset::UnrealEngine,
        ThemePreset::ModernDark,
        ThemePreset::HighContrast,
        // New vibrant themes
        ThemePreset::Dracula,
        ThemePreset::OneDarkPro,
        ThemePreset::Nord,
        ThemePreset::CatppuccinMocha
    };
}

void Theme::ApplyPreset(ThemePreset preset) {
    current_preset_ = preset;

    switch (preset) {
        case ThemePreset::CyxWizDark:      ApplyCyxWizDark(); break;
        case ThemePreset::CyxWizLight:     ApplyCyxWizLight(); break;
        case ThemePreset::VSCodeDark:      ApplyVSCodeDark(); break;
        case ThemePreset::UnrealEngine:    ApplyUnrealEngine(); break;
        case ThemePreset::ModernDark:      ApplyModernDark(); break;
        case ThemePreset::HighContrast:    ApplyHighContrast(); break;
        // New vibrant themes
        case ThemePreset::Dracula:         ApplyDracula(); break;
        case ThemePreset::OneDarkPro:      ApplyOneDarkPro(); break;
        case ThemePreset::Nord:            ApplyNord(); break;
        case ThemePreset::CatppuccinMocha: ApplyCatppuccinMocha(); break;
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

// ============================================================================
// CyxWiz Dark Theme - Custom branded dark theme
// ============================================================================
void Theme::ApplyCyxWizDark() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // CyxWiz brand colors - Clean, minimal borders
    ImVec4 bg_dark       = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);   // Very dark background
    ImVec4 bg_medium     = ImVec4(0.14f, 0.14f, 0.16f, 1.00f);   // Panel background
    ImVec4 bg_light      = ImVec4(0.18f, 0.18f, 0.21f, 1.00f);   // Lighter elements
    ImVec4 border        = ImVec4(0.20f, 0.20f, 0.23f, 0.50f);   // Very subtle borders (semi-transparent)
    ImVec4 text          = ImVec4(0.92f, 0.92f, 0.94f, 1.00f);   // Main text
    ImVec4 text_dim      = ImVec4(0.60f, 0.60f, 0.65f, 1.00f);   // Dimmed text
    ImVec4 accent        = ImVec4(0.20f, 0.55f, 0.85f, 1.00f);   // Blue accent
    ImVec4 accent_hover  = ImVec4(0.30f, 0.65f, 0.95f, 1.00f);   // Hover state
    ImVec4 accent_active = ImVec4(0.15f, 0.45f, 0.75f, 1.00f);   // Active state
    ImVec4 success       = ImVec4(0.20f, 0.70f, 0.40f, 1.00f);   // Green success
    ImVec4 warning       = ImVec4(0.90f, 0.70f, 0.20f, 1.00f);   // Yellow warning
    ImVec4 error_col     = ImVec4(0.85f, 0.30f, 0.30f, 1.00f);   // Red error

    // Text
    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_medium;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, etc.)
    colors[ImGuiCol_FrameBg]                = bg_light;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_light.x + 0.05f, bg_light.y + 0.05f, bg_light.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_light.x + 0.10f, bg_light.y + 0.10f, bg_light.z + 0.10f, 1.00f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_dark.x + 0.02f, bg_dark.y + 0.02f, bg_dark.z + 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.45f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = accent;

    // Slider
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_ButtonHovered]          = accent_hover;
    colors[ImGuiCol_ButtonActive]           = accent_active;

    // Header (selectable, tree nodes)
    colors[ImGuiCol_Header]                 = ImVec4(accent.x, accent.y, accent.z, 0.30f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent.x, accent.y, accent.z, 0.50f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent.x, accent.y, accent.z, 0.70f);

    // Separator - Very subtle
    colors[ImGuiCol_Separator]              = ImVec4(0.20f, 0.20f, 0.23f, 0.30f);  // Almost invisible
    colors[ImGuiCol_SeparatorHovered]       = accent;
    colors[ImGuiCol_SeparatorActive]        = accent_active;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.20f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.60f);
    colors[ImGuiCol_ResizeGripActive]       = accent;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_light;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(accent.x, accent.y, accent.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = bg_light;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent.x, accent.y, accent.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot
    colors[ImGuiCol_PlotLines]              = accent;
    colors[ImGuiCol_PlotLinesHovered]       = accent_hover;
    colors[ImGuiCol_PlotHistogram]          = success;
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(success.x + 0.10f, success.y + 0.10f, success.z, 1.00f);

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_light;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent.x, accent.y, accent.z, 0.35f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = accent;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = accent;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    // Configuration - Clean, minimal appearance
    config_.window_rounding = 4.0f;
    config_.frame_rounding = 3.0f;
    config_.popup_rounding = 4.0f;
    config_.scrollbar_rounding = 4.0f;
    config_.grab_rounding = 3.0f;
    config_.tab_rounding = 3.0f;
    config_.window_border_size = 0.0f;   // No window borders - cleaner look
    config_.frame_border_size = 0.0f;
    config_.popup_border_size = 1.0f;    // Keep popup borders for visibility
}

// ============================================================================
// CyxWiz Light Theme
// ============================================================================
void Theme::ApplyCyxWizLight() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    ImVec4 bg_light      = ImVec4(0.96f, 0.96f, 0.97f, 1.00f);
    ImVec4 bg_medium     = ImVec4(0.92f, 0.92f, 0.94f, 1.00f);
    ImVec4 bg_dark       = ImVec4(0.88f, 0.88f, 0.90f, 1.00f);
    ImVec4 border        = ImVec4(0.75f, 0.75f, 0.78f, 1.00f);
    ImVec4 text          = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    ImVec4 text_dim      = ImVec4(0.45f, 0.45f, 0.48f, 1.00f);
    ImVec4 accent        = ImVec4(0.20f, 0.50f, 0.80f, 1.00f);
    ImVec4 accent_hover  = ImVec4(0.25f, 0.55f, 0.85f, 1.00f);
    ImVec4 accent_active = ImVec4(0.15f, 0.40f, 0.70f, 1.00f);

    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;
    colors[ImGuiCol_WindowBg]               = bg_light;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(1.00f, 1.00f, 1.00f, 0.98f);
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = bg_medium;
    colors[ImGuiCol_FrameBgHovered]         = bg_dark;
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_dark.x - 0.05f, bg_dark.y - 0.05f, bg_dark.z - 0.05f, 1.00f);
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = bg_medium;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);
    colors[ImGuiCol_MenuBarBg]              = bg_medium;
    colors[ImGuiCol_ScrollbarBg]            = bg_light;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.70f, 0.70f, 0.72f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.60f, 0.60f, 0.62f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.50f, 0.50f, 0.52f, 1.00f);
    colors[ImGuiCol_CheckMark]              = accent;
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;
    colors[ImGuiCol_Button]                 = accent;
    colors[ImGuiCol_ButtonHovered]          = accent_hover;
    colors[ImGuiCol_ButtonActive]           = accent_active;
    colors[ImGuiCol_Header]                 = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent.x, accent.y, accent.z, 0.45f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent;
    colors[ImGuiCol_SeparatorActive]        = accent_active;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = accent;
    colors[ImGuiCol_Tab]                    = bg_medium;
    colors[ImGuiCol_TabHovered]             = accent_hover;
    colors[ImGuiCol_TabActive]              = accent;
    colors[ImGuiCol_TabUnfocused]           = bg_medium;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_light;
    colors[ImGuiCol_PlotLines]              = accent;
    colors[ImGuiCol_PlotLinesHovered]       = accent_hover;
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.20f, 0.65f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.25f, 0.75f, 0.40f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = bg_medium;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.60f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(0.00f, 0.00f, 0.00f, 0.03f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent.x, accent.y, accent.z, 0.30f);
    colors[ImGuiCol_DragDropTarget]         = accent;
    colors[ImGuiCol_NavHighlight]           = accent;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(0.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.40f);

    config_.window_rounding = 6.0f;
    config_.frame_rounding = 4.0f;
    config_.popup_rounding = 6.0f;
    config_.scrollbar_rounding = 6.0f;
    config_.grab_rounding = 4.0f;
    config_.tab_rounding = 4.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// VS Code Dark Theme - Inspired by Visual Studio Code
// ============================================================================
void Theme::ApplyVSCodeDark() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // VS Code color palette
    ImVec4 bg_editor     = ImVec4(0.118f, 0.118f, 0.118f, 1.00f);  // #1e1e1e
    ImVec4 bg_sidebar    = ImVec4(0.153f, 0.153f, 0.153f, 1.00f);  // #272727
    ImVec4 bg_activitybar= ImVec4(0.200f, 0.200f, 0.200f, 1.00f);  // #333333
    ImVec4 border        = ImVec4(0.267f, 0.267f, 0.267f, 1.00f);  // #444444
    ImVec4 text          = ImVec4(0.847f, 0.847f, 0.847f, 1.00f);  // #d8d8d8
    ImVec4 text_dim      = ImVec4(0.502f, 0.502f, 0.502f, 1.00f);  // #808080
    ImVec4 accent        = ImVec4(0.075f, 0.463f, 0.788f, 1.00f);  // #1377c9 (VS Code blue)
    ImVec4 accent_hover  = ImVec4(0.110f, 0.529f, 0.882f, 1.00f);
    ImVec4 accent_active = ImVec4(0.059f, 0.392f, 0.694f, 1.00f);

    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;
    colors[ImGuiCol_WindowBg]               = bg_sidebar;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_editor.x, bg_editor.y, bg_editor.z, 0.98f);
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = bg_editor;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_editor.x + 0.05f, bg_editor.y + 0.05f, bg_editor.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_editor.x + 0.10f, bg_editor.y + 0.10f, bg_editor.z + 0.10f, 1.00f);
    colors[ImGuiCol_TitleBg]                = bg_activitybar;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_activitybar.x + 0.02f, bg_activitybar.y + 0.02f, bg_activitybar.z + 0.02f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_activitybar.x, bg_activitybar.y, bg_activitybar.z, 0.75f);
    colors[ImGuiCol_MenuBarBg]              = bg_activitybar;
    colors[ImGuiCol_ScrollbarBg]            = bg_sidebar;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.55f, 0.55f, 0.55f, 1.00f);
    colors[ImGuiCol_CheckMark]              = accent;
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;
    colors[ImGuiCol_Button]                 = accent;
    colors[ImGuiCol_ButtonHovered]          = accent_hover;
    colors[ImGuiCol_ButtonActive]           = accent_active;
    colors[ImGuiCol_Header]                 = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent;
    colors[ImGuiCol_SeparatorActive]        = accent_active;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = accent;
    colors[ImGuiCol_Tab]                    = bg_activitybar;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent.x, accent.y, accent.z, 0.80f);
    colors[ImGuiCol_TabActive]              = bg_editor;
    colors[ImGuiCol_TabUnfocused]           = bg_activitybar;
    colors[ImGuiCol_TabUnfocusedActive]     = bg_sidebar;
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_editor;
    colors[ImGuiCol_PlotLines]              = ImVec4(0.608f, 0.733f, 0.349f, 1.00f);  // #9bbb59
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.708f, 0.833f, 0.449f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.843f, 0.584f, 0.282f, 1.00f);  // #d79548
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.943f, 0.684f, 0.382f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = bg_activitybar;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent.x, accent.y, accent.z, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = accent;
    colors[ImGuiCol_NavHighlight]           = accent;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    config_.window_rounding = 0.0f;
    config_.frame_rounding = 0.0f;
    config_.popup_rounding = 0.0f;
    config_.scrollbar_rounding = 0.0f;
    config_.grab_rounding = 0.0f;
    config_.tab_rounding = 0.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// Unreal Engine Theme - Inspired by Unreal Editor
// ============================================================================
void Theme::ApplyUnrealEngine() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Unreal Editor color palette
    ImVec4 bg_dark       = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    ImVec4 bg_medium     = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    ImVec4 bg_panel      = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
    ImVec4 border        = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    ImVec4 text          = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
    ImVec4 text_dim      = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    ImVec4 accent        = ImVec4(0.13f, 0.59f, 0.95f, 1.00f);  // Unreal blue
    ImVec4 accent_hover  = ImVec4(0.20f, 0.65f, 1.00f, 1.00f);
    ImVec4 accent_active = ImVec4(0.10f, 0.50f, 0.85f, 1.00f);
    ImVec4 highlight     = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);  // Orange highlight

    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;
    colors[ImGuiCol_WindowBg]               = bg_panel;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = bg_medium;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_medium.x + 0.05f, bg_medium.y + 0.05f, bg_medium.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_medium.x + 0.10f, bg_medium.y + 0.10f, bg_medium.z + 0.10f, 1.00f);
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = bg_medium;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);
    colors[ImGuiCol_MenuBarBg]              = bg_dark;
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_CheckMark]              = highlight;
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;
    colors[ImGuiCol_Button]                 = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(highlight.x, highlight.y, highlight.z, 0.50f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(highlight.x, highlight.y, highlight.z, 0.70f);
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = highlight;
    colors[ImGuiCol_SeparatorActive]        = highlight;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = accent;
    colors[ImGuiCol_Tab]                    = bg_medium;
    colors[ImGuiCol_TabHovered]             = ImVec4(highlight.x, highlight.y, highlight.z, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(highlight.x, highlight.y, highlight.z, 0.70f);
    colors[ImGuiCol_TabUnfocused]           = bg_dark;
    colors[ImGuiCol_TabUnfocusedActive]     = bg_panel;
    colors[ImGuiCol_DockingPreview]         = ImVec4(highlight.x, highlight.y, highlight.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;
    colors[ImGuiCol_PlotLines]              = accent;
    colors[ImGuiCol_PlotLinesHovered]       = accent_hover;
    colors[ImGuiCol_PlotHistogram]          = highlight;
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.70f, 0.20f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = bg_medium;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(highlight.x, highlight.y, highlight.z, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = highlight;
    colors[ImGuiCol_NavHighlight]           = highlight;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    config_.window_rounding = 2.0f;
    config_.frame_rounding = 2.0f;
    config_.popup_rounding = 2.0f;
    config_.scrollbar_rounding = 2.0f;
    config_.grab_rounding = 2.0f;
    config_.tab_rounding = 2.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// Modern Dark Theme - Clean and minimal
// ============================================================================
void Theme::ApplyModernDark() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    ImVec4 bg_dark       = ImVec4(0.06f, 0.06f, 0.08f, 1.00f);
    ImVec4 bg_medium     = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
    ImVec4 bg_light      = ImVec4(0.14f, 0.14f, 0.16f, 1.00f);
    ImVec4 border        = ImVec4(0.20f, 0.20f, 0.22f, 1.00f);
    ImVec4 text          = ImVec4(0.95f, 0.95f, 0.97f, 1.00f);
    ImVec4 text_dim      = ImVec4(0.55f, 0.55f, 0.58f, 1.00f);
    ImVec4 accent        = ImVec4(0.40f, 0.55f, 0.80f, 1.00f);  // Soft blue
    ImVec4 accent_hover  = ImVec4(0.50f, 0.65f, 0.90f, 1.00f);
    ImVec4 accent_active = ImVec4(0.30f, 0.45f, 0.70f, 1.00f);

    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;
    colors[ImGuiCol_WindowBg]               = bg_medium;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = bg_light;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_light.x + 0.04f, bg_light.y + 0.04f, bg_light.z + 0.04f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_light.x + 0.08f, bg_light.y + 0.08f, bg_light.z + 0.08f, 1.00f);
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_dark.x + 0.02f, bg_dark.y + 0.02f, bg_dark.z + 0.02f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);
    colors[ImGuiCol_MenuBarBg]              = bg_dark;
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.28f, 0.28f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.38f, 0.38f, 0.40f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.48f, 0.48f, 0.50f, 1.00f);
    colors[ImGuiCol_CheckMark]              = accent;
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;
    colors[ImGuiCol_Button]                 = ImVec4(accent.x, accent.y, accent.z, 0.60f);
    colors[ImGuiCol_ButtonHovered]          = accent_hover;
    colors[ImGuiCol_ButtonActive]           = accent_active;
    colors[ImGuiCol_Header]                 = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent.x, accent.y, accent.z, 0.40f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent.x, accent.y, accent.z, 0.55f);
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent;
    colors[ImGuiCol_SeparatorActive]        = accent_active;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.20f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.50f);
    colors[ImGuiCol_ResizeGripActive]       = accent;
    colors[ImGuiCol_Tab]                    = bg_light;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent.x, accent.y, accent.z, 0.65f);
    colors[ImGuiCol_TabActive]              = ImVec4(accent.x, accent.y, accent.z, 0.80f);
    colors[ImGuiCol_TabUnfocused]           = bg_light;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent.x, accent.y, accent.z, 0.45f);
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;
    colors[ImGuiCol_PlotLines]              = accent;
    colors[ImGuiCol_PlotLinesHovered]       = accent_hover;
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.45f, 0.75f, 0.45f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.55f, 0.85f, 0.55f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = bg_light;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent.x, accent.y, accent.z, 0.30f);
    colors[ImGuiCol_DragDropTarget]         = accent;
    colors[ImGuiCol_NavHighlight]           = accent;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    config_.window_rounding = 8.0f;
    config_.frame_rounding = 4.0f;
    config_.popup_rounding = 8.0f;
    config_.scrollbar_rounding = 8.0f;
    config_.grab_rounding = 4.0f;
    config_.tab_rounding = 6.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// High Contrast Theme - Accessibility
// ============================================================================
void Theme::ApplyHighContrast() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    ImVec4 bg_black      = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    ImVec4 border        = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    ImVec4 text          = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    ImVec4 text_dim      = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
    ImVec4 accent        = ImVec4(0.00f, 0.80f, 1.00f, 1.00f);  // Cyan
    ImVec4 accent_hover  = ImVec4(0.20f, 0.90f, 1.00f, 1.00f);
    ImVec4 accent_active = ImVec4(0.00f, 0.60f, 0.80f, 1.00f);
    ImVec4 yellow        = ImVec4(1.00f, 1.00f, 0.00f, 1.00f);

    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;
    colors[ImGuiCol_WindowBg]               = bg_black;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = bg_black;
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_TitleBg]                = bg_black;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = bg_black;
    colors[ImGuiCol_MenuBarBg]              = bg_black;
    colors[ImGuiCol_ScrollbarBg]            = bg_black;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_CheckMark]              = yellow;
    colors[ImGuiCol_SliderGrab]             = accent;
    colors[ImGuiCol_SliderGrabActive]       = accent_active;
    colors[ImGuiCol_Button]                 = accent;
    colors[ImGuiCol_ButtonHovered]          = accent_hover;
    colors[ImGuiCol_ButtonActive]           = accent_active;
    colors[ImGuiCol_Header]                 = ImVec4(accent.x, accent.y, accent.z, 0.40f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent.x, accent.y, accent.z, 0.60f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent.x, accent.y, accent.z, 0.80f);
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = yellow;
    colors[ImGuiCol_SeparatorActive]        = yellow;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent.x, accent.y, accent.z, 0.40f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_ResizeGripActive]       = accent;
    colors[ImGuiCol_Tab]                    = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_TabHovered]             = accent_hover;
    colors[ImGuiCol_TabActive]              = accent;
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent.x, accent.y, accent.z, 0.70f);
    colors[ImGuiCol_DockingPreview]         = yellow;
    colors[ImGuiCol_DockingEmptyBg]         = bg_black;
    colors[ImGuiCol_PlotLines]              = yellow;
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 1.00f, 0.50f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.00f, 1.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.50f, 1.00f, 0.50f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.70f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.05f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(yellow.x, yellow.y, yellow.z, 0.50f);
    colors[ImGuiCol_DragDropTarget]         = yellow;
    colors[ImGuiCol_NavHighlight]           = yellow;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.50f, 0.50f, 0.50f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.80f);

    config_.window_rounding = 0.0f;
    config_.frame_rounding = 0.0f;
    config_.popup_rounding = 0.0f;
    config_.scrollbar_rounding = 0.0f;
    config_.grab_rounding = 0.0f;
    config_.tab_rounding = 0.0f;
    config_.window_border_size = 2.0f;
    config_.frame_border_size = 1.0f;
}

// ============================================================================
// Dracula Theme - Vibrant purple/pink theme
// ============================================================================
void Theme::ApplyDracula() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Dracula official color palette
    ImVec4 bg_dark       = ImVec4(0.16f, 0.16f, 0.21f, 1.00f);  // #282a36
    ImVec4 bg_medium     = ImVec4(0.18f, 0.18f, 0.24f, 1.00f);  // Slightly lighter
    ImVec4 bg_light      = ImVec4(0.22f, 0.22f, 0.28f, 1.00f);  // Input backgrounds
    ImVec4 border        = ImVec4(0.27f, 0.28f, 0.33f, 1.00f);  // #44465a
    ImVec4 text          = ImVec4(0.97f, 0.97f, 0.95f, 1.00f);  // #f8f8f2
    ImVec4 text_dim      = ImVec4(0.60f, 0.60f, 0.65f, 1.00f);  // Dimmed text
    ImVec4 purple        = ImVec4(0.74f, 0.58f, 0.98f, 1.00f);  // #bd93f9 (PRIMARY)
    ImVec4 pink          = ImVec4(1.00f, 0.47f, 0.78f, 1.00f);  // #ff79c6
    ImVec4 cyan          = ImVec4(0.55f, 0.91f, 0.99f, 1.00f);  // #8be9fd
    ImVec4 green         = ImVec4(0.31f, 0.98f, 0.48f, 1.00f);  // #50fa7b
    ImVec4 yellow        = ImVec4(0.95f, 0.98f, 0.55f, 1.00f);  // #f1fa8c
    ImVec4 orange        = ImVec4(1.00f, 0.72f, 0.42f, 1.00f);  // #ffb86c
    ImVec4 red           = ImVec4(1.00f, 0.33f, 0.33f, 1.00f);  // #ff5555

    // Text
    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_medium;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, etc.)
    colors[ImGuiCol_FrameBg]                = bg_light;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_light.x + 0.05f, bg_light.y + 0.05f, bg_light.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_light.x + 0.10f, bg_light.y + 0.10f, bg_light.z + 0.10f, 1.00f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_dark.x + 0.02f, bg_dark.y + 0.02f, bg_dark.z + 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.55f, 0.55f, 0.60f, 1.00f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = green;

    // Slider
    colors[ImGuiCol_SliderGrab]             = purple;
    colors[ImGuiCol_SliderGrabActive]       = pink;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(purple.x, purple.y, purple.z, 0.70f);
    colors[ImGuiCol_ButtonHovered]          = pink;
    colors[ImGuiCol_ButtonActive]           = ImVec4(purple.x * 0.8f, purple.y * 0.8f, purple.z, 1.00f);

    // Header (selectable, tree nodes)
    colors[ImGuiCol_Header]                 = ImVec4(purple.x, purple.y, purple.z, 0.35f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(purple.x, purple.y, purple.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(purple.x, purple.y, purple.z, 0.75f);

    // Separator
    colors[ImGuiCol_Separator]              = ImVec4(border.x, border.y, border.z, 0.60f);
    colors[ImGuiCol_SeparatorHovered]       = pink;
    colors[ImGuiCol_SeparatorActive]        = cyan;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(purple.x, purple.y, purple.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(purple.x, purple.y, purple.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = purple;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_light;
    colors[ImGuiCol_TabHovered]             = ImVec4(pink.x, pink.y, pink.z, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(purple.x, purple.y, purple.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = bg_light;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(purple.x, purple.y, purple.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(pink.x, pink.y, pink.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot
    colors[ImGuiCol_PlotLines]              = cyan;
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(cyan.x + 0.10f, cyan.y, cyan.z, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = green;
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(green.x + 0.10f, green.y, green.z, 1.00f);

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_light;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(purple.x, purple.y, purple.z, 0.40f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = pink;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = purple;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.65f);

    // Vibrant, rounded style
    config_.window_rounding = 6.0f;
    config_.frame_rounding = 4.0f;
    config_.popup_rounding = 6.0f;
    config_.scrollbar_rounding = 6.0f;
    config_.grab_rounding = 4.0f;
    config_.tab_rounding = 4.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// One Dark Pro Theme - VSCode's most popular theme (Blue/Cyan)
// ============================================================================
void Theme::ApplyOneDarkPro() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // One Dark Pro color palette
    ImVec4 bg_dark       = ImVec4(0.16f, 0.17f, 0.20f, 1.00f);  // #282c34
    ImVec4 bg_medium     = ImVec4(0.18f, 0.19f, 0.22f, 1.00f);  // Slightly lighter
    ImVec4 bg_light      = ImVec4(0.21f, 0.22f, 0.26f, 1.00f);  // Input backgrounds
    ImVec4 border        = ImVec4(0.25f, 0.27f, 0.31f, 1.00f);  // Subtle border
    ImVec4 text          = ImVec4(0.67f, 0.70f, 0.75f, 1.00f);  // #abb2bf
    ImVec4 text_dim      = ImVec4(0.45f, 0.48f, 0.53f, 1.00f);  // Dimmed text
    ImVec4 blue          = ImVec4(0.38f, 0.69f, 0.94f, 1.00f);  // #61afef (PRIMARY)
    ImVec4 cyan          = ImVec4(0.34f, 0.71f, 0.76f, 1.00f);  // #56b6c2
    ImVec4 green         = ImVec4(0.60f, 0.76f, 0.47f, 1.00f);  // #98c379
    ImVec4 yellow        = ImVec4(0.90f, 0.75f, 0.48f, 1.00f);  // #e5c07b
    ImVec4 orange        = ImVec4(0.82f, 0.60f, 0.40f, 1.00f);  // #d19a66
    ImVec4 red           = ImVec4(0.88f, 0.42f, 0.46f, 1.00f);  // #e06c75
    ImVec4 purple        = ImVec4(0.78f, 0.47f, 0.86f, 1.00f);  // #c678dd

    // Text
    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_medium;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, etc.)
    colors[ImGuiCol_FrameBg]                = bg_light;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_light.x + 0.05f, bg_light.y + 0.05f, bg_light.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_light.x + 0.10f, bg_light.y + 0.10f, bg_light.z + 0.10f, 1.00f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_dark.x + 0.02f, bg_dark.y + 0.02f, bg_dark.z + 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.55f, 0.55f, 0.60f, 1.00f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = green;

    // Slider
    colors[ImGuiCol_SliderGrab]             = blue;
    colors[ImGuiCol_SliderGrabActive]       = cyan;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(blue.x, blue.y, blue.z, 0.70f);
    colors[ImGuiCol_ButtonHovered]          = blue;
    colors[ImGuiCol_ButtonActive]           = ImVec4(blue.x * 0.8f, blue.y * 0.8f, blue.z, 1.00f);

    // Header (selectable, tree nodes)
    colors[ImGuiCol_Header]                 = ImVec4(blue.x, blue.y, blue.z, 0.35f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(blue.x, blue.y, blue.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(blue.x, blue.y, blue.z, 0.75f);

    // Separator
    colors[ImGuiCol_Separator]              = ImVec4(border.x, border.y, border.z, 0.60f);
    colors[ImGuiCol_SeparatorHovered]       = cyan;
    colors[ImGuiCol_SeparatorActive]        = blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(blue.x, blue.y, blue.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(blue.x, blue.y, blue.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_light;
    colors[ImGuiCol_TabHovered]             = ImVec4(blue.x, blue.y, blue.z, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(blue.x, blue.y, blue.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = bg_light;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(blue.x, blue.y, blue.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(cyan.x, cyan.y, cyan.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot
    colors[ImGuiCol_PlotLines]              = cyan;
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(cyan.x + 0.10f, cyan.y, cyan.z, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = blue;
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(blue.x + 0.10f, blue.y, blue.z, 1.00f);

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_light;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(blue.x, blue.y, blue.z, 0.40f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = cyan;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.65f);

    // Modern, smooth style
    config_.window_rounding = 5.0f;
    config_.frame_rounding = 3.0f;
    config_.popup_rounding = 5.0f;
    config_.scrollbar_rounding = 5.0f;
    config_.grab_rounding = 3.0f;
    config_.tab_rounding = 3.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// Nord Theme - Cool pastels inspired by arctic colors
// ============================================================================
void Theme::ApplyNord() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Nord color palette (Polar Night + Snow Storm + Frost + Aurora)
    ImVec4 bg_dark       = ImVec4(0.18f, 0.20f, 0.25f, 1.00f);  // #2e3440
    ImVec4 bg_medium     = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);  // #3b4252
    ImVec4 bg_light      = ImVec4(0.26f, 0.30f, 0.37f, 1.00f);  // #434c5e
    ImVec4 border        = ImVec4(0.30f, 0.34f, 0.42f, 1.00f);  // #4c566a
    ImVec4 text          = ImVec4(0.85f, 0.87f, 0.91f, 1.00f);  // #d8dee9
    ImVec4 text_dim      = ImVec4(0.60f, 0.63f, 0.68f, 1.00f);  // Dimmed text
    ImVec4 frost_blue    = ImVec4(0.53f, 0.75f, 0.82f, 1.00f);  // #88c0d0 (PRIMARY)
    ImVec4 frost_cyan    = ImVec4(0.56f, 0.74f, 0.73f, 1.00f);  // #8fbcbb
    ImVec4 frost_light   = ImVec4(0.51f, 0.63f, 0.76f, 1.00f);  // #81a1c1
    ImVec4 frost_deep    = ImVec4(0.37f, 0.51f, 0.67f, 1.00f);  // #5e81ac
    ImVec4 aurora_red    = ImVec4(0.75f, 0.38f, 0.42f, 1.00f);  // #bf616a
    ImVec4 aurora_orange = ImVec4(0.82f, 0.53f, 0.44f, 1.00f);  // #d08770
    ImVec4 aurora_yellow = ImVec4(0.92f, 0.80f, 0.55f, 1.00f);  // #ebcb8b
    ImVec4 aurora_green  = ImVec4(0.64f, 0.75f, 0.55f, 1.00f);  // #a3be8c
    ImVec4 aurora_purple = ImVec4(0.71f, 0.56f, 0.68f, 1.00f);  // #b48ead

    // Text
    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = text_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_medium;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, etc.)
    colors[ImGuiCol_FrameBg]                = bg_light;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_light.x + 0.05f, bg_light.y + 0.05f, bg_light.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(bg_light.x + 0.10f, bg_light.y + 0.10f, bg_light.z + 0.10f, 1.00f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(bg_dark.x + 0.02f, bg_dark.y + 0.02f, bg_dark.z + 0.04f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.35f, 0.40f, 0.47f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.45f, 0.50f, 0.57f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.55f, 0.60f, 0.67f, 1.00f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = aurora_green;

    // Slider
    colors[ImGuiCol_SliderGrab]             = frost_blue;
    colors[ImGuiCol_SliderGrabActive]       = frost_light;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.70f);
    colors[ImGuiCol_ButtonHovered]          = frost_blue;
    colors[ImGuiCol_ButtonActive]           = frost_deep;

    // Header (selectable, tree nodes)
    colors[ImGuiCol_Header]                 = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.35f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.75f);

    // Separator
    colors[ImGuiCol_Separator]              = ImVec4(border.x, border.y, border.z, 0.60f);
    colors[ImGuiCol_SeparatorHovered]       = frost_cyan;
    colors[ImGuiCol_SeparatorActive]        = frost_blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = frost_blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_light;
    colors[ImGuiCol_TabHovered]             = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = bg_light;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(frost_cyan.x, frost_cyan.y, frost_cyan.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot
    colors[ImGuiCol_PlotLines]              = frost_cyan;
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(frost_cyan.x + 0.10f, frost_cyan.y, frost_cyan.z, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = frost_blue;
    colors[ImGuiCol_PlotHistogramHovered]   = frost_light;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_light;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.40f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = frost_cyan;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = frost_blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.65f);

    // Minimal, calm style
    config_.window_rounding = 4.0f;
    config_.frame_rounding = 3.0f;
    config_.popup_rounding = 4.0f;
    config_.scrollbar_rounding = 4.0f;
    config_.grab_rounding = 3.0f;
    config_.tab_rounding = 3.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// Catppuccin Mocha Theme - Pastel rainbow cozy theme
// ============================================================================
void Theme::ApplyCatppuccinMocha() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Catppuccin Mocha color palette
    ImVec4 base          = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);  // #1e1e2e
    ImVec4 mantle        = ImVec4(0.09f, 0.09f, 0.15f, 1.00f);  // #181825
    ImVec4 crust         = ImVec4(0.07f, 0.07f, 0.11f, 1.00f);  // #11111b
    ImVec4 text          = ImVec4(0.80f, 0.84f, 0.96f, 1.00f);  // #cdd6f4
    ImVec4 subtext1      = ImVec4(0.73f, 0.76f, 0.87f, 1.00f);  // #bac2de
    ImVec4 subtext0      = ImVec4(0.65f, 0.68f, 0.78f, 1.00f);  // #a6adc8
    ImVec4 overlay2      = ImVec4(0.58f, 0.60f, 0.69f, 1.00f);  // #9399b2
    ImVec4 overlay1      = ImVec4(0.47f, 0.49f, 0.58f, 1.00f);  // #7f849c
    ImVec4 surface2      = ImVec4(0.36f, 0.38f, 0.46f, 1.00f);  // #585b70
    ImVec4 surface1      = ImVec4(0.28f, 0.30f, 0.38f, 1.00f);  // #45475a
    ImVec4 surface0      = ImVec4(0.20f, 0.22f, 0.28f, 1.00f);  // #313244

    // Accent colors
    ImVec4 sapphire      = ImVec4(0.45f, 0.78f, 0.93f, 1.00f);  // #74c7ec
    ImVec4 blue          = ImVec4(0.54f, 0.71f, 0.98f, 1.00f);  // #89b4fa (PRIMARY)
    ImVec4 lavender      = ImVec4(0.71f, 0.75f, 1.00f, 1.00f);  // #b4befe
    ImVec4 mauve         = ImVec4(0.80f, 0.65f, 0.97f, 1.00f);  // #cba6f7
    ImVec4 pink          = ImVec4(0.96f, 0.76f, 0.90f, 1.00f);  // #f5c2e7
    ImVec4 red           = ImVec4(0.95f, 0.55f, 0.66f, 1.00f);  // #f38ba8
    ImVec4 peach         = ImVec4(0.98f, 0.70f, 0.53f, 1.00f);  // #fab387
    ImVec4 yellow        = ImVec4(0.98f, 0.89f, 0.69f, 1.00f);  // #f9e2af
    ImVec4 green         = ImVec4(0.65f, 0.89f, 0.63f, 1.00f);  // #a6e3a1
    ImVec4 teal          = ImVec4(0.58f, 0.89f, 0.84f, 1.00f);  // #94e2d5

    // Text
    colors[ImGuiCol_Text]                   = text;
    colors[ImGuiCol_TextDisabled]           = subtext0;

    // Window
    colors[ImGuiCol_WindowBg]               = base;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(mantle.x, mantle.y, mantle.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = surface0;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame (input boxes, etc.)
    colors[ImGuiCol_FrameBg]                = surface0;
    colors[ImGuiCol_FrameBgHovered]         = surface1;
    colors[ImGuiCol_FrameBgActive]          = surface2;

    // Title bar
    colors[ImGuiCol_TitleBg]                = mantle;
    colors[ImGuiCol_TitleBgActive]          = crust;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(mantle.x, mantle.y, mantle.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = mantle;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = mantle;
    colors[ImGuiCol_ScrollbarGrab]          = surface1;
    colors[ImGuiCol_ScrollbarGrabHovered]   = surface2;
    colors[ImGuiCol_ScrollbarGrabActive]    = overlay1;

    // Check mark
    colors[ImGuiCol_CheckMark]              = green;

    // Slider
    colors[ImGuiCol_SliderGrab]             = blue;
    colors[ImGuiCol_SliderGrabActive]       = lavender;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(blue.x, blue.y, blue.z, 0.70f);
    colors[ImGuiCol_ButtonHovered]          = blue;
    colors[ImGuiCol_ButtonActive]           = sapphire;

    // Header (selectable, tree nodes)
    colors[ImGuiCol_Header]                 = ImVec4(blue.x, blue.y, blue.z, 0.35f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(blue.x, blue.y, blue.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(blue.x, blue.y, blue.z, 0.75f);

    // Separator
    colors[ImGuiCol_Separator]              = surface0;
    colors[ImGuiCol_SeparatorHovered]       = teal;
    colors[ImGuiCol_SeparatorActive]        = blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(blue.x, blue.y, blue.z, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(blue.x, blue.y, blue.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = surface0;
    colors[ImGuiCol_TabHovered]             = ImVec4(mauve.x, mauve.y, mauve.z, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(blue.x, blue.y, blue.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = surface0;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(blue.x, blue.y, blue.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(pink.x, pink.y, pink.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = mantle;

    // Plot
    colors[ImGuiCol_PlotLines]              = teal;
    colors[ImGuiCol_PlotLinesHovered]       = sapphire;
    colors[ImGuiCol_PlotHistogram]          = green;
    colors[ImGuiCol_PlotHistogramHovered]   = yellow;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = surface0;
    colors[ImGuiCol_TableBorderStrong]      = surface1;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(surface0.x, surface0.y, surface0.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.03f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(blue.x, blue.y, blue.z, 0.40f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = pink;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.65f);

    // Cozy, rounded style
    config_.window_rounding = 8.0f;
    config_.frame_rounding = 5.0f;
    config_.popup_rounding = 8.0f;
    config_.scrollbar_rounding = 8.0f;
    config_.grab_rounding = 5.0f;
    config_.tab_rounding = 5.0f;
    config_.window_border_size = 0.0f;
    config_.frame_border_size = 0.0f;
}

// ============================================================================
// ImNodes Styling - Matches node editor to current theme
// ============================================================================

// Helper to convert ImVec4 color to ImU32
static inline ImU32 ColorToU32(const ImVec4& col) {
    return IM_COL32(
        (int)(col.x * 255.0f),
        (int)(col.y * 255.0f),
        (int)(col.z * 255.0f),
        (int)(col.w * 255.0f)
    );
}

void Theme::ApplyImNodesStyle() {
    ImNodesStyle& style = ImNodes::GetStyle();
    // Note: imgui_style is available for future use if needed to match colors
    (void)ImGui::GetStyle();

    // Match ImNodes rounding to ImGui theme
    style.NodeCornerRounding = config_.frame_rounding;
    style.NodePadding = ImVec2(8.0f, 8.0f);
    style.NodeBorderThickness = 1.0f;

    // Link styling
    style.LinkThickness = 3.0f;
    style.LinkLineSegmentsPerLength = 0.1f;
    style.LinkHoverDistance = 10.0f;

    // Pin styling
    style.PinCircleRadius = 5.0f;
    style.PinQuadSideLength = 7.0f;
    style.PinTriangleSideLength = 9.0f;
    style.PinLineThickness = 1.5f;
    style.PinHoverRadius = 10.0f;
    style.PinOffset = 0.0f;

    // Grid
    style.GridSpacing = 32.0f;
    style.Flags = ImNodesStyleFlags_GridLines | ImNodesStyleFlags_GridLinesPrimary | ImNodesStyleFlags_NodeOutline;

    // Theme-specific colors
    switch (current_preset_) {
        case ThemePreset::CyxWizDark:
        case ThemePreset::ModernDark: {
            // Dark blue-accented theme
            ImVec4 node_bg = ImVec4(0.15f, 0.15f, 0.18f, 1.0f);
            ImVec4 node_bg_hover = ImVec4(0.18f, 0.18f, 0.22f, 1.0f);
            ImVec4 node_bg_selected = ImVec4(0.20f, 0.20f, 0.25f, 1.0f);
            ImVec4 title_bar = ImVec4(0.20f, 0.55f, 0.85f, 1.0f);  // Blue accent
            ImVec4 title_bar_hover = ImVec4(0.25f, 0.60f, 0.90f, 1.0f);
            ImVec4 title_bar_selected = ImVec4(0.30f, 0.65f, 0.95f, 1.0f);
            ImVec4 link_color = ImVec4(0.45f, 0.70f, 0.95f, 1.0f);
            ImVec4 pin_color = ImVec4(0.50f, 0.75f, 1.00f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(node_bg_hover);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(node_bg_selected);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.30f, 0.30f, 0.35f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(title_bar);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(title_bar_hover);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(title_bar_selected);
            style.Colors[ImNodesCol_Link] = ColorToU32(link_color);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.55f, 0.80f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(ImVec4(0.65f, 0.85f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_Pin] = ColorToU32(pin_color);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.70f, 0.90f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(0.20f, 0.55f, 0.85f, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(ImVec4(0.20f, 0.55f, 0.85f, 1.0f));
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.08f, 0.08f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.20f, 0.20f, 0.25f, 1.0f));
            break;
        }

        case ThemePreset::CyxWizLight: {
            // Light theme
            ImVec4 node_bg = ImVec4(0.96f, 0.96f, 0.98f, 1.0f);
            ImVec4 node_bg_hover = ImVec4(0.94f, 0.94f, 0.96f, 1.0f);
            ImVec4 node_bg_selected = ImVec4(0.92f, 0.92f, 0.95f, 1.0f);
            ImVec4 title_bar = ImVec4(0.20f, 0.50f, 0.80f, 1.0f);
            ImVec4 link_color = ImVec4(0.25f, 0.55f, 0.85f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(node_bg_hover);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(node_bg_selected);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.70f, 0.70f, 0.75f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(title_bar);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.25f, 0.55f, 0.85f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(0.30f, 0.60f, 0.90f, 1.0f));
            style.Colors[ImNodesCol_Link] = ColorToU32(link_color);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.35f, 0.65f, 0.95f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(ImVec4(0.40f, 0.70f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_Pin] = ColorToU32(ImVec4(0.30f, 0.60f, 0.90f, 1.0f));
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.40f, 0.70f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(0.20f, 0.50f, 0.80f, 0.25f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(ImVec4(0.20f, 0.50f, 0.80f, 1.0f));
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.92f, 0.92f, 0.94f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.85f, 0.85f, 0.88f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.78f, 0.78f, 0.82f, 1.0f));
            break;
        }

        case ThemePreset::VSCodeDark: {
            // VS Code inspired - no rounding
            style.NodeCornerRounding = 0.0f;

            ImVec4 node_bg = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
            ImVec4 title_bar = ImVec4(0.075f, 0.46f, 0.79f, 1.0f);  // VS Code blue

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.18f, 0.18f, 0.18f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.27f, 0.27f, 0.27f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(title_bar);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.11f, 0.53f, 0.88f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(0.15f, 0.60f, 0.95f, 1.0f));
            style.Colors[ImNodesCol_Link] = ColorToU32(ImVec4(0.61f, 0.73f, 0.35f, 1.0f));  // Green links
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.71f, 0.83f, 0.45f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(ImVec4(0.81f, 0.93f, 0.55f, 1.0f));
            style.Colors[ImNodesCol_Pin] = ColorToU32(ImVec4(0.84f, 0.58f, 0.28f, 1.0f));  // Orange pins
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.94f, 0.68f, 0.38f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(0.075f, 0.46f, 0.79f, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(title_bar);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.18f, 0.18f, 0.18f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            break;
        }

        case ThemePreset::UnrealEngine: {
            // Unreal Engine style - orange highlights
            style.NodeCornerRounding = 2.0f;

            ImVec4 node_bg = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
            ImVec4 title_bar = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
            ImVec4 orange = ImVec4(1.00f, 0.60f, 0.00f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.18f, 0.16f, 0.12f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(title_bar);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(orange.x, orange.y, orange.z, 0.50f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(orange.x, orange.y, orange.z, 0.70f));
            style.Colors[ImNodesCol_Link] = ColorToU32(ImVec4(0.13f, 0.59f, 0.95f, 1.0f));  // Unreal blue links
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(orange);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(orange);
            style.Colors[ImNodesCol_Pin] = ColorToU32(ImVec4(0.13f, 0.59f, 0.95f, 1.0f));
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(orange);
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(orange.x, orange.y, orange.z, 0.25f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(orange);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.14f, 0.14f, 0.14f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.20f, 0.20f, 0.20f, 1.0f));
            break;
        }

        case ThemePreset::HighContrast: {
            // High contrast - sharp, no rounding
            style.NodeCornerRounding = 0.0f;
            style.NodeBorderThickness = 2.0f;
            style.LinkThickness = 4.0f;

            ImVec4 cyan = ImVec4(0.00f, 0.80f, 1.00f, 1.0f);
            ImVec4 yellow = ImVec4(1.00f, 1.00f, 0.00f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(ImVec4(0.00f, 0.00f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.10f, 0.10f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.15f, 0.15f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(1.00f, 1.00f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(cyan);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.20f, 0.90f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(yellow);
            style.Colors[ImNodesCol_Link] = ColorToU32(cyan);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(yellow);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(yellow);
            style.Colors[ImNodesCol_Pin] = ColorToU32(cyan);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(yellow);
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(yellow.x, yellow.y, yellow.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(yellow);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.00f, 0.00f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.40f, 0.40f, 0.40f, 1.0f));
            break;
        }

        case ThemePreset::Dracula: {
            // Dracula theme - vibrant purple/pink
            style.NodeCornerRounding = 6.0f;

            ImVec4 node_bg = ImVec4(0.18f, 0.18f, 0.24f, 1.0f);  // bg_medium
            ImVec4 node_bg_hover = ImVec4(0.22f, 0.22f, 0.28f, 1.0f);  // bg_light
            ImVec4 node_bg_selected = ImVec4(0.26f, 0.26f, 0.32f, 1.0f);
            ImVec4 purple = ImVec4(0.74f, 0.58f, 0.98f, 1.0f);  // #bd93f9
            ImVec4 pink = ImVec4(1.00f, 0.47f, 0.78f, 1.0f);  // #ff79c6
            ImVec4 cyan = ImVec4(0.55f, 0.91f, 0.99f, 1.0f);  // #8be9fd
            ImVec4 green = ImVec4(0.31f, 0.98f, 0.48f, 1.0f);  // #50fa7b

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(node_bg_hover);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(node_bg_selected);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.27f, 0.28f, 0.33f, 1.0f));  // border
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(purple);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(pink);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(0.84f, 0.68f, 1.00f, 1.0f));  // Lighter purple
            style.Colors[ImNodesCol_Link] = ColorToU32(cyan);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.65f, 1.00f, 1.00f, 1.0f));  // Brighter cyan
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(pink);
            style.Colors[ImNodesCol_Pin] = ColorToU32(green);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.41f, 1.00f, 0.58f, 1.0f));  // Brighter green
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(purple.x, purple.y, purple.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(purple);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.16f, 0.16f, 0.21f, 1.0f));  // bg_dark
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.20f, 0.20f, 0.26f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.24f, 0.24f, 0.30f, 1.0f));
            break;
        }

        case ThemePreset::OneDarkPro: {
            // One Dark Pro theme - blue/cyan accents
            style.NodeCornerRounding = 5.0f;

            ImVec4 node_bg = ImVec4(0.18f, 0.19f, 0.22f, 1.0f);  // bg_medium
            ImVec4 node_bg_hover = ImVec4(0.21f, 0.22f, 0.26f, 1.0f);  // bg_light
            ImVec4 node_bg_selected = ImVec4(0.25f, 0.26f, 0.30f, 1.0f);
            ImVec4 blue = ImVec4(0.38f, 0.69f, 0.94f, 1.0f);  // #61afef
            ImVec4 cyan = ImVec4(0.34f, 0.71f, 0.76f, 1.0f);  // #56b6c2
            ImVec4 green = ImVec4(0.60f, 0.76f, 0.47f, 1.0f);  // #98c379
            ImVec4 purple = ImVec4(0.78f, 0.47f, 0.86f, 1.0f);  // #c678dd

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(node_bg_hover);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(node_bg_selected);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.25f, 0.27f, 0.31f, 1.0f));  // border
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(blue);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.48f, 0.79f, 1.00f, 1.0f));  // Brighter blue
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(cyan);
            style.Colors[ImNodesCol_Link] = ColorToU32(cyan);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.44f, 0.81f, 0.86f, 1.0f));  // Brighter cyan
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(blue);
            style.Colors[ImNodesCol_Pin] = ColorToU32(purple);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.88f, 0.57f, 0.96f, 1.0f));  // Brighter purple
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(blue.x, blue.y, blue.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(blue);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.16f, 0.17f, 0.20f, 1.0f));  // bg_dark
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.20f, 0.21f, 0.24f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.24f, 0.25f, 0.28f, 1.0f));
            break;
        }

        case ThemePreset::Nord: {
            // Nord theme - cool frost blue accents
            style.NodeCornerRounding = 4.0f;

            ImVec4 node_bg = ImVec4(0.23f, 0.26f, 0.32f, 1.0f);  // bg_medium
            ImVec4 node_bg_hover = ImVec4(0.26f, 0.30f, 0.37f, 1.0f);  // bg_light
            ImVec4 node_bg_selected = ImVec4(0.30f, 0.34f, 0.42f, 1.0f);  // border
            ImVec4 frost_blue = ImVec4(0.53f, 0.75f, 0.82f, 1.0f);  // #88c0d0
            ImVec4 frost_cyan = ImVec4(0.56f, 0.74f, 0.73f, 1.0f);  // #8fbcbb
            ImVec4 frost_light = ImVec4(0.51f, 0.63f, 0.76f, 1.0f);  // #81a1c1
            ImVec4 aurora_green = ImVec4(0.64f, 0.75f, 0.55f, 1.0f);  // #a3be8c
            ImVec4 aurora_purple = ImVec4(0.71f, 0.56f, 0.68f, 1.0f);  // #b48ead

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(node_bg_hover);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(node_bg_selected);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.30f, 0.34f, 0.42f, 1.0f));  // border
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(frost_blue);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.63f, 0.85f, 0.92f, 1.0f));  // Brighter frost
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(frost_light);
            style.Colors[ImNodesCol_Link] = ColorToU32(frost_cyan);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(frost_blue);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(frost_light);
            style.Colors[ImNodesCol_Pin] = ColorToU32(aurora_purple);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.81f, 0.66f, 0.78f, 1.0f));  // Brighter purple
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(frost_blue.x, frost_blue.y, frost_blue.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(frost_blue);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.18f, 0.20f, 0.25f, 1.0f));  // bg_dark
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.22f, 0.24f, 0.30f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.26f, 0.28f, 0.34f, 1.0f));
            break;
        }

        case ThemePreset::CatppuccinMocha: {
            // Catppuccin Mocha theme - pastel rainbow
            style.NodeCornerRounding = 8.0f;

            ImVec4 base = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);  // #1e1e2e
            ImVec4 surface0 = ImVec4(0.20f, 0.22f, 0.28f, 1.0f);  // #313244
            ImVec4 surface1 = ImVec4(0.28f, 0.30f, 0.38f, 1.0f);  // #45475a
            ImVec4 surface2 = ImVec4(0.36f, 0.38f, 0.46f, 1.0f);  // #585b70
            ImVec4 blue = ImVec4(0.54f, 0.71f, 0.98f, 1.0f);  // #89b4fa
            ImVec4 sapphire = ImVec4(0.45f, 0.78f, 0.93f, 1.0f);  // #74c7ec
            ImVec4 teal = ImVec4(0.58f, 0.89f, 0.84f, 1.0f);  // #94e2d5
            ImVec4 mauve = ImVec4(0.80f, 0.65f, 0.97f, 1.0f);  // #cba6f7
            ImVec4 pink = ImVec4(0.96f, 0.76f, 0.90f, 1.0f);  // #f5c2e7
            ImVec4 lavender = ImVec4(0.71f, 0.75f, 1.00f, 1.0f);  // #b4befe

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(surface0);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(surface1);
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(surface2);
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.36f, 0.38f, 0.46f, 1.0f));  // surface2
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(blue);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(lavender);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(mauve);
            style.Colors[ImNodesCol_Link] = ColorToU32(teal);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(sapphire);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(blue);
            style.Colors[ImNodesCol_Pin] = ColorToU32(pink);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(1.00f, 0.86f, 0.95f, 1.0f));  // Brighter pink
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(blue.x, blue.y, blue.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(blue);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(base);
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.16f, 0.16f, 0.22f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.20f, 0.20f, 0.26f, 1.0f));
            break;
        }

        default:
            break;
    }

    // Mini-map styling (common for all themes)
    style.Colors[ImNodesCol_MiniMapBackground] = ColorToU32(ImVec4(0.10f, 0.10f, 0.12f, 0.80f));
    style.Colors[ImNodesCol_MiniMapBackgroundHovered] = ColorToU32(ImVec4(0.15f, 0.15f, 0.18f, 0.90f));
    style.Colors[ImNodesCol_MiniMapOutline] = ColorToU32(ImVec4(0.30f, 0.30f, 0.35f, 1.0f));
    style.Colors[ImNodesCol_MiniMapOutlineHovered] = ColorToU32(ImVec4(0.40f, 0.40f, 0.45f, 1.0f));
    style.Colors[ImNodesCol_MiniMapNodeBackground] = style.Colors[ImNodesCol_NodeBackground];
    style.Colors[ImNodesCol_MiniMapNodeBackgroundHovered] = style.Colors[ImNodesCol_NodeBackgroundHovered];
    style.Colors[ImNodesCol_MiniMapNodeBackgroundSelected] = style.Colors[ImNodesCol_NodeBackgroundSelected];
    style.Colors[ImNodesCol_MiniMapNodeOutline] = style.Colors[ImNodesCol_NodeOutline];
    style.Colors[ImNodesCol_MiniMapLink] = style.Colors[ImNodesCol_Link];
    style.Colors[ImNodesCol_MiniMapLinkSelected] = style.Colors[ImNodesCol_LinkSelected];
    style.Colors[ImNodesCol_MiniMapCanvas] = ColorToU32(ImVec4(0.08f, 0.08f, 0.10f, 0.50f));
    style.Colors[ImNodesCol_MiniMapCanvasOutline] = ColorToU32(ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
}

// ============================================================================
// Dock Style Integration - Matches dock tabs to current theme
// ============================================================================
void Theme::ApplyDockStyle() {
    DockStyle& dock_style = GetDockStyle();

    // Map theme presets to dock style presets
    switch (current_preset_) {
        case ThemePreset::UnrealEngine:
            dock_style.ApplyPreset(DockStylePreset::UnrealEngine);
            break;

        case ThemePreset::VSCodeDark:
            dock_style.ApplyPreset(DockStylePreset::VSCode);
            break;

        case ThemePreset::CyxWizDark:
        case ThemePreset::ModernDark: {
            // Use Unreal-style but with CyxWiz blue accent
            dock_style.ApplyPreset(DockStylePreset::UnrealEngine);

            // Override with CyxWiz blue accent
            DockTabStyle style = dock_style.GetStyle();
            style.active_indicator_color = ImVec4(0.20f, 0.55f, 0.85f, 1.0f);  // CyxWiz blue
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxWizLight: {
            // Light theme variant
            dock_style.ApplyPreset(DockStylePreset::Unity);

            DockTabStyle style = dock_style.GetStyle();
            // Adjust for light theme
            style.tab_bg = ImVec4(0.88f, 0.88f, 0.90f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.92f, 0.92f, 0.94f, 1.0f);
            style.tab_bg_active = ImVec4(0.96f, 0.96f, 0.98f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.85f, 0.85f, 0.87f, 1.0f);
            style.tab_text = ImVec4(0.35f, 0.35f, 0.38f, 1.0f);
            style.tab_text_active = ImVec4(0.10f, 0.10f, 0.12f, 1.0f);
            style.active_indicator_color = ImVec4(0.20f, 0.50f, 0.80f, 1.0f);
            style.dock_bg = ImVec4(0.92f, 0.92f, 0.94f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::HighContrast: {
            // High contrast with bold indicator
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
            style.tab_bg_active = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
            style.tab_text = ImVec4(0.80f, 0.80f, 0.80f, 1.0f);
            style.tab_text_active = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            style.active_indicator_color = ImVec4(0.0f, 0.80f, 1.0f, 1.0f);  // Cyan
            style.active_indicator_height = 3.0f;
            style.dock_bg = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::Dracula: {
            // Dracula theme with vibrant purple/pink accents
            dock_style.ApplyPreset(DockStylePreset::UnrealEngine);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.22f, 0.22f, 0.28f, 1.0f);  // bg_light
            style.tab_bg_hovered = ImVec4(0.26f, 0.26f, 0.32f, 1.0f);
            style.tab_bg_active = ImVec4(0.18f, 0.18f, 0.24f, 1.0f);  // bg_medium
            style.tab_bg_unfocused = ImVec4(0.16f, 0.16f, 0.21f, 1.0f);  // bg_dark
            style.tab_text = ImVec4(0.60f, 0.60f, 0.65f, 1.0f);  // text_dim
            style.tab_text_active = ImVec4(0.97f, 0.97f, 0.95f, 1.0f);  // text
            style.active_indicator_color = ImVec4(0.74f, 0.58f, 0.98f, 1.0f);  // Purple
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 6.0f;
            style.dock_bg = ImVec4(0.16f, 0.16f, 0.21f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::OneDarkPro: {
            // One Dark Pro theme with blue/cyan accents
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.21f, 0.22f, 0.26f, 1.0f);  // bg_light
            style.tab_bg_hovered = ImVec4(0.25f, 0.26f, 0.30f, 1.0f);
            style.tab_bg_active = ImVec4(0.18f, 0.19f, 0.22f, 1.0f);  // bg_medium
            style.tab_bg_unfocused = ImVec4(0.16f, 0.17f, 0.20f, 1.0f);  // bg_dark
            style.tab_text = ImVec4(0.45f, 0.48f, 0.53f, 1.0f);  // text_dim
            style.tab_text_active = ImVec4(0.67f, 0.70f, 0.75f, 1.0f);  // text
            style.active_indicator_color = ImVec4(0.38f, 0.69f, 0.94f, 1.0f);  // Blue
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 5.0f;
            style.dock_bg = ImVec4(0.16f, 0.17f, 0.20f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::Nord: {
            // Nord theme with frost blue accents
            dock_style.ApplyPreset(DockStylePreset::Unity);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.26f, 0.30f, 0.37f, 1.0f);  // bg_light
            style.tab_bg_hovered = ImVec4(0.30f, 0.34f, 0.42f, 1.0f);
            style.tab_bg_active = ImVec4(0.23f, 0.26f, 0.32f, 1.0f);  // bg_medium
            style.tab_bg_unfocused = ImVec4(0.18f, 0.20f, 0.25f, 1.0f);  // bg_dark
            style.tab_text = ImVec4(0.60f, 0.63f, 0.68f, 1.0f);  // text_dim
            style.tab_text_active = ImVec4(0.85f, 0.87f, 0.91f, 1.0f);  // text
            style.active_indicator_color = ImVec4(0.53f, 0.75f, 0.82f, 1.0f);  // Frost Blue
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 4.0f;
            style.dock_bg = ImVec4(0.18f, 0.20f, 0.25f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CatppuccinMocha: {
            // Catppuccin Mocha theme with pastel blue/pink accents
            dock_style.ApplyPreset(DockStylePreset::Unity);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.20f, 0.22f, 0.28f, 1.0f);  // surface0
            style.tab_bg_hovered = ImVec4(0.28f, 0.30f, 0.38f, 1.0f);  // surface1
            style.tab_bg_active = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);  // base
            style.tab_bg_unfocused = ImVec4(0.09f, 0.09f, 0.15f, 1.0f);  // mantle
            style.tab_text = ImVec4(0.65f, 0.68f, 0.78f, 1.0f);  // subtext0
            style.tab_text_active = ImVec4(0.80f, 0.84f, 0.96f, 1.0f);  // text
            style.active_indicator_color = ImVec4(0.54f, 0.71f, 0.98f, 1.0f);  // Blue
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 8.0f;
            style.dock_bg = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        default:
            dock_style.ApplyPreset(DockStylePreset::Default);
            break;
    }
}

} // namespace gui
