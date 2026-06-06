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

        // ============== CyxOS Platform Themes ==============

        case ThemePreset::CyxOSAqua: {
            // macOS Big Sur style - clean, rounded nodes
            style.NodeCornerRounding = 10.0f;
            style.NodeBorderThickness = 1.0f;
            style.LinkThickness = 3.0f;

            ImVec4 node_bg = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
            ImVec4 accent = ImVec4(0.00f, 0.48f, 1.00f, 1.0f);  // #007AFF
            ImVec4 border = ImVec4(0.26f, 0.26f, 0.27f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.22f, 0.22f, 0.22f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.25f, 0.25f, 0.26f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(border);
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(accent);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.10f, 0.55f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(0.20f, 0.60f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_Link] = ColorToU32(ImVec4(0.20f, 0.78f, 0.35f, 1.0f));  // Green success
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.30f, 0.88f, 0.45f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(accent);
            style.Colors[ImNodesCol_Pin] = ColorToU32(accent);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.34f, 0.34f, 0.84f, 1.0f));  // Purple secondary
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(accent.x, accent.y, accent.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(accent);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.11f, 0.11f, 0.12f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.16f, 0.16f, 0.17f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.22f, 0.22f, 0.23f, 1.0f));
            break;
        }

        case ThemePreset::CyxOSFluent: {
            // Windows 11 Fluent Design - subtle rounded corners
            style.NodeCornerRounding = 8.0f;
            style.NodeBorderThickness = 1.0f;
            style.LinkThickness = 2.5f;

            ImVec4 node_bg = ImVec4(0.20f, 0.20f, 0.20f, 1.0f);
            ImVec4 accent = ImVec4(0.00f, 0.47f, 0.83f, 1.0f);  // #0078D4
            ImVec4 accent_light = ImVec4(0.38f, 0.80f, 1.00f, 1.0f);  // #60CDFF

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(node_bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.25f, 0.25f, 0.25f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.28f, 0.28f, 0.28f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.24f, 0.24f, 0.24f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(accent);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(accent.x * 1.1f, accent.y * 1.1f, accent.z * 1.1f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(accent_light);
            style.Colors[ImNodesCol_Link] = ColorToU32(accent_light);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.48f, 0.90f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(accent);
            style.Colors[ImNodesCol_Pin] = ColorToU32(ImVec4(0.42f, 0.80f, 0.37f, 1.0f));  // Success green
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.52f, 0.90f, 0.47f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(accent.x, accent.y, accent.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(accent);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.13f, 0.13f, 0.13f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.18f, 0.18f, 0.18f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.22f, 0.22f, 0.22f, 1.0f));
            break;
        }

        case ThemePreset::CyxOSCoder: {
            // Developer IDE - syntax highlighting colors
            style.NodeCornerRounding = 4.0f;
            style.NodeBorderThickness = 1.0f;
            style.LinkThickness = 2.0f;

            ImVec4 bg = ImVec4(0.16f, 0.16f, 0.21f, 1.0f);
            ImVec4 accent = ImVec4(0.54f, 0.71f, 0.98f, 1.0f);  // #89B4FA
            ImVec4 func_cyan = ImVec4(0.54f, 0.86f, 0.92f, 1.0f);  // #89DCEB
            ImVec4 keyword = ImVec4(0.80f, 0.65f, 0.97f, 1.0f);  // #CBA6F7
            ImVec4 string = ImVec4(0.65f, 0.89f, 0.63f, 1.0f);  // #A6E3A1

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.20f, 0.20f, 0.26f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.24f, 0.24f, 0.30f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.27f, 0.28f, 0.35f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(keyword);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.90f, 0.75f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(accent);
            style.Colors[ImNodesCol_Link] = ColorToU32(func_cyan);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.64f, 0.96f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(accent);
            style.Colors[ImNodesCol_Pin] = ColorToU32(string);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.75f, 0.99f, 0.73f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(accent.x, accent.y, accent.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(accent);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.12f, 0.12f, 0.18f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.16f, 0.16f, 0.22f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.20f, 0.20f, 0.26f, 1.0f));
            break;
        }

        case ThemePreset::CyxOSOffice: {
            // Professional enterprise - clean, corporate
            style.NodeCornerRounding = 6.0f;
            style.NodeBorderThickness = 1.0f;
            style.LinkThickness = 2.5f;

            ImVec4 bg = ImVec4(0.22f, 0.25f, 0.32f, 1.0f);
            ImVec4 accent = ImVec4(0.15f, 0.39f, 0.92f, 1.0f);  // #2563EB
            ImVec4 success = ImVec4(0.06f, 0.72f, 0.51f, 1.0f);  // #10B981

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(bg);
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.26f, 0.29f, 0.36f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.30f, 0.33f, 0.40f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(ImVec4(0.29f, 0.34f, 0.39f, 1.0f));
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(accent);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(ImVec4(0.23f, 0.51f, 0.96f, 1.0f));
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(ImVec4(0.30f, 0.58f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_Link] = ColorToU32(success);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(ImVec4(0.16f, 0.82f, 0.61f, 1.0f));
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(accent);
            style.Colors[ImNodesCol_Pin] = ColorToU32(accent);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.25f, 0.49f, 1.00f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(accent.x, accent.y, accent.z, 0.30f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(accent);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.12f, 0.16f, 0.22f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.18f, 0.22f, 0.28f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.24f, 0.28f, 0.34f, 1.0f));
            break;
        }

        // ============== CyxOS Retro TUI Themes ==============

        case ThemePreset::CyxOSTuiClassic: {
            // Green phosphor terminal - sharp edges, glow effect
            style.NodeCornerRounding = 0.0f;
            style.NodeBorderThickness = 1.5f;
            style.LinkThickness = 2.0f;

            ImVec4 green = ImVec4(0.20f, 1.00f, 0.20f, 1.0f);
            ImVec4 green_glow = ImVec4(0.00f, 1.00f, 0.00f, 1.0f);
            ImVec4 green_dim = ImVec4(0.10f, 0.55f, 0.10f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(ImVec4(0.05f, 0.05f, 0.05f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.08f, 0.08f, 0.08f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.10f, 0.12f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(green_dim);
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(green_dim);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(green);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(green_glow);
            style.Colors[ImNodesCol_Link] = ColorToU32(green);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(green_glow);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(green_glow);
            style.Colors[ImNodesCol_Pin] = ColorToU32(green_glow);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.30f, 1.00f, 0.30f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(0.00f, 1.00f, 0.00f, 0.25f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(green_glow);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.04f, 0.04f, 0.04f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.08f, 0.12f, 0.08f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.10f, 0.20f, 0.10f, 1.0f));
            break;
        }

        case ThemePreset::CyxOSTuiMatrix: {
            // Matrix digital rain - deep black, neon green
            style.NodeCornerRounding = 0.0f;
            style.NodeBorderThickness = 1.5f;
            style.LinkThickness = 2.0f;

            ImVec4 matrix = ImVec4(0.00f, 1.00f, 0.25f, 1.0f);  // #00FF41
            ImVec4 bright = ImVec4(0.50f, 1.00f, 0.00f, 1.0f);  // #7FFF00
            ImVec4 dim = ImVec4(0.00f, 0.23f, 0.00f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(ImVec4(0.00f, 0.04f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.00f, 0.08f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.00f, 0.12f, 0.02f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(dim);
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(dim);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(matrix);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(bright);
            style.Colors[ImNodesCol_Link] = ColorToU32(matrix);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(bright);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(bright);
            style.Colors[ImNodesCol_Pin] = ColorToU32(bright);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(0.60f, 1.00f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(0.00f, 1.00f, 0.25f, 0.25f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(matrix);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.00f, 0.00f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.00f, 0.08f, 0.00f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.00f, 0.15f, 0.00f, 1.0f));
            break;
        }

        case ThemePreset::CyxOSTuiAmber: {
            // Amber CRT phosphor - warm retro
            style.NodeCornerRounding = 0.0f;
            style.NodeBorderThickness = 1.5f;
            style.LinkThickness = 2.0f;

            ImVec4 amber = ImVec4(1.00f, 0.69f, 0.00f, 1.0f);  // #FFB000
            ImVec4 bright = ImVec4(1.00f, 0.75f, 0.00f, 1.0f);  // #FFC000
            ImVec4 dim = ImVec4(0.55f, 0.37f, 0.00f, 1.0f);

            style.Colors[ImNodesCol_NodeBackground] = ColorToU32(ImVec4(0.08f, 0.06f, 0.03f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundHovered] = ColorToU32(ImVec4(0.12f, 0.09f, 0.05f, 1.0f));
            style.Colors[ImNodesCol_NodeBackgroundSelected] = ColorToU32(ImVec4(0.15f, 0.11f, 0.06f, 1.0f));
            style.Colors[ImNodesCol_NodeOutline] = ColorToU32(dim);
            style.Colors[ImNodesCol_TitleBar] = ColorToU32(dim);
            style.Colors[ImNodesCol_TitleBarHovered] = ColorToU32(amber);
            style.Colors[ImNodesCol_TitleBarSelected] = ColorToU32(bright);
            style.Colors[ImNodesCol_Link] = ColorToU32(amber);
            style.Colors[ImNodesCol_LinkHovered] = ColorToU32(bright);
            style.Colors[ImNodesCol_LinkSelected] = ColorToU32(bright);
            style.Colors[ImNodesCol_Pin] = ColorToU32(bright);
            style.Colors[ImNodesCol_PinHovered] = ColorToU32(ImVec4(1.00f, 0.85f, 0.10f, 1.0f));
            style.Colors[ImNodesCol_BoxSelector] = ColorToU32(ImVec4(1.00f, 0.69f, 0.00f, 0.25f));
            style.Colors[ImNodesCol_BoxSelectorOutline] = ColorToU32(amber);
            style.Colors[ImNodesCol_GridBackground] = ColorToU32(ImVec4(0.05f, 0.04f, 0.02f, 1.0f));
            style.Colors[ImNodesCol_GridLine] = ColorToU32(ImVec4(0.10f, 0.07f, 0.03f, 1.0f));
            style.Colors[ImNodesCol_GridLinePrimary] = ColorToU32(ImVec4(0.15f, 0.10f, 0.04f, 1.0f));
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

        // ============== CyxOS Platform Themes ==============

        case ThemePreset::CyxOSAqua: {
            // macOS Big Sur style - rounded, clean
            dock_style.ApplyPreset(DockStylePreset::Unity);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
            style.tab_bg_active = ImVec4(0.11f, 0.11f, 0.12f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
            style.tab_text = ImVec4(0.60f, 0.60f, 0.62f, 1.0f);
            style.tab_text_active = ImVec4(0.96f, 0.96f, 0.97f, 1.0f);
            style.active_indicator_color = ImVec4(0.00f, 0.48f, 1.00f, 1.0f);  // #007AFF
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 6.0f;
            style.dock_bg = ImVec4(0.11f, 0.11f, 0.12f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxOSFluent: {
            // Windows 11 Fluent Design
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
            style.tab_bg_active = ImVec4(0.13f, 0.13f, 0.13f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.16f, 0.16f, 0.16f, 1.0f);
            style.tab_text = ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
            style.tab_text_active = ImVec4(1.00f, 1.00f, 1.00f, 1.0f);
            style.active_indicator_color = ImVec4(0.00f, 0.47f, 0.83f, 1.0f);  // #0078D4
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 4.0f;
            style.dock_bg = ImVec4(0.13f, 0.13f, 0.13f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxOSCoder: {
            // Developer IDE - syntax colored
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.16f, 0.16f, 0.21f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.20f, 0.20f, 0.26f, 1.0f);
            style.tab_bg_active = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.14f, 0.14f, 0.19f, 1.0f);
            style.tab_text = ImVec4(0.42f, 0.44f, 0.53f, 1.0f);
            style.tab_text_active = ImVec4(0.80f, 0.84f, 0.96f, 1.0f);
            style.active_indicator_color = ImVec4(0.80f, 0.65f, 0.97f, 1.0f);  // #CBA6F7 purple
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 3.0f;
            style.dock_bg = ImVec4(0.12f, 0.12f, 0.18f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxOSOffice: {
            // Professional enterprise
            dock_style.ApplyPreset(DockStylePreset::Unity);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.22f, 0.25f, 0.32f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.26f, 0.29f, 0.36f, 1.0f);
            style.tab_bg_active = ImVec4(0.12f, 0.16f, 0.22f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.18f, 0.21f, 0.28f, 1.0f);
            style.tab_text = ImVec4(0.60f, 0.65f, 0.71f, 1.0f);
            style.tab_text_active = ImVec4(0.97f, 0.97f, 0.98f, 1.0f);
            style.active_indicator_color = ImVec4(0.15f, 0.39f, 0.92f, 1.0f);  // #2563EB
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 4.0f;
            style.dock_bg = ImVec4(0.12f, 0.16f, 0.22f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        // ============== CyxOS Retro TUI Themes ==============

        case ThemePreset::CyxOSTuiClassic: {
            // Green phosphor terminal - sharp edges
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.05f, 0.05f, 0.05f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.08f, 0.12f, 0.08f, 1.0f);
            style.tab_bg_active = ImVec4(0.04f, 0.04f, 0.04f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.05f, 0.05f, 0.05f, 1.0f);
            style.tab_text = ImVec4(0.10f, 0.55f, 0.10f, 1.0f);
            style.tab_text_active = ImVec4(0.20f, 1.00f, 0.20f, 1.0f);
            style.active_indicator_color = ImVec4(0.00f, 1.00f, 0.00f, 1.0f);  // Glow green
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 0.0f;
            style.dock_bg = ImVec4(0.04f, 0.04f, 0.04f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxOSTuiMatrix: {
            // Matrix digital rain - neon green
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.00f, 0.04f, 0.00f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.00f, 0.10f, 0.00f, 1.0f);
            style.tab_bg_active = ImVec4(0.00f, 0.00f, 0.00f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.00f, 0.02f, 0.00f, 1.0f);
            style.tab_text = ImVec4(0.00f, 0.23f, 0.00f, 1.0f);
            style.tab_text_active = ImVec4(0.00f, 1.00f, 0.25f, 1.0f);
            style.active_indicator_color = ImVec4(0.50f, 1.00f, 0.00f, 1.0f);  // #7FFF00
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 0.0f;
            style.dock_bg = ImVec4(0.00f, 0.00f, 0.00f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        case ThemePreset::CyxOSTuiAmber: {
            // Amber CRT phosphor - warm glow
            dock_style.ApplyPreset(DockStylePreset::VSCode);

            DockTabStyle style = dock_style.GetStyle();
            style.tab_bg = ImVec4(0.08f, 0.06f, 0.03f, 1.0f);
            style.tab_bg_hovered = ImVec4(0.12f, 0.09f, 0.05f, 1.0f);
            style.tab_bg_active = ImVec4(0.05f, 0.04f, 0.02f, 1.0f);
            style.tab_bg_unfocused = ImVec4(0.06f, 0.05f, 0.03f, 1.0f);
            style.tab_text = ImVec4(0.55f, 0.37f, 0.00f, 1.0f);
            style.tab_text_active = ImVec4(1.00f, 0.69f, 0.00f, 1.0f);
            style.active_indicator_color = ImVec4(1.00f, 0.75f, 0.00f, 1.0f);  // #FFC000
            style.active_indicator_height = 2.0f;
            style.tab_rounding = 0.0f;
            style.dock_bg = ImVec4(0.05f, 0.04f, 0.02f, 1.0f);
            dock_style.SetStyle(style);
            break;
        }

        default:
            dock_style.ApplyPreset(DockStylePreset::Default);
            break;
    }
}

} // namespace gui
