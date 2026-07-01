// Theme styling for dock tabs.

#include "theme.h"
#include "dock_style.h"

#include <imgui.h>

namespace gui {

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
        case ThemePreset::CyxWizLaunch:
        case ThemePreset::ModernDark: {
            // Use Unreal-style but with CyxWiz blue accent
            dock_style.ApplyPreset(DockStylePreset::UnrealEngine);

            // Override with CyxWiz blue accent
            DockTabStyle style = dock_style.GetStyle();
            style.active_indicator_color =
                current_preset_ == ThemePreset::CyxWizLaunch
                    ? ImVec4(0.04f, 0.48f, 1.00f, 1.0f)
                    : ImVec4(0.20f, 0.55f, 0.85f, 1.0f);
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
