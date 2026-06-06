// Theme preset color definitions.

#include "theme.h"

namespace gui {

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
// CyxOS Aqua Theme - macOS Big Sur inspired
// Clean, minimal design with subtle transparency and SF-style aesthetics
// ============================================================================
void Theme::ApplyCyxOSAqua() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // macOS Big Sur color palette (dark mode)
    ImVec4 bg_primary    = ImVec4(0.11f, 0.11f, 0.12f, 1.00f);  // #1D1D1F
    ImVec4 bg_secondary  = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);  // #2D2D2D
    ImVec4 bg_tertiary   = ImVec4(0.22f, 0.22f, 0.23f, 1.00f);  // #383839
    ImVec4 text_primary  = ImVec4(0.96f, 0.96f, 0.97f, 1.00f);  // #F5F5F7
    ImVec4 text_secondary= ImVec4(0.60f, 0.60f, 0.62f, 1.00f);  // #999A9E
    ImVec4 border        = ImVec4(0.26f, 0.26f, 0.27f, 1.00f);  // #424245

    // macOS accent colors
    ImVec4 accent_blue   = ImVec4(0.00f, 0.48f, 1.00f, 1.00f);  // #007AFF
    ImVec4 accent_purple = ImVec4(0.34f, 0.34f, 0.84f, 1.00f);  // #5856D6
    ImVec4 success       = ImVec4(0.20f, 0.78f, 0.35f, 1.00f);  // #34C759
    ImVec4 warning       = ImVec4(1.00f, 0.58f, 0.00f, 1.00f);  // #FF9500
    ImVec4 error         = ImVec4(1.00f, 0.23f, 0.19f, 1.00f);  // #FF3B30

    // Text
    colors[ImGuiCol_Text]                   = text_primary;
    colors[ImGuiCol_TextDisabled]           = text_secondary;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_primary;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_secondary.x, bg_secondary.y, bg_secondary.z, 0.95f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_secondary;
    colors[ImGuiCol_FrameBgHovered]         = bg_tertiary;
    colors[ImGuiCol_FrameBgActive]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.40f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_primary;
    colors[ImGuiCol_TitleBgActive]          = bg_secondary;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_primary.x, bg_primary.y, bg_primary.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_primary;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.50f, 0.50f, 0.52f, 0.50f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.60f, 0.60f, 0.62f, 0.60f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.70f, 0.70f, 0.72f, 0.70f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = accent_blue;

    // Slider
    colors[ImGuiCol_SliderGrab]             = accent_blue;
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(accent_blue.x * 1.2f, accent_blue.y * 1.2f, accent_blue.z, 1.00f);

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.85f);
    colors[ImGuiCol_ButtonHovered]          = accent_blue;
    colors[ImGuiCol_ButtonActive]           = ImVec4(accent_blue.x * 0.8f, accent_blue.y * 0.8f, accent_blue.z * 0.8f, 1.00f);

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.30f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.50f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);

    // Separator
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent_blue;
    colors[ImGuiCol_SeparatorActive]        = accent_blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.50f, 0.50f, 0.50f, 0.20f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.65f);
    colors[ImGuiCol_ResizeGripActive]       = accent_blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_secondary;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.90f);
    colors[ImGuiCol_TabUnfocused]           = bg_secondary;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.50f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_primary;

    // Plot
    colors[ImGuiCol_PlotLines]              = accent_blue;
    colors[ImGuiCol_PlotLinesHovered]       = accent_purple;
    colors[ImGuiCol_PlotHistogram]          = success;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_secondary;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.35f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = accent_blue;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = accent_blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    // macOS-style rounded corners
    config_.window_rounding = 10.0f;
    config_.frame_rounding = 6.0f;
    config_.popup_rounding = 10.0f;
    config_.scrollbar_rounding = 10.0f;
    config_.grab_rounding = 6.0f;
    config_.tab_rounding = 6.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
    config_.window_padding = ImVec2(10.0f, 10.0f);
    config_.frame_padding = ImVec2(8.0f, 5.0f);
    config_.item_spacing = ImVec2(10.0f, 6.0f);
}

// ============================================================================
// CyxOS Fluent Theme - Windows 11 Fluent Design
// Mica-style effects, subtle transparency, modern Microsoft aesthetics
// ============================================================================
void Theme::ApplyCyxOSFluent() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Windows 11 Fluent Design palette (dark)
    ImVec4 bg_mica       = ImVec4(0.13f, 0.13f, 0.13f, 1.00f);  // #202020
    ImVec4 bg_surface    = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);  // #2D2D2D
    ImVec4 bg_card       = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);  // #333333
    ImVec4 text_primary  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);  // #FFFFFF
    ImVec4 text_secondary= ImVec4(0.70f, 0.70f, 0.70f, 1.00f);  // #B3B3B3
    ImVec4 border        = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);  // #3D3D3D

    // Fluent accent colors
    ImVec4 accent_blue   = ImVec4(0.00f, 0.47f, 0.83f, 1.00f);  // #0078D4
    ImVec4 accent_light  = ImVec4(0.38f, 0.80f, 1.00f, 1.00f);  // #60CDFF
    ImVec4 success       = ImVec4(0.42f, 0.80f, 0.37f, 1.00f);  // #6CCB5F
    ImVec4 warning       = ImVec4(0.99f, 0.88f, 0.00f, 1.00f);  // #FCE100
    ImVec4 error         = ImVec4(0.91f, 0.07f, 0.14f, 1.00f);  // #E81123

    // Text
    colors[ImGuiCol_Text]                   = text_primary;
    colors[ImGuiCol_TextDisabled]           = text_secondary;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_mica;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_surface.x, bg_surface.y, bg_surface.z, 0.98f);

    // Borders - Fluent uses subtle borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame - Fluent card style
    colors[ImGuiCol_FrameBg]                = bg_card;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(bg_card.x + 0.05f, bg_card.y + 0.05f, bg_card.z + 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.40f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_mica;
    colors[ImGuiCol_TitleBgActive]          = bg_surface;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_mica.x, bg_mica.y, bg_mica.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_mica;

    // Scrollbar - Fluent thin style
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.55f, 0.55f, 0.55f, 0.40f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.65f, 0.65f, 0.65f, 0.50f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.75f, 0.75f, 0.75f, 0.60f);

    // Check mark
    colors[ImGuiCol_CheckMark]              = accent_light;

    // Slider
    colors[ImGuiCol_SliderGrab]             = accent_blue;
    colors[ImGuiCol_SliderGrabActive]       = accent_light;

    // Button - Fluent accent button
    colors[ImGuiCol_Button]                 = accent_blue;
    colors[ImGuiCol_ButtonHovered]          = ImVec4(accent_blue.x * 1.1f, accent_blue.y * 1.1f, accent_blue.z * 1.1f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(accent_blue.x * 0.85f, accent_blue.y * 0.85f, accent_blue.z * 0.85f, 1.00f);

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.25f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.45f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.65f);

    // Separator
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent_light;
    colors[ImGuiCol_SeparatorActive]        = accent_blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.50f, 0.50f, 0.50f, 0.15f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.55f);
    colors[ImGuiCol_ResizeGripActive]       = accent_blue;

    // Tabs - Fluent tab style
    colors[ImGuiCol_Tab]                    = bg_surface;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.65f);
    colors[ImGuiCol_TabActive]              = accent_blue;
    colors[ImGuiCol_TabUnfocused]           = bg_surface;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.45f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_mica;

    // Plot
    colors[ImGuiCol_PlotLines]              = accent_light;
    colors[ImGuiCol_PlotLinesHovered]       = accent_blue;
    colors[ImGuiCol_PlotHistogram]          = success;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_surface;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.30f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = accent_light;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = accent_blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.55f);

    // Fluent Design rounded corners (8px standard)
    config_.window_rounding = 8.0f;
    config_.frame_rounding = 4.0f;
    config_.popup_rounding = 8.0f;
    config_.scrollbar_rounding = 8.0f;
    config_.grab_rounding = 4.0f;
    config_.tab_rounding = 4.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
    config_.window_padding = ImVec2(12.0f, 12.0f);
    config_.frame_padding = ImVec2(12.0f, 6.0f);
    config_.item_spacing = ImVec2(8.0f, 6.0f);
}

// ============================================================================
// CyxOS Coder Theme - Developer IDE
// Syntax highlighting colors, optimized for long coding sessions
// ============================================================================
void Theme::ApplyCyxOSCoder() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Developer IDE color palette
    ImVec4 bg_dark       = ImVec4(0.12f, 0.12f, 0.18f, 1.00f);  // #1E1E2E
    ImVec4 bg_medium     = ImVec4(0.16f, 0.16f, 0.21f, 1.00f);  // #282A36
    ImVec4 bg_light      = ImVec4(0.20f, 0.20f, 0.26f, 1.00f);  // #313244
    ImVec4 text_primary  = ImVec4(0.80f, 0.84f, 0.96f, 1.00f);  // #CDD6F4
    ImVec4 text_dim      = ImVec4(0.42f, 0.44f, 0.53f, 1.00f);  // #6C7086
    ImVec4 comment       = ImVec4(0.38f, 0.45f, 0.64f, 1.00f);  // #6272A4
    ImVec4 border        = ImVec4(0.27f, 0.28f, 0.35f, 1.00f);  // #45475A

    // Syntax highlighting colors
    ImVec4 accent_blue   = ImVec4(0.54f, 0.71f, 0.98f, 1.00f);  // #89B4FA
    ImVec4 accent_pink   = ImVec4(0.96f, 0.76f, 0.91f, 1.00f);  // #F5C2E7
    ImVec4 string_green  = ImVec4(0.65f, 0.89f, 0.63f, 1.00f);  // #A6E3A1
    ImVec4 number_orange = ImVec4(0.98f, 0.70f, 0.53f, 1.00f);  // #FAB387
    ImVec4 keyword_purple= ImVec4(0.80f, 0.65f, 0.97f, 1.00f);  // #CBA6F7
    ImVec4 function_cyan = ImVec4(0.54f, 0.86f, 0.92f, 1.00f);  // #89DCEB
    ImVec4 error_red     = ImVec4(0.95f, 0.55f, 0.66f, 1.00f);  // #F38BA8
    ImVec4 warning_yellow= ImVec4(0.98f, 0.89f, 0.69f, 1.00f);  // #F9E2AF

    // Text
    colors[ImGuiCol_Text]                   = text_primary;
    colors[ImGuiCol_TextDisabled]           = text_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_dark;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_medium.x, bg_medium.y, bg_medium.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_medium;
    colors[ImGuiCol_FrameBgHovered]         = bg_light;
    colors[ImGuiCol_FrameBgActive]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.35f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = bg_medium;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = border;
    colors[ImGuiCol_ScrollbarGrabHovered]   = comment;
    colors[ImGuiCol_ScrollbarGrabActive]    = accent_blue;

    // Check mark - use string green for success
    colors[ImGuiCol_CheckMark]              = string_green;

    // Slider - use accent blue
    colors[ImGuiCol_SliderGrab]             = accent_blue;
    colors[ImGuiCol_SliderGrabActive]       = function_cyan;

    // Button - use keyword purple
    colors[ImGuiCol_Button]                 = ImVec4(keyword_purple.x, keyword_purple.y, keyword_purple.z, 0.75f);
    colors[ImGuiCol_ButtonHovered]          = keyword_purple;
    colors[ImGuiCol_ButtonActive]           = ImVec4(keyword_purple.x * 0.85f, keyword_purple.y * 0.85f, keyword_purple.z * 0.85f, 1.00f);

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.30f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.50f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);

    // Separator
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = function_cyan;
    colors[ImGuiCol_SeparatorActive]        = accent_blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.20f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.55f);
    colors[ImGuiCol_ResizeGripActive]       = accent_blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_medium;
    colors[ImGuiCol_TabHovered]             = ImVec4(keyword_purple.x, keyword_purple.y, keyword_purple.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.85f);
    colors[ImGuiCol_TabUnfocused]           = bg_medium;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.45f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent_pink.x, accent_pink.y, accent_pink.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot - use syntax colors
    colors[ImGuiCol_PlotLines]              = function_cyan;
    colors[ImGuiCol_PlotLinesHovered]       = accent_pink;
    colors[ImGuiCol_PlotHistogram]          = string_green;
    colors[ImGuiCol_PlotHistogramHovered]   = warning_yellow;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_medium;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.35f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = accent_pink;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = accent_blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);

    // Developer-focused minimal rounding
    config_.window_rounding = 4.0f;
    config_.frame_rounding = 3.0f;
    config_.popup_rounding = 4.0f;
    config_.scrollbar_rounding = 4.0f;
    config_.grab_rounding = 3.0f;
    config_.tab_rounding = 3.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
    config_.window_padding = ImVec2(8.0f, 8.0f);
    config_.frame_padding = ImVec2(6.0f, 4.0f);
    config_.item_spacing = ImVec2(8.0f, 4.0f);
}

// ============================================================================
// CyxOS Office Theme - Professional Enterprise
// Clean, productivity-focused design for business applications
// ============================================================================
void Theme::ApplyCyxOSOffice() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Professional enterprise palette (dark)
    ImVec4 bg_primary    = ImVec4(0.12f, 0.16f, 0.22f, 1.00f);  // #1F2937
    ImVec4 bg_secondary  = ImVec4(0.22f, 0.25f, 0.32f, 1.00f);  // #374151
    ImVec4 bg_tertiary   = ImVec4(0.29f, 0.33f, 0.40f, 1.00f);  // #4B5563
    ImVec4 text_primary  = ImVec4(0.97f, 0.97f, 0.98f, 1.00f);  // #F9FAFB
    ImVec4 text_secondary= ImVec4(0.60f, 0.65f, 0.71f, 1.00f);  // #9CA3AF
    ImVec4 border        = ImVec4(0.29f, 0.34f, 0.39f, 1.00f);  // #4B5563

    // Enterprise accent colors
    ImVec4 accent_blue   = ImVec4(0.15f, 0.39f, 0.92f, 1.00f);  // #2563EB
    ImVec4 accent_light  = ImVec4(0.23f, 0.51f, 0.96f, 1.00f);  // #3B82F6
    ImVec4 success       = ImVec4(0.06f, 0.72f, 0.51f, 1.00f);  // #10B981
    ImVec4 warning       = ImVec4(0.96f, 0.62f, 0.04f, 1.00f);  // #F59E0B
    ImVec4 error         = ImVec4(0.94f, 0.27f, 0.27f, 1.00f);  // #EF4444

    // Text
    colors[ImGuiCol_Text]                   = text_primary;
    colors[ImGuiCol_TextDisabled]           = text_secondary;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_primary;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_secondary.x, bg_secondary.y, bg_secondary.z, 0.98f);

    // Borders
    colors[ImGuiCol_Border]                 = border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_secondary;
    colors[ImGuiCol_FrameBgHovered]         = bg_tertiary;
    colors[ImGuiCol_FrameBgActive]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.35f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_primary;
    colors[ImGuiCol_TitleBgActive]          = bg_secondary;
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_primary.x, bg_primary.y, bg_primary.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_primary;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_primary;
    colors[ImGuiCol_ScrollbarGrab]          = bg_tertiary;
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(bg_tertiary.x + 0.10f, bg_tertiary.y + 0.10f, bg_tertiary.z + 0.10f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = accent_blue;

    // Check mark
    colors[ImGuiCol_CheckMark]              = success;

    // Slider
    colors[ImGuiCol_SliderGrab]             = accent_blue;
    colors[ImGuiCol_SliderGrabActive]       = accent_light;

    // Button - professional blue
    colors[ImGuiCol_Button]                 = accent_blue;
    colors[ImGuiCol_ButtonHovered]          = accent_light;
    colors[ImGuiCol_ButtonActive]           = ImVec4(accent_blue.x * 0.85f, accent_blue.y * 0.85f, accent_blue.z * 0.85f, 1.00f);

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.25f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.45f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.65f);

    // Separator
    colors[ImGuiCol_Separator]              = border;
    colors[ImGuiCol_SeparatorHovered]       = accent_light;
    colors[ImGuiCol_SeparatorActive]        = accent_blue;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.50f, 0.50f, 0.50f, 0.15f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.55f);
    colors[ImGuiCol_ResizeGripActive]       = accent_blue;

    // Tabs
    colors[ImGuiCol_Tab]                    = bg_secondary;
    colors[ImGuiCol_TabHovered]             = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.65f);
    colors[ImGuiCol_TabActive]              = accent_blue;
    colors[ImGuiCol_TabUnfocused]           = bg_secondary;
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.45f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_primary;

    // Plot
    colors[ImGuiCol_PlotLines]              = accent_light;
    colors[ImGuiCol_PlotLinesHovered]       = accent_blue;
    colors[ImGuiCol_PlotHistogram]          = success;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = bg_secondary;
    colors[ImGuiCol_TableBorderStrong]      = border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(border.x, border.y, border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.02f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(accent_blue.x, accent_blue.y, accent_blue.z, 0.30f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = accent_light;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = accent_blue;
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.55f);

    // Professional modest rounding
    config_.window_rounding = 6.0f;
    config_.frame_rounding = 4.0f;
    config_.popup_rounding = 6.0f;
    config_.scrollbar_rounding = 6.0f;
    config_.grab_rounding = 4.0f;
    config_.tab_rounding = 4.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 0.0f;
    config_.window_padding = ImVec2(10.0f, 10.0f);
    config_.frame_padding = ImVec2(10.0f, 6.0f);
    config_.item_spacing = ImVec2(8.0f, 6.0f);
}

// ============================================================================
// CyxOS TUI Classic Theme - Green Phosphor Terminal
// IBM 3278 inspired, classic green on black terminal
// ============================================================================
void Theme::ApplyCyxOSTuiClassic() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Classic green phosphor CRT palette
    ImVec4 bg_black      = ImVec4(0.04f, 0.04f, 0.04f, 1.00f);  // #0A0A0A
    ImVec4 bg_dark       = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);  // #0D0D0D
    ImVec4 green_primary = ImVec4(0.20f, 1.00f, 0.20f, 1.00f);  // #33FF33
    ImVec4 green_dim     = ImVec4(0.10f, 0.55f, 0.10f, 1.00f);  // #1A8C1A
    ImVec4 green_glow    = ImVec4(0.00f, 1.00f, 0.00f, 1.00f);  // #00FF00
    ImVec4 green_border  = ImVec4(0.10f, 0.30f, 0.10f, 1.00f);  // #1A4D1A
    ImVec4 warning       = ImVec4(0.80f, 1.00f, 0.20f, 1.00f);  // #CCFF33
    ImVec4 error         = ImVec4(1.00f, 0.20f, 0.20f, 1.00f);  // #FF3333

    // Text
    colors[ImGuiCol_Text]                   = green_primary;
    colors[ImGuiCol_TextDisabled]           = green_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_black;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.98f);

    // Borders - phosphor glow effect
    colors[ImGuiCol_Border]                 = green_border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.20f, 0.00f, 0.30f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_dark;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(green_glow.x, green_glow.y, green_glow.z, 0.25f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_black;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(green_border.x, green_border.y, green_border.z, 0.80f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_black.x, bg_black.y, bg_black.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_black;

    // Scrollbar - thin terminal style
    colors[ImGuiCol_ScrollbarBg]            = bg_black;
    colors[ImGuiCol_ScrollbarGrab]          = green_dim;
    colors[ImGuiCol_ScrollbarGrabHovered]   = green_primary;
    colors[ImGuiCol_ScrollbarGrabActive]    = green_glow;

    // Check mark
    colors[ImGuiCol_CheckMark]              = green_glow;

    // Slider
    colors[ImGuiCol_SliderGrab]             = green_primary;
    colors[ImGuiCol_SliderGrabActive]       = green_glow;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.60f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(green_primary.x, green_primary.y, green_primary.z, 0.70f);
    colors[ImGuiCol_ButtonActive]           = green_glow;

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.50f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(green_primary.x, green_primary.y, green_primary.z, 0.60f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(green_glow.x, green_glow.y, green_glow.z, 0.70f);

    // Separator
    colors[ImGuiCol_Separator]              = green_border;
    colors[ImGuiCol_SeparatorHovered]       = green_primary;
    colors[ImGuiCol_SeparatorActive]        = green_glow;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.30f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(green_primary.x, green_primary.y, green_primary.z, 0.60f);
    colors[ImGuiCol_ResizeGripActive]       = green_glow;

    // Tabs
    colors[ImGuiCol_Tab]                    = ImVec4(green_border.x, green_border.y, green_border.z, 0.70f);
    colors[ImGuiCol_TabHovered]             = ImVec4(green_primary.x, green_primary.y, green_primary.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(green_glow.x, green_glow.y, green_glow.z, 0.80f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(green_border.x, green_border.y, green_border.z, 0.50f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.70f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(green_glow.x, green_glow.y, green_glow.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_black;

    // Plot
    colors[ImGuiCol_PlotLines]              = green_glow;
    colors[ImGuiCol_PlotLinesHovered]       = warning;
    colors[ImGuiCol_PlotHistogram]          = green_primary;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(green_border.x, green_border.y, green_border.z, 0.50f);
    colors[ImGuiCol_TableBorderStrong]      = green_border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(green_border.x, green_border.y, green_border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(0.00f, 0.10f, 0.00f, 0.10f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(green_glow.x, green_glow.y, green_glow.z, 0.35f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = green_glow;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = green_glow;
    colors[ImGuiCol_NavWindowingHighlight]  = green_primary;
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.10f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.05f, 0.00f, 0.70f);

    // Terminal style - no rounding
    config_.window_rounding = 0.0f;
    config_.frame_rounding = 0.0f;
    config_.popup_rounding = 0.0f;
    config_.scrollbar_rounding = 0.0f;
    config_.grab_rounding = 0.0f;
    config_.tab_rounding = 0.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 1.0f;
    config_.window_padding = ImVec2(8.0f, 8.0f);
    config_.frame_padding = ImVec2(6.0f, 4.0f);
    config_.item_spacing = ImVec2(8.0f, 4.0f);
}

// ============================================================================
// CyxOS TUI Matrix Theme - The Matrix Digital Rain
// Neo's terminal, digital rain aesthetic
// ============================================================================
void Theme::ApplyCyxOSTuiMatrix() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Matrix digital rain palette
    ImVec4 bg_void       = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);  // #000000
    ImVec4 bg_subtle     = ImVec4(0.00f, 0.07f, 0.00f, 1.00f);  // #001100
    ImVec4 green_matrix  = ImVec4(0.00f, 1.00f, 0.25f, 1.00f);  // #00FF41
    ImVec4 green_dim     = ImVec4(0.00f, 0.23f, 0.00f, 1.00f);  // #003B00
    ImVec4 green_bright  = ImVec4(0.50f, 1.00f, 0.00f, 1.00f);  // #7FFF00
    ImVec4 green_border  = ImVec4(0.00f, 0.20f, 0.00f, 1.00f);  // #003300
    ImVec4 warning       = ImVec4(0.68f, 1.00f, 0.18f, 1.00f);  // #ADFF2F
    ImVec4 error         = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);  // #FF0000

    // Text
    colors[ImGuiCol_Text]                   = green_matrix;
    colors[ImGuiCol_TextDisabled]           = green_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_void;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_subtle.x, bg_subtle.y, bg_subtle.z, 0.98f);

    // Borders - matrix glow
    colors[ImGuiCol_Border]                 = green_border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.30f, 0.00f, 0.40f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_subtle;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.00f, 0.12f, 0.00f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(green_matrix.x, green_matrix.y, green_matrix.z, 0.25f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_void;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(green_border.x, green_border.y, green_border.z, 0.90f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_void.x, bg_void.y, bg_void.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_void;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_void;
    colors[ImGuiCol_ScrollbarGrab]          = green_dim;
    colors[ImGuiCol_ScrollbarGrabHovered]   = green_matrix;
    colors[ImGuiCol_ScrollbarGrabActive]    = green_bright;

    // Check mark
    colors[ImGuiCol_CheckMark]              = green_bright;

    // Slider
    colors[ImGuiCol_SliderGrab]             = green_matrix;
    colors[ImGuiCol_SliderGrabActive]       = green_bright;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.70f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(green_matrix.x, green_matrix.y, green_matrix.z, 0.70f);
    colors[ImGuiCol_ButtonActive]           = green_bright;

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.55f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(green_matrix.x, green_matrix.y, green_matrix.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(green_bright.x, green_bright.y, green_bright.z, 0.70f);

    // Separator
    colors[ImGuiCol_Separator]              = green_border;
    colors[ImGuiCol_SeparatorHovered]       = green_matrix;
    colors[ImGuiCol_SeparatorActive]        = green_bright;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.35f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(green_matrix.x, green_matrix.y, green_matrix.z, 0.60f);
    colors[ImGuiCol_ResizeGripActive]       = green_bright;

    // Tabs
    colors[ImGuiCol_Tab]                    = ImVec4(green_border.x, green_border.y, green_border.z, 0.75f);
    colors[ImGuiCol_TabHovered]             = ImVec4(green_matrix.x, green_matrix.y, green_matrix.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(green_bright.x, green_bright.y, green_bright.z, 0.80f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(green_border.x, green_border.y, green_border.z, 0.50f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(green_dim.x, green_dim.y, green_dim.z, 0.75f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(green_bright.x, green_bright.y, green_bright.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_void;

    // Plot
    colors[ImGuiCol_PlotLines]              = green_bright;
    colors[ImGuiCol_PlotLinesHovered]       = warning;
    colors[ImGuiCol_PlotHistogram]          = green_matrix;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(green_border.x, green_border.y, green_border.z, 0.55f);
    colors[ImGuiCol_TableBorderStrong]      = green_border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(green_border.x, green_border.y, green_border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(0.00f, 0.08f, 0.00f, 0.15f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(green_bright.x, green_bright.y, green_bright.z, 0.40f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = green_bright;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = green_bright;
    colors[ImGuiCol_NavWindowingHighlight]  = green_matrix;
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.00f, 0.15f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.08f, 0.00f, 0.75f);

    // Matrix style - sharp edges
    config_.window_rounding = 0.0f;
    config_.frame_rounding = 0.0f;
    config_.popup_rounding = 0.0f;
    config_.scrollbar_rounding = 0.0f;
    config_.grab_rounding = 0.0f;
    config_.tab_rounding = 0.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 1.0f;
    config_.window_padding = ImVec2(8.0f, 8.0f);
    config_.frame_padding = ImVec2(6.0f, 4.0f);
    config_.item_spacing = ImVec2(8.0f, 4.0f);
}

// ============================================================================
// CyxOS TUI Amber Theme - Amber CRT P3 Phosphor
// Warm retro terminal with amber glow
// ============================================================================
void Theme::ApplyCyxOSTuiAmber() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Amber CRT phosphor palette
    ImVec4 bg_dark       = ImVec4(0.05f, 0.04f, 0.02f, 1.00f);  // #0D0906
    ImVec4 bg_surface    = ImVec4(0.10f, 0.07f, 0.04f, 1.00f);  // #1A110A
    ImVec4 amber_primary = ImVec4(1.00f, 0.69f, 0.00f, 1.00f);  // #FFB000
    ImVec4 amber_dim     = ImVec4(0.55f, 0.37f, 0.00f, 1.00f);  // #8B5E00
    ImVec4 amber_bright  = ImVec4(1.00f, 0.75f, 0.00f, 1.00f);  // #FFC000
    ImVec4 amber_border  = ImVec4(0.30f, 0.22f, 0.00f, 1.00f);  // #4D3800
    ImVec4 warning       = ImVec4(1.00f, 0.88f, 0.00f, 1.00f);  // #FFE000
    ImVec4 error         = ImVec4(1.00f, 0.25f, 0.00f, 1.00f);  // #FF4000

    // Text
    colors[ImGuiCol_Text]                   = amber_primary;
    colors[ImGuiCol_TextDisabled]           = amber_dim;

    // Window
    colors[ImGuiCol_WindowBg]               = bg_dark;
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(bg_surface.x, bg_surface.y, bg_surface.z, 0.98f);

    // Borders - amber glow effect
    colors[ImGuiCol_Border]                 = amber_border;
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.30f, 0.15f, 0.00f, 0.35f);

    // Frame
    colors[ImGuiCol_FrameBg]                = bg_surface;
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.15f, 0.10f, 0.05f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(amber_bright.x, amber_bright.y, amber_bright.z, 0.25f);

    // Title bar
    colors[ImGuiCol_TitleBg]                = bg_dark;
    colors[ImGuiCol_TitleBgActive]          = ImVec4(amber_border.x, amber_border.y, amber_border.z, 0.85f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(bg_dark.x, bg_dark.y, bg_dark.z, 0.75f);

    // Menu bar
    colors[ImGuiCol_MenuBarBg]              = bg_dark;

    // Scrollbar
    colors[ImGuiCol_ScrollbarBg]            = bg_dark;
    colors[ImGuiCol_ScrollbarGrab]          = amber_dim;
    colors[ImGuiCol_ScrollbarGrabHovered]   = amber_primary;
    colors[ImGuiCol_ScrollbarGrabActive]    = amber_bright;

    // Check mark
    colors[ImGuiCol_CheckMark]              = amber_bright;

    // Slider
    colors[ImGuiCol_SliderGrab]             = amber_primary;
    colors[ImGuiCol_SliderGrabActive]       = amber_bright;

    // Button
    colors[ImGuiCol_Button]                 = ImVec4(amber_dim.x, amber_dim.y, amber_dim.z, 0.65f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(amber_primary.x, amber_primary.y, amber_primary.z, 0.70f);
    colors[ImGuiCol_ButtonActive]           = amber_bright;

    // Header
    colors[ImGuiCol_Header]                 = ImVec4(amber_dim.x, amber_dim.y, amber_dim.z, 0.50f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(amber_primary.x, amber_primary.y, amber_primary.z, 0.55f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(amber_bright.x, amber_bright.y, amber_bright.z, 0.70f);

    // Separator
    colors[ImGuiCol_Separator]              = amber_border;
    colors[ImGuiCol_SeparatorHovered]       = amber_primary;
    colors[ImGuiCol_SeparatorActive]        = amber_bright;

    // Resize grip
    colors[ImGuiCol_ResizeGrip]             = ImVec4(amber_dim.x, amber_dim.y, amber_dim.z, 0.30f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(amber_primary.x, amber_primary.y, amber_primary.z, 0.60f);
    colors[ImGuiCol_ResizeGripActive]       = amber_bright;

    // Tabs
    colors[ImGuiCol_Tab]                    = ImVec4(amber_border.x, amber_border.y, amber_border.z, 0.70f);
    colors[ImGuiCol_TabHovered]             = ImVec4(amber_primary.x, amber_primary.y, amber_primary.z, 0.70f);
    colors[ImGuiCol_TabActive]              = ImVec4(amber_bright.x, amber_bright.y, amber_bright.z, 0.80f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(amber_border.x, amber_border.y, amber_border.z, 0.50f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(amber_dim.x, amber_dim.y, amber_dim.z, 0.70f);

    // Docking
    colors[ImGuiCol_DockingPreview]         = ImVec4(amber_bright.x, amber_bright.y, amber_bright.z, 0.70f);
    colors[ImGuiCol_DockingEmptyBg]         = bg_dark;

    // Plot
    colors[ImGuiCol_PlotLines]              = amber_bright;
    colors[ImGuiCol_PlotLinesHovered]       = warning;
    colors[ImGuiCol_PlotHistogram]          = amber_primary;
    colors[ImGuiCol_PlotHistogramHovered]   = warning;

    // Table
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(amber_border.x, amber_border.y, amber_border.z, 0.50f);
    colors[ImGuiCol_TableBorderStrong]      = amber_border;
    colors[ImGuiCol_TableBorderLight]       = ImVec4(amber_border.x, amber_border.y, amber_border.z, 0.50f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(0.10f, 0.06f, 0.00f, 0.12f);

    // Text selection
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(amber_bright.x, amber_bright.y, amber_bright.z, 0.35f);

    // Drag drop
    colors[ImGuiCol_DragDropTarget]         = amber_bright;

    // Navigation
    colors[ImGuiCol_NavHighlight]           = amber_bright;
    colors[ImGuiCol_NavWindowingHighlight]  = amber_primary;
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.15f, 0.08f, 0.00f, 0.50f);

    // Modal
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.08f, 0.04f, 0.00f, 0.70f);

    // CRT terminal style - no rounding
    config_.window_rounding = 0.0f;
    config_.frame_rounding = 0.0f;
    config_.popup_rounding = 0.0f;
    config_.scrollbar_rounding = 0.0f;
    config_.grab_rounding = 0.0f;
    config_.tab_rounding = 0.0f;
    config_.window_border_size = 1.0f;
    config_.frame_border_size = 1.0f;
    config_.window_padding = ImVec2(8.0f, 8.0f);
    config_.frame_padding = ImVec2(6.0f, 4.0f);
    config_.item_spacing = ImVec2(8.0f, 4.0f);
}

// ============================================================================
// ImNodes Styling - Matches node editor to current theme
} // namespace gui
