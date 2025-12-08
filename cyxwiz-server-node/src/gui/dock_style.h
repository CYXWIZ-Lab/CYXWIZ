#pragma once

#include <imgui.h>
#include <imgui_internal.h>
#include <string>
#include <vector>
#include <functional>

namespace gui {

/**
 * Docking style presets inspired by professional applications
 */
enum class DockStylePreset {
    Default,        // Standard ImGui docking
    UnrealEngine,   // Unreal Engine 5 style - flat tabs, orange accents
    Unity,          // Unity Editor style
    VSCode,         // VS Code style - minimal tabs
    Blender         // Blender style - compact
};

/**
 * Sidebar position options
 */
enum class SidebarPosition {
    Left,
    Right,
    Hidden          // Completely hidden
};

/**
 * Configuration for dock tab appearance
 */
struct DockTabStyle {
    // Tab bar height
    float tab_bar_height = 24.0f;

    // Tab appearance
    float tab_rounding = 0.0f;          // 0 = flat, >0 = rounded corners
    float tab_min_width = 80.0f;
    float tab_max_width = 200.0f;
    float tab_padding_x = 8.0f;
    float tab_padding_y = 4.0f;

    // Active tab indicator (Unreal-style colored line at bottom)
    bool show_active_indicator = true;
    float active_indicator_height = 2.0f;
    ImVec4 active_indicator_color = ImVec4(1.0f, 0.6f, 0.0f, 1.0f);  // Orange

    // Close button
    bool show_close_button = true;
    float close_button_size = 12.0f;
    float close_button_padding = 4.0f;

    // Tab colors
    ImVec4 tab_bg = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
    ImVec4 tab_bg_hovered = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
    ImVec4 tab_bg_active = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
    ImVec4 tab_bg_unfocused = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);
    ImVec4 tab_text = ImVec4(0.70f, 0.70f, 0.70f, 1.0f);
    ImVec4 tab_text_active = ImVec4(0.95f, 0.95f, 0.95f, 1.0f);

    // Separator between tabs
    bool show_tab_separator = true;
    float tab_separator_width = 1.0f;
    ImVec4 tab_separator_color = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);

    // Dock area styling
    ImVec4 dock_bg = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);
    ImVec4 dock_border = ImVec4(0.06f, 0.06f, 0.06f, 1.0f);
    float dock_border_size = 1.0f;
    float dock_splitter_size = 4.0f;

    // Overflow menu button
    bool show_overflow_button = true;
    ImVec4 overflow_button_color = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
};

/**
 * Panel visibility state for sidebar toggles
 */
struct PanelVisibility {
    std::string name;
    std::string icon;           // Icon character or empty
    bool* visible_ptr;          // Pointer to panel's visibility flag
    std::function<void()> on_toggle;  // Optional callback
};

/**
 * DockStyle - Manages Unreal Engine-style docking appearance
 *
 * Features:
 * - Flat tab bars with active indicator line
 * - Custom close buttons on tabs
 * - Sidebar panel toggles (like Unreal's panel icons)
 * - Custom dock node window menu
 */
class DockStyle {
public:
    DockStyle();
    ~DockStyle() = default;

    // Apply a dock style preset
    void ApplyPreset(DockStylePreset preset);

    // Get/Set current style
    DockTabStyle& GetStyle() { return style_; }
    const DockTabStyle& GetStyle() const { return style_; }
    void SetStyle(const DockTabStyle& style);

    // Apply current style to ImGui
    void ApplyToImGui();

    // Register panels for sidebar toggles
    void RegisterPanel(const std::string& name, const std::string& icon, bool* visible_ptr,
                       std::function<void()> on_toggle = nullptr);
    void UnregisterPanel(const std::string& name);
    void ClearPanels();

    // Get registered panels
    const std::vector<PanelVisibility>& GetPanels() const { return panels_; }

    // Sidebar configuration
    void SetSidebarPosition(SidebarPosition position) { sidebar_position_ = position; }
    SidebarPosition GetSidebarPosition() const { return sidebar_position_; }
    void SetSidebarAutoHide(bool auto_hide) { sidebar_auto_hide_ = auto_hide; }
    bool GetSidebarAutoHide() const { return sidebar_auto_hide_; }

    // Render sidebar panel toggles (call this in your main render loop)
    // Returns true if any panel visibility changed
    bool RenderSidebarToggles();

    // Custom dock node rendering hooks
    // Call these to customize dock node appearance
    void BeginDockNodeOverride();
    void EndDockNodeOverride();

    // Render custom tab bar for a dock node
    // This replaces the default ImGui dock tab bar rendering
    void RenderCustomTabBar(ImGuiDockNode* node);

    // Static: Install custom dock node handler
    // Call once during initialization
    static void InstallCustomHandler();

    // Static: Custom window menu handler for dock nodes
    static void CustomDockNodeWindowMenuHandler(ImGuiContext* ctx, ImGuiDockNode* node, ImGuiTabBar* tab_bar);

private:
    void ApplyUnrealEnginePreset();
    void ApplyUnityPreset();
    void ApplyVSCodePreset();
    void ApplyBlenderPreset();

    // Helper to draw a close button
    bool DrawCloseButton(ImDrawList* draw_list, ImVec2 pos, float size, bool hovered);

    // Helper to draw active tab indicator
    void DrawActiveIndicator(ImDrawList* draw_list, ImVec2 tab_min, ImVec2 tab_max);

    DockStylePreset current_preset_ = DockStylePreset::UnrealEngine;
    DockTabStyle style_;
    std::vector<PanelVisibility> panels_;

    // Sidebar state
    SidebarPosition sidebar_position_ = SidebarPosition::Right;
    bool sidebar_auto_hide_ = true;  // Auto-hide by default
    float sidebar_hover_timer_ = 0.0f;
    float sidebar_visibility_ = 0.0f;  // 0.0 = hidden, 1.0 = fully visible
    bool sidebar_hovered_ = false;

    // Style backup for push/pop
    ImGuiStyle style_backup_;
    bool style_pushed_ = false;
};

// Global dock style instance
DockStyle& GetDockStyle();

} // namespace gui
