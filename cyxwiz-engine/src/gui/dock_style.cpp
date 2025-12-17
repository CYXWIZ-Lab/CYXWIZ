#include "dock_style.h"
#include <algorithm>
#include <cctype>
#include <spdlog/spdlog.h>

namespace gui {

// Global dock style instance
static DockStyle g_dock_style;

DockStyle& GetDockStyle() {
    return g_dock_style;
}

DockStyle::DockStyle() {
    // Apply Unreal Engine preset by default
    ApplyUnrealEnginePreset();
}

void DockStyle::ApplyPreset(DockStylePreset preset) {
    current_preset_ = preset;

    switch (preset) {
        case DockStylePreset::UnrealEngine:
            ApplyUnrealEnginePreset();
            break;
        case DockStylePreset::Unity:
            ApplyUnityPreset();
            break;
        case DockStylePreset::VSCode:
            ApplyVSCodePreset();
            break;
        case DockStylePreset::Blender:
            ApplyBlenderPreset();
            break;
        case DockStylePreset::Default:
        default:
            // Reset to default ImGui style
            style_ = DockTabStyle{};
            break;
    }

    ApplyToImGui();
}

void DockStyle::SetStyle(const DockTabStyle& style) {
    style_ = style;
    ApplyToImGui();
}

void DockStyle::ApplyToImGui() {
    ImGuiStyle& imgui_style = ImGui::GetStyle();

    // Apply tab styling
    imgui_style.TabRounding = style_.tab_rounding;
    imgui_style.TabBorderSize = style_.show_tab_separator ? style_.tab_separator_width : 0.0f;

    // Tab close button visibility (ImGui 1.91.9+ API)
    if (style_.show_close_button) {
        imgui_style.TabCloseButtonMinWidthSelected = 0.0f;      // Show on hover when selected
        imgui_style.TabCloseButtonMinWidthUnselected = style_.tab_min_width;  // Show on hover if wide enough
    } else {
        imgui_style.TabCloseButtonMinWidthSelected = FLT_MAX;   // Never show
        imgui_style.TabCloseButtonMinWidthUnselected = FLT_MAX; // Never show
    }

    // Tab colors
    imgui_style.Colors[ImGuiCol_Tab] = style_.tab_bg;
    imgui_style.Colors[ImGuiCol_TabHovered] = style_.tab_bg_hovered;
    imgui_style.Colors[ImGuiCol_TabActive] = style_.tab_bg_active;
    imgui_style.Colors[ImGuiCol_TabUnfocused] = style_.tab_bg_unfocused;
    imgui_style.Colors[ImGuiCol_TabUnfocusedActive] = style_.tab_bg_active;

    // Docking colors
    imgui_style.Colors[ImGuiCol_DockingEmptyBg] = style_.dock_bg;

    // Tab separator (uses border color)
    if (style_.show_tab_separator) {
        imgui_style.Colors[ImGuiCol_Border] = style_.tab_separator_color;
    }

    spdlog::debug("Applied dock style to ImGui");
}

void DockStyle::ApplyUnrealEnginePreset() {
    // Unreal Engine 5 style
    // - Very flat, minimal tabs
    // - Orange accent for active tab indicator
    // - Dark gray colors

    style_.tab_bar_height = 26.0f;
    style_.tab_rounding = 0.0f;  // Completely flat
    style_.tab_min_width = 100.0f;
    style_.tab_max_width = 250.0f;
    style_.tab_padding_x = 12.0f;
    style_.tab_padding_y = 5.0f;

    // Active indicator (the orange line at the top of active tab)
    style_.show_active_indicator = true;
    style_.active_indicator_height = 2.0f;
    style_.active_indicator_color = ImVec4(1.0f, 0.55f, 0.0f, 1.0f);  // Unreal orange

    // Close button
    style_.show_close_button = true;
    style_.close_button_size = 14.0f;
    style_.close_button_padding = 4.0f;

    // Tab colors - dark grays like Unreal
    style_.tab_bg = ImVec4(0.14f, 0.14f, 0.14f, 1.0f);             // Inactive tab
    style_.tab_bg_hovered = ImVec4(0.20f, 0.20f, 0.20f, 1.0f);     // Hovered
    style_.tab_bg_active = ImVec4(0.24f, 0.24f, 0.24f, 1.0f);      // Active tab
    style_.tab_bg_unfocused = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);   // Unfocused window
    style_.tab_text = ImVec4(0.60f, 0.60f, 0.60f, 1.0f);           // Inactive text
    style_.tab_text_active = ImVec4(0.95f, 0.95f, 0.95f, 1.0f);    // Active text

    // Tab separator
    style_.show_tab_separator = false;  // Unreal doesn't show separators
    style_.tab_separator_width = 1.0f;
    style_.tab_separator_color = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);

    // Dock area - Minimal borders for clean look
    style_.dock_bg = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);
    style_.dock_border = ImVec4(0.12f, 0.12f, 0.12f, 0.0f);  // Transparent border
    style_.dock_border_size = 0.0f;  // No dock borders
    style_.dock_splitter_size = 2.0f;  // Thinner splitter

    // Overflow
    style_.show_overflow_button = true;
    style_.overflow_button_color = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
}

void DockStyle::ApplyUnityPreset() {
    // Unity Editor style
    // - Slightly rounded tabs
    // - Blue accent

    style_.tab_bar_height = 22.0f;
    style_.tab_rounding = 4.0f;
    style_.tab_min_width = 80.0f;
    style_.tab_max_width = 200.0f;
    style_.tab_padding_x = 10.0f;
    style_.tab_padding_y = 4.0f;

    style_.show_active_indicator = true;
    style_.active_indicator_height = 2.0f;
    style_.active_indicator_color = ImVec4(0.22f, 0.55f, 0.92f, 1.0f);  // Unity blue

    style_.show_close_button = true;
    style_.close_button_size = 12.0f;
    style_.close_button_padding = 4.0f;

    style_.tab_bg = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
    style_.tab_bg_hovered = ImVec4(0.28f, 0.28f, 0.28f, 1.0f);
    style_.tab_bg_active = ImVec4(0.32f, 0.32f, 0.32f, 1.0f);
    style_.tab_bg_unfocused = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
    style_.tab_text = ImVec4(0.65f, 0.65f, 0.65f, 1.0f);
    style_.tab_text_active = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

    style_.show_tab_separator = true;
    style_.tab_separator_width = 1.0f;
    style_.tab_separator_color = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);

    style_.dock_bg = ImVec4(0.16f, 0.16f, 0.16f, 1.0f);
    style_.dock_border = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
    style_.dock_border_size = 1.0f;
    style_.dock_splitter_size = 3.0f;

    style_.show_overflow_button = true;
    style_.overflow_button_color = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
}

void DockStyle::ApplyVSCodePreset() {
    // VS Code style
    // - Sharp edges, no rounding
    // - Activity bar accent

    style_.tab_bar_height = 35.0f;
    style_.tab_rounding = 0.0f;
    style_.tab_min_width = 120.0f;
    style_.tab_max_width = 300.0f;
    style_.tab_padding_x = 16.0f;
    style_.tab_padding_y = 8.0f;

    style_.show_active_indicator = true;
    style_.active_indicator_height = 1.0f;
    style_.active_indicator_color = ImVec4(0.0f, 0.48f, 0.80f, 1.0f);  // VS Code blue

    style_.show_close_button = true;
    style_.close_button_size = 14.0f;
    style_.close_button_padding = 6.0f;

    style_.tab_bg = ImVec4(0.15f, 0.15f, 0.15f, 1.0f);
    style_.tab_bg_hovered = ImVec4(0.20f, 0.20f, 0.20f, 1.0f);
    style_.tab_bg_active = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);  // Active is darker in VS Code
    style_.tab_bg_unfocused = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
    style_.tab_text = ImVec4(0.55f, 0.55f, 0.55f, 1.0f);
    style_.tab_text_active = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

    style_.show_tab_separator = true;
    style_.tab_separator_width = 1.0f;
    style_.tab_separator_color = ImVec4(0.10f, 0.10f, 0.10f, 1.0f);

    style_.dock_bg = ImVec4(0.12f, 0.12f, 0.12f, 1.0f);
    style_.dock_border = ImVec4(0.08f, 0.08f, 0.08f, 1.0f);
    style_.dock_border_size = 0.0f;
    style_.dock_splitter_size = 4.0f;

    style_.show_overflow_button = true;
    style_.overflow_button_color = ImVec4(0.50f, 0.50f, 0.50f, 1.0f);
}

void DockStyle::ApplyBlenderPreset() {
    // Blender style
    // - Very compact
    // - Rounded tabs

    style_.tab_bar_height = 20.0f;
    style_.tab_rounding = 4.0f;
    style_.tab_min_width = 60.0f;
    style_.tab_max_width = 150.0f;
    style_.tab_padding_x = 6.0f;
    style_.tab_padding_y = 2.0f;

    style_.show_active_indicator = false;  // Blender uses background color change
    style_.active_indicator_height = 0.0f;
    style_.active_indicator_color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    style_.show_close_button = true;
    style_.close_button_size = 10.0f;
    style_.close_button_padding = 2.0f;

    style_.tab_bg = ImVec4(0.27f, 0.27f, 0.27f, 1.0f);
    style_.tab_bg_hovered = ImVec4(0.35f, 0.35f, 0.35f, 1.0f);
    style_.tab_bg_active = ImVec4(0.40f, 0.40f, 0.40f, 1.0f);
    style_.tab_bg_unfocused = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
    style_.tab_text = ImVec4(0.75f, 0.75f, 0.75f, 1.0f);
    style_.tab_text_active = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);

    style_.show_tab_separator = false;
    style_.tab_separator_width = 0.0f;
    style_.tab_separator_color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    style_.dock_bg = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
    style_.dock_border = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
    style_.dock_border_size = 1.0f;
    style_.dock_splitter_size = 2.0f;

    style_.show_overflow_button = true;
    style_.overflow_button_color = ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
}

void DockStyle::RegisterPanel(const std::string& name, const std::string& icon,
                              bool* visible_ptr, std::function<void()> on_toggle) {
    // Check if already registered
    auto it = std::find_if(panels_.begin(), panels_.end(),
                           [&name](const PanelVisibility& p) { return p.name == name; });

    if (it != panels_.end()) {
        // Update existing
        it->icon = icon;
        it->visible_ptr = visible_ptr;
        it->on_toggle = on_toggle;
    } else {
        // Add new
        panels_.push_back({name, icon, visible_ptr, on_toggle});
    }
}

void DockStyle::UnregisterPanel(const std::string& name) {
    panels_.erase(
        std::remove_if(panels_.begin(), panels_.end(),
                       [&name](const PanelVisibility& p) { return p.name == name; }),
        panels_.end());
}

void DockStyle::ClearPanels() {
    panels_.clear();
}

bool DockStyle::RenderSidebarToggles() {
    bool any_changed = false;

    if (panels_.empty() || sidebar_position_ == SidebarPosition::Hidden) {
        return false;
    }

    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGuiIO& io = ImGui::GetIO();

    // Sidebar dimensions
    const float sidebar_width = 40.0f;
    const float icon_size = 24.0f;
    const float icon_padding = 8.0f;
    const float top_offset = 32.0f;  // Space for toolbar
    const float hover_zone = 50.0f;  // Edge zone to trigger hover

    bool left_side = (sidebar_position_ == SidebarPosition::Left);

    // Calculate hover detection zone
    ImVec2 hover_zone_min, hover_zone_max;
    if (left_side) {
        hover_zone_min = ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + top_offset);
        hover_zone_max = ImVec2(viewport->WorkPos.x + hover_zone + sidebar_width,
                                viewport->WorkPos.y + viewport->WorkSize.y);
    } else {
        hover_zone_min = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - sidebar_width - hover_zone,
                                viewport->WorkPos.y + top_offset);
        hover_zone_max = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x,
                                viewport->WorkPos.y + viewport->WorkSize.y);
    }

    // Check if mouse is in hover zone
    ImVec2 mouse_pos = io.MousePos;
    bool in_hover_zone = (mouse_pos.x >= hover_zone_min.x && mouse_pos.x <= hover_zone_max.x &&
                          mouse_pos.y >= hover_zone_min.y && mouse_pos.y <= hover_zone_max.y);

    // Update hover state with hysteresis
    if (in_hover_zone) {
        sidebar_hovered_ = true;
        sidebar_hover_timer_ = 0.3f;
    } else if (sidebar_hover_timer_ > 0.0f) {
        sidebar_hover_timer_ -= io.DeltaTime;
        if (sidebar_hover_timer_ <= 0.0f) {
            sidebar_hovered_ = false;
        }
    }

    // Animate visibility
    float target_visibility = (sidebar_auto_hide_ && !sidebar_hovered_) ? 0.0f : 1.0f;
    float speed = 8.0f;

    if (sidebar_visibility_ < target_visibility) {
        sidebar_visibility_ = std::min(sidebar_visibility_ + speed * io.DeltaTime, target_visibility);
    } else if (sidebar_visibility_ > target_visibility) {
        sidebar_visibility_ = std::max(sidebar_visibility_ - speed * io.DeltaTime, target_visibility);
    }

    // Don't render if fully hidden
    if (sidebar_visibility_ < 0.01f) {
        return false;
    }

    // Calculate animated sidebar position
    float slide_offset = (1.0f - sidebar_visibility_) * sidebar_width;
    ImVec2 sidebar_pos;
    if (left_side) {
        sidebar_pos = ImVec2(viewport->WorkPos.x - slide_offset, viewport->WorkPos.y + top_offset);
    } else {
        sidebar_pos = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - sidebar_width + slide_offset,
                             viewport->WorkPos.y + top_offset);
    }

    float sidebar_height = viewport->WorkSize.y - top_offset;
    float alpha = sidebar_visibility_;

    // Create actual window for sidebar (not just drawing on foreground)
    ImGui::SetNextWindowPos(sidebar_pos);
    ImGui::SetNextWindowSize(ImVec2(sidebar_width, sidebar_height));
    ImGui::SetNextWindowBgAlpha(0.95f * alpha);

    ImGuiWindowFlags sidebar_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);  // No sidebar border
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.10f, 0.10f, 0.10f, 0.95f * alpha));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.15f, 0.15f, 0.15f, 0.3f * alpha));  // Very subtle

    if (ImGui::Begin("##SidebarPanel", nullptr, sidebar_flags)) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        // Don't process interactions if not fully visible
        if (sidebar_visibility_ >= 0.5f) {
            float y_offset = icon_padding;

            for (auto& panel : panels_) {
                bool is_visible = panel.visible_ptr ? *panel.visible_ptr : false;

                ImVec2 icon_pos = ImVec2(
                    sidebar_pos.x + (sidebar_width - icon_size) * 0.5f,
                    sidebar_pos.y + y_offset
                );

                ImVec2 icon_min = icon_pos;
                ImVec2 icon_max = ImVec2(icon_pos.x + icon_size, icon_pos.y + icon_size);

                // Use cursor position for button
                ImGui::SetCursorScreenPos(icon_min);
                ImGui::PushID(panel.name.c_str());

                bool hovered = false;
                bool clicked = false;

                // Create invisible button for interaction
                if (ImGui::InvisibleButton("##toggle", ImVec2(icon_size, icon_size))) {
                    clicked = true;
                }
                hovered = ImGui::IsItemHovered();

                ImGui::PopID();

                // Determine colors
                ImU32 icon_bg_color;
                ImU32 icon_text_color;
                ImU32 indicator_color = ImGui::ColorConvertFloat4ToU32(
                    ImVec4(style_.active_indicator_color.x, style_.active_indicator_color.y,
                           style_.active_indicator_color.z, style_.active_indicator_color.w * alpha));

                if (is_visible) {
                    if (hovered) {
                        icon_bg_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.22f, 0.22f, 0.22f, alpha));
                    } else {
                        icon_bg_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.18f, 0.18f, 0.18f, alpha));
                    }
                    icon_text_color = IM_COL32(255, 255, 255, (int)(255 * alpha));
                } else {
                    if (hovered) {
                        icon_bg_color = ImGui::ColorConvertFloat4ToU32(ImVec4(0.16f, 0.16f, 0.16f, alpha));
                    } else {
                        icon_bg_color = IM_COL32(0, 0, 0, 0);
                    }
                    icon_text_color = IM_COL32(128, 128, 128, (int)(255 * alpha));
                }

                // Draw icon background
                if (is_visible || hovered) {
                    draw_list->AddRectFilled(icon_min, icon_max, icon_bg_color, 4.0f);
                }

                // Draw active indicator bar on edge
                if (is_visible && left_side) {
                    draw_list->AddRectFilled(
                        ImVec2(sidebar_pos.x, icon_min.y),
                        ImVec2(sidebar_pos.x + 3.0f, icon_max.y),
                        indicator_color);
                } else if (is_visible && !left_side) {
                    draw_list->AddRectFilled(
                        ImVec2(sidebar_pos.x + sidebar_width - 3.0f, icon_min.y),
                        ImVec2(sidebar_pos.x + sidebar_width, icon_max.y),
                        indicator_color);
                }

                // Draw icon text
                std::string label = panel.icon.empty() ?
                                    std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(panel.name[0])))) :
                                    panel.icon;

                ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
                ImVec2 text_pos = ImVec2(
                    icon_min.x + (icon_size - text_size.x) * 0.5f,
                    icon_min.y + (icon_size - text_size.y) * 0.5f
                );

                // Use ImGui text rendering for proper merged font support
                ImGui::SetCursorScreenPos(text_pos);
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(icon_text_color));
                ImGui::TextUnformatted(label.c_str());
                ImGui::PopStyleColor();

                // Handle click
                if (clicked) {
                    if (panel.visible_ptr) {
                        *panel.visible_ptr = !*panel.visible_ptr;
                    }
                    if (panel.on_toggle) {
                        panel.on_toggle();
                    }
                    any_changed = true;
                }

                // Tooltip on hover
                if (hovered) {
                    ImGui::SetNextWindowBgAlpha(0.9f);
                    ImGui::BeginTooltip();
                    ImGui::Text("%s", panel.name.c_str());
                    ImGui::TextDisabled(is_visible ? "Click to hide" : "Click to show");
                    ImGui::EndTooltip();
                }

                y_offset += icon_size + icon_padding;
            }

            // Right-click context menu
            if (ImGui::IsWindowHovered() && io.MouseClicked[1]) {
                ImGui::OpenPopup("##SidebarContextMenu");
            }

            // Render context menu
            ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.12f, 0.12f, 0.98f));
            ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 4.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));

            if (ImGui::BeginPopup("##SidebarContextMenu")) {
                ImGui::TextDisabled("Sidebar Position");
                ImGui::Separator();

                if (ImGui::MenuItem("Left Side", nullptr, sidebar_position_ == SidebarPosition::Left)) {
                    sidebar_position_ = SidebarPosition::Left;
                }
                if (ImGui::MenuItem("Right Side", nullptr, sidebar_position_ == SidebarPosition::Right)) {
                    sidebar_position_ = SidebarPosition::Right;
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Hide Sidebar")) {
                    sidebar_position_ = SidebarPosition::Hidden;
                }

                ImGui::Separator();
                ImGui::TextDisabled("Behavior");
                if (ImGui::MenuItem("Auto-hide", nullptr, sidebar_auto_hide_)) {
                    sidebar_auto_hide_ = !sidebar_auto_hide_;
                }

                ImGui::EndPopup();
            }

            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);

    return any_changed;
}

void DockStyle::BeginDockNodeOverride() {
    if (style_pushed_) return;

    style_backup_ = ImGui::GetStyle();

    ImGuiStyle& style = ImGui::GetStyle();

    // Override tab styling for dock nodes
    style.TabRounding = style_.tab_rounding;
    style.Colors[ImGuiCol_Tab] = style_.tab_bg;
    style.Colors[ImGuiCol_TabHovered] = style_.tab_bg_hovered;
    style.Colors[ImGuiCol_TabActive] = style_.tab_bg_active;
    style.Colors[ImGuiCol_TabUnfocused] = style_.tab_bg_unfocused;
    style.Colors[ImGuiCol_TabUnfocusedActive] = style_.tab_bg_active;

    style_pushed_ = true;
}

void DockStyle::EndDockNodeOverride() {
    if (!style_pushed_) return;

    ImGui::GetStyle() = style_backup_;
    style_pushed_ = false;
}

bool DockStyle::DrawCloseButton(ImDrawList* draw_list, ImVec2 pos, float size, bool hovered) {
    // Calculate center
    ImVec2 center = ImVec2(pos.x + size * 0.5f, pos.y + size * 0.5f);
    float cross_size = size * 0.3f;

    // Colors
    ImU32 color = hovered ?
                  IM_COL32(255, 255, 255, 255) :
                  IM_COL32(180, 180, 180, 255);

    // Draw background on hover
    if (hovered) {
        draw_list->AddCircleFilled(center, size * 0.4f, IM_COL32(255, 80, 80, 200));
        color = IM_COL32(255, 255, 255, 255);
    }

    // Draw X
    draw_list->AddLine(
        ImVec2(center.x - cross_size, center.y - cross_size),
        ImVec2(center.x + cross_size, center.y + cross_size),
        color, 1.5f);
    draw_list->AddLine(
        ImVec2(center.x + cross_size, center.y - cross_size),
        ImVec2(center.x - cross_size, center.y + cross_size),
        color, 1.5f);

    return hovered;
}

void DockStyle::DrawActiveIndicator(ImDrawList* draw_list, ImVec2 tab_min, ImVec2 tab_max) {
    if (!style_.show_active_indicator) return;

    // Draw indicator at the TOP of the tab (Unreal style)
    ImU32 indicator_color = ImGui::ColorConvertFloat4ToU32(style_.active_indicator_color);

    draw_list->AddRectFilled(
        ImVec2(tab_min.x, tab_min.y),
        ImVec2(tab_max.x, tab_min.y + style_.active_indicator_height),
        indicator_color);
}

void DockStyle::RenderCustomTabBar(ImGuiDockNode* node) {
    if (!node || !node->TabBar) return;

    ImGuiTabBar* tab_bar = node->TabBar;
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Get tab bar bounds
    ImRect tab_bar_rect = tab_bar->BarRect;

    // Draw tab bar background
    draw_list->AddRectFilled(
        tab_bar_rect.Min,
        tab_bar_rect.Max,
        ImGui::ColorConvertFloat4ToU32(style_.dock_bg));

    // Draw active tab indicator
    if (tab_bar->VisibleTabId != 0) {
        ImGuiTabItem* active_tab = ImGui::TabBarFindTabByID(tab_bar, tab_bar->VisibleTabId);
        if (active_tab) {
            // Calculate tab position
            ImVec2 tab_min = ImVec2(tab_bar_rect.Min.x + active_tab->Offset, tab_bar_rect.Min.y);
            ImVec2 tab_max = ImVec2(tab_min.x + active_tab->Width, tab_bar_rect.Max.y);

            DrawActiveIndicator(draw_list, tab_min, tab_max);
        }
    }
}

void DockStyle::InstallCustomHandler() {
    ImGuiContext& g = *GImGui;
    g.DockNodeWindowMenuHandler = CustomDockNodeWindowMenuHandler;
    spdlog::info("Installed custom dock node window menu handler");
}

void DockStyle::CustomDockNodeWindowMenuHandler(ImGuiContext* ctx, ImGuiDockNode* node, ImGuiTabBar* tab_bar) {
    // Custom menu for dock node window button (the hamburger menu)
    // This appears when you click the small triangle/menu button on dock tabs

    (void)ctx;  // Unused parameter

    if (ImGui::BeginPopup("DockNodeWindowMenu")) {
        // Panel visibility toggles
        if (node->Windows.Size > 0) {
            ImGui::TextDisabled("Windows");
            ImGui::Separator();

            for (int i = 0; i < node->Windows.Size; i++) {
                ImGuiWindow* window = node->Windows[i];
                bool is_selected = (tab_bar->VisibleTabId == window->TabId);

                if (ImGui::MenuItem(window->Name, nullptr, is_selected)) {
                    // Focus this window/tab
                    tab_bar->NextSelectedTabId = window->TabId;
                }
            }

            ImGui::Separator();
        }

        // Standard options
        if (ImGui::MenuItem("Close All")) {
            for (int i = 0; i < node->Windows.Size; i++) {
                ImGuiWindow* window = node->Windows[i];
                if (window->HasCloseButton) {
                    // Request close (use DockTabWantClose for docked windows in ImGui 1.91.9+)
                    window->DockTabWantClose = true;
                }
            }
        }

        // Tab bar visibility toggle
        if (!(node->MergedFlags & ImGuiDockNodeFlags_NoTabBar)) {
            if (ImGui::MenuItem(node->IsHiddenTabBar() ? "Show Tab Bar" : "Hide Tab Bar")) {
                node->WantHiddenTabBarToggle = true;
            }
        }

        ImGui::EndPopup();
    }
}

} // namespace gui
