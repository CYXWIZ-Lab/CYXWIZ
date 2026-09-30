#include "appearance_settings.h"
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
                              bool* visible_ptr, std::function<void()> on_toggle,
                              const std::string& group, const std::string& shortcut) {
    // Check if already registered
    auto it = std::find_if(panels_.begin(), panels_.end(),
                           [&name](const PanelVisibility& p) { return p.name == name; });

    if (it != panels_.end()) {
        // Update existing
        it->icon = icon;
        it->visible_ptr = visible_ptr;
        it->on_toggle = on_toggle;
        it->group = group;
        it->shortcut = shortcut;
    } else {
        // Add new
        panels_.push_back({name, icon, visible_ptr, on_toggle, group, shortcut});
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
    const ImGuiStyle& style = ImGui::GetStyle();

    // Sidebar geometry: one 26 px square per panel, a thin line between
    // groups, the action entries (Command Palette) pinned at the bottom.
    const float sidebar_width = kSidebarWidth;
    const float icon_size = 26.0f;
    const float icon_gap = 3.0f;
    const float group_gap = 9.0f;
    const float edge_pad = 5.0f;
    const float top_offset = 32.0f;  // Space for the menu bar
    const float hover_zone = 50.0f;  // Edge zone that reveals the sidebar

    const bool left_side = (sidebar_position_ == SidebarPosition::Left);

    // Hover detection with hysteresis (auto-hide)
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
    const ImVec2 mouse_pos = io.MousePos;
    const bool in_hover_zone = (mouse_pos.x >= hover_zone_min.x && mouse_pos.x <= hover_zone_max.x &&
                                mouse_pos.y >= hover_zone_min.y && mouse_pos.y <= hover_zone_max.y);
    if (in_hover_zone) {
        sidebar_hovered_ = true;
        sidebar_hover_timer_ = 0.3f;
    } else if (sidebar_hover_timer_ > 0.0f) {
        sidebar_hover_timer_ -= io.DeltaTime;
        if (sidebar_hover_timer_ <= 0.0f) sidebar_hovered_ = false;
    }

    // Slide in and out
    const float target_visibility = (sidebar_auto_hide_ && !sidebar_hovered_) ? 0.0f : 1.0f;
    const float speed = 8.0f;
    if (sidebar_visibility_ < target_visibility) {
        sidebar_visibility_ = std::min(sidebar_visibility_ + speed * io.DeltaTime, target_visibility);
    } else if (sidebar_visibility_ > target_visibility) {
        sidebar_visibility_ = std::max(sidebar_visibility_ - speed * io.DeltaTime, target_visibility);
    }
    if (sidebar_visibility_ < 0.01f) return false;

    const float slide_offset = (1.0f - sidebar_visibility_) * sidebar_width;
    const ImVec2 sidebar_pos = left_side
        ? ImVec2(viewport->WorkPos.x - slide_offset, viewport->WorkPos.y + top_offset)
        : ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - sidebar_width + slide_offset,
                 viewport->WorkPos.y + top_offset);
    const float sidebar_height = viewport->WorkSize.y - top_offset;
    const float alpha = sidebar_visibility_;

    // Content height: panels with their group lines, then the pinned actions.
    float content_height = edge_pad;
    {
        std::string group;
        bool first = true;
        for (const auto& panel : panels_) {
            if (!panel.visible_ptr) continue;  // pinned action
            if (!first && panel.group != group) content_height += group_gap;
            group = panel.group;
            first = false;
            content_height += icon_size + icon_gap;
        }
        for (const auto& panel : panels_) {
            if (!panel.visible_ptr) content_height += icon_size + icon_gap;
        }
        content_height += edge_pad;
    }

    // Colours come from the active theme.
    auto themed = [&style, alpha](ImGuiCol col) {
        ImVec4 c = style.Colors[col];
        c.w *= alpha;
        return ImGui::ColorConvertFloat4ToU32(c);
    };
    ImVec4 window_bg = style.Colors[ImGuiCol_MenuBarBg];
    window_bg.w = 0.97f * alpha;

    ImGui::SetNextWindowPos(sidebar_pos);
    ImGui::SetNextWindowSize(ImVec2(sidebar_width, sidebar_height));
    ImGui::SetNextWindowContentSize(ImVec2(0.0f, content_height));
    ImGui::SetNextWindowBgAlpha(window_bg.w);

    const ImGuiWindowFlags sidebar_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                                           ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
                                           ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 4.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, window_bg);

    if (ImGui::Begin("##SidebarPanel", nullptr, sidebar_flags)) {
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        if (sidebar_visibility_ >= 0.5f) {
            const ImVec2 content_origin = ImGui::GetCursorScreenPos();
            const float x = content_origin.x + (sidebar_width - icon_size) * 0.5f;

            auto draw_toggle = [&](PanelVisibility& panel, float y) {
                const bool is_action = panel.visible_ptr == nullptr;
                const bool is_visible = panel.visible_ptr ? *panel.visible_ptr : false;
                const ImVec2 icon_min(x, y);
                const ImVec2 icon_max(x + icon_size, y + icon_size);

                ImGui::SetCursorScreenPos(icon_min);
                ImGui::PushID(panel.name.c_str());
                const bool clicked = ImGui::InvisibleButton("##toggle", ImVec2(icon_size, icon_size));
                const bool hovered = ImGui::IsItemHovered();
                ImGui::PopID();

                if (is_visible || hovered) {
                    draw_list->AddRectFilled(icon_min, icon_max,
                                             themed(hovered ? ImGuiCol_HeaderHovered : ImGuiCol_Header), 5.0f);
                }
                if (is_visible) {
                    // Accent bar on the outer edge marks a shown panel.
                    const float bar_x = left_side ? sidebar_pos.x : sidebar_pos.x + sidebar_width - 3.0f;
                    draw_list->AddRectFilled(ImVec2(bar_x, icon_min.y + 5.0f), ImVec2(bar_x + 3.0f, icon_max.y - 5.0f),
                                             themed(ImGuiCol_CheckMark), 1.5f);
                }

                const std::string label = panel.icon.empty()
                    ? std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(panel.name[0]))))
                    : panel.icon;
                const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
                ImGui::SetCursorScreenPos(ImVec2(icon_min.x + (icon_size - text_size.x) * 0.5f,
                                                 icon_min.y + (icon_size - text_size.y) * 0.5f));
                ImVec4 text_color = style.Colors[(is_visible || is_action) ? ImGuiCol_Text : ImGuiCol_TextDisabled];
                text_color.w *= alpha;
                ImGui::PushStyleColor(ImGuiCol_Text, text_color);
                ImGui::TextUnformatted(label.c_str());
                ImGui::PopStyleColor();

                if (clicked) {
                    if (panel.visible_ptr) *panel.visible_ptr = !*panel.visible_ptr;
                    if (panel.on_toggle) panel.on_toggle();
                    any_changed = true;
                }
                if (hovered) {
                    ImGui::BeginTooltip();
                    if (panel.shortcut.empty()) ImGui::TextUnformatted(panel.name.c_str());
                    else ImGui::Text("%s  (%s)", panel.name.c_str(), panel.shortcut.c_str());
                    if (!is_action) ImGui::TextDisabled(is_visible ? "Click to hide" : "Click to show");
                    ImGui::EndTooltip();
                }
            };

            float y = content_origin.y + edge_pad;
            std::string group;
            bool first = true;
            for (auto& panel : panels_) {
                if (!panel.visible_ptr) continue;
                if (!first && panel.group != group) {
                    const float line_y = y + (group_gap - 1.0f) * 0.5f;
                    draw_list->AddLine(ImVec2(x + 2.0f, line_y), ImVec2(x + icon_size - 2.0f, line_y),
                                       themed(ImGuiCol_Separator));
                    y += group_gap;
                }
                group = panel.group;
                first = false;
                draw_toggle(panel, y);
                y += icon_size + icon_gap;
            }

            // Pinned actions sit at the bottom when there is room.
            int actions = 0;
            for (const auto& panel : panels_) if (!panel.visible_ptr) ++actions;
            if (actions > 0) {
                const float window_bottom = ImGui::GetWindowPos().y + ImGui::GetWindowHeight();
                const float bottom_y = window_bottom - edge_pad - actions * (icon_size + icon_gap) + icon_gap;
                y = std::max(y, bottom_y);
                for (auto& panel : panels_) {
                    if (panel.visible_ptr) continue;
                    draw_toggle(panel, y);
                    y += icon_size + icon_gap;
                }
            }

            // Right-click: side and auto-hide
            if (ImGui::IsWindowHovered() && io.MouseClicked[1]) {
                ImGui::OpenPopup("##SidebarContextMenu");
            }
            if (ImGui::BeginPopup("##SidebarContextMenu")) {
                ImGui::TextDisabled("Sidebar side");
                ImGui::Separator();
                if (ImGui::MenuItem("Left", nullptr, sidebar_position_ == SidebarPosition::Left)) {
                    SetSidebarOnLeft(true);  // saved Engine-wide
                }
                if (ImGui::MenuItem("Right", nullptr, sidebar_position_ == SidebarPosition::Right)) {
                    SetSidebarOnLeft(false);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Auto-hide", nullptr, sidebar_auto_hide_)) {
                    sidebar_auto_hide_ = !sidebar_auto_hide_;
                }
                if (ImGui::MenuItem("Hide sidebar")) {
                    sidebar_position_ = SidebarPosition::Hidden;
                }
                ImGui::EndPopup();
            }
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(1);
    ImGui::PopStyleVar(4);

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
