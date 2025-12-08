#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../icons.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

void ToolbarPanel::RenderViewMenu() {
    if (ImGui::BeginMenu("View")) {
        // Increase padding for menu items
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // ========== Panels Section ==========
        if (ImGui::BeginMenu(ICON_FA_TABLE_COLUMNS " Panels")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));

            // Get registered panels from DockStyle (same as sidebar)
            auto& dock_style = gui::GetDockStyle();
            const auto& panels = dock_style.GetPanels();

            if (panels.empty()) {
                ImGui::TextDisabled("No panels registered");
            } else {
                for (const auto& panel : panels) {
                    bool is_visible = panel.visible_ptr ? *panel.visible_ptr : false;
                    std::string label = panel.icon + " " + panel.name;

                    if (ImGui::MenuItem(label.c_str(), nullptr, is_visible)) {
                        // Toggle visibility
                        if (panel.visible_ptr) {
                            *panel.visible_ptr = !*panel.visible_ptr;
                        }
                        // Call optional toggle callback
                        if (panel.on_toggle) {
                            panel.on_toggle();
                        }
                    }
                }
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Layout Section ==========
        if (ImGui::BeginMenu(ICON_FA_TABLE_COLUMNS " Layout")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));

            if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Layout")) {
                if (save_layout_callback_) {
                    save_layout_callback_();
                    spdlog::info("Layout saved to imgui.ini");
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Save window layout (applies to all projects)");
            }

            if (ImGui::MenuItem(ICON_FA_ARROWS_ROTATE " Reset to Default")) {
                if (reset_layout_callback_) {
                    reset_layout_callback_();
                }
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        // ========== Theme Section ==========
        if (ImGui::BeginMenu(ICON_FA_PALETTE " Theme")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));

            auto& theme = gui::GetTheme();
            auto current_preset = theme.GetCurrentPreset();

            for (auto preset : gui::Theme::GetAvailablePresets()) {
                bool is_selected = (current_preset == preset);
                if (ImGui::MenuItem(gui::Theme::GetPresetName(preset), nullptr, is_selected)) {
                    theme.ApplyPreset(preset);
                    spdlog::info("Theme changed to: {}", gui::Theme::GetPresetName(preset));
                    // Notify callback to save the theme
                    if (app_theme_changed_callback_) {
                        app_theme_changed_callback_(static_cast<int>(preset));
                    }
                }
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        // Theme Editor (opens full panel for advanced customization)
        if (ImGui::MenuItem(ICON_FA_BRUSH " Theme Editor...")) {
            if (open_theme_editor_callback_) {
                open_theme_editor_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open advanced theme editor for customizing ImGui and ImNodes colors");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Project Settings Section ==========
        bool has_project = ProjectManager::Instance().HasActiveProject();

        if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Project Settings", "Ctrl+Shift+S", false, has_project)) {
            if (save_project_settings_callback_) {
                save_project_settings_callback_();
                spdlog::info("Project settings saved to .cyxwiz");
            }
        }
        if (!has_project && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Open a project first to save settings");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Minimaps Section ==========
        if (ImGui::BeginMenu(ICON_FA_EYE " Minimaps")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));

            // Node Editor Minimap
            if (node_editor_minimap_ptr_) {
                bool node_minimap = *node_editor_minimap_ptr_;
                if (ImGui::MenuItem(ICON_FA_DIAGRAM_PROJECT " Node Editor Minimap", nullptr, node_minimap)) {
                    *node_editor_minimap_ptr_ = !node_minimap;
                }
            } else {
                ImGui::MenuItem(ICON_FA_DIAGRAM_PROJECT " Node Editor Minimap", nullptr, false, false);
            }

            // Script Editor Minimap
            if (script_editor_minimap_ptr_) {
                bool script_minimap = *script_editor_minimap_ptr_;
                if (ImGui::MenuItem(ICON_FA_CODE " Script Editor Minimap", nullptr, script_minimap)) {
                    *script_editor_minimap_ptr_ = !script_minimap;
                }
            } else {
                ImGui::MenuItem(ICON_FA_CODE " Script Editor Minimap", nullptr, false, false);
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Developer Tools ==========
        if (ImGui::MenuItem(ICON_FA_GAUGE_HIGH " Performance Profiler")) {
            if (open_profiler_callback_) {
                open_profiler_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Open per-layer timing profiler for training analysis");
        }

        if (ImGui::MenuItem(ICON_FA_MICROCHIP " Memory Monitor")) {
            if (open_memory_monitor_callback_) {
                open_memory_monitor_callback_();
            }
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Monitor CPU and GPU memory usage in real-time");
        }

        ImGui::Spacing();

        // Debug logging toggles
        if (idle_log_ptr_) {
            bool idle_log = *idle_log_ptr_;
            if (ImGui::MenuItem(ICON_FA_PAUSE " Log Idle Mode Transitions", nullptr, idle_log)) {
                *idle_log_ptr_ = !idle_log;
                if (*idle_log_ptr_) {
                    spdlog::info("Idle mode logging ENABLED");
                } else {
                    spdlog::info("Idle mode logging DISABLED");
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Log when app enters/exits power-saving idle mode");
            }
        }

        if (verbose_python_log_ptr_) {
            bool verbose_log = *verbose_python_log_ptr_;
            if (ImGui::MenuItem(ICON_FA_TERMINAL " Verbose Python Logging", nullptr, verbose_log)) {
                *verbose_python_log_ptr_ = !verbose_log;
                if (*verbose_python_log_ptr_) {
                    spdlog::info("Verbose Python logging ENABLED (includes Variable Explorer)");
                } else {
                    spdlog::info("Verbose Python logging DISABLED");
                }
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Log all Python commands including internal Variable Explorer queries");
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== View Options ==========
        if (ImGui::MenuItem(ICON_FA_EXPAND " Fullscreen", "F11")) {
            // TODO: Toggle fullscreen mode
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
