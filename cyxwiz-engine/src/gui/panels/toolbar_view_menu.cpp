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
                }
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
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

        // ========== View Options ==========
        if (ImGui::MenuItem(ICON_FA_EXPAND " Fullscreen", "F11")) {
            // TODO: Toggle fullscreen mode
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
