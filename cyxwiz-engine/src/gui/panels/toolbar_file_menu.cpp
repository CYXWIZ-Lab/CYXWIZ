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
#include "../../core/file_dialogs.h"
#include "../icons.h"

namespace cyxwiz {

void ToolbarPanel::RenderFileMenu() {
    if (ImGui::BeginMenu("File")) {
        // Increase padding for menu items
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // ========== Project Section ==========
        if (ImGui::MenuItem(ICON_FA_FILE " New Project", "Ctrl+Shift+N")) {
            show_new_project_dialog_ = true;
        }

        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open Project...", "Ctrl+Shift+O")) {
            auto result = FileDialogs::OpenProject();
            if (result) {
                auto& pm = ProjectManager::Instance();
                if (pm.OpenProject(*result)) {
                    spdlog::info("Project opened: {}", *result);
                } else {
                    spdlog::error("Failed to open project: {}", *result);
                }
            }
        }

        if (ImGui::MenuItem(ICON_FA_XMARK " Close Project", nullptr, false, ProjectManager::Instance().HasActiveProject())) {
            ProjectManager::Instance().CloseProject();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Script Section ==========
        if (ImGui::MenuItem(ICON_FA_FILE_CODE " New Script...", "Ctrl+N")) {
            show_new_script_dialog_ = true;
            memset(new_script_name_, 0, sizeof(new_script_name_));
            new_script_type_ = 0;  // Default to .cyx
        }

        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open Script...", "Ctrl+O")) {
            auto result = FileDialogs::OpenScript();
            if (result) {
                if (open_script_in_editor_callback_) {
                    open_script_in_editor_callback_(*result);
                }
                spdlog::info("Opening script: {}", *result);
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Save Section ==========
        if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save", "Ctrl+S", false, ProjectManager::Instance().HasActiveProject())) {
            auto& pm = ProjectManager::Instance();
            if (pm.SaveProject()) {
                spdlog::info("Project saved");
            }
        }

        if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save As...", "Ctrl+Shift+S", false, ProjectManager::Instance().HasActiveProject())) {
            show_save_as_dialog_ = true;
            // Pre-fill with current project name and a suggestion for new location
            auto& pm = ProjectManager::Instance();
            strncpy(save_as_name_buffer_, (pm.GetProjectName() + "_copy").c_str(), sizeof(save_as_name_buffer_) - 1);
            save_as_name_buffer_[sizeof(save_as_name_buffer_) - 1] = '\0';
            // Use parent of current project root as default location
            std::filesystem::path current_root(pm.GetProjectRoot());
            std::string parent_path = current_root.parent_path().string();
            strncpy(save_as_path_buffer_, parent_path.c_str(), sizeof(save_as_path_buffer_) - 1);
            save_as_path_buffer_[sizeof(save_as_path_buffer_) - 1] = '\0';
        }

        if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save All", "Ctrl+Alt+S")) {
            if (save_all_callback_) {
                save_all_callback_();
            }
        }

        ImGui::Spacing();

        // Auto-save toggle with checkmark
        if (ImGui::MenuItem(ICON_FA_CLOCK " Auto Save", nullptr, auto_save_enabled_)) {
            auto_save_enabled_ = !auto_save_enabled_;
            spdlog::info("Auto-save {}", auto_save_enabled_ ? "enabled" : "disabled");
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Import/Export Section ==========
        if (ImGui::MenuItem(ICON_FA_FILE_IMPORT " Import Model...", "Ctrl+I")) {
            if (import_model_callback_) {
                import_model_callback_();
            }
        }

        if (ImGui::MenuItem(ICON_FA_FILE_EXPORT " Export Model...", "Ctrl+E")) {
            if (export_model_callback_) {
                export_model_callback_(0);  // 0 = default format (CyxModel)
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Recent Projects Section ==========
        auto& pm = ProjectManager::Instance();
        const auto& recent = pm.GetRecentProjects();

        if (ImGui::BeginMenu(ICON_FA_CLOCK " Recent Projects", !recent.empty())) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));

            for (const auto& rp : recent) {
                // Show project name with path as tooltip
                std::string label = rp.name;
                if (ImGui::MenuItem(label.c_str())) {
                    if (pm.OpenProject(rp.path)) {
                        spdlog::info("Opened recent project: {}", rp.name);
                    } else {
                        spdlog::error("Failed to open recent project: {}", rp.path);
                    }
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", rp.path.c_str());
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Clear Recent Projects")) {
                pm.ClearRecentProjects();
            }

            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Settings Section ==========
        if (ImGui::MenuItem(ICON_FA_USER " Account Settings...")) {
            show_account_settings_dialog_ = true;
            if (account_settings_callback_) {
                account_settings_callback_();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Exit ==========
        if (ImGui::MenuItem(ICON_FA_XMARK " Exit", "Alt+F4")) {
            // Check for unsaved changes
            bool has_unsaved = has_unsaved_changes_callback_ && has_unsaved_changes_callback_();
            if (has_unsaved) {
                show_exit_confirmation_dialog_ = true;
            } else {
                // No unsaved changes, exit directly
                if (exit_callback_) {
                    exit_callback_();
                }
            }
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
