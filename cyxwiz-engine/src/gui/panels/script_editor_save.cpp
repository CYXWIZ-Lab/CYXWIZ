// Script Editor unsaved-file checks and save confirmation dialogs.

#include "script_editor.h"

#include <string>
#include <vector>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {
// ==================== Unsaved Files Check ====================

bool ScriptEditorPanel::HasUnsavedFiles() const {
    for (const auto& tab : tabs_) {
        if (tab->is_modified || tab->is_new) {
            // Check if tab has any content (not just an empty new file)
            std::string text = tab->editor.GetText();
            // Trim whitespace
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                // Has content that is unsaved
                return true;
            }
        }
    }
    return false;
}

std::vector<std::string> ScriptEditorPanel::GetUnsavedFileNames() const {
    std::vector<std::string> names;
    for (const auto& tab : tabs_) {
        if (tab->is_modified || tab->is_new) {
            // Check if tab has any content
            std::string text = tab->editor.GetText();
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                names.push_back(tab->filename);
            }
        }
    }
    return names;
}

void ScriptEditorPanel::SaveAllFiles() {
    int original_active = active_tab_index_;

    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        auto& tab = tabs_[i];
        if (tab->is_modified || tab->is_new) {
            // Check if tab has content
            std::string text = tab->editor.GetText();
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                // Has content, save it
                active_tab_index_ = i;
                SaveFile();
            }
        }
    }

    // Restore original active tab
    if (original_active >= 0 && original_active < static_cast<int>(tabs_.size())) {
        active_tab_index_ = original_active;
    }
}

bool ScriptEditorPanel::HasEmptyNewTab() const {
    for (const auto& tab : tabs_) {
        if (tab->is_new && !tab->is_modified) {
            return true;
        }
    }
    return false;
}

// ==================== Save Confirmation Dialogs ====================

void ScriptEditorPanel::RenderSaveBeforeRunDialog() {
    if (!show_save_before_run_dialog_) return;

    ImGui::OpenPopup("Save Before Running?");

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Save Before Running?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        std::string filename = "Untitled";
        if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
            filename = tabs_[active_tab_index_]->filename;
        }

        ImGui::Text("The script '%s' has unsaved changes.", filename.c_str());
        ImGui::Text("Would you like to save before running?");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Calculate button sizes for better layout
        float button_width = 100.0f;
        float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
        float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;
        ImGui::SetCursorPosX(start_x);

        if (ImGui::Button("Save & Run", ImVec2(button_width, 0))) {
            show_save_before_run_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Save the file first
            SaveFile();

            // Check if save was successful (file is no longer new/modified after SaveFileAs completes)
            if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[active_tab_index_];
                // If user completed save (file has a path now), run the script
                if (!tab->filepath.empty()) {
                    DoRunScript();
                } else {
                    spdlog::info("Save cancelled, script not run");
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            show_save_before_run_dialog_ = false;
            ImGui::CloseCurrentPopup();
            spdlog::info("Run cancelled by user");
        }

        ImGui::EndPopup();
    }
}

void ScriptEditorPanel::RenderSaveBeforeCloseDialog() {
    if (!show_save_before_close_dialog_) return;

    ImGui::OpenPopup("Save Changes?");

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Save Changes?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        std::string filename = "Untitled";
        if (pending_close_tab_index_ >= 0 && pending_close_tab_index_ < static_cast<int>(tabs_.size())) {
            filename = tabs_[pending_close_tab_index_]->filename;
        }

        ImGui::Text("Save changes to '%s'?", filename.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Calculate button sizes for better layout
        float button_width = 80.0f;
        float total_width = button_width * 3 + ImGui::GetStyle().ItemSpacing.x * 2;
        float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;
        ImGui::SetCursorPosX(start_x);

        if (ImGui::Button("Yes", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Switch to the tab being closed and save it
            int original_active = active_tab_index_;
            active_tab_index_ = pending_close_tab_index_;
            SaveFile();

            // Check if save was successful
            if (pending_close_tab_index_ >= 0 && pending_close_tab_index_ < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[pending_close_tab_index_];
                if (!tab->filepath.empty()) {
                    // Save completed, close the tab
                    DoCloseFile(pending_close_tab_index_);
                } else {
                    // Save was cancelled, don't close
                    spdlog::info("Save cancelled, tab not closed");
                    active_tab_index_ = original_active;
                }
            }
            pending_close_tab_index_ = -1;
        }

        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Close without saving
            if (pending_close_tab_index_ >= 0) {
                DoCloseFile(pending_close_tab_index_);
            }
            pending_close_tab_index_ = -1;
            spdlog::info("Closed without saving");
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();
            pending_close_tab_index_ = -1;
            spdlog::info("Close cancelled by user");
        }

        ImGui::EndPopup();
    }
}
} // namespace cyxwiz