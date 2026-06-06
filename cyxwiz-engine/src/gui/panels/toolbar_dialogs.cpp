// Toolbar modal dialogs and dialog-adjacent background handling.

#include "toolbar.h"
#include "../icons.h"
#include "../../auth/auth_client.h"
#include "../../core/engine_config.h"
#include "../../core/project_manager.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cyxwiz/cyxwiz.h>

namespace cyxwiz {
void ToolbarPanel::RenderProjectDialogs() {
    // Render dialogs if open
    if (show_new_project_dialog_) {
        ImGui::OpenPopup("New Project");
        if (ImGui::BeginPopupModal("New Project", &show_new_project_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Create a new CyxWiz project");
            ImGui::Separator();

            ImGui::InputText("Project Name", project_name_buffer_, sizeof(project_name_buffer_));

            ImGui::InputText("Location", project_path_buffer_, sizeof(project_path_buffer_));
            ImGui::SameLine();
            if (ImGui::Button("Browse...")) {
                std::string selected_folder = OpenFolderDialog();
                if (!selected_folder.empty()) {
                    strncpy(project_path_buffer_, selected_folder.c_str(), sizeof(project_path_buffer_) - 1);
                    project_path_buffer_[sizeof(project_path_buffer_) - 1] = '\0';
                }
            }

            ImGui::Separator();

            if (ImGui::Button("Create", ImVec2(120, 0))) {
                std::string proj_name = project_name_buffer_;
                std::string proj_path = project_path_buffer_;

                if (!proj_name.empty() && !proj_path.empty()) {
                    auto& pm = ProjectManager::Instance();
                    if (pm.CreateProject(proj_name, proj_path)) {
                        spdlog::info("Project created: {}/{}", proj_path, proj_name);
                        show_new_project_dialog_ = false;
                        // Clear buffers
                        memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
                        memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
                    }
                } else {
                    spdlog::warn("Project name and location are required");
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_new_project_dialog_ = false;
                // Clear buffers
                memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
                memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
            }

            ImGui::EndPopup();
        }
    }

    if (show_about_dialog_) {
        ImGui::OpenPopup("About CyxWiz");
        if (ImGui::BeginPopupModal("About CyxWiz", &show_about_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("CyxWiz Engine");
            ImGui::Text("Version 0.1.0");
            ImGui::Separator();
            ImGui::Text("Decentralized ML Compute Platform");
            ImGui::Text("Built with C++, ImGui, ArrayFire, and Solana");
            ImGui::Separator();

            if (ImGui::Button("OK", ImVec2(120, 0))) {
                show_about_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }

    // Login Required popup - shown when user tries to access server features without logging in
    if (show_login_required_popup_) {
        ImGui::OpenPopup("Login Required");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        ImGuiWindowFlags popup_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove;
        if (ImGui::BeginPopupModal("Login Required", &show_login_required_popup_, popup_flags)) {
            // Warning icon and message
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
            ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Text("Authentication Required");

            ImGui::Separator();
            ImGui::Spacing();

            // Dynamic message based on action
            ImGui::TextWrapped("You need to be logged in to %s.", login_required_action_.c_str());
            ImGui::Spacing();
            ImGui::TextWrapped("Please login to your CyxWiz account or create a new account to continue.");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Buttons
            float button_width = 120.0f;
            float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
            float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;
            ImGui::SetCursorPosX(start_x);

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.8f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.6f, 0.9f, 1.0f));
            if (ImGui::Button(ICON_FA_RIGHT_TO_BRACKET " Login", ImVec2(button_width, 0))) {
                show_login_required_popup_ = false;
                show_account_settings_dialog_ = true;  // Open login dialog
            }
            ImGui::PopStyleColor(2);

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
                show_login_required_popup_ = false;
            }

            ImGui::EndPopup();
        }
    }

    // Save As dialog
    if (show_save_as_dialog_) {
        ImGui::OpenPopup("Save Project As");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Save Project As", &show_save_as_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Save a copy of the project with a new name");
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Text("New Project Name:");
            ImGui::SetNextItemWidth(350);
            ImGui::InputText("##saveasname", save_as_name_buffer_, sizeof(save_as_name_buffer_));

            ImGui::Spacing();

            ImGui::Text("Location:");
            ImGui::SetNextItemWidth(280);
            ImGui::InputText("##saveaspath", save_as_path_buffer_, sizeof(save_as_path_buffer_));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##saveas")) {
                std::string selected_folder = OpenFolderDialog();
                if (!selected_folder.empty()) {
                    strncpy(save_as_path_buffer_, selected_folder.c_str(), sizeof(save_as_path_buffer_) - 1);
                    save_as_path_buffer_[sizeof(save_as_path_buffer_) - 1] = '\0';
                }
            }

            ImGui::Spacing();

            // Preview the new project path
            std::string new_name = save_as_name_buffer_;
            std::string new_path = save_as_path_buffer_;
            if (!new_name.empty() && !new_path.empty()) {
                std::filesystem::path preview_path = std::filesystem::path(new_path) / new_name;
                ImGui::TextDisabled("Will create: %s", preview_path.string().c_str());
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            bool valid_input = strlen(save_as_name_buffer_) > 0 && strlen(save_as_path_buffer_) > 0;

            if (!valid_input) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("Save", ImVec2(120, 0))) {
                auto& pm = ProjectManager::Instance();
                if (pm.SaveProjectAs(save_as_name_buffer_, save_as_path_buffer_)) {
                    spdlog::info("Project saved as: {}", save_as_name_buffer_);
                    show_save_as_dialog_ = false;
                    memset(save_as_name_buffer_, 0, sizeof(save_as_name_buffer_));
                    memset(save_as_path_buffer_, 0, sizeof(save_as_path_buffer_));
                } else {
                    spdlog::error("Failed to save project as: {}", save_as_name_buffer_);
                }
            }

            if (!valid_input) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_save_as_dialog_ = false;
                memset(save_as_name_buffer_, 0, sizeof(save_as_name_buffer_));
                memset(save_as_path_buffer_, 0, sizeof(save_as_path_buffer_));
            }

            ImGui::EndPopup();
        }
    }
}

void ToolbarPanel::HandleAutoSaveTimer() {
    // Auto Save timer logic
    if (auto_save_enabled_) {
        float delta_time = ImGui::GetIO().DeltaTime;
        auto_save_timer_ += delta_time;

        if (auto_save_timer_ >= auto_save_interval_) {
            auto_save_timer_ = 0.0f;

            // Trigger save all callback
            if (save_all_callback_) {
                save_all_callback_();
                spdlog::info("Auto-save triggered");
            }
        }
    } else {
        // Reset timer when disabled
        auto_save_timer_ = 0.0f;
    }
}

void ToolbarPanel::RenderEditorDialogs() {
    // New Script dialog
    if (show_new_script_dialog_) {
        ImGui::OpenPopup("New Script");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("New Script", &show_new_script_dialog_, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Create a new script file");
            ImGui::Separator();
            ImGui::Spacing();

            // Script name input
            ImGui::Text("Script Name:");
            ImGui::SetNextItemWidth(300);
            ImGui::InputText("##scriptname", new_script_name_, sizeof(new_script_name_));

            ImGui::Spacing();

            // Script type selection
            ImGui::Text("Script Type:");
            ImGui::RadioButton(".cyx (CyxWiz Script)", &new_script_type_, 0);
            ImGui::SameLine();
            ImGui::RadioButton(".py (Python Script)", &new_script_type_, 1);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Preview the filename
            std::string script_name = new_script_name_;
            std::string extension = (new_script_type_ == 0) ? ".cyx" : ".py";
            if (!script_name.empty()) {
                // Add extension if not already present
                std::string lower_name = script_name;
                std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
                if (lower_name.length() < 4 ||
                    (lower_name.substr(lower_name.length() - 4) != ".cyx" &&
                     lower_name.substr(lower_name.length() - 3) != ".py")) {
                    script_name += extension;
                }
                ImGui::TextDisabled("Will create: %s", script_name.c_str());
            }

            ImGui::Spacing();

            // Check if we have an active project
            auto& pm = ProjectManager::Instance();
            bool has_project = pm.HasActiveProject();

            if (!has_project) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.5f, 0.0f, 1.0f));
                ImGui::TextWrapped("Note: No active project. Script will be created in the current directory.");
                ImGui::PopStyleColor();
                ImGui::Spacing();
            }

            // Create and Cancel buttons
            bool name_valid = strlen(new_script_name_) > 0;

            if (!name_valid) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button("Create", ImVec2(120, 0))) {
                // Build full path
                std::string filename = new_script_name_;
                std::string ext = (new_script_type_ == 0) ? ".cyx" : ".py";

                // Add extension if not present
                std::string lower_fn = filename;
                std::transform(lower_fn.begin(), lower_fn.end(), lower_fn.begin(), ::tolower);
                if (lower_fn.length() < 4 ||
                    (lower_fn.substr(lower_fn.length() - 4) != ".cyx" &&
                     lower_fn.substr(lower_fn.length() - 3) != ".py")) {
                    filename += ext;
                }

                std::filesystem::path file_path;
                if (has_project) {
                    file_path = std::filesystem::path(pm.GetScriptsPath()) / filename;
                } else {
                    file_path = std::filesystem::path(filename);
                }

                // Create the file with default content
                std::ofstream file(file_path);
                if (file.is_open()) {
                    if (new_script_type_ == 0) {
                        // .cyx file default content
                        file << "# CyxWiz Script\n";
                        file << "# " << filename << "\n\n";
                        file << "# Define your ML pipeline here\n\n";
                    } else {
                        // .py file default content
                        file << "# Python Script\n";
                        file << "# " << filename << "\n\n";
                        file << "import pycyxwiz\n\n";
                        file << "def main():\n";
                        file << "    pass\n\n";
                        file << "if __name__ == '__main__':\n";
                        file << "    main()\n";
                    }
                    file.close();

                    spdlog::info("Created script: {}", file_path.string());

                    // Open the script in editor
                    if (open_script_in_editor_callback_) {
                        open_script_in_editor_callback_(file_path.string());
                    }

                    // Refresh asset browser
                    if (new_script_callback_) {
                        new_script_callback_();
                    }

                    show_new_script_dialog_ = false;
                    memset(new_script_name_, 0, sizeof(new_script_name_));
                } else {
                    spdlog::error("Failed to create script: {}", file_path.string());
                }
            }

            if (!name_valid) {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_new_script_dialog_ = false;
                memset(new_script_name_, 0, sizeof(new_script_name_));
            }

            ImGui::EndPopup();
        }
    }

    RenderAccountDialogs();

    // Exit confirmation dialog
    if (show_exit_confirmation_dialog_) {
        ImGui::OpenPopup("##ExitConfirmation");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 20));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12, 12));

        ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;

        if (ImGui::BeginPopupModal("##ExitConfirmation", &show_exit_confirmation_dialog_, flags)) {
            // Warning icon and title
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
            ImGui::Text(ICON_FA_TRIANGLE_EXCLAMATION);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Text("Unsaved Changes");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped("You have unsaved changes. Do you want to save before closing?");

            ImGui::Spacing();
            ImGui::Spacing();

            // Button row
            float button_width = 100.0f;
            float total_width = button_width * 3 + ImGui::GetStyle().ItemSpacing.x * 2;
            float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;

            ImGui::SetCursorPosX(start_x);

            // Save & Exit button (primary action)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.45f, 0.85f, 1.0f));
            if (ImGui::Button("Save & Exit", ImVec2(button_width, 32))) {
                // Save all and then exit
                if (save_all_callback_) {
                    save_all_callback_();
                }
                show_exit_confirmation_dialog_ = false;
                if (exit_callback_) {
                    exit_callback_();
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine();

            // Don't Save button (secondary action)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.18f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.15f, 0.15f, 1.0f));
            if (ImGui::Button("Don't Save", ImVec2(button_width, 32))) {
                // Exit without saving
                show_exit_confirmation_dialog_ = false;
                if (exit_callback_) {
                    exit_callback_();
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::SameLine();

            // Cancel button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.38f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.20f, 0.23f, 1.0f));
            if (ImGui::Button("Cancel", ImVec2(button_width, 32))) {
                show_exit_confirmation_dialog_ = false;
            }
            ImGui::PopStyleColor(3);

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(5);
    }

    RenderSearchDialogs();

    RenderPreferencesDialog();

    // ========== Go to Line Dialog ==========
    if (show_go_to_line_dialog_) {
        ImGui::OpenPopup("Go to Line");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Go to Line", &show_go_to_line_dialog_, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Enter line number:");
            ImGui::Spacing();

            ImGui::SetNextItemWidth(200);
            bool enter_pressed = ImGui::InputInt("##linenumber", &go_to_line_number_, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue);
            if (go_to_line_number_ < 1) go_to_line_number_ = 1;

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Go", ImVec2(80, 0)) || enter_pressed) {
                if (go_to_line_callback_) {
                    go_to_line_callback_(go_to_line_number_);
                }
                show_go_to_line_dialog_ = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(80, 0))) {
                show_go_to_line_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }
}

} // namespace cyxwiz
