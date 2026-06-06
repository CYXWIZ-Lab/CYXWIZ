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

    // ========== Preferences Dialog ==========
    if (show_preferences_dialog_) {
        // Note: shortcuts_ is initialized in RenderEditMenu() when Preferences is clicked

        // Python settings have been moved to PythonSettingsPanel
        // This initialization code is no longer needed

        ImGui::OpenPopup("Preferences");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(650, 500), ImGuiCond_Appearing);

        if (ImGui::BeginPopupModal("Preferences", &show_preferences_dialog_)) {
            // Tab bar for different preference sections
            if (ImGui::BeginTabBar("PreferenceTabs")) {

                // ========== General Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_GEAR " General")) {
                    preferences_tab_ = 0;
                    ImGui::Spacing();

                    ImGui::Text("Startup");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Restore last session on startup", &general_restore_last_session_);
                    ImGui::Checkbox("Check for updates on startup", &general_check_updates_);

                    ImGui::Spacing();
                    ImGui::Text("Recent Files Limit:");
                    ImGui::SetNextItemWidth(100);
                    ImGui::InputInt("##recent_limit", &general_recent_files_limit_);
                    if (general_recent_files_limit_ < 1) general_recent_files_limit_ = 1;
                    if (general_recent_files_limit_ > 50) general_recent_files_limit_ = 50;

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Exit Behavior");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Confirm before exit with unsaved changes", &general_confirm_on_exit_);

                    ImGui::EndTabItem();
                }

                // ========== Editor Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_PEN " Editor")) {
                    preferences_tab_ = 1;
                    ImGui::Spacing();

                    ImGui::Text("Theme & Colors");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Editor Theme:");
                    ImGui::SetNextItemWidth(200);
                    const char* theme_items[] = { "Dark", "Light", "Retro Blue", "Monokai", "Dracula", "One Dark", "GitHub" };
                    int prev_theme = editor_theme_;
                    if (ImGui::Combo("##editor_theme", &editor_theme_, theme_items, IM_ARRAYSIZE(theme_items))) {
                        if (editor_theme_callback_ && editor_theme_ != prev_theme) {
                            editor_theme_callback_(editor_theme_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Font & Display");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Font Size:");
                    ImGui::SetNextItemWidth(200);
                    const char* font_size_items[] = { "Small (1.0x)", "Medium (1.3x)", "Large (1.6x)", "Extra Large (2.0x)" };
                    int font_size_index = 2;  // Default to Large
                    if (editor_font_size_ <= 10) font_size_index = 0;
                    else if (editor_font_size_ <= 14) font_size_index = 1;
                    else if (editor_font_size_ <= 18) font_size_index = 2;
                    else font_size_index = 3;

                    if (ImGui::Combo("##font_size", &font_size_index, font_size_items, IM_ARRAYSIZE(font_size_items))) {
                        float scales[] = { 1.0f, 1.3f, 1.6f, 2.0f };
                        int sizes[] = { 10, 13, 16, 20 };
                        editor_font_size_ = sizes[font_size_index];
                        if (editor_font_scale_callback_) {
                            editor_font_scale_callback_(scales[font_size_index]);
                        }
                    }

                    ImGui::Spacing();

                    ImGui::Text("Tab Size:");
                    ImGui::SetNextItemWidth(200);
                    const char* tab_size_items[] = { "2 Spaces", "4 Spaces", "8 Spaces" };
                    int tab_size_index = (editor_tab_size_ == 2) ? 0 : (editor_tab_size_ == 8) ? 2 : 1;
                    int prev_tab_index = tab_size_index;
                    if (ImGui::Combo("##tab_size", &tab_size_index, tab_size_items, IM_ARRAYSIZE(tab_size_items))) {
                        int sizes[] = { 2, 4, 8 };
                        editor_tab_size_ = sizes[tab_size_index];
                        if (editor_tab_size_callback_ && tab_size_index != prev_tab_index) {
                            editor_tab_size_callback_(editor_tab_size_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Editor Features");
                    ImGui::Separator();
                    ImGui::Spacing();

                    bool prev_show_whitespace = editor_show_whitespace_;
                    if (ImGui::Checkbox("Show Whitespace Characters", &editor_show_whitespace_)) {
                        if (editor_show_whitespace_callback_ && editor_show_whitespace_ != prev_show_whitespace) {
                            editor_show_whitespace_callback_(editor_show_whitespace_);
                        }
                    }

                    bool prev_word_wrap = editor_word_wrap_;
                    if (ImGui::Checkbox("Word Wrap", &editor_word_wrap_)) {
                        if (editor_word_wrap_callback_ && editor_word_wrap_ != prev_word_wrap) {
                            editor_word_wrap_callback_(editor_word_wrap_);
                        }
                    }

                    bool prev_auto_indent = editor_auto_indent_;
                    if (ImGui::Checkbox("Auto Indent", &editor_auto_indent_)) {
                        if (editor_auto_indent_callback_ && editor_auto_indent_ != prev_auto_indent) {
                            editor_auto_indent_callback_(editor_auto_indent_);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::TextDisabled("Line numbers are always shown. Current line is highlighted.");
                    ImGui::TextDisabled("These settings will be saved with your project.");

                    ImGui::EndTabItem();
                }

                // ========== Appearance Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_PALETTE " Appearance")) {
                    preferences_tab_ = 2;
                    ImGui::Spacing();

                    ImGui::Text("User Interface");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("UI Scale:");
                    ImGui::SetNextItemWidth(200);
                    ImGui::SliderFloat("##ui_scale", &appearance_ui_scale_, 0.8f, 2.0f, "%.1fx");
                    ImGui::SameLine();
                    if (ImGui::Button("Reset##scale")) {
                        appearance_ui_scale_ = 1.0f;
                    }

                    ImGui::Spacing();
                    ImGui::Checkbox("Smooth Scrolling", &appearance_smooth_scrolling_);

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Layout");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Sidebar Position:");
                    ImGui::RadioButton("Left", &appearance_sidebar_position_, 0);
                    ImGui::SameLine();
                    ImGui::RadioButton("Right", &appearance_sidebar_position_, 1);

                    ImGui::Spacing();
                    ImGui::TextDisabled("Note: Editor theme can be changed in the Editor tab.");

                    ImGui::EndTabItem();
                }

                // ========== Files Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_FILE " Files")) {
                    preferences_tab_ = 3;
                    ImGui::Spacing();

                    ImGui::Text("File Encoding");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Default Encoding:");
                    const char* encodings[] = { "UTF-8", "UTF-16", "ASCII" };
                    ImGui::SetNextItemWidth(150);
                    ImGui::Combo("##encoding", &files_default_encoding_, encodings, IM_ARRAYSIZE(encodings));

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Line Endings");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Default Line Ending:");
                    const char* line_endings[] = { "Auto (OS default)", "LF (Unix/macOS)", "CRLF (Windows)" };
                    ImGui::SetNextItemWidth(200);
                    ImGui::Combo("##line_ending", &files_line_ending_, line_endings, IM_ARRAYSIZE(line_endings));

                    ImGui::Spacing();
                    ImGui::Spacing();
                    ImGui::Text("Save Options");
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Checkbox("Trim trailing whitespace on save", &files_trim_trailing_whitespace_);
                    ImGui::Checkbox("Insert final newline on save", &files_insert_final_newline_);

                    ImGui::EndTabItem();
                }

                // ========== Python/Scripting Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_CODE " Python")) {
                    preferences_tab_ = 4;
                    ImGui::Spacing();

                    python_settings_panel_.Render();

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Startup Script
                    ImGui::Text("Startup Script (run on launch):");
                    ImGui::SetNextItemWidth(-100);
                    ImGui::InputText("##startup_script", python_startup_script_, sizeof(python_startup_script_));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##startup")) {
                        std::string path = OpenFileDialog("Python Scripts (*.py)\0*.py\0CyxWiz Scripts (*.cyx)\0*.cyx\0All Files (*.*)\0*.*\0", "Select Startup Script");
                        if (!path.empty()) {
                            strncpy(python_startup_script_, path.c_str(), sizeof(python_startup_script_) - 1);
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Auto-import options
                    ImGui::Text("Auto-Import Libraries:");
                    ImGui::Checkbox("Import NumPy as 'np'", &python_auto_import_numpy_);
                    ImGui::Checkbox("Import CyxWiz module", &python_auto_import_cyxwiz_);

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Output limit
                    ImGui::Text("Console Output Limit (lines):");
                    ImGui::SetNextItemWidth(150);
                    ImGui::InputInt("##output_limit", &python_output_limit_);
                    if (python_output_limit_ < 100) python_output_limit_ = 100;
                    if (python_output_limit_ > 10000) python_output_limit_ = 10000;
                    ImGui::TextDisabled("Range: 100 - 10000 lines");

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    if (ImGui::Button("Show Runtime Details")) {
                        if (python_diagnostics_callback_) {
                            python_diagnostics_text_ = python_diagnostics_callback_();
                        } else {
                            python_diagnostics_text_ = "Python diagnostics are not available.";
                        }
                        show_python_diagnostics_popup_ = true;
                        ImGui::OpenPopup("Python Runtime Details");
                    }

                    if (ImGui::BeginPopupModal("Python Runtime Details", &show_python_diagnostics_popup_, ImGuiWindowFlags_AlwaysAutoResize)) {
                        ImGui::BeginChild("##python_diag_text", ImVec2(560, 300), true, ImGuiWindowFlags_HorizontalScrollbar);
                        ImGui::TextUnformatted(python_diagnostics_text_.c_str());
                        ImGui::EndChild();

                        if (ImGui::Button("Copy")) {
                            ImGui::SetClipboardText(python_diagnostics_text_.c_str());
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Close")) {
                            show_python_diagnostics_popup_ = false;
                            ImGui::CloseCurrentPopup();
                        }

                        ImGui::EndPopup();
                    }

                    ImGui::EndTabItem();
                }

                // ========== Keyboard Shortcuts Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_KEYBOARD " Shortcuts")) {
                    preferences_tab_ = 5;
                    ImGui::Spacing();

                    ImGui::TextDisabled("Double-click a shortcut to edit. Some shortcuts are system-level and cannot be changed.");
                    ImGui::Spacing();

                    // Table of shortcuts with category grouping
                    if (ImGui::BeginTable("ShortcutsTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 320))) {
                        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 180);
                        ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 150);
                        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        std::string current_category = "";
                        for (int i = 0; i < static_cast<int>(shortcuts_.size()); ++i) {
                            auto& shortcut = shortcuts_[i];

                            // Check if we're entering a new category
                            if (shortcut.category != current_category) {
                                current_category = shortcut.category;

                                // Render category header row
                                ImGui::TableNextRow();
                                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, ImGui::GetColorU32(ImGuiCol_TableHeaderBg));

                                ImGui::TableNextColumn();
                                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                                ImGui::TextUnformatted(ICON_FA_FOLDER);
                                ImGui::SameLine();
                                ImGui::Text("%s", current_category.c_str());
                                ImGui::PopStyleColor();

                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("---");

                                ImGui::TableNextColumn();
                                ImGui::TextDisabled("---");
                            }

                            ImGui::TableNextRow();

                            // Action column (indented to show hierarchy)
                            ImGui::TableNextColumn();
                            ImGui::Text("  %s", shortcut.action.c_str());

                            // Shortcut column
                            ImGui::TableNextColumn();
                            if (editing_shortcut_index_ == i) {
                                // Edit mode
                                ImGui::SetNextItemWidth(-1);
                                if (ImGui::InputText("##edit_shortcut", shortcut_edit_buffer_, sizeof(shortcut_edit_buffer_), ImGuiInputTextFlags_EnterReturnsTrue)) {
                                    shortcut.shortcut = shortcut_edit_buffer_;
                                    editing_shortcut_index_ = -1;
                                }
                                if (ImGui::IsItemDeactivated() && !ImGui::IsItemActive()) {
                                    editing_shortcut_index_ = -1;
                                }
                            } else {
                                // Display mode
                                if (shortcut.editable) {
                                    if (ImGui::Selectable(shortcut.shortcut.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                                        editing_shortcut_index_ = i;
                                        strncpy(shortcut_edit_buffer_, shortcut.shortcut.c_str(), sizeof(shortcut_edit_buffer_) - 1);
                                    }
                                } else {
                                    ImGui::TextDisabled("%s", shortcut.shortcut.c_str());
                                }
                            }

                            // Description column
                            ImGui::TableNextColumn();
                            ImGui::TextDisabled("%s", shortcut.description.c_str());
                        }

                        ImGui::EndTable();
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Reset to Defaults")) {
                        shortcuts_.clear();  // Will be re-initialized on next open
                    }

                    ImGui::EndTabItem();
                }

                // ========== Devices Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_MICROCHIP " Devices")) {
                    preferences_tab_ = 6;
                    ImGui::Spacing();

                    // Show currently active backend
                    ImGui::Text("Currently Active Backend:");
                    ImGui::SameLine();
                    try {
                        auto* current_device = cyxwiz::Device::GetCurrentDevice();
                        if (current_device) {
                            const char* backend_name = "Unknown";
                            ImVec4 backend_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
                            switch (current_device->GetType()) {
                                case cyxwiz::DeviceType::CPU:
                                    backend_name = "CPU";
                                    backend_color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::CUDA:
                                    backend_name = "CUDA (NVIDIA)";
                                    backend_color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::OPENCL:
                                    backend_name = "OpenCL";
                                    backend_color = ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                                    break;
                                case cyxwiz::DeviceType::METAL:
                                    backend_name = "Metal (Apple)";
                                    backend_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                                    break;
                                default:
                                    break;
                            }
                            ImGui::TextColored(backend_color, "%s %s", ICON_FA_CIRCLE_CHECK, backend_name);
                        } else {
                            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.4f, 1.0f), "%s Not Set (using default)", ICON_FA_CIRCLE_EXCLAMATION);
                        }
                    } catch (...) {
                        ImGui::TextDisabled("Unable to query");
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    ImGui::Text("Available Compute Devices");
                    ImGui::Separator();
                    ImGui::Spacing();
                    ImGui::TextDisabled("Select which device to use for training. CUDA devices are fastest for NVIDIA GPUs.");
                    ImGui::Spacing();

                    // Initialize device list if needed
                    if (!devices_initialized_) {
                        cached_devices_.clear();
                        try {
                            auto devices = cyxwiz::Device::GetAvailableDevices();
                            for (const auto& dev : devices) {
                                CachedDevice cached;
                                cached.type = static_cast<int>(dev.type);
                                cached.device_id = dev.device_id;
                                cached.name = dev.name;
                                cached.memory_total = dev.memory_total;
                                cached.memory_available = dev.memory_available;
                                cached_devices_.push_back(cached);
                            }

                            // Default to first CUDA device if available, otherwise keep CPU (index 0)
                            selected_device_index_ = 0;
                            for (size_t i = 0; i < cached_devices_.size(); ++i) {
                                if (cached_devices_[i].type == 1) {  // 1 = CUDA
                                    selected_device_index_ = static_cast<int>(i);
                                    // Set CUDA as the active device
                                    cyxwiz::Device cuda_device(cyxwiz::DeviceType::CUDA, cached_devices_[i].device_id);
                                    cuda_device.SetActive();
                                    spdlog::info("Default device set to CUDA: {}", cached_devices_[i].name);
                                    break;
                                }
                            }
                        } catch (...) {
                            spdlog::warn("Failed to enumerate devices");
                        }
                        devices_initialized_ = true;
                    }

                    // Device selection list
                    if (cached_devices_.empty()) {
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "No compute devices found!");
                    } else {
                        for (size_t i = 0; i < cached_devices_.size(); ++i) {
                            const auto& dev = cached_devices_[i];

                            const char* type_icon = ICON_FA_MICROCHIP;
                            const char* type_name = "Unknown";
                            ImVec4 type_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);

                            switch (dev.type) {
                                case 0: // CPU
                                    type_icon = ICON_FA_MICROCHIP;
                                    type_name = "CPU";
                                    type_color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
                                    break;
                                case 1: // CUDA
                                    type_icon = ICON_FA_BOLT;
                                    type_name = "CUDA";
                                    type_color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
                                    break;
                                case 2: // OpenCL
                                    type_icon = ICON_FA_DESKTOP;
                                    type_name = "OpenCL";
                                    type_color = ImVec4(1.0f, 0.8f, 0.4f, 1.0f);
                                    break;
                                case 3: // Metal
                                    type_icon = ICON_FA_MICROCHIP;
                                    type_name = "Metal";
                                    type_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
                                    break;
                            }

                            ImGui::PushID(static_cast<int>(i));

                            // Radio button for selection
                            bool is_selected = (selected_device_index_ == static_cast<int>(i));
                            if (ImGui::RadioButton("##device_select", is_selected)) {
                                selected_device_index_ = static_cast<int>(i);
                                // Apply device selection
                                try {
                                    cyxwiz::Device device(static_cast<cyxwiz::DeviceType>(dev.type), dev.device_id);
                                    device.SetActive();
                                    spdlog::info("Selected device: {} [{}]", dev.name, type_name);
                                } catch (const std::exception& e) {
                                    spdlog::error("Failed to set device: {}", e.what());
                                }
                            }
                            ImGui::SameLine();

                            // Device info
                            ImGui::TextColored(type_color, "%s", type_icon);
                            ImGui::SameLine();
                            ImGui::Text("%s", dev.name.c_str());
                            ImGui::SameLine();
                            ImGui::TextDisabled("[%s]", type_name);

                            // Memory info on same line if available
                            if (dev.memory_total > 0) {
                                ImGui::SameLine();
                                double mem_gb = dev.memory_total / (1024.0 * 1024.0 * 1024.0);
                                ImGui::TextDisabled("(%.1f GB)", mem_gb);
                            }

                            // Show "Active" badge if this is the current device
                            auto* current_device = cyxwiz::Device::GetCurrentDevice();
                            if (current_device &&
                                current_device->GetType() == static_cast<cyxwiz::DeviceType>(dev.type) &&
                                current_device->GetDeviceId() == dev.device_id) {
                                ImGui::SameLine();
                                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "%s Active", ICON_FA_CIRCLE_CHECK);
                            }

                            ImGui::PopID();
                        }
                    }

                    ImGui::Spacing();
                    ImGui::Spacing();

                    // Refresh button
                    if (ImGui::Button(ICON_FA_ARROWS_ROTATE " Refresh Devices")) {
                        devices_initialized_ = false;
                    }

                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();

                    // Current device info
                    if (selected_device_index_ >= 0 && selected_device_index_ < static_cast<int>(cached_devices_.size())) {
                        const auto& dev = cached_devices_[selected_device_index_];
                        ImGui::Text("Selected Device Details:");
                        ImGui::Indent();
                        ImGui::BulletText("Name: %s", dev.name.c_str());
                        if (dev.memory_total > 0) {
                            double total_gb = dev.memory_total / (1024.0 * 1024.0 * 1024.0);
                            double avail_gb = dev.memory_available / (1024.0 * 1024.0 * 1024.0);
                            ImGui::BulletText("Memory: %.2f GB total, %.2f GB available", total_gb, avail_gb);
                        }
                        ImGui::Unindent();
                    }

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Dialog buttons
            float button_width = 100.0f;
            float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
            ImGui::SetCursorPosX((ImGui::GetWindowWidth() - total_width) * 0.5f);

            if (ImGui::Button("OK", ImVec2(button_width, 0))) {
                // Python settings are now in PythonSettingsPanel
                // Save preferences to project if one is open
                if (save_project_settings_callback_) {
                    save_project_settings_callback_();
                }
                show_preferences_dialog_ = false;
                spdlog::info("Preferences saved");
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
                show_preferences_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }

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
