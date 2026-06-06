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

    // Account Settings dialog
    if (show_account_settings_dialog_) {
        ImGui::OpenPopup("##AccountSettings");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        // Professional styling
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 24));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;

        if (ImGui::BeginPopupModal("##AccountSettings", &show_account_settings_dialog_, flags)) {

            if (!is_logged_in_) {
                // ========== Sign In View ==========

                // Logo/Brand area
                ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);  // Use default font
                float window_width = ImGui::GetWindowWidth();

                // Center the title
                const char* title = "CyxWiz";
                float title_width = ImGui::CalcTextSize(title).x;
                ImGui::SetCursorPosX((window_width - title_width) * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                ImGui::Text("%s", title);
                ImGui::PopStyleColor();
                ImGui::PopFont();

                ImGui::Spacing();

                // Subtitle
                const char* subtitle = "Sign in to your account";
                float subtitle_width = ImGui::CalcTextSize(subtitle).x;
                ImGui::SetCursorPosX((window_width - subtitle_width) * 0.5f);
                ImGui::TextDisabled("%s", subtitle);

                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();

                // Input fields with consistent width
                float input_width = 320.0f;
                float start_x = (window_width - input_width) * 0.5f;

                // Email or Phone field
                ImGui::SetCursorPosX(start_x);
                ImGui::Text("Email or Phone");
                ImGui::SetCursorPosX(start_x);

                float button_size = 28.0f;
                float field_width = input_width - button_size - 4.0f;

                ImGui::SetNextItemWidth(field_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                ImGui::InputText("##identifier", login_identifier_, sizeof(login_identifier_));
                ImGui::PopStyleColor(2);

                // Paste button for email
                ImGui::SameLine(0, 4.0f);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
                if (ImGui::Button(ICON_FA_PASTE "##paste_email", ImVec2(button_size, 0))) {
                    if (ImGui::GetClipboardText()) {
                        strncpy(login_identifier_, ImGui::GetClipboardText(), sizeof(login_identifier_) - 1);
                        login_identifier_[sizeof(login_identifier_) - 1] = '\0';
                    }
                }
                ImGui::PopStyleColor(2);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Paste from clipboard");
                }

                ImGui::Spacing();

                // Password field
                ImGui::SetCursorPosX(start_x);
                ImGui::Text("Password");
                ImGui::SetCursorPosX(start_x);

                float password_field_width = input_width - (button_size * 2) - 8.0f;
                ImGui::SetNextItemWidth(password_field_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                ImGuiInputTextFlags password_flags = ImGuiInputTextFlags_EnterReturnsTrue;
                if (!show_password_) {
                    password_flags |= ImGuiInputTextFlags_Password;
                }
                bool enter_pressed = ImGui::InputText("##password", login_password_, sizeof(login_password_), password_flags);
                ImGui::PopStyleColor(2);

                // Show/Hide password toggle button
                ImGui::SameLine(0, 4.0f);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
                if (ImGui::Button(show_password_ ? ICON_FA_EYE_SLASH "##toggle_pw" : ICON_FA_EYE "##toggle_pw", ImVec2(button_size, 0))) {
                    show_password_ = !show_password_;
                }
                ImGui::PopStyleColor(2);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip(show_password_ ? "Hide password" : "Show password");
                }

                // Paste button for password
                ImGui::SameLine(0, 4.0f);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
                if (ImGui::Button(ICON_FA_PASTE "##paste_password", ImVec2(button_size, 0))) {
                    if (ImGui::GetClipboardText()) {
                        strncpy(login_password_, ImGui::GetClipboardText(), sizeof(login_password_) - 1);
                        login_password_[sizeof(login_password_) - 1] = '\0';
                    }
                }
                ImGui::PopStyleColor(2);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Paste from clipboard");
                }

                // Error message
                if (!login_error_message_.empty()) {
                    ImGui::Spacing();
                    ImGui::SetCursorPosX(start_x);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui::Text("%s", login_error_message_.c_str());
                    ImGui::PopStyleColor();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                // Sign In button
                ImGui::SetCursorPosX(start_x);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.45f, 0.85f, 1.0f));

                bool can_login = strlen(login_identifier_) > 0 && strlen(login_password_) > 0 && !is_logging_in_;

                if (is_logging_in_) {
                    // Show loading state
                    ImGui::BeginDisabled();
                    ImGui::Button("Signing in...", ImVec2(input_width, 38));
                    ImGui::EndDisabled();
                } else if ((ImGui::Button("Sign In", ImVec2(input_width, 38)) || enter_pressed) && can_login) {
                    // Validate input
                    std::string identifier = login_identifier_;
                    std::string password = login_password_;

                    if (identifier.empty()) {
                        login_error_message_ = "Please enter your email or phone number";
                    } else if (password.empty()) {
                        login_error_message_ = "Please enter your password";
                    } else {
                        // Auto-detect if email or phone
                        bool is_email = identifier.find('@') != std::string::npos;

                        if (is_email) {
                            // Start async login
                            is_logging_in_ = true;
                            login_error_message_.clear();
                            login_success_message_.clear();
                            auto& auth = auth::AuthClient::Instance();
                            login_future_ = auth.LoginWithEmail(identifier, password);
                            spdlog::info("Starting login for: {}", identifier);
                        } else {
                            login_error_message_ = "Please enter a valid email address";
                        }
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();
                ImGui::Spacing();

                // Links row
                ImGui::SetCursorPosX(start_x);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));

                ImGui::Text("Forgot password?");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                    ImGui::SetTooltip("Reset your password");
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemClicked()) {
                    spdlog::info("Forgot password clicked");
                }

                ImGui::SameLine();
                ImGui::SetCursorPosX(start_x + input_width - ImGui::CalcTextSize("Create account").x);

                ImGui::Text("Create account");
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                    ImGui::SetTooltip("Sign up for CyxWiz");
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemClicked()) {
                    auth::AuthClient::OpenRegistrationPage();
                }

                ImGui::PopStyleColor();

            } else {
                // ========== Logged In View ==========
                auto& auth = auth::AuthClient::Instance();
                auto user = auth.GetUserInfo();

                // Get initials from name
                std::string initials;
                if (!user.name.empty()) {
                    std::istringstream iss(user.name);
                    std::string word;
                    while (iss >> word && initials.length() < 2) {
                        if (!word.empty()) {
                            initials += static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
                        }
                    }
                }
                if (initials.empty() && !user.email.empty()) {
                    initials = std::string(1, static_cast<char>(std::toupper(static_cast<unsigned char>(user.email[0]))));
                }
                if (initials.empty()) initials = "U";

                // Header with user avatar placeholder
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                ImGui::Text(ICON_FA_USER "  Account Settings");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                // User card with better info
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##UserCard", ImVec2(340, 90), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 12));

                // Avatar circle with initials
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float avatar_radius = 28.0f;
                ImVec2 avatar_center(pos.x + avatar_radius, pos.y + avatar_radius);
                draw_list->AddCircleFilled(avatar_center, avatar_radius, IM_COL32(40, 80, 160, 255));

                // Draw initials centered
                float font_scale = 1.4f;
                ImVec2 text_size = ImGui::CalcTextSize(initials.c_str());
                ImVec2 text_pos(avatar_center.x - text_size.x * font_scale * 0.5f,
                              avatar_center.y - text_size.y * font_scale * 0.5f);
                draw_list->AddText(ImGui::GetFont(), ImGui::GetFontSize() * font_scale,
                                  text_pos, IM_COL32(255, 255, 255, 255), initials.c_str());

                // User info next to avatar
                ImGui::SetCursorPos(ImVec2(72, 12));

                // Name (or username if no name)
                std::string display_name = user.name.empty() ? user.username : user.name;
                if (display_name.empty()) display_name = "User";
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                ImGui::Text("%s", display_name.c_str());
                ImGui::PopStyleColor();

                // Email
                ImGui::SetCursorPos(ImVec2(72, 32));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
                ImGui::Text("%s", user.email.c_str());
                ImGui::PopStyleColor();

                // Role badge
                ImGui::SetCursorPos(ImVec2(72, 54));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
                ImGui::Text(ICON_FA_CIRCLE_CHECK);
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
                std::string role_display = user.role.empty() ? "User" : user.role;
                role_display[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(role_display[0])));
                ImGui::Text("%s", role_display.c_str());
                ImGui::PopStyleColor();

                ImGui::EndChild();
                ImGui::PopStyleColor();

                ImGui::Spacing();
                ImGui::Spacing();

                // Wallet Section
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
                ImGui::Text(ICON_FA_WALLET "  WALLET");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##WalletCard", ImVec2(340, 70), true, ImGuiWindowFlags_NoScrollbar);

                if (!user.wallet_address.empty()) {
                    // Show connected CyxWallet
                    ImGui::SetCursorPos(ImVec2(12, 10));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.7f, 0.4f, 1.0f));
                    ImGui::Text(ICON_FA_CIRCLE_CHECK " CyxWallet");
                    ImGui::PopStyleColor();

                    // Truncate wallet address for display
                    ImGui::SameLine();
                    std::string wallet_display = user.wallet_address;
                    if (wallet_display.length() > 20) {
                        wallet_display = wallet_display.substr(0, 8) + "..." + wallet_display.substr(wallet_display.length() - 6);
                    }
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.55f, 1.0f));
                    ImGui::Text("%s", wallet_display.c_str());
                    ImGui::PopStyleColor();

                    // Copy button
                    ImGui::SameLine();
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                    if (ImGui::Button(ICON_FA_COPY "##copy_wallet")) {
                        ImGui::SetClipboardText(user.wallet_address.c_str());
                        spdlog::info("Wallet address copied to clipboard");
                    }
                    ImGui::PopStyleColor(2);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Copy to clipboard");
                    }

                    // Link External Wallet - subtle text link
                    ImGui::SetCursorPos(ImVec2(12, 38));
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.55f, 0.8f, 1.0f));
                    if (ImGui::SmallButton(ICON_FA_LINK " Link external wallet")) {
                        show_wallet_connect_dialog_ = true;
                        show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    on_login_success_callback_(auth::AuthClient::Instance().GetJwtToken());
                }
                        wallet_connect_step_ = 0;
                        memset(wallet_address_buffer_, 0, sizeof(wallet_address_buffer_));
                        memset(wallet_signature_buffer_, 0, sizeof(wallet_signature_buffer_));
                        wallet_nonce_.clear();
                        wallet_error_message_.clear();
                        spdlog::info("Connect external wallet dialog opened");
                    }
                    ImGui::PopStyleColor(4);
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Connect Phantom or other Solana wallet");
                    }
                } else {
                    ImGui::SetCursorPos(ImVec2(12, 12));
                    ImGui::TextDisabled("No wallet connected");
                    ImGui::SetCursorPos(ImVec2(12, 35));

                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.18f, 0.22f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
                    if (ImGui::Button(ICON_FA_WALLET " Connect Wallet", ImVec2(140, 26))) {
                        show_wallet_connect_dialog_ = true;
                        show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    on_login_success_callback_(auth::AuthClient::Instance().GetJwtToken());
                }
                        wallet_connect_step_ = 0;
                        memset(wallet_address_buffer_, 0, sizeof(wallet_address_buffer_));
                        memset(wallet_signature_buffer_, 0, sizeof(wallet_signature_buffer_));
                        wallet_nonce_.clear();
                        wallet_error_message_.clear();
                        spdlog::info("Connect wallet dialog opened");
                    }
                    ImGui::PopStyleColor(2);
                }

                ImGui::EndChild();
                ImGui::PopStyleColor();

                ImGui::Spacing();
                ImGui::Spacing();

                // Server Section
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.75f, 1.0f));
                ImGui::Text(ICON_FA_SERVER "  SERVER");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##ServerCard", ImVec2(340, 65), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 10));
                ImGui::Text("Default Server");
                ImGui::SetCursorPos(ImVec2(12, 32));

                static char server_address[256] = "";
                static bool server_address_initialized = false;
                if (!server_address_initialized) {
                    std::strncpy(server_address, core::EngineConfig::Instance().GetCentralServerAddress().c_str(), sizeof(server_address) - 1);
                    server_address[sizeof(server_address) - 1] = '\0';
                    server_address_initialized = true;
                }
                ImGui::SetNextItemWidth(316);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.10f, 1.0f));
                ImGui::InputText("##server", server_address, sizeof(server_address));
                ImGui::PopStyleColor();

                ImGui::EndChild();
                ImGui::PopStyleColor();

                ImGui::Spacing();
                ImGui::Spacing();
                ImGui::Spacing();

                // Sign Out button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.18f, 0.18f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.25f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.45f, 0.15f, 0.15f, 1.0f));
                if (ImGui::Button("Sign Out", ImVec2(340, 36))) {
                    auth.Logout();
                    is_logged_in_ = false;
                    logged_in_user_.clear();
                    memset(login_identifier_, 0, sizeof(login_identifier_));
                    memset(login_password_, 0, sizeof(login_password_));
                    login_error_message_.clear();
                    login_success_message_.clear();
                    spdlog::info("User signed out");
                    // Notify application of logout
                    if (on_logout_callback_) {
                        on_logout_callback_();
                    }
                }
                ImGui::PopStyleColor(3);
            }

            ImGui::Spacing();

            // Close button (subtle, bottom right)
            float close_width = 70;
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - close_width - 24);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            if (ImGui::Button("Close", ImVec2(close_width, 28))) {
                show_account_settings_dialog_ = false;
                if (on_login_success_callback_) {
                    auto& auth = auth::AuthClient::Instance();
                    on_login_success_callback_(auth.GetJwtToken());
                }
            }
            ImGui::PopStyleColor(3);

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(5);
    }

    // Wallet Connect Dialog
    if (show_wallet_connect_dialog_) {
        ImGui::OpenPopup("##WalletConnect");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(24, 24));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12, 8));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 12));

        ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;

        if (ImGui::BeginPopupModal("##WalletConnect", &show_wallet_connect_dialog_, flags)) {
            // Check for async operation results
            if (wallet_nonce_future_.valid() &&
                wallet_nonce_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                auto result = wallet_nonce_future_.get();
                if (result.success) {
                    wallet_nonce_ = result.nonce;
                    wallet_sign_message_ = result.message;
                    wallet_connect_step_ = 1;  // Move to sign step
                    spdlog::info("Got wallet nonce, ready for signing");
                } else {
                    wallet_error_message_ = result.error;
                    wallet_connect_step_ = 0;  // Back to address entry
                }
            }

            if (wallet_link_future_.valid() &&
                wallet_link_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                auto result = wallet_link_future_.get();
                if (result.success) {
                    spdlog::info("Wallet login successful: {}", result.wallet_address);
                    show_wallet_connect_dialog_ = false;
                    // Update logged-in state
                    is_logged_in_ = true;
                    auto& auth_client = auth::AuthClient::Instance();
                    auto user = auth_client.GetUserInfo();
                    logged_in_user_ = user.email.empty() ? user.wallet_address : user.email;
                } else {
                    wallet_error_message_ = result.error;
                    wallet_connect_step_ = 1;  // Stay on sign step
                }
            }

            // Header
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
            ImGui::Text(ICON_FA_WALLET);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Text("Connect Solana Wallet");

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            float input_width = 380.0f;

            if (wallet_connect_step_ == 0) {
                // Step 1: Enter wallet address
                ImGui::TextWrapped("Enter your Solana wallet address to connect:");
                ImGui::Spacing();

                ImGui::Text("Wallet Address");
                ImGui::SetNextItemWidth(input_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::InputText("##wallet_address", wallet_address_buffer_, sizeof(wallet_address_buffer_));
                ImGui::PopStyleColor();

                if (!wallet_error_message_.empty()) {
                    ImGui::Spacing();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui::TextWrapped("%s", wallet_error_message_.c_str());
                    ImGui::PopStyleColor();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                // Get Nonce button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
                bool can_get_nonce = strlen(wallet_address_buffer_) > 30;  // Basic validation
                if (!can_get_nonce) ImGui::BeginDisabled();
                if (ImGui::Button("Get Signing Message", ImVec2(input_width, 36))) {
                    wallet_error_message_.clear();
                    auto& auth_client = auth::AuthClient::Instance();
                    wallet_nonce_future_ = auth_client.GetWalletNonce(wallet_address_buffer_);
                    spdlog::info("Requesting nonce for wallet: {}", wallet_address_buffer_);
                }
                if (!can_get_nonce) ImGui::EndDisabled();
                ImGui::PopStyleColor(2);

            } else if (wallet_connect_step_ == 1) {
                // Step 2: Sign the message
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.5f, 1.0f));
                ImGui::Text("Step 1:");
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::TextWrapped("Copy this message and sign it in your Phantom wallet");
                ImGui::Spacing();

                // Show the message to sign from server
                ImGui::Text("Message to Sign:");
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.08f, 0.08f, 0.10f, 1.0f));
                ImGui::InputTextMultiline("##sign_message", const_cast<char*>(wallet_sign_message_.c_str()),
                    wallet_sign_message_.size() + 1, ImVec2(input_width, 80),
                    ImGuiInputTextFlags_ReadOnly);
                ImGui::PopStyleColor();

                // Copy button
                if (ImGui::Button(ICON_FA_COPY " Copy Message")) {
                    ImGui::SetClipboardText(wallet_sign_message_.c_str());
                    spdlog::info("Sign message copied to clipboard");
                    spdlog::info("Copied message: {}", wallet_sign_message_);
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.5f, 1.0f));
                ImGui::Text("Step 2:");
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::TextWrapped("Paste the SIGNATURE from Phantom (not the message!)");

                ImGui::Spacing();
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 0.6f, 0.65f, 1.0f));
                ImGui::TextWrapped("The signature is a base58 string like: 3AhUen...");
                ImGui::PopStyleColor();

                ImGui::Text("Signature:");
                ImGui::SetNextItemWidth(input_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::InputText("##signature", wallet_signature_buffer_, sizeof(wallet_signature_buffer_));
                ImGui::PopStyleColor();

                if (!wallet_error_message_.empty()) {
                    ImGui::Spacing();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui::TextWrapped("%s", wallet_error_message_.c_str());
                    ImGui::PopStyleColor();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                // Verify button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.52f, 0.96f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.60f, 1.0f, 1.0f));
                bool can_verify = strlen(wallet_signature_buffer_) > 10;
                if (!can_verify) ImGui::BeginDisabled();
                if (ImGui::Button("Verify & Link Wallet", ImVec2(input_width, 36))) {
                    wallet_error_message_.clear();
                    auto& auth_client = auth::AuthClient::Instance();
                    // Debug logging
                    spdlog::info("=== Wallet Login Debug ===");
                    spdlog::info("Wallet Address: {}", wallet_address_buffer_);
                    spdlog::info("Nonce: {}", wallet_nonce_);
                    spdlog::info("Signature (first 50 chars): {}", std::string(wallet_signature_buffer_).substr(0, 50));
                    spdlog::info("Signature length: {}", strlen(wallet_signature_buffer_));
                    spdlog::info("Message to sign was: {}", wallet_sign_message_);
                    wallet_link_future_ = auth_client.LinkWallet(
                        wallet_address_buffer_, wallet_signature_buffer_, wallet_nonce_);
                    wallet_connect_step_ = 2;  // Show verifying state
                    spdlog::info("Verifying wallet signature...");
                }
                if (!can_verify) ImGui::EndDisabled();
                ImGui::PopStyleColor(2);

                // Back button
                ImGui::SameLine();
                if (ImGui::Button("Back")) {
                    wallet_connect_step_ = 0;
                    wallet_error_message_.clear();
                }

            } else if (wallet_connect_step_ == 2) {
                // Verifying...
                ImGui::TextWrapped("Verifying signature...");
                ImGui::Spacing();
                ImGui::ProgressBar(-1.0f * ImGui::GetTime(), ImVec2(input_width, 4));
            }

            ImGui::Spacing();

            // Cancel button
            float cancel_width = 80;
            ImGui::SetCursorPosX(ImGui::GetWindowWidth() - cancel_width - 24);
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            if (ImGui::Button("Cancel", ImVec2(cancel_width, 28))) {
                show_wallet_connect_dialog_ = false;
            }
            ImGui::PopStyleColor(3);

            ImGui::EndPopup();
        }

        ImGui::PopStyleVar(5);
    }

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

    // ========== Find Dialog ==========
    if (show_find_dialog_) {
        ImGui::OpenPopup("Find");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Find", &show_find_dialog_, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Find in current editor:");
            ImGui::Spacing();

            // Search text input
            ImGui::SetNextItemWidth(-1);
            bool enter_pressed = ImGui::InputText("##findtext", find_text_buffer_, sizeof(find_text_buffer_),
                ImGuiInputTextFlags_EnterReturnsTrue);

            ImGui::Spacing();

            // Options
            ImGui::Checkbox("Case sensitive", &find_case_sensitive_);
            ImGui::SameLine();
            ImGui::Checkbox("Whole word", &find_whole_word_);
            ImGui::SameLine();
            ImGui::Checkbox("Regex", &find_use_regex_);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Buttons
            float button_width = 100.0f;

            if (ImGui::Button("Find Next", ImVec2(button_width, 0)) || enter_pressed) {
                if (find_callback_ && strlen(find_text_buffer_) > 0) {
                    find_callback_(find_text_buffer_, find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            ImGui::BeginDisabled();
            ImGui::Button("Find Previous", ImVec2(button_width, 0));
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Find Previous is planned");
            }
            ImGui::SameLine();
            if (ImGui::Button("Close", ImVec2(button_width, 0))) {
                show_find_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }

    // ========== Replace Dialog ==========
    if (show_replace_dialog_) {
        ImGui::OpenPopup("Replace");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Replace", &show_replace_dialog_, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Find and replace in current editor:");
            ImGui::Spacing();

            // Search text input
            ImGui::Text("Find:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##findtext_replace", find_text_buffer_, sizeof(find_text_buffer_));

            ImGui::Spacing();

            // Replace text input
            ImGui::Text("Replace with:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##replacetext", replace_text_buffer_, sizeof(replace_text_buffer_));

            ImGui::Spacing();

            // Options
            ImGui::Checkbox("Case sensitive##replace", &find_case_sensitive_);
            ImGui::SameLine();
            ImGui::Checkbox("Whole word##replace", &find_whole_word_);
            ImGui::SameLine();
            ImGui::Checkbox("Regex##replace", &find_use_regex_);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Buttons
            float button_width = 90.0f;

            if (ImGui::Button("Find Next", ImVec2(button_width, 0))) {
                if (find_callback_ && strlen(find_text_buffer_) > 0) {
                    find_callback_(find_text_buffer_, find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Replace", ImVec2(button_width, 0))) {
                if (replace_callback_ && strlen(find_text_buffer_) > 0) {
                    replace_callback_(find_text_buffer_, replace_text_buffer_, find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Replace All", ImVec2(button_width, 0))) {
                if (replace_all_callback_ && strlen(find_text_buffer_) > 0) {
                    replace_all_callback_(find_text_buffer_, replace_text_buffer_, find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Close", ImVec2(button_width, 0))) {
                show_replace_dialog_ = false;
            }

            ImGui::EndPopup();
        }
    }

    // ========== Find in Files Dialog ==========
    if (show_find_in_files_dialog_) {
        ImGui::OpenPopup("Find in Files");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_Appearing);

        if (ImGui::BeginPopupModal("Find in Files", &show_find_in_files_dialog_, ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Search across project files:");
            ImGui::Spacing();

            // Search text input
            ImGui::Text("Search for:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##findtext_files", find_text_buffer_, sizeof(find_text_buffer_));

            ImGui::Spacing();

            // File pattern
            ImGui::Text("File patterns (e.g., *.py;*.cyx):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##filepattern", find_in_files_pattern_, sizeof(find_in_files_pattern_));

            ImGui::Spacing();

            // Search path
            ImGui::Text("Search in:");
            ImGui::SetNextItemWidth(-70);
            ImGui::InputText("##searchpath", find_in_files_path_, sizeof(find_in_files_path_));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##findinfiles")) {
                std::string selected_folder = OpenFolderDialog();
                if (!selected_folder.empty()) {
                    strncpy(find_in_files_path_, selected_folder.c_str(), sizeof(find_in_files_path_) - 1);
                    find_in_files_path_[sizeof(find_in_files_path_) - 1] = '\0';
                }
            }

            ImGui::Spacing();

            // Options
            ImGui::Checkbox("Case sensitive##files", &find_case_sensitive_);
            ImGui::SameLine();
            ImGui::Checkbox("Whole word##files", &find_whole_word_);
            ImGui::SameLine();
            ImGui::Checkbox("Regex##files", &find_use_regex_);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Search button
            if (ImGui::Button("Search", ImVec2(100, 0))) {
                if (strlen(find_text_buffer_) > 0 && strlen(find_in_files_path_) > 0) {
                    SearchInFiles(find_text_buffer_, find_in_files_path_, find_in_files_pattern_,
                                  find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Close", ImVec2(100, 0))) {
                show_find_in_files_dialog_ = false;
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Results area
            if (search_results_.empty()) {
                ImGui::Text("Results:");
            } else {
                ImGui::Text("Results: %zu matches", search_results_.size());
            }
            ImGui::BeginChild("##searchresults", ImVec2(-1, -1), true);

            if (search_in_progress_) {
                ImGui::TextDisabled("Searching...");
            } else if (search_results_.empty()) {
                ImGui::TextDisabled("No results. Enter search text and click Search.");
            } else {
                std::string current_file;
                for (const auto& result : search_results_) {
                    // Group by file
                    if (result.file_path != current_file) {
                        current_file = result.file_path;
                        ImGui::Spacing();
                        // Display relative path if in project
                        std::filesystem::path file_path(result.file_path);
                        std::string display_path = file_path.filename().string();
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), ICON_FA_FILE_CODE " %s", display_path.c_str());
                        ImGui::SameLine();
                        ImGui::TextDisabled("(%s)", result.file_path.c_str());
                    }

                    // Display line with clickable result
                    ImGui::Indent(20.0f);
                    std::string label = std::to_string(result.line_number) + ": " + result.line_content;
                    if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_None)) {
                        // Open file at line
                        if (open_script_in_editor_callback_) {
                            open_script_in_editor_callback_(result.file_path);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Click to open file at line %d", result.line_number);
                    }
                    ImGui::Unindent(20.0f);
                }
            }

            ImGui::EndChild();

            ImGui::EndPopup();
        }
    }

    // ========== Replace in Files Dialog ==========
    if (show_replace_in_files_dialog_) {
        ImGui::OpenPopup("Replace in Files");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(550, 450), ImGuiCond_Appearing);

        if (ImGui::BeginPopupModal("Replace in Files", &show_replace_in_files_dialog_, ImGuiWindowFlags_NoMove)) {
            ImGui::Text("Find and replace across project files:");
            ImGui::Spacing();

            // Search text input
            ImGui::Text("Find:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##findtext_replacefiles", find_text_buffer_, sizeof(find_text_buffer_));

            ImGui::Spacing();

            // Replace text input
            ImGui::Text("Replace with:");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##replacetext_files", replace_text_buffer_, sizeof(replace_text_buffer_));

            ImGui::Spacing();

            // File pattern
            ImGui::Text("File patterns (e.g., *.py;*.cyx):");
            ImGui::SetNextItemWidth(-1);
            ImGui::InputText("##filepattern_replace", find_in_files_pattern_, sizeof(find_in_files_pattern_));

            ImGui::Spacing();

            // Search path
            ImGui::Text("Search in:");
            ImGui::SetNextItemWidth(-70);
            ImGui::InputText("##searchpath_replace", find_in_files_path_, sizeof(find_in_files_path_));
            ImGui::SameLine();
            if (ImGui::Button("Browse...##replaceinfiles")) {
                std::string selected_folder = OpenFolderDialog();
                if (!selected_folder.empty()) {
                    strncpy(find_in_files_path_, selected_folder.c_str(), sizeof(find_in_files_path_) - 1);
                    find_in_files_path_[sizeof(find_in_files_path_) - 1] = '\0';
                }
            }

            ImGui::Spacing();

            // Options
            ImGui::Checkbox("Case sensitive##replacefiles", &find_case_sensitive_);
            ImGui::SameLine();
            ImGui::Checkbox("Whole word##replacefiles", &find_whole_word_);
            ImGui::SameLine();
            ImGui::Checkbox("Regex##replacefiles", &find_use_regex_);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Action buttons
            if (ImGui::Button("Find All", ImVec2(100, 0))) {
                if (strlen(find_text_buffer_) > 0 && strlen(find_in_files_path_) > 0) {
                    SearchInFiles(find_text_buffer_, find_in_files_path_, find_in_files_pattern_,
                                  find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
            }
            ImGui::SameLine();
            ImGui::BeginDisabled();
            ImGui::Button("Replace All", ImVec2(100, 0));
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Replace in files is planned");
            }
            ImGui::SameLine();
            if (ImGui::Button("Close", ImVec2(100, 0))) {
                show_replace_in_files_dialog_ = false;
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // Results area
            if (search_results_.empty()) {
                ImGui::Text("Results:");
            } else {
                ImGui::Text("Results: %zu matches", search_results_.size());
            }
            ImGui::BeginChild("##replaceresults", ImVec2(-1, -1), true);

            if (search_in_progress_) {
                ImGui::TextDisabled("Searching...");
            } else if (search_results_.empty()) {
                ImGui::TextDisabled("No results. Enter search text and click Find All.");
            } else {
                std::string current_file;
                for (const auto& result : search_results_) {
                    // Group by file
                    if (result.file_path != current_file) {
                        current_file = result.file_path;
                        ImGui::Spacing();
                        std::filesystem::path file_path(result.file_path);
                        std::string display_path = file_path.filename().string();
                        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), ICON_FA_FILE_CODE " %s", display_path.c_str());
                        ImGui::SameLine();
                        ImGui::TextDisabled("(%s)", result.file_path.c_str());
                    }

                    // Display line with clickable result
                    ImGui::Indent(20.0f);
                    std::string label = std::to_string(result.line_number) + ": " + result.line_content;
                    if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_None)) {
                        if (open_script_in_editor_callback_) {
                            open_script_in_editor_callback_(result.file_path);
                        }
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::SetTooltip("Click to open file at line %d", result.line_number);
                    }
                    ImGui::Unindent(20.0f);
                }
            }

            ImGui::EndChild();

            ImGui::EndPopup();
        }
    }

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
