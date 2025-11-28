#include "toolbar.h"
#include "plot_window.h"
#include "../theme.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cctype>
#include "../dock_style.h"
#include "../../core/project_manager.h"
#include "../icons.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

ToolbarPanel::ToolbarPanel()
    : Panel("Toolbar", true)
    , show_new_project_dialog_(false)
    , show_about_dialog_(false)
{
    memset(project_name_buffer_, 0, sizeof(project_name_buffer_));
    memset(project_path_buffer_, 0, sizeof(project_path_buffer_));
}

void ToolbarPanel::Render() {
    if (!visible_) return;

    if (ImGui::BeginMainMenuBar()) {
        RenderFileMenu();
        RenderEditMenu();
        RenderViewMenu();
        RenderNodesMenu();
        RenderTrainMenu();
        RenderDatasetMenu();
        RenderScriptMenu();
        RenderPlotsMenu();
        RenderDeployMenu();
        RenderHelpMenu();

        // Show current project name in menu bar if active
        auto& pm = ProjectManager::Instance();
        if (pm.HasActiveProject()) {
            ImGui::Separator();
            ImGui::TextDisabled("| Project: %s", pm.GetProjectName().c_str());
        }

        ImGui::EndMainMenuBar();
    }

    // Render all plot windows
    for (auto& plot_window : plot_windows_) {
        if (plot_window) {
            plot_window->Render();
        }
    }

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
                ImGui::SetNextItemWidth(input_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                ImGui::InputText("##identifier", login_identifier_, sizeof(login_identifier_));
                ImGui::PopStyleColor(2);

                ImGui::Spacing();

                // Password field
                ImGui::SetCursorPosX(start_x);
                ImGui::Text("Password");
                ImGui::SetCursorPosX(start_x);
                ImGui::SetNextItemWidth(input_width);
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
                bool enter_pressed = ImGui::InputText("##password", login_password_, sizeof(login_password_),
                    ImGuiInputTextFlags_Password | ImGuiInputTextFlags_EnterReturnsTrue);
                ImGui::PopStyleColor(2);

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

                if (ImGui::Button("Sign In", ImVec2(input_width, 38)) || enter_pressed) {
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
                        bool is_phone = !is_email && identifier.length() >= 10 &&
                            std::all_of(identifier.begin(), identifier.end(),
                                [](char c) { return std::isdigit(c) || c == '+' || c == '-' || c == ' ' || c == '(' || c == ')'; });

                        if (!is_email && !is_phone) {
                            login_error_message_ = "Please enter a valid email or phone number";
                        } else {
                            // TODO: Actual authentication API call
                            is_logged_in_ = true;
                            logged_in_user_ = identifier;
                            login_error_message_.clear();
                            spdlog::info("User signed in: {} ({})", identifier, is_email ? "email" : "phone");
                            memset(login_password_, 0, sizeof(login_password_));
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
                    spdlog::info("Create account clicked");
                }

                ImGui::PopStyleColor();

            } else {
                // ========== Logged In View ==========

                // Header with user avatar placeholder
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.9f, 0.9f, 1.0f));
                ImGui::Text(ICON_FA_USER "  Account");
                ImGui::PopStyleColor();

                ImGui::Spacing();

                // User card
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f));
                ImGui::BeginChild("##UserCard", ImVec2(-1, 60), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 12));

                // Avatar circle placeholder
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                draw_list->AddCircleFilled(ImVec2(pos.x + 18, pos.y + 18), 18,
                    IM_COL32(64, 100, 180, 255));
                draw_list->AddText(ImVec2(pos.x + 10, pos.y + 8),
                    IM_COL32(255, 255, 255, 255),
                    std::string(1, static_cast<char>(std::toupper(logged_in_user_[0]))).c_str());

                ImGui::SetCursorPos(ImVec2(52, 14));
                ImGui::Text("Signed in as");
                ImGui::SetCursorPos(ImVec2(52, 32));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.85f, 1.0f, 1.0f));
                ImGui::Text("%s", logged_in_user_.c_str());
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
                ImGui::BeginChild("##WalletCard", ImVec2(-1, 70), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 10));
                ImGui::TextDisabled("Connect your Solana wallet");
                ImGui::SetCursorPos(ImVec2(12, 30));

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.18f, 0.22f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.25f, 0.30f, 1.0f));
                if (ImGui::Button(ICON_FA_LINK " Connect Wallet", ImVec2(150, 28))) {
                    spdlog::info("Connect wallet clicked");
                }
                ImGui::PopStyleColor(2);

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
                ImGui::BeginChild("##ServerCard", ImVec2(-1, 65), true, ImGuiWindowFlags_NoScrollbar);

                ImGui::SetCursorPos(ImVec2(12, 10));
                ImGui::Text("Default Server");
                ImGui::SetCursorPos(ImVec2(12, 32));

                static char server_address[256] = "localhost:50051";
                ImGui::SetNextItemWidth(-24);
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
                if (ImGui::Button("Sign Out", ImVec2(-1, 36))) {
                    is_logged_in_ = false;
                    logged_in_user_.clear();
                    memset(login_identifier_, 0, sizeof(login_identifier_));
                    memset(login_password_, 0, sizeof(login_password_));
                    login_error_message_.clear();
                    spdlog::info("User signed out");
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
}

void ToolbarPanel::RenderFileMenu() {
    if (ImGui::BeginMenu("File")) {
        // Increase padding for menu items
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // ========== Project Section ==========
        if (ImGui::MenuItem(ICON_FA_FILE " New Project", "Ctrl+Shift+N")) {
            show_new_project_dialog_ = true;
        }

        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Open Project...", "Ctrl+Shift+O")) {
            std::string file_path = OpenFileDialog("CyxWiz Projects (*.cyxwiz)\0*.cyxwiz\0All Files (*.*)\0*.*\0", "Open Project");
            if (!file_path.empty()) {
                auto& pm = ProjectManager::Instance();
                if (pm.OpenProject(file_path)) {
                    spdlog::info("Project opened: {}", file_path);
                } else {
                    spdlog::error("Failed to open project: {}", file_path);
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
            std::string file_path = OpenFileDialog(
                "CyxWiz Scripts (*.cyx)\0*.cyx\0Python Scripts (*.py)\0*.py\0All Scripts (*.cyx;*.py)\0*.cyx;*.py\0All Files (*.*)\0*.*\0",
                "Open Script");
            if (!file_path.empty()) {
                if (open_script_in_editor_callback_) {
                    open_script_in_editor_callback_(file_path);
                }
                spdlog::info("Opening script: {}", file_path);
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
        if (ImGui::BeginMenu(ICON_FA_DOWNLOAD " Import")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
            if (ImGui::MenuItem("ONNX Model...")) {
                // TODO: Import ONNX
            }
            if (ImGui::MenuItem("PyTorch Model...")) {
                // TODO: Import PyTorch
            }
            if (ImGui::MenuItem("TensorFlow Model...")) {
                // TODO: Import TF
            }
            ImGui::PopStyleVar();
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu(ICON_FA_DOWNLOAD " Export")) {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
            if (ImGui::MenuItem("ONNX...")) {
                // TODO: Export ONNX
            }
            if (ImGui::MenuItem("GGUF...")) {
                // TODO: Export GGUF
            }
            if (ImGui::MenuItem("LoRA Adapter...")) {
                // TODO: Export LoRA
            }
            ImGui::PopStyleVar();
            ImGui::EndMenu();
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

void ToolbarPanel::RenderEditMenu() {
    if (ImGui::BeginMenu("Edit")) {
        if (ImGui::MenuItem("Undo", "Ctrl+Z")) {
            // TODO: Undo
        }

        if (ImGui::MenuItem("Redo", "Ctrl+Y")) {
            // TODO: Redo
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Cut", "Ctrl+X")) {
            // TODO: Cut
        }

        if (ImGui::MenuItem("Copy", "Ctrl+C")) {
            // TODO: Copy
        }

        if (ImGui::MenuItem("Paste", "Ctrl+V")) {
            // TODO: Paste
        }

        if (ImGui::MenuItem("Delete", "Delete")) {
            // TODO: Delete
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Select All", "Ctrl+A")) {
            // TODO: Select all
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Preferences...")) {
            // TODO: Open preferences
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderViewMenu() {
    if (ImGui::BeginMenu("View")) {
        // Panel visibility toggles
        // TODO: Get references to actual panels and toggle their visibility
        ImGui::MenuItem("Asset Browser", nullptr, true);
        ImGui::MenuItem("Node Editor", nullptr, true);
        ImGui::MenuItem("Properties", nullptr, true);
        ImGui::MenuItem("Console", nullptr, true);
        ImGui::MenuItem("Training Dashboard", nullptr, true);
        ImGui::MenuItem("Viewport (Profiler)", nullptr, true);
        ImGui::MenuItem("Wallet", nullptr, true);

        ImGui::Separator();

        if (ImGui::BeginMenu("Layout")) {
            if (ImGui::MenuItem("Reset to Default Layout")) {
                // Call the reset layout callback if set
                if (reset_layout_callback_) {
                    reset_layout_callback_();
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Save Layout...")) {
                // TODO: Save current layout to file
            }
            if (ImGui::MenuItem("Load Layout...")) {
                // TODO: Load saved layout from file
            }
            ImGui::EndMenu();
        }

        // Theme selector
        if (ImGui::BeginMenu("Theme")) {
            auto& theme = gui::GetTheme();
            auto current_preset = theme.GetCurrentPreset();

            for (auto preset : gui::Theme::GetAvailablePresets()) {
                bool is_selected = (current_preset == preset);
                if (ImGui::MenuItem(gui::Theme::GetPresetName(preset), nullptr, is_selected)) {
                    theme.ApplyPreset(preset);
                    spdlog::info("Theme changed to: {}", gui::Theme::GetPresetName(preset));
                }
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Fullscreen", "F11")) {
            // TODO: Toggle fullscreen mode
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderNodesMenu() {
    if (ImGui::BeginMenu("Nodes")) {
        if (ImGui::BeginMenu("Add Layer")) {
            if (ImGui::MenuItem("Dense/Linear")) {
                // TODO: Add dense layer
            }
            if (ImGui::MenuItem("Convolutional")) {
                // TODO: Add conv layer
            }
            if (ImGui::MenuItem("Pooling")) {
                // TODO: Add pooling layer
            }
            if (ImGui::MenuItem("Dropout")) {
                // TODO: Add dropout
            }
            if (ImGui::MenuItem("Batch Normalization")) {
                // TODO: Add batch norm
            }
            if (ImGui::MenuItem("Attention")) {
                // TODO: Add attention
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Group Selected", "Ctrl+G")) {
            // TODO: Group nodes
        }

        if (ImGui::MenuItem("Ungroup", "Ctrl+Shift+G")) {
            // TODO: Ungroup nodes
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
            // TODO: Duplicate selected
        }

        if (ImGui::MenuItem("Delete Selected", "Delete")) {
            // TODO: Delete selected nodes
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderTrainMenu() {
    if (ImGui::BeginMenu("Train")) {
        if (ImGui::MenuItem("Start Training", "F5")) {
            // TODO: Start training
        }

        if (ImGui::MenuItem("Pause", "F6")) {
            // TODO: Pause training
        }

        if (ImGui::MenuItem("Stop", "Shift+F5")) {
            // TODO: Stop training
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Training Settings...")) {
            // TODO: Open training settings
        }

        if (ImGui::MenuItem("Optimizer Settings...")) {
            // TODO: Open optimizer settings
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Resume from Checkpoint...")) {
            // TODO: Resume training
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDatasetMenu() {
    if (ImGui::BeginMenu("Dataset")) {
        if (ImGui::MenuItem("Import Dataset...")) {
            // TODO: Import dataset
        }

        if (ImGui::MenuItem("Create Custom Dataset...")) {
            // TODO: Create dataset
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Preprocess...")) {
            // TODO: Preprocess dataset
        }

        if (ImGui::MenuItem("Tokenize...")) {
            // TODO: Tokenize dataset
        }

        if (ImGui::MenuItem("Augment...")) {
            // TODO: Data augmentation
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Dataset Statistics")) {
            // TODO: Show statistics
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderScriptMenu() {
    if (ImGui::BeginMenu("Script")) {
        if (ImGui::MenuItem("Open Python Console", "F12")) {
            // TODO: Show Python console
        }

        ImGui::Separator();

        if (ImGui::MenuItem("New Script...")) {
            // TODO: Create new script
        }

        if (ImGui::MenuItem("Run Script...", "Ctrl+R")) {
            // TODO: Run script
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Script Editor...")) {
            // TODO: Open script editor
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderPlotsMenu() {
    if (ImGui::BeginMenu("Plots")) {
        // Test Control - Interactive option
        if (ImGui::MenuItem("Test Control")) {
            if (toggle_plot_test_control_callback_) {
                toggle_plot_test_control_callback_();
            }
        }

        ImGui::Separator();
        ImGui::TextDisabled("Available Plot Types (View Only)");
        ImGui::Separator();

        // Basic 2D Plots (from cheatsheet)
        if (ImGui::BeginMenu("Basic 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot() - Line plot");
        ImGui::TextDisabled("scatter() - Scatter plot");
        ImGui::TextDisabled("bar() / barh() - Bar chart");
        ImGui::TextDisabled("imshow() - Image display");
        ImGui::TextDisabled("contour() / contourf() - Contour plot");
        ImGui::TextDisabled("pcolormesh() - Pseudocolor plot");
        ImGui::TextDisabled("quiver() - Vector field");
        ImGui::TextDisabled("pie() - Pie chart");
        ImGui::TextDisabled("fill_between() - Filled area");
        ImGui::Unindent();

        ImGui::Separator();

        // Advanced 2D Plots
        if (ImGui::BeginMenu("Advanced 2D", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("step() - Step plot");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("errorbar() - Error bar plot");
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("barbs() - Barbs plot");
        ImGui::TextDisabled("eventplot() - Event plot");
        ImGui::TextDisabled("hexbin() - Hexagonal binning");
        ImGui::Unindent();

        ImGui::Separator();

        // 3D Plots
        if (ImGui::BeginMenu("3D Plots", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("plot3D() - 3D line plot");
        ImGui::TextDisabled("scatter3D() - 3D scatter");
        ImGui::TextDisabled("plot_surface() - Surface plot");
        ImGui::TextDisabled("plot_wireframe() - Wireframe");
        ImGui::TextDisabled("contour3D() - 3D contour");
        ImGui::Unindent();

        ImGui::Separator();

        // Polar Plots
        if (ImGui::BeginMenu("Polar", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("polar() - Polar plot");
        ImGui::Unindent();

        ImGui::Separator();

        // Statistical Plots
        if (ImGui::BeginMenu("Statistical", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("hist() - Histogram");
        ImGui::TextDisabled("boxplot() - Box plot");
        ImGui::TextDisabled("violinplot() - Violin plot");
        ImGui::TextDisabled("kde plot - Density estimation");
        ImGui::Unindent();

        ImGui::Separator();

        // Specialized Plots
        if (ImGui::BeginMenu("Specialized", false)) { ImGui::EndMenu(); }
        ImGui::Indent();
        ImGui::TextDisabled("heatmap - Heat map");
        ImGui::TextDisabled("streamplot() - Stream plot");
        ImGui::TextDisabled("specgram() - Spectrogram");
        ImGui::TextDisabled("spy() - Sparse matrix viz");
        ImGui::Unindent();

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderDeployMenu() {
    if (ImGui::BeginMenu("Deploy")) {
        if (ImGui::MenuItem("Connect to Server...")) {
            if (connect_to_server_callback_) {
                connect_to_server_callback_();
            }
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Export Model")) {
            if (ImGui::MenuItem("ONNX Format")) {
                // TODO: Export ONNX
            }
            if (ImGui::MenuItem("GGUF Format")) {
                // TODO: Export GGUF
            }
            if (ImGui::MenuItem("LoRA Adapter")) {
                // TODO: Export LoRA
            }
            if (ImGui::MenuItem("Safetensors")) {
                // TODO: Export safetensors
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::BeginMenu("Quantize")) {
            if (ImGui::MenuItem("INT8")) {
                // TODO: Quantize INT8
            }
            if (ImGui::MenuItem("INT4")) {
                // TODO: Quantize INT4
            }
            if (ImGui::MenuItem("FP16")) {
                // TODO: Quantize FP16
            }
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Deploy to Server Node...")) {
            // TODO: Deploy to node
        }

        if (ImGui::MenuItem("Publish to Marketplace...")) {
            // TODO: Publish model
        }

        ImGui::EndMenu();
    }
}

void ToolbarPanel::RenderHelpMenu() {
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Documentation", "F1")) {
            // TODO: Open docs
        }

        if (ImGui::MenuItem("Keyboard Shortcuts")) {
            // TODO: Show shortcuts
        }

        if (ImGui::MenuItem("API Reference")) {
            // TODO: Open API docs
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Report Issue...")) {
            // TODO: Open issue tracker
        }

        if (ImGui::MenuItem("Check for Updates...")) {
            // TODO: Check updates
        }

        ImGui::Separator();

        if (ImGui::MenuItem("About CyxWiz")) {
            show_about_dialog_ = true;
        }

        ImGui::EndMenu();
    }
}

std::string ToolbarPanel::OpenFolderDialog() {
#ifdef _WIN32
    BROWSEINFO bi = { 0 };
    bi.lpszTitle = "Select Project Location";
    bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    LPITEMIDLIST pidl = SHBrowseForFolder(&bi);
    if (pidl != nullptr) {
        char path[MAX_PATH];
        if (SHGetPathFromIDList(pidl, path)) {
            CoTaskMemFree(pidl);
            return std::string(path);
        }
        CoTaskMemFree(pidl);
    }
    return "";
#else
    // For Linux/Mac, we'd need a different implementation
    // For now, return empty string
    spdlog::warn("Folder dialog not implemented for this platform");
    return "";
#endif
}


std::string ToolbarPanel::OpenFileDialog(const char* filter, const char* title) {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char szFile[260] = { 0 };

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = filter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = nullptr;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = nullptr;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(szFile);
    }
    return "";
#else
    (void)filter;
    (void)title;
    spdlog::warn("File dialog not implemented for this platform");
    return "";
#endif
}

void ToolbarPanel::CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type) {
    // Create new plot window with auto-generated data
    auto plot_window = std::make_shared<PlotWindow>(title, type, true);
    plot_windows_.push_back(plot_window);
    spdlog::info("Created new plot window: {}", title);
}

} // namespace cyxwiz
