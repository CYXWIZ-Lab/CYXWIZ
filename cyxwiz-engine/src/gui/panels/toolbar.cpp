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
#include <regex>
#include <sstream>
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

void ToolbarPanel::SetEditorFontScale(float scale) {
    // Convert font scale to font size index for UI
    // 1.0 -> 10, 1.3 -> 13, 1.6 -> 16, 2.0 -> 20
    if (scale <= 1.15f) editor_font_size_ = 10;
    else if (scale <= 1.45f) editor_font_size_ = 13;
    else if (scale <= 1.8f) editor_font_size_ = 16;
    else editor_font_size_ = 20;
}

// Helper function to check if a file matches any of the given patterns
static bool MatchesFilePattern(const std::string& filename, const std::string& patterns) {
    if (patterns.empty()) return true;

    // Split patterns by semicolon
    std::vector<std::string> pattern_list;
    std::stringstream ss(patterns);
    std::string pattern;
    while (std::getline(ss, pattern, ';')) {
        // Trim whitespace
        size_t start = pattern.find_first_not_of(" \t");
        size_t end = pattern.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            pattern_list.push_back(pattern.substr(start, end - start + 1));
        }
    }

    // Check if filename matches any pattern
    for (const auto& pat : pattern_list) {
        // Convert glob pattern to regex
        std::string regex_pattern;
        for (char c : pat) {
            switch (c) {
                case '*': regex_pattern += ".*"; break;
                case '?': regex_pattern += "."; break;
                case '.': regex_pattern += "\\."; break;
                default: regex_pattern += c; break;
            }
        }
        regex_pattern = "^" + regex_pattern + "$";

        try {
            std::regex re(regex_pattern, std::regex::icase);
            if (std::regex_match(filename, re)) {
                return true;
            }
        } catch (const std::regex_error&) {
            // If regex fails, try simple extension match
            if (pat.length() > 1 && pat[0] == '*') {
                std::string ext = pat.substr(1);
                if (filename.length() >= ext.length() &&
                    filename.substr(filename.length() - ext.length()) == ext) {
                    return true;
                }
            }
        }
    }

    return false;
}

// Helper function to search in a single line
static bool SearchInLine(const std::string& line, const std::string& search_text,
                         bool case_sensitive, bool whole_word, bool use_regex,
                         int& match_start, int& match_length) {
    if (search_text.empty()) return false;

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);
            std::smatch match;
            if (std::regex_search(line, match, re)) {
                match_start = static_cast<int>(match.position(0));
                match_length = static_cast<int>(match.length(0));
                return true;
            }
        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex pattern: {}", e.what());
            return false;
        }
    } else {
        std::string search_line = line;
        std::string search_term = search_text;

        if (!case_sensitive) {
            std::transform(search_line.begin(), search_line.end(), search_line.begin(), ::tolower);
            std::transform(search_term.begin(), search_term.end(), search_term.begin(), ::tolower);
        }

        size_t pos = search_line.find(search_term);
        if (pos != std::string::npos) {
            if (whole_word) {
                // Check word boundaries
                bool start_ok = (pos == 0) || !std::isalnum(static_cast<unsigned char>(search_line[pos - 1]));
                bool end_ok = (pos + search_term.length() >= search_line.length()) ||
                              !std::isalnum(static_cast<unsigned char>(search_line[pos + search_term.length()]));
                if (!start_ok || !end_ok) {
                    return false;
                }
            }
            match_start = static_cast<int>(pos);
            match_length = static_cast<int>(search_term.length());
            return true;
        }
    }

    return false;
}

void ToolbarPanel::SearchInFiles(const std::string& search_text, const std::string& search_path,
                                  const std::string& file_patterns, bool case_sensitive,
                                  bool whole_word, bool use_regex) {
    search_results_.clear();
    search_in_progress_ = true;

    if (search_text.empty() || search_path.empty()) {
        search_in_progress_ = false;
        return;
    }

    namespace fs = std::filesystem;

    try {
        int files_searched = 0;
        int max_results = 1000;  // Limit results to prevent UI slowdown

        for (const auto& entry : fs::recursive_directory_iterator(search_path,
                fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            if (!MatchesFilePattern(filename, file_patterns)) continue;

            files_searched++;

            // Read file and search
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            std::string line;
            int line_number = 0;

            while (std::getline(file, line) && search_results_.size() < max_results) {
                line_number++;

                int match_start = 0, match_length = 0;
                if (SearchInLine(line, search_text, case_sensitive, whole_word, use_regex,
                                 match_start, match_length)) {
                    SearchResult result;
                    result.file_path = entry.path().string();
                    result.line_number = line_number;
                    result.line_content = line;
                    result.match_start = match_start;
                    result.match_length = match_length;

                    // Truncate line if too long
                    if (result.line_content.length() > 200) {
                        result.line_content = result.line_content.substr(0, 200) + "...";
                    }

                    search_results_.push_back(result);
                }
            }

            if (search_results_.size() >= max_results) {
                spdlog::info("Search stopped: max results ({}) reached", max_results);
                break;
            }
        }

        spdlog::info("Search complete: found {} results in {} files",
                     search_results_.size(), files_searched);

    } catch (const fs::filesystem_error& e) {
        spdlog::error("Filesystem error during search: {}", e.what());
    }

    search_in_progress_ = false;
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
            if (ImGui::Button("Find Previous", ImVec2(button_width, 0))) {
                // TODO: Find previous
                if (find_callback_ && strlen(find_text_buffer_) > 0) {
                    find_callback_(find_text_buffer_, find_case_sensitive_, find_whole_word_, find_use_regex_);
                }
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
            if (ImGui::Button("Replace All", ImVec2(100, 0))) {
                // First search for all matches
                if (strlen(find_text_buffer_) > 0 && strlen(find_in_files_path_) > 0) {
                    SearchInFiles(find_text_buffer_, find_in_files_path_, find_in_files_pattern_,
                                  find_case_sensitive_, find_whole_word_, find_use_regex_);
                    // TODO: Implement actual replace in files (requires file modification)
                    spdlog::info("Replace All: Found {} occurrences. Replace functionality not yet implemented.",
                                 search_results_.size());
                }
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
        // Initialize default shortcuts if empty
        if (shortcuts_.empty()) {
            shortcuts_ = {
                // File operations
                {"New File", "Ctrl+N", "Create a new file", false},
                {"Open File", "Ctrl+O", "Open an existing file", false},
                {"Save", "Ctrl+S", "Save current file", false},
                {"Save As", "Ctrl+Shift+S", "Save with new name", false},
                {"Close File", "Ctrl+W", "Close current file", false},
                // Edit operations
                {"Undo", "Ctrl+Z", "Undo last action", false},
                {"Redo", "Ctrl+Y", "Redo last action", false},
                {"Cut", "Ctrl+X", "Cut selection", false},
                {"Copy", "Ctrl+C", "Copy selection", false},
                {"Paste", "Ctrl+V", "Paste from clipboard", false},
                {"Select All", "Ctrl+A", "Select all text", false},
                {"Find", "Ctrl+F", "Find text", false},
                {"Replace", "Ctrl+H", "Find and replace", false},
                {"Go to Line", "Ctrl+G", "Jump to line number", false},
                // Code editing
                {"Toggle Comment", "Ctrl+/", "Toggle line comment", false},
                {"Block Comment", "Shift+Alt+A", "Toggle block comment", false},
                {"Duplicate Line", "Ctrl+D", "Duplicate current line", true},
                {"Move Line Up", "Alt+Up", "Move line up", true},
                {"Move Line Down", "Alt+Down", "Move line down", true},
                {"Indent", "Tab", "Increase indentation", false},
                {"Outdent", "Shift+Tab", "Decrease indentation", false},
                {"Join Lines", "Ctrl+J", "Join selected lines", true},
                // Script execution
                {"Run Script", "F5", "Execute current script", false},
                {"Run Selection", "Ctrl+Enter", "Execute selection", false},
                {"Stop Script", "Ctrl+Break", "Stop running script", false},
            };
        }

        ImGui::OpenPopup("Preferences");
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(650, 500), ImGuiCond_Appearing);

        if (ImGui::BeginPopupModal("Preferences", &show_preferences_dialog_, ImGuiWindowFlags_NoMove)) {
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

                    // Python Interpreter Path
                    ImGui::Text("Python Interpreter Path:");
                    ImGui::SetNextItemWidth(-100);
                    ImGui::InputText("##python_path", python_interpreter_path_, sizeof(python_interpreter_path_));
                    ImGui::SameLine();
                    if (ImGui::Button("Browse##python")) {
                        std::string path = OpenFileDialog("Python Executable (python.exe)\0python.exe\0All Files (*.*)\0*.*\0", "Select Python Interpreter");
                        if (!path.empty()) {
                            strncpy(python_interpreter_path_, path.c_str(), sizeof(python_interpreter_path_) - 1);
                        }
                    }
                    ImGui::TextDisabled("Leave empty to use system default");

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

                    ImGui::EndTabItem();
                }

                // ========== Keyboard Shortcuts Tab ==========
                if (ImGui::BeginTabItem(ICON_FA_KEYBOARD " Shortcuts")) {
                    preferences_tab_ = 5;
                    ImGui::Spacing();

                    ImGui::TextDisabled("Double-click a shortcut to edit. Some shortcuts are system-level and cannot be changed.");
                    ImGui::Spacing();

                    // Table of shortcuts
                    if (ImGui::BeginTable("ShortcutsTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 280))) {
                        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 180);
                        ImGui::TableSetupColumn("Shortcut", ImGuiTableColumnFlags_WidthFixed, 150);
                        ImGui::TableSetupColumn("Description", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableHeadersRow();

                        for (int i = 0; i < static_cast<int>(shortcuts_.size()); ++i) {
                            auto& shortcut = shortcuts_[i];
                            ImGui::TableNextRow();

                            // Action column
                            ImGui::TableNextColumn();
                            ImGui::Text("%s", shortcut.action.c_str());

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

} // namespace cyxwiz
