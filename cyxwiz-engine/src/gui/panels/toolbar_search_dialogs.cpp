// Toolbar find and replace modal rendering.

#include "toolbar.h"
#include "../icons.h"

#include <cstring>
#include <filesystem>
#include <string>

#include <imgui.h>

namespace cyxwiz {
void ToolbarPanel::RenderSearchDialogs() {
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
}

} // namespace cyxwiz