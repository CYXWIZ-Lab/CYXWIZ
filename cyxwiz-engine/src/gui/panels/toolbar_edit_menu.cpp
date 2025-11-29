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

void ToolbarPanel::RenderEditMenu() {
    if (ImGui::BeginMenu("Edit")) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

        // ========== Undo/Redo Section ==========
        if (ImGui::MenuItem(ICON_FA_ROTATE_LEFT " Undo", "Ctrl+Z")) {
            if (undo_callback_) undo_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_ROTATE_RIGHT " Redo", "Ctrl+Y")) {
            if (redo_callback_) redo_callback_();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Clipboard Section ==========
        if (ImGui::MenuItem(ICON_FA_SCISSORS " Cut", "Ctrl+X")) {
            if (cut_callback_) cut_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_COPY " Copy", "Ctrl+C")) {
            if (copy_callback_) copy_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_PASTE " Paste", "Ctrl+V")) {
            if (paste_callback_) paste_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_TRASH " Delete", "Delete")) {
            if (delete_callback_) delete_callback_();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Selection Section ==========
        if (ImGui::MenuItem(ICON_FA_OBJECT_GROUP " Select All", "Ctrl+A")) {
            if (select_all_callback_) select_all_callback_();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Navigation Section ==========
        if (ImGui::MenuItem(ICON_FA_ARROW_DOWN " Go to Line...", "Ctrl+G")) {
            show_go_to_line_dialog_ = true;
            go_to_line_number_ = 1;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Line Operations Section ==========
        if (ImGui::MenuItem(ICON_FA_COPY " Duplicate Line", "Ctrl+D")) {
            if (duplicate_line_callback_) duplicate_line_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_ARROW_UP " Move Line Up", "Alt+Up")) {
            if (move_line_up_callback_) move_line_up_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_ARROW_DOWN " Move Line Down", "Alt+Down")) {
            if (move_line_down_callback_) move_line_down_callback_();
        }

        ImGui::Spacing();

        if (ImGui::MenuItem(ICON_FA_CHEVRON_RIGHT " Indent", "Tab")) {
            if (indent_callback_) indent_callback_();
        }

        if (ImGui::MenuItem(ICON_FA_CHEVRON_LEFT " Outdent", "Shift+Tab")) {
            if (outdent_callback_) outdent_callback_();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Transform Text Submenu ==========
        if (ImGui::BeginMenu(ICON_FA_WAND_MAGIC_SPARKLES " Transform Text")) {
            if (ImGui::MenuItem("UPPERCASE")) {
                if (transform_uppercase_callback_) transform_uppercase_callback_();
            }
            if (ImGui::MenuItem("lowercase")) {
                if (transform_lowercase_callback_) transform_lowercase_callback_();
            }
            if (ImGui::MenuItem("Title Case")) {
                if (transform_titlecase_callback_) transform_titlecase_callback_();
            }
            ImGui::EndMenu();
        }

        // ========== Sort/Join Lines ==========
        if (ImGui::BeginMenu(ICON_FA_SORT " Sort Lines")) {
            if (ImGui::MenuItem(ICON_FA_SORT_UP " Ascending (A-Z)")) {
                if (sort_lines_asc_callback_) sort_lines_asc_callback_();
            }
            if (ImGui::MenuItem(ICON_FA_SORT_DOWN " Descending (Z-A)")) {
                if (sort_lines_desc_callback_) sort_lines_desc_callback_();
            }
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem(ICON_FA_LINK " Join Lines", "Ctrl+J")) {
            if (join_lines_callback_) join_lines_callback_();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Find/Replace Section ==========
        if (ImGui::MenuItem(ICON_FA_MAGNIFYING_GLASS " Find...", "Ctrl+F")) {
            show_find_dialog_ = true;
        }

        if (ImGui::MenuItem(ICON_FA_RIGHT_LEFT " Replace...", "Ctrl+H")) {
            show_replace_dialog_ = true;
        }

        ImGui::Spacing();

        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Find in Files...", "Ctrl+Shift+F")) {
            show_find_in_files_dialog_ = true;
            // Pre-fill with project path if available
            auto& pm = ProjectManager::Instance();
            if (pm.HasActiveProject()) {
                strncpy(find_in_files_path_, pm.GetProjectRoot().c_str(), sizeof(find_in_files_path_) - 1);
                find_in_files_path_[sizeof(find_in_files_path_) - 1] = '\0';
            }
        }

        if (ImGui::MenuItem(ICON_FA_FOLDER_OPEN " Replace in Files...", "Ctrl+Shift+H")) {
            show_replace_in_files_dialog_ = true;
            // Pre-fill with project path if available
            auto& pm = ProjectManager::Instance();
            if (pm.HasActiveProject()) {
                strncpy(find_in_files_path_, pm.GetProjectRoot().c_str(), sizeof(find_in_files_path_) - 1);
                find_in_files_path_[sizeof(find_in_files_path_) - 1] = '\0';
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Comment Section ==========
        if (ImGui::MenuItem(ICON_FA_COMMENT " Toggle Line Comment", "Ctrl+/")) {
            if (toggle_line_comment_callback_) {
                toggle_line_comment_callback_();
            }
        }

        if (ImGui::MenuItem(ICON_FA_COMMENTS " Toggle Block Comment", "Shift+Alt+A")) {
            if (toggle_block_comment_callback_) {
                toggle_block_comment_callback_();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // ========== Preferences ==========
        if (ImGui::MenuItem(ICON_FA_GEAR " Preferences...")) {
            show_preferences_dialog_ = true;
            // Initialize shortcuts list if empty
            if (shortcuts_.empty()) {
                shortcuts_ = {
                    // File operations
                    {"New File", "Ctrl+N", "Create new file", false},
                    {"Open File", "Ctrl+O", "Open existing file", false},
                    {"Save", "Ctrl+S", "Save current file", false},
                    // Edit operations
                    {"Undo", "Ctrl+Z", "Undo last action", false},
                    {"Redo", "Ctrl+Y", "Redo last undone action", false},
                    {"Cut", "Ctrl+X", "Cut selection to clipboard", false},
                    {"Copy", "Ctrl+C", "Copy selection to clipboard", false},
                    {"Paste", "Ctrl+V", "Paste from clipboard", false},
                    {"Select All", "Ctrl+A", "Select all text", false},
                    // Navigation
                    {"Go to Line", "Ctrl+G", "Jump to line number", true},
                    // Line operations
                    {"Duplicate Line", "Ctrl+D", "Duplicate current line", true},
                    {"Move Line Up", "Alt+Up", "Move line up", true},
                    {"Move Line Down", "Alt+Down", "Move line down", true},
                    {"Join Lines", "Ctrl+J", "Join selected lines", true},
                    // Find/Replace
                    {"Find", "Ctrl+F", "Open Find dialog", true},
                    {"Replace", "Ctrl+H", "Open Replace dialog", true},
                    {"Find in Files", "Ctrl+Shift+F", "Search across project files", true},
                    {"Replace in Files", "Ctrl+Shift+H", "Replace across project files", true},
                    // Comments
                    {"Toggle Line Comment", "Ctrl+/", "Comment/uncomment current line", true},
                    {"Toggle Block Comment", "Shift+Alt+A", "Add/remove block comment", true},
                    // Script execution
                    {"Run Script", "F5", "Execute current script", true},
                    {"Stop Script", "Shift+F5", "Stop running script", true},
                };
            }
        }

        ImGui::PopStyleVar();
        ImGui::EndMenu();
    }
}

} // namespace cyxwiz
