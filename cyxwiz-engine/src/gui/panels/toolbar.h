#pragma once

#include "../panel.h"
#include "plot_window.h"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace cyxwiz {

/**
 * Top Toolbar Panel
 * Renders main menu bar with File, Edit, View, Nodes, Train, Dataset, Script, Deploy, Plots, Help
 */
class ToolbarPanel : public Panel {
public:
    ToolbarPanel();
    ~ToolbarPanel() override = default;

    void Render() override;

    // Callbacks
    void SetResetLayoutCallback(std::function<void()> callback) { reset_layout_callback_ = callback; }
    void SetSaveLayoutCallback(std::function<void()> callback) { save_layout_callback_ = callback; }
    void SetSaveProjectSettingsCallback(std::function<void()> callback) { save_project_settings_callback_ = callback; }
    void SetTogglePlotTestControlCallback(std::function<void()> callback) { toggle_plot_test_control_callback_ = callback; }
    void SetConnectToServerCallback(std::function<void()> callback) { connect_to_server_callback_ = callback; }
    void SetImportDatasetCallback(std::function<void()> callback) { import_dataset_callback_ = callback; }
    void SetNewScriptCallback(std::function<void()> callback) { new_script_callback_ = callback; }
    void SetOpenScriptCallback(std::function<void()> callback) { open_script_callback_ = callback; }
    void SetOpenScriptInEditorCallback(std::function<void(const std::string&)> callback) { open_script_in_editor_callback_ = callback; }
    void SetSaveAllCallback(std::function<void()> callback) { save_all_callback_ = callback; }
    void SetAccountSettingsCallback(std::function<void()> callback) { account_settings_callback_ = callback; }
    void SetExitCallback(std::function<void()> callback) { exit_callback_ = callback; }
    void SetHasUnsavedChangesCallback(std::function<bool()> callback) { has_unsaved_changes_callback_ = callback; }

    // Edit menu callbacks
    void SetUndoCallback(std::function<void()> callback) { undo_callback_ = callback; }
    void SetRedoCallback(std::function<void()> callback) { redo_callback_ = callback; }
    void SetCutCallback(std::function<void()> callback) { cut_callback_ = callback; }
    void SetCopyCallback(std::function<void()> callback) { copy_callback_ = callback; }
    void SetPasteCallback(std::function<void()> callback) { paste_callback_ = callback; }
    void SetDeleteCallback(std::function<void()> callback) { delete_callback_ = callback; }
    void SetSelectAllCallback(std::function<void()> callback) { select_all_callback_ = callback; }
    void SetToggleLineCommentCallback(std::function<void()> callback) { toggle_line_comment_callback_ = callback; }
    void SetToggleBlockCommentCallback(std::function<void()> callback) { toggle_block_comment_callback_ = callback; }
    void SetFindCallback(std::function<void(const std::string&, bool, bool, bool)> callback) { find_callback_ = callback; }
    void SetFindNextCallback(std::function<void()> callback) { find_next_callback_ = callback; }
    void SetReplaceCallback(std::function<void(const std::string&, const std::string&, bool, bool, bool)> callback) { replace_callback_ = callback; }
    void SetReplaceAllCallback(std::function<void(const std::string&, const std::string&, bool, bool, bool)> callback) { replace_all_callback_ = callback; }

    // New Edit menu callbacks
    void SetGoToLineCallback(std::function<void(int)> callback) { go_to_line_callback_ = callback; }
    void SetDuplicateLineCallback(std::function<void()> callback) { duplicate_line_callback_ = callback; }
    void SetMoveLineUpCallback(std::function<void()> callback) { move_line_up_callback_ = callback; }
    void SetMoveLineDownCallback(std::function<void()> callback) { move_line_down_callback_ = callback; }
    void SetIndentCallback(std::function<void()> callback) { indent_callback_ = callback; }
    void SetOutdentCallback(std::function<void()> callback) { outdent_callback_ = callback; }
    void SetTransformUppercaseCallback(std::function<void()> callback) { transform_uppercase_callback_ = callback; }
    void SetTransformLowercaseCallback(std::function<void()> callback) { transform_lowercase_callback_ = callback; }
    void SetTransformTitleCaseCallback(std::function<void()> callback) { transform_titlecase_callback_ = callback; }
    void SetSortLinesAscCallback(std::function<void()> callback) { sort_lines_asc_callback_ = callback; }
    void SetSortLinesDescCallback(std::function<void()> callback) { sort_lines_desc_callback_ = callback; }
    void SetJoinLinesCallback(std::function<void()> callback) { join_lines_callback_ = callback; }

    // Find dialog visibility
    bool IsFindDialogOpen() const { return show_find_dialog_; }
    bool IsReplaceDialogOpen() const { return show_replace_dialog_; }
    bool IsFindInFilesDialogOpen() const { return show_find_in_files_dialog_; }
    bool IsReplaceInFilesDialogOpen() const { return show_replace_in_files_dialog_; }

    void OpenFindDialog() { show_find_dialog_ = true; }
    void OpenReplaceDialog() { show_replace_dialog_ = true; }
    void OpenFindInFilesDialog() { show_find_in_files_dialog_ = true; }
    void OpenReplaceInFilesDialog() { show_replace_in_files_dialog_ = true; }

    // Auto-save state
    bool IsAutoSaveEnabled() const { return auto_save_enabled_; }
    void SetAutoSaveEnabled(bool enabled) { auto_save_enabled_ = enabled; }

    // Editor settings callbacks (for Preferences -> Script Editor synchronization)
    void SetEditorThemeCallback(std::function<void(int)> callback) { editor_theme_callback_ = callback; }
    void SetEditorTabSizeCallback(std::function<void(int)> callback) { editor_tab_size_callback_ = callback; }
    void SetEditorFontScaleCallback(std::function<void(float)> callback) { editor_font_scale_callback_ = callback; }
    void SetEditorShowWhitespaceCallback(std::function<void(bool)> callback) { editor_show_whitespace_callback_ = callback; }
    void SetEditorWordWrapCallback(std::function<void(bool)> callback) { editor_word_wrap_callback_ = callback; }
    void SetEditorAutoIndentCallback(std::function<void(bool)> callback) { editor_auto_indent_callback_ = callback; }

    // Application theme callback (called when View -> Theme changes)
    void SetAppThemeChangedCallback(std::function<void(int)> callback) { app_theme_changed_callback_ = callback; }

    // Minimap visibility pointers (for View -> Minimaps menu)
    void SetNodeEditorMinimapPtr(bool* ptr) { node_editor_minimap_ptr_ = ptr; }
    void SetScriptEditorMinimapPtr(bool* ptr) { script_editor_minimap_ptr_ = ptr; }

    // Initialize editor settings from script editor's current values
    void SetEditorTheme(int theme) { editor_theme_ = theme; }
    void SetEditorTabSize(int size) { editor_tab_size_ = size; }
    void SetEditorFontScale(float scale);  // Converts scale to font size
    void SetEditorShowWhitespace(bool show) { editor_show_whitespace_ = show; }
    void SetEditorWordWrap(bool wrap) { editor_word_wrap_ = wrap; }
    void SetEditorAutoIndent(bool indent) { editor_auto_indent_ = indent; }

    // Access to created plot windows
    const std::vector<std::shared_ptr<PlotWindow>>& GetPlotWindows() const { return plot_windows_; }

private:
    void RenderFileMenu();
    void RenderEditMenu();
    void RenderViewMenu();
    void RenderNodesMenu();
    void RenderTrainMenu();
    void RenderDatasetMenu();
    void RenderScriptMenu();
    void RenderPlotsMenu();
    void RenderDeployMenu();

    // File search functionality
    void SearchInFiles(const std::string& search_text, const std::string& search_path,
                       const std::string& file_patterns, bool case_sensitive,
                       bool whole_word, bool use_regex);
    void RenderHelpMenu();

    // Helper functions
    std::string OpenFolderDialog();
    std::string OpenFileDialog(const char* filter, const char* title);
    void CreatePlotWindow(const std::string& title, PlotWindow::PlotWindowType type);

    bool show_new_project_dialog_;
    bool show_about_dialog_;
    bool show_account_settings_dialog_ = false;
    bool show_exit_confirmation_dialog_ = false;
    bool auto_save_enabled_ = false;
    float auto_save_interval_ = 60.0f;  // Auto save every 60 seconds
    float auto_save_timer_ = 0.0f;      // Current timer countdown

    // Account/Auth state
    bool is_logged_in_ = false;
    char login_identifier_[256] = "";  // Email or phone (auto-detected)
    char login_password_[256] = "";
    std::string logged_in_user_;
    std::string login_error_message_;

    std::function<void()> reset_layout_callback_;
    std::function<void()> save_layout_callback_;
    std::function<void()> save_project_settings_callback_;
    std::function<void()> toggle_plot_test_control_callback_;
    std::function<void()> connect_to_server_callback_;
    std::function<void()> import_dataset_callback_;
    std::function<void()> new_script_callback_;
    std::function<void()> open_script_callback_;
    std::function<void(const std::string&)> open_script_in_editor_callback_;
    std::function<void()> save_all_callback_;
    std::function<void()> account_settings_callback_;
    std::function<void()> exit_callback_;
    std::function<bool()> has_unsaved_changes_callback_;

    // Project creation state
    char project_name_buffer_[256];
    char project_path_buffer_[512];

    // Save As dialog state
    bool show_save_as_dialog_ = false;
    char save_as_name_buffer_[256] = "";
    char save_as_path_buffer_[512] = "";

    // New script dialog state
    bool show_new_script_dialog_ = false;
    char new_script_name_[256] = "";
    int new_script_type_ = 0;  // 0 = .cyx, 1 = .py

    // Plot windows management
    std::vector<std::shared_ptr<PlotWindow>> plot_windows_;

    // Edit menu callbacks
    std::function<void()> undo_callback_;
    std::function<void()> redo_callback_;
    std::function<void()> cut_callback_;
    std::function<void()> copy_callback_;
    std::function<void()> paste_callback_;
    std::function<void()> delete_callback_;
    std::function<void()> select_all_callback_;
    std::function<void()> toggle_line_comment_callback_;
    std::function<void()> toggle_block_comment_callback_;
    std::function<void(const std::string&, bool, bool, bool)> find_callback_;  // text, case_sensitive, whole_word, regex
    std::function<void()> find_next_callback_;  // Find next occurrence
    std::function<void(const std::string&, const std::string&, bool, bool, bool)> replace_callback_;  // find, replace, case_sensitive, whole_word, regex
    std::function<void(const std::string&, const std::string&, bool, bool, bool)> replace_all_callback_;

    // New Edit menu callbacks
    std::function<void(int)> go_to_line_callback_;
    std::function<void()> duplicate_line_callback_;
    std::function<void()> move_line_up_callback_;
    std::function<void()> move_line_down_callback_;
    std::function<void()> indent_callback_;
    std::function<void()> outdent_callback_;
    std::function<void()> transform_uppercase_callback_;
    std::function<void()> transform_lowercase_callback_;
    std::function<void()> transform_titlecase_callback_;
    std::function<void()> sort_lines_asc_callback_;
    std::function<void()> sort_lines_desc_callback_;
    std::function<void()> join_lines_callback_;

    // Find/Replace dialog state
    bool show_find_dialog_ = false;
    bool show_replace_dialog_ = false;
    bool show_find_in_files_dialog_ = false;
    bool show_replace_in_files_dialog_ = false;

    // Find/Replace buffers
    char find_text_buffer_[512] = "";
    char replace_text_buffer_[512] = "";
    char find_in_files_pattern_[256] = "*.py;*.cyx";  // File filter pattern
    char find_in_files_path_[512] = "";  // Search path

    // Find/Replace options
    bool find_case_sensitive_ = false;
    bool find_whole_word_ = false;
    bool find_use_regex_ = false;

    // Find in Files results
    struct SearchResult {
        std::string file_path;
        int line_number;
        std::string line_content;
        int match_start;
        int match_length;
    };
    std::vector<SearchResult> search_results_;
    bool search_in_progress_ = false;

    // Preferences dialog state
    bool show_preferences_dialog_ = false;
    int preferences_tab_ = 0;  // 0 = Python/Scripting, 1 = Keyboard Shortcuts

    // Python/Scripting preferences
    char python_interpreter_path_[512] = "";
    char python_startup_script_[512] = "";
    bool python_auto_import_numpy_ = true;
    bool python_auto_import_cyxwiz_ = true;
    int python_output_limit_ = 1000;  // Max lines in output

    // Keyboard shortcuts (action name -> shortcut string)
    struct ShortcutEntry {
        std::string category;     // Category name (e.g., "General", "Script Editor", "Node Editor")
        std::string action;
        std::string shortcut;
        std::string description;
        bool editable;
    };
    std::vector<ShortcutEntry> shortcuts_;
    int editing_shortcut_index_ = -1;
    char shortcut_edit_buffer_[64] = "";

    // Go to Line dialog state
    bool show_go_to_line_dialog_ = false;
    int go_to_line_number_ = 1;

    // Editor preferences
    int editor_theme_ = 3;  // Default to Monokai (index 3)
    int editor_font_size_ = 16;  // Maps to font_scale: 8=1.0, 12=1.3, 16=1.6, 20=2.0
    int editor_tab_size_ = 4;
    bool editor_word_wrap_ = false;
    bool editor_show_line_numbers_ = true;
    bool editor_show_whitespace_ = true;
    bool editor_auto_indent_ = true;
    bool editor_highlight_current_line_ = true;
    bool editor_show_minimap_ = false;

    // Editor settings callbacks
    std::function<void(int)> editor_theme_callback_;
    std::function<void(int)> editor_tab_size_callback_;
    std::function<void(float)> editor_font_scale_callback_;
    std::function<void(bool)> editor_show_whitespace_callback_;
    std::function<void(bool)> editor_word_wrap_callback_;
    std::function<void(bool)> editor_auto_indent_callback_;

    // Application theme callback
    std::function<void(int)> app_theme_changed_callback_;

    // General preferences
    bool general_restore_last_session_ = true;
    bool general_check_updates_ = true;
    int general_recent_files_limit_ = 10;
    bool general_confirm_on_exit_ = true;

    // Appearance preferences
    float appearance_ui_scale_ = 1.0f;
    bool appearance_smooth_scrolling_ = true;
    int appearance_sidebar_position_ = 0;  // 0 = Left, 1 = Right

    // Minimap visibility pointers
    bool* node_editor_minimap_ptr_ = nullptr;
    bool* script_editor_minimap_ptr_ = nullptr;

    // Files preferences
    int files_default_encoding_ = 0;  // 0 = UTF-8, 1 = UTF-16, 2 = ASCII
    int files_line_ending_ = 0;  // 0 = Auto, 1 = LF, 2 = CRLF
    bool files_trim_trailing_whitespace_ = false;
    bool files_insert_final_newline_ = true;
};

} // namespace cyxwiz
