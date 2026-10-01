#pragma once

#include "../panel.h"
#include <chrono>
#include "../../core/async_task_manager.h"
#include "../../core/large_text_file.h"
#include "../../core/script_text_file.h"
#include "../../scripting/cell_manager.h"
#include "../../scripting/debugger.h"
#include "../../scripting/script_manager.h"
#include "../code_editor.h"
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <functional>
#include <atomic>
#include <cstdint>

namespace scripting {
    class IScriptOutputSink;
    class ScriptingEngine;
}

namespace cyxwiz {

/**
 * Script Editor Panel
 * Multi-tab text editor for .cyx scripts with Python syntax highlighting
 * Supports section execution (%%),comments (#), and file operations
 */
class ScriptEditorPanel : public Panel {
public:
    ScriptEditorPanel();
    ~ScriptEditorPanel() override;

    void Render() override;

    // Set scripting engine (shared with other panels)
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

    void SetScriptOutputSink(scripting::IScriptOutputSink* output_sink);

    // File operations
    void NewFile();
    void OpenFile(const std::string& filepath = "");
    void SaveFile();
    void SaveFileAs();
    void CloseFile(int tab_index);

    // Load generated code (from Node Editor)
    void LoadGeneratedCode(const std::string& code, const std::string& framework_name);

    // Execution
    void RunScript();
    void StopScript();
    bool IsScriptRunning() const { return script_running_; }
    // True when the active tab holds a script that can be run or edited.
    bool HasEditableScript() const { return IsActiveTabEditable(); }
    void RunSelection();
    void RunCurrentSection();  // Execute code between %% markers
    void Debug();

    // Check for unsaved files (for application close confirmation)
    bool HasUnsavedFiles() const;
    std::vector<std::string> GetUnsavedFileNames() const;
    void SaveAllFiles();  // Save all unsaved files
    bool HasEmptyNewTab() const;  // Check if there's an empty untitled tab

    // Inline find and replace widget (Ctrl+F / Ctrl+H).
    void OpenFind(bool replace);

    // Find/Replace operations
    bool FindInEditor(const std::string& search_text, bool case_sensitive, bool whole_word, bool use_regex);
    bool FindNext();
    bool FindPrevious();
    // Find dialog "Find Previous": sets the search, then steps back one match.
    bool FindPreviousOf(const std::string& search_text, bool case_sensitive, bool whole_word, bool use_regex);
    // Replace in Files support: an open tab with unsaved edits must not be
    // overwritten on disk; an open unmodified tab is reloaded after the write.
    bool HasUnsavedChangesFor(const std::string& filepath) const;
    bool ReloadOpenFile(const std::string& filepath);
    bool Replace(const std::string& search_text, const std::string& replace_text,
                 bool case_sensitive, bool whole_word, bool use_regex);
    int ReplaceAll(const std::string& search_text, const std::string& replace_text,
                   bool case_sensitive, bool whole_word, bool use_regex);

    // Comment operations
    void ToggleLineComment();
    void ToggleBlockComment();

    // Edit operations (for Edit menu)
    void Undo();
    void Redo();
    void Cut();
    void Copy();
    void Paste();
    void Delete();
    void SelectAll();

    // Navigation
    void GoToLine(int line_number);

    // Line operations
    void DuplicateLine();
    void MoveLineUp();
    void MoveLineDown();
    void Indent();
    void Outdent();

    // Text transformation
    void TransformToUppercase();
    void TransformToLowercase();
    void TransformToTitleCase();

    // Multi-line operations
    void SortLinesAscending();
    void SortLinesDescending();
    void JoinLines();

    // Settings access (for Preferences synchronization)
    // Engine-wide code text size (Preferences > Appearance).
    void SetFontScale(float scale);
    void SetTabSize(int size);
    void SetShowWhitespace(bool show);
    void SetWordWrap(bool wrap);
    void SetAutoIndent(bool indent);
    void SetSyntaxHighlighting(bool enabled);
    void SetShowMinimap(bool show) { show_minimap_ = show; }
    float GetFontScale() const;
    int GetTabSize() const { return tab_size_; }
    bool GetShowWhitespace() const { return show_whitespace_; }
    bool GetWordWrap() const { return word_wrap_; }
    bool GetAutoIndent() const { return auto_indent_; }
    bool GetSyntaxHighlighting() const { return syntax_highlighting_; }
    bool GetShowMinimap() const { return show_minimap_; }
    bool* GetShowMinimapPtr() { return &show_minimap_; }

    // Open files state (for project persistence)
    std::vector<std::string> GetOpenFilePaths() const;
    int GetActiveTabIndex() const { return active_tab_index_; }
    void SetActiveTabIndex(int index);

    // Callbacks for settings changes (Script Editor -> Preferences sync)
    void SetOnSettingsChangedCallback(std::function<void()> callback) { on_settings_changed_callback_ = callback; }

    // Auto-completion state query
    bool IsCompletionPopupOpen() const { return show_completion_popup_; }

    // Check if this panel has focus (including child windows)
    bool IsFocused() const { return is_focused_; }

private:
    // Tab/File representation
    struct EditorTab {
        std::uint64_t document_id = 0; // Stable across tab reordering and closure
        std::string filename;        // Display name (e.g., "script.cyx")
        std::string filepath;        // Full path (empty if unsaved)
        CodeEditor editor;           // the code view and its document (TOFIX133 P1)
        bool is_modified;            // Unsaved changes flag
        bool is_new;                 // New file (not saved yet)

        // Async loading state
        bool is_loading = false;         // True while loading file content
        float load_progress = 0.0f;      // Loading progress (0-1)
        std::string load_status;         // Status text during loading
        bool load_failed = false;        // the file was not read: never write this tab over it
        scriptfile::TextFormat format;   // BOM and line endings of the file, kept on save
        std::uint64_t load_task_id = 0;

        // Large files use a bounded, read-only, virtualized text view.
        bool is_large_file = false;
        bool large_page_loading = false;
        std::uint64_t large_page_task_id = 0;
        std::uint64_t large_page_generation = 0;
        std::uint64_t requested_page_start = 0;
        std::uint64_t go_to_line = 1; // One-based value shown to users
        std::uint64_t scroll_to_line = 0;
        bool request_large_scroll = false;
        std::string large_error;
        LargeTextFileIndex large_index;
        LargeTextFilePage large_page;

        // Cell-based mode (Jupyter-like)
        bool cell_mode = false;          // True for cell-based editing
        CellManager cell_manager;        // Cell management
        int selected_cell = -1;          // Currently selected cell
        int editing_cell = -1;           // Cell being edited (-1 = command mode)
        int last_editing_cell = -1;      // Track previous editing cell to detect mode change
        float cell_scroll_y = 0.0f;      // Scroll position in cell view

        // Breakpoints for traditional script mode (1-based line numbers)
        std::vector<int> breakpoints;

        EditorTab() : is_modified(false), is_new(true) {}
    };

    // Async loading helper
    void OpenFileAsync(std::uint64_t document_id, const std::string& filepath);
    void FinalizeAsyncLoad(std::uint64_t document_id, std::string content);
    void StartEditorRun(const std::string& code);  // sets script_running_ only if the run started
    void RetryLoad(int tab_index);  // a failed or cancelled load, again
    void OpenLargeFileAsync(std::uint64_t document_id, const std::string& filepath);
    void RequestLargeFilePage(std::uint64_t document_id, std::uint64_t first_line);
    void RenderLargeFileViewer(EditorTab& tab);
    void CancelTabTasks(EditorTab& tab);
    int FindTabIndex(std::uint64_t document_id) const;
    bool IsActiveTabEditable() const;
    bool IsActiveTabTextMode() const;  // editable and showing its text buffer (not notebook mode)

    // Rendering functions
    void RenderTabBar();
    void RenderMenuBar();
    void RenderEditorToolbar();
    void RenderEditor();
    void RenderStatusBar();
    void HandleKeyboardShortcuts() override;

    // Cell-based editor rendering
    void RenderCellBasedEditor();
    void RenderCell(Cell& cell, int index);
    void RenderCodeCell(Cell& cell, int index);
    void RenderMarkdownCell(Cell& cell, int index);
    void RenderCellOutput(const CellOutput& output);
    void RenderCellToolbar(int index);
    void HandleCellKeyboardShortcuts();
    void ToggleCellMode();

    // Debugger UI rendering
    void RenderDebugToolbar();
    void ToggleBreakpointAtCursor();

    // File operations helpers
    bool LoadFileContent(const std::string& filepath, std::string& content);
    bool SaveFileContent(const std::string& filepath, const std::string& content,
                         const scriptfile::TextFormat& format, std::string* error);
    std::string OpenFileDialog();
    std::string SaveFileDialog();

    // Section execution helpers
    struct Section {
        int start_line;
        int end_line;
        std::string code;
    };
    std::vector<Section> ParseSections(const std::string& text);
    Section GetCurrentSection();
    std::string DedentCode(const std::string& code);  // Remove common leading whitespace
    void SyncActiveCellEditor(EditorTab& tab);
    std::string GetTabContentForPersistence(EditorTab& tab);
    std::string GetTabExecutableText(EditorTab& tab);
    bool IsTabContentBlank(EditorTab& tab) const;

    // Data
    std::vector<std::unique_ptr<EditorTab>> tabs_;
    int active_tab_index_;
    std::uint64_t next_document_id_ = 1;
    std::shared_ptr<std::atomic<bool>> async_owner_alive_ =
        std::make_shared<std::atomic<bool>>(true);
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    std::unique_ptr<scripting::DebuggerManager> debugger_;
    scripting::IScriptOutputSink* script_output_sink_ = nullptr;

    // Debugger state
    bool debug_mode_active_ = false;      // True when debugging is active
    int debug_current_line_ = -1;         // Current line being debugged (-1 = none)
    std::string debug_current_cell_;      // Current cell ID being debugged

    // UI state
    bool show_editor_menu_;
    bool request_focus_;
    bool request_window_focus_;
    int close_tab_index_;  // Tab to close (-1 = none)

    // Execution output
    std::string last_execution_output_;
    bool show_output_notification_;
    float output_notification_time_;

    // Async execution state
    bool script_running_;
    float running_indicator_time_;
    // Name shown for the running script in the Console (file, selection or
    // section) and when it started, for its outcome line.
    std::string running_script_name_;
    std::chrono::steady_clock::time_point running_script_started_{};

    // View settings (code colours follow the Engine theme, TOFIX133 P1)
    float font_scale_ = 1.3f;  // Medium: 16 px native atlas font for crisp rendering
    bool show_whitespace_ = true;
    bool syntax_highlighting_ = true;
    bool word_wrap_ = false;
    bool auto_indent_ = true;
    int tab_size_ = 4;  // 2, 4, or 8
    bool show_minimap_ = true;  // Show code minimap on the right
    float minimap_width_ = 100.0f;  // Width of minimap in pixels

    // Save/Close dialog state
    bool show_save_before_run_dialog_ = false;      // "Save before running?" dialog
    bool show_save_before_close_dialog_ = false;    // "Save changes?" dialog when closing
    int pending_close_tab_index_ = -1;              // Tab waiting to be closed after dialog
    bool run_after_save_ = false;                   // Flag to run script after saving

    // Empty script warning popup state
    bool show_empty_script_warning_ = false;

    // Dialog rendering helpers
    void RenderSaveBeforeRunDialog();
    void RenderSaveBeforeCloseDialog();
    void DoRunScript();      // Internal run after save check passed
    void DoCloseFile(int tab_index);  // Internal close after save check passed

    // Apply settings to all tabs
    void ApplyTabSizeToAllTabs();
    void ApplySyntaxHighlightingToAllTabs();
    void ConfigureEditor(CodeEditor& editor) const;  // tab size, whitespace, colouring

    // Inline find widget (TOFIX133 P2)
    struct FindState {
        bool open = false;
        bool show_replace = false;
        int focus_input = 0;  // 1 find box, 2 replace box
        char query[256] = {};
        char replacement[256] = {};
        bool case_sensitive = false;
        bool whole_word = false;
        bool regex = false;
        uint64_t version = ~0ull;
        uint64_t doc_id = 0;
        std::string key;
        std::string error;
        std::vector<std::pair<editor::Pos, editor::Pos>> matches;
        int current = -1;
        float last_height = 0.0f;
    } find_;
    void CloseFind();
    void UpdateFindMarks(CodeEditor& code);
    void FindStep(bool forward);
    void RenderFindWidget(CodeEditor& code, const ImVec2& code_min, float code_width, bool narrow);

    // Find/Replace state
    std::string last_search_text_;
    bool last_case_sensitive_ = false;
    bool last_whole_word_ = false;
    bool last_use_regex_ = false;

    // Settings changed callback (for syncing with Preferences)
    std::function<void()> on_settings_changed_callback_;

    // Auto-completion state
    scripting::ScriptManager script_manager_;
    std::vector<scripting::CompletionItem> completion_items_;
    bool show_completion_popup_ = false;
    bool completion_just_opened_ = false;  // Skip close check for one frame after opening
    bool completion_just_accepted_ = false;  // Skip editor keyboard input for one frame after accepting
    int selected_completion_ = 0;
    std::string completion_prefix_;
    editor::Pos completion_start_pos_;

    // Focus tracking
    bool is_focused_ = false;

    static constexpr std::uint64_t kEditableFileLimitBytes = 4ULL * 1024ULL * 1024ULL;
    static constexpr std::uint64_t kLargeTextCheckpointStride = 1024;
    static constexpr std::size_t kLargeTextPageLines = 512;
    static constexpr std::size_t kLargeTextMaxLineBytes = 16 * 1024;

    // Auto-completion helpers
    void UpdateAutoCompletion(bool force = false);
    void RenderCompletionPopup();
    void ApplyCompletion(const scripting::CompletionItem& item);
    void CloseCompletionPopup();
};

} // namespace cyxwiz
