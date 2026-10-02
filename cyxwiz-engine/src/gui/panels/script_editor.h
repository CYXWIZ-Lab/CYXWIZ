#pragma once

#include "../panel.h"
#include <chrono>
#include "../../core/async_task_manager.h"
#include "../../core/large_text_file.h"
#include "../../core/script_text_file.h"
#include "../../scripting/cell_manager.h"
#include "../../core/notebook_presentation.h"
#include "../../core/language_results.h"
#include "../../scripting/language_service.h"
#include "../../scripting/debugger.h"
#include "../../scripting/script_manager.h"
#include "../code_editor.h"
#include "../variables_view.h"
#include <string>
#include <unordered_map>
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
class DataTable;
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
    // Edit > Go to Line (the Engine's dialog), offered in the code's right-click menu.
    void SetGoToLineRequest(std::function<void()> request) { go_to_line_request_ = std::move(request); }
    // A notebook table result opens in the Table Viewer (TOFIX133 P4 board 5).
    // Variable Explorer (TOFIX133 P5): the open notebooks it can show, and
    // a name inserted at the cursor of the script or cell being edited.
    std::vector<VariablesView::Scope> NotebookScopes() const;
    void InsertTextAtCursor(const std::string& text);

    void SetOpenTableCallback(std::function<void(std::shared_ptr<DataTable>)> callback) {
        open_table_callback_ = std::move(callback);
    }
    // Opens a file and moves to a line (1-based) once it is loaded.
    void OpenFileAtLine(const std::string& path, int line);

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
    bool HasOpenFiles() const {  // a file from disk, not the starting Untitled tab
        for (const auto& tab : tabs_)
            if (tab && !tab->is_new) return true;
        return false;
    }

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
        bool format_changed = false;     // BOM or line endings changed in the status bar, not saved yet
        std::filesystem::file_time_type disk_time{};  // write time seen at load or save
        bool disk_changed = false;       // changed (or gone) on disk while the tab has edits
        bool disk_missing = false;
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
        bool restore_cell_scroll = false;  // set when the notebook view returns to a kept position
        std::unordered_map<std::string, float> cell_heights;  // last frame's row height per cell id
        // Notebook side panels (board 4).
        bool show_variables = false;
        bool show_outline = false;
        int scroll_to_cell = -1;  // the Outline asks the cell list to scroll there
        // The notebook's Variables (TOFIX133 P5): the shared view on its namespace.
        std::unique_ptr<VariablesView> variables_view;
        // Problems (TOFIX133 P3 step 3.4): the script's last pyflakes check.
        std::vector<lang::Problem> problems;
        std::uint64_t problems_version = 0;  // the text version they are for (+1; 0 = not checked)
        bool show_problems = false;
        bool problems_show_warnings = true;

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
    void RenderBreadcrumbs(EditorTab& tab);
    void RefreshPythonStatus();
    std::string python_version_;      // "3.12.8" once Python runs
    std::string python_environment_;  // project name, or "system Python"
    bool python_started_ = false;
    void RenderEditor();
    void RenderStatusBar();
    void HandleKeyboardShortcuts() override;

    // Cell-based editor rendering
    void RenderCellBasedEditor();
    void RenderNotebookToolbar(EditorTab& tab);       // script_editor_notebook.cpp (board 4)
    static ImVec4 NotebookToneColour(int tone);       // nbview::Tone -> colour
    void RenderCell(Cell& cell, int index);
    bool RenderCellEditorBlock(Cell& cell, int index, float width, bool editing, bool python);
    void RenderCellActions(Cell& cell, int index, const ImVec2& top_right);
    void RenderInsertBar(int after_index);
    void RenderNotebookOutputs(Cell& cell, int index, float width);  // script_editor_notebook_outputs.cpp (board 5)
    void RenderErrorOutput(Cell& cell, int index, CellOutput& out, float width);
    void RenderTableOutput(Cell& cell, CellOutput& out, float width);
    void RenderPlotOutput(Cell& cell, CellOutput& out, float width);
    void OpenResultInTableViewer(Cell& cell, const CellOutput& out);
    void OpenTraceFrame(const nbview::FrameLink& link);
    void RenderPlotWindows();
    std::function<void(std::shared_ptr<DataTable>)> open_table_callback_;
    std::string table_open_error_;
    std::string table_open_error_cell_;
    std::string pending_goto_path_;
    std::string deferred_open_path_;  // a traceback frame's file, opened next frame
    int deferred_open_line_ = 0;
    int pending_goto_line_ = 0;
    // Plots opened in their own window: a copy of the image, so clearing the
    // cell does not take the window's picture with it.
    struct PlotWindow {
        int id = 0;
        std::string title;
        std::vector<unsigned char> png;
        unsigned int texture = 0;
        int width = 0;
        int height = 0;
        bool open = true;
    };
    std::vector<PlotWindow> plot_windows_;
    int next_plot_window_ = 1;
    void RenderNotebookVariables(EditorTab& tab, float height);  // script_editor_notebook_side.cpp
    void RenderNotebookOutline(EditorTab& tab, float width, float height);
    void RestartNotebook(EditorTab& tab);  // Restart + the Variables view forgets
    struct CellClipboard {
        bool has = false;
        CellType type = CellType::Code;
        std::string source;
    };
    CellClipboard cell_clipboard_;
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
    std::string GetTabContentForPersistence(EditorTab& tab, const std::string& path);
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
    int window_focus_frames_ = 0;
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

    std::function<void()> go_to_line_request_;
    void RenderCodeContextMenu(EditorTab& tab);

    // Files changed on disk (TOFIX133 P2 step 2.6)
    void NoteDiskTime(EditorTab& tab);
    void CheckFilesOnDisk();
    void ReloadFromDisk(EditorTab& tab);
    void RenderDiskChangedBand(EditorTab& tab);
    void OpenDiskCopy(EditorTab& tab);
    double disk_check_time_ = -10.0;

    // Settings changed callback (for syncing with Preferences)
    std::function<void()> on_settings_changed_callback_;

    // Auto-completion state
    scripting::ScriptManager script_manager_;
    // Completion (TOFIX133 P3, board 6): entries from Jedi, or the old keyword
    // completer when the tools are missing; details of the selected entry.
    std::vector<lang::Completion> completion_entries_;
    bool completion_entries_from_fallback_ = false;
    bool completion_scroll_to_selected_ = false;
    bool completion_details_shown_ = true;  // Ctrl+Space hides/shows them
    std::string completion_details_for_;
    lang::Description completion_details_;
    std::uint64_t completion_details_request_ = 0;
    float completion_details_height_ = 120.0f;
    std::string completion_file_stem_;  // names of this file show no module
    bool show_completion_popup_ = false;
    bool completion_just_opened_ = false;  // Skip close check for one frame after opening
    bool completion_just_accepted_ = false;  // Skip editor keyboard input for one frame after accepting
    int selected_completion_ = 0;
    std::string completion_prefix_;
    editor::Pos completion_start_pos_;
    // Jedi completion (TOFIX133 P3): the request in flight and where it was asked.
    std::uint64_t completion_request_ = 0;
    editor::Pos completion_request_pos_;
    std::uint64_t completion_request_version_ = 0;
    void PollLanguageResults();
    // script_editor_language.cpp (boards 6-8)
    CodeEditor* ActiveCodeEditor();  // the text, or the notebook cell being edited
    scripting::LanguageService::Request LanguageRequest(scripting::LanguageService::Kind kind, const CodeEditor& code,
                                                        const editor::Pos& pos);
    bool LanguageReady();
    void RequestCompletionDetails();
    void AcceptCompletion();
    void OpenCompletionList(bool fallback);
    bool HandleLanguageResult(const scripting::LanguageService::Result& result);
    // Signature help (step 3.5), hover and go to definition (step 3.6):
    // script_editor_language_cards.cpp. A target is the script or one cell.
    struct CodeTarget {
        std::uint64_t document_id = 0;
        std::string cell_id;
        bool operator==(const CodeTarget&) const = default;
    };
    enum class HoverState { Resting, Waiting, Shown, Nothing };
    struct HoverCard {
        CodeTarget target;      // document 0: none
        int line = -1;
        int key_col = -1;       // the word's first column, or -2 - problem index
        editor::Pos pos;        // asked at (the word's start)
        std::uint64_t version = 0;
        double since = 0.0;
        ImVec2 below;           // screen point under the hovered character
        HoverState state = HoverState::Resting;
        std::uint64_t request = 0;
        lang::Hover info;
        bool is_problem = false;
        lang::Problem problem;
        ImVec2 card_min, card_max;  // last frame's card, so the mouse can move onto it
    };
    CodeTarget ActiveCodeTarget();
    CodeEditor* CodeEditorFor(const CodeTarget& target);
    void RequestSignatures(bool manual);
    void CloseSignatureHelp();
    void UpdateSignatureHelp();
    bool HandleSignaturesResult(const scripting::LanguageService::Result& result);
    void RenderSignatureCard();
    void UpdateHover(CodeEditor& code, const std::vector<lang::Problem>& problems, const std::string& cell_id);
    bool HandleHoverResult(const scripting::LanguageService::Result& result);
    void RenderHoverCard();
    void GoToDefinition(const CodeTarget& target, CodeEditor& code, const editor::Pos& pos);
    bool HandleDefinitionResult(const scripting::LanguageService::Result& result);
    void ShowLanguageNote(const CodeEditor& code, std::string text);
    void RenderLanguageNote();
    bool HandleCardResult(const scripting::LanguageService::Result& result);
    // After a code view drew: hover tracking and Ctrl+click (cell_id empty for the script).
    void AfterCodeRender(CodeEditor& code, const std::vector<lang::Problem>& problems, const std::string& cell_id);
    void RenderLanguageCards();
    bool signature_open_ = false;
    bool signature_manual_ = false;  // Ctrl+Shift+Space: ")" does not close it
    std::vector<lang::Signature> signatures_;
    CodeTarget signature_target_;
    std::uint64_t signature_request_ = 0;
    std::uint64_t signature_request_version_ = 0;
    editor::Pos signature_request_pos_;
    CodeTarget signature_seen_target_;
    std::uint64_t signature_seen_version_ = 0;
    editor::Pos signature_seen_pos_;
    HoverCard hover_;
    int hover_seen_frame_ = -1;
    CodeTarget definition_target_;
    std::uint64_t definition_request_ = 0;
    std::string definition_name_;
    std::string language_note_;  // "No definition found ...", shown for a moment
    double language_note_until_ = 0.0;
    ImVec2 language_note_at_;
    // Problems (step 3.4)
    struct PendingDiagnostics {
        std::uint64_t document_id = 0;
        std::string cell_id;
        std::uint64_t version = 0;
    };
    struct ProblemCounts {
        int errors = 0;
        int warnings = 0;
    };
    void UpdateDiagnostics(EditorTab& tab);
    bool HandleDiagnosticsResult(const scripting::LanguageService::Result& result);
    void ApplyProblemSquiggles(const EditorTab& tab, CodeEditor& code, const std::vector<lang::Problem>& problems);
    ProblemCounts CountProblems(const EditorTab& tab) const;
    float ProblemCountsWidth(const EditorTab& tab) const;
    bool ProblemCountsItem(EditorTab& tab);
    void RenderProblemsPanel(EditorTab& tab, float height);
    std::unordered_map<std::uint64_t, PendingDiagnostics> pending_diagnostics_;
    std::uint64_t diag_seen_doc_ = 0;
    std::string diag_seen_cell_;
    std::uint64_t diag_seen_version_ = 0;
    double diag_changed_at_ = 0.0;
    std::uint64_t diag_in_flight_doc_ = 0;
    std::string diag_in_flight_cell_;
    std::uint64_t diag_in_flight_version_ = 0;

    // Focus tracking
    bool is_focused_ = false;

    // Status bar: the Python the editor runs with (refreshed every 2 s)
    std::string python_status_;
    std::string python_tooltip_;
    double python_status_time_ = -10.0;

    static constexpr std::uint64_t kEditableFileLimitBytes = 4ULL * 1024ULL * 1024ULL;
    static constexpr std::uint64_t kLargeTextCheckpointStride = 1024;
    static constexpr std::size_t kLargeTextPageLines = 512;
    static constexpr std::size_t kLargeTextMaxLineBytes = 16 * 1024;

    // Auto-completion helpers
    void UpdateAutoCompletion(bool force = false);
    void RenderCompletionPopup();
    void CloseCompletionPopup();
};

} // namespace cyxwiz
