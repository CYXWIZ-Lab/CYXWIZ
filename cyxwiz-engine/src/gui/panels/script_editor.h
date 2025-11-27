#pragma once

#include "../panel.h"
#include <TextEditor.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace scripting {
    class ScriptingEngine;
}

namespace cyxwiz {

class CommandWindowPanel;  // Forward declaration

/**
 * Script Editor Panel
 * Multi-tab text editor for .cyx scripts with Python syntax highlighting
 * Supports section execution (%%),comments (#), and file operations
 */
class ScriptEditorPanel : public Panel {
public:
    ScriptEditorPanel();
    ~ScriptEditorPanel() override = default;

    void Render() override;

    // Set scripting engine (shared with other panels)
    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);

    // Set command window for output display
    void SetCommandWindow(CommandWindowPanel* command_window);

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
    void RunSelection();
    void RunCurrentSection();  // Execute code between %% markers
    void Debug();

    // Check for unsaved files (for application close confirmation)
    bool HasUnsavedFiles() const;
    std::vector<std::string> GetUnsavedFileNames() const;
    void SaveAllFiles();  // Save all unsaved files

private:
    // Tab/File representation
    struct EditorTab {
        std::string filename;        // Display name (e.g., "script.cyx")
        std::string filepath;        // Full path (empty if unsaved)
        TextEditor editor;           // ImGuiColorTextEdit instance
        bool is_modified;            // Unsaved changes flag
        bool is_new;                 // New file (not saved yet)

        EditorTab() : is_modified(false), is_new(true) {}
    };

    // Rendering functions
    void RenderTabBar();
    void RenderMenuBar();
    void RenderEditor();
    void RenderStatusBar();
    void HandleKeyboardShortcuts();

    // File operations helpers
    bool LoadFileContent(const std::string& filepath, std::string& content);
    bool SaveFileContent(const std::string& filepath, const std::string& content);
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

    // Python language definition for syntax highlighting
    static TextEditor::LanguageDefinition CreatePythonLanguage();

    // Data
    std::vector<std::unique_ptr<EditorTab>> tabs_;
    int active_tab_index_;
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    CommandWindowPanel* command_window_;  // For output display

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

    // View settings
    enum class EditorTheme { Dark, Light, RetroBlu, Monokai, Dracula, OneDark, GitHub };
    EditorTheme current_theme_ = EditorTheme::Monokai;  // Default to Monokai
    float font_scale_ = 1.6f;  // 1.0 = Small, 1.3 = Medium, 1.6 = Large (default), 2.0 = Extra Large
    bool show_whitespace_ = true;
    bool syntax_highlighting_ = true;
    int tab_size_ = 4;  // 2, 4, or 8

    // Save/Close dialog state
    bool show_save_before_run_dialog_ = false;      // "Save before running?" dialog
    bool show_save_before_close_dialog_ = false;    // "Save changes?" dialog when closing
    int pending_close_tab_index_ = -1;              // Tab waiting to be closed after dialog
    bool run_after_save_ = false;                   // Flag to run script after saving

    // Dialog rendering helpers
    void RenderSaveBeforeRunDialog();
    void RenderSaveBeforeCloseDialog();
    void DoRunScript();      // Internal run after save check passed
    void DoCloseFile(int tab_index);  // Internal close after save check passed

    // Apply settings to all tabs
    void ApplyThemeToAllTabs();
    void ApplyTabSizeToAllTabs();
    void ApplySyntaxHighlightingToAllTabs();

    // Custom theme palettes
    static TextEditor::Palette GetMonokaiPalette();
    static TextEditor::Palette GetDraculaPalette();
    static TextEditor::Palette GetOneDarkPalette();
    static TextEditor::Palette GetGitHubPalette();
};

} // namespace cyxwiz
