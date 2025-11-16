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

    // File operations
    void NewFile();
    void OpenFile(const std::string& filepath = "");
    void SaveFile();
    void SaveFileAs();
    void CloseFile(int tab_index);

    // Execution
    void RunScript();
    void RunSelection();
    void RunCurrentSection();  // Execute code between %% markers
    void Debug();

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

    // UI state
    bool show_editor_menu_;
    bool request_focus_;
    int close_tab_index_;  // Tab to close (-1 = none)
};

} // namespace cyxwiz
