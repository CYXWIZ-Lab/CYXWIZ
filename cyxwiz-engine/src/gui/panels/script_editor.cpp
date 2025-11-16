#include "script_editor.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

ScriptEditorPanel::ScriptEditorPanel()
    : Panel("Script Editor", true)
    , active_tab_index_(-1)
    , show_editor_menu_(false)
    , request_focus_(false)
    , close_tab_index_(-1)
{
    // Create initial empty tab
    NewFile();
}

void ScriptEditorPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void ScriptEditorPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    // Menu bar (when editor is active)
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootWindow)) {
        show_editor_menu_ = true;
    }

    if (show_editor_menu_) {
        RenderMenuBar();
    }

    // Tab bar
    RenderTabBar();

    // Editor content
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        RenderEditor();
    }

    // Status bar
    RenderStatusBar();

    // Handle deferred tab close
    if (close_tab_index_ >= 0) {
        CloseFile(close_tab_index_);
        close_tab_index_ = -1;
    }

    ImGui::End();
}

void ScriptEditorPanel::RenderMenuBar() {
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New", "Ctrl+N")) {
                NewFile();
            }
            if (ImGui::MenuItem("Open", "Ctrl+O")) {
                OpenFile();
            }
            if (ImGui::MenuItem("Save", "Ctrl+S", false, active_tab_index_ >= 0)) {
                SaveFile();
            }
            if (ImGui::MenuItem("Save As", "Ctrl+Shift+S", false, active_tab_index_ >= 0)) {
                SaveFileAs();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Close", "Ctrl+W", false, active_tab_index_ >= 0)) {
                close_tab_index_ = active_tab_index_;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
            bool has_active_tab = active_tab_index_ >= 0;
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, has_active_tab && tabs_[active_tab_index_]->editor.CanUndo())) {
                tabs_[active_tab_index_]->editor.Undo();
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, has_active_tab && tabs_[active_tab_index_]->editor.CanRedo())) {
                tabs_[active_tab_index_]->editor.Redo();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "Ctrl+X", false, has_active_tab)) {
                tabs_[active_tab_index_]->editor.Cut();
            }
            if (ImGui::MenuItem("Copy", "Ctrl+C", false, has_active_tab)) {
                tabs_[active_tab_index_]->editor.Copy();
            }
            if (ImGui::MenuItem("Paste", "Ctrl+V", false, has_active_tab)) {
                tabs_[active_tab_index_]->editor.Paste();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Run")) {
            if (ImGui::MenuItem("Run Script", "F5", false, active_tab_index_ >= 0)) {
                RunScript();
            }
            if (ImGui::MenuItem("Run Selection", "F9", false, active_tab_index_ >= 0)) {
                RunSelection();
            }
            if (ImGui::MenuItem("Run Section", "Ctrl+Enter", false, active_tab_index_ >= 0)) {
                RunCurrentSection();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Debug", "F10", false, active_tab_index_ >= 0)) {
                Debug();
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void ScriptEditorPanel::RenderTabBar() {
    if (ImGui::BeginTabBar("ScriptEditorTabs", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyScroll)) {

        // Render existing tabs
        for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
            auto& tab = tabs_[i];

            // Add asterisk for modified files
            std::string tab_label = tab->filename;
            if (tab->is_modified) {
                tab_label += "*";
            }

            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
            if (request_focus_ && i == active_tab_index_) {
                tab_flags |= ImGuiTabItemFlags_SetSelected;
                request_focus_ = false;
            }

            bool open = true;
            if (ImGui::BeginTabItem(tab_label.c_str(), &open, tab_flags)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }

            // Handle tab close
            if (!open) {
                close_tab_index_ = i;
            }
        }

        // "+" button to add new tab
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) {
            NewFile();
        }

        ImGui::EndTabBar();
    }
}

void ScriptEditorPanel::RenderEditor() {
    auto& tab = tabs_[active_tab_index_];

    // Render the TextEditor
    ImVec2 editor_size = ImVec2(0, -ImGui::GetFrameHeightWithSpacing());
    tab->editor.Render("##editor", editor_size);

    // Track modifications
    if (tab->editor.IsTextChanged()) {
        tab->is_modified = true;
    }
}

void ScriptEditorPanel::RenderStatusBar() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        auto& tab = tabs_[active_tab_index_];
        auto cursor_pos = tab->editor.GetCursorPosition();

        ImGui::Text("Line: %d | Column: %d | %s | %d lines",
            cursor_pos.mLine + 1,
            cursor_pos.mColumn + 1,
            tab->is_modified ? "Modified" : "Saved",
            tab->editor.GetTotalLines());

        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        ImGui::Text("%s", tab->filepath.empty() ? "Untitled" : tab->filepath.c_str());
    }
}

void ScriptEditorPanel::NewFile() {
    auto tab = std::make_unique<EditorTab>();
    tab->filename = "Untitled" + std::to_string(tabs_.size() + 1) + ".cyx";
    tab->filepath = "";
    tab->is_new = true;
    tab->is_modified = false;

    // Configure editor
    tab->editor.SetLanguageDefinition(CreatePythonLanguage());
    tab->editor.SetPalette(TextEditor::GetDarkPalette());
    tab->editor.SetShowWhitespaces(false);
    tab->editor.SetTabSize(4);

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    request_focus_ = true;

    spdlog::info("Created new script file: {}", tabs_[active_tab_index_]->filename);
}

void ScriptEditorPanel::OpenFile(const std::string& filepath) {
    std::string path = filepath;

    // If no path provided, show file dialog
    if (path.empty()) {
        path = OpenFileDialog();
        if (path.empty()) return;  // User cancelled
    }

    // Check if file is already open
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == path) {
            active_tab_index_ = i;
            request_focus_ = true;
            spdlog::info("File already open: {}", path);
            return;
        }
    }

    // Load file content
    std::string content;
    if (!LoadFileContent(path, content)) {
        spdlog::error("Failed to load file: {}", path);
        return;
    }

    // Create new tab
    auto tab = std::make_unique<EditorTab>();
    tab->filename = std::filesystem::path(path).filename().string();
    tab->filepath = path;
    tab->is_new = false;
    tab->is_modified = false;

    // Configure editor
    tab->editor.SetLanguageDefinition(CreatePythonLanguage());
    tab->editor.SetPalette(TextEditor::GetDarkPalette());
    tab->editor.SetShowWhitespaces(false);
    tab->editor.SetTabSize(4);
    tab->editor.SetText(content);

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    request_focus_ = true;

    spdlog::info("Opened file: {}", path);
}

void ScriptEditorPanel::SaveFile() {
    if (active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];

    // If new file without path, use Save As
    if (tab->is_new || tab->filepath.empty()) {
        SaveFileAs();
        return;
    }

    // Save to existing path
    std::string content = tab->editor.GetText();
    if (SaveFileContent(tab->filepath, content)) {
        tab->is_modified = false;
        spdlog::info("Saved file: {}", tab->filepath);
    } else {
        spdlog::error("Failed to save file: {}", tab->filepath);
    }
}

void ScriptEditorPanel::SaveFileAs() {
    if (active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];

    // Show save dialog
    std::string path = SaveFileDialog();
    if (path.empty()) return;  // User cancelled

    // Ensure .cyx extension
    std::filesystem::path fspath(path);
    if (fspath.extension() != ".cyx") {
        path += ".cyx";
    }

    // Save content
    std::string content = tab->editor.GetText();
    if (SaveFileContent(path, content)) {
        tab->filepath = path;
        tab->filename = std::filesystem::path(path).filename().string();
        tab->is_new = false;
        tab->is_modified = false;
        spdlog::info("Saved file as: {}", path);
    } else {
        spdlog::error("Failed to save file: {}", path);
    }
}

void ScriptEditorPanel::CloseFile(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    // TODO: Prompt to save if modified
    if (tabs_[tab_index]->is_modified) {
        // For now, just warn
        spdlog::warn("Closing modified file: {}", tabs_[tab_index]->filename);
    }

    tabs_.erase(tabs_.begin() + tab_index);

    // Adjust active tab index
    if (tabs_.empty()) {
        // Create new empty tab if all closed
        NewFile();
    } else if (active_tab_index_ >= static_cast<int>(tabs_.size())) {
        active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    }
}

void ScriptEditorPanel::RunScript() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];
    std::string script = tab->editor.GetText();

    spdlog::info("Running script: {}", tab->filename);
    auto result = scripting_engine_->ExecuteScript(script);

    if (!result.success) {
        spdlog::error("Script error: {}", result.error_message);
        // TODO: Display error in UI
    } else {
        spdlog::info("Script executed successfully");
        // TODO: Display output
    }
}

void ScriptEditorPanel::RunSelection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];
    std::string selected_text = tab->editor.GetSelectedText();

    if (selected_text.empty()) {
        spdlog::warn("No text selected");
        return;
    }

    spdlog::info("Running selection");
    auto result = scripting_engine_->ExecuteCommand(selected_text);

    if (!result.success) {
        spdlog::error("Execution error: {}", result.error_message);
    }
}

void ScriptEditorPanel::RunCurrentSection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    Section section = GetCurrentSection();

    if (section.code.empty()) {
        spdlog::warn("No section found at cursor");
        return;
    }

    spdlog::info("Running section (lines {}-{})", section.start_line, section.end_line);
    auto result = scripting_engine_->ExecuteScript(section.code);

    if (!result.success) {
        spdlog::error("Section execution error: {}", result.error_message);
    }
}

void ScriptEditorPanel::Debug() {
    // TODO: Implement debugger integration
    spdlog::info("Debug mode not yet implemented");
}

// Helper functions

bool ScriptEditorPanel::LoadFileContent(const std::string& filepath, std::string& content) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

bool ScriptEditorPanel::SaveFileContent(const std::string& filepath, const std::string& content) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file << content;
    file.close();
    return true;
}

std::string ScriptEditorPanel::OpenFileDialog() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char filename[MAX_PATH] = "";

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "CyxWiz Scripts (*.cyx)\0*.cyx\0Python Files (*.py)\0*.py\0Text Files (*.txt)\0*.txt\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
#else
    // TODO: Implement for Linux/macOS using native dialogs or portable file browser
    spdlog::error("File dialog not implemented for this platform");
#endif
    return "";
}

std::string ScriptEditorPanel::SaveFileDialog() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char filename[MAX_PATH] = "Untitled.cyx";

    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrFilter = "CyxWiz Scripts (*.cyx)\0*.cyx\0All Files (*.*)\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrDefExt = "cyx";
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

    if (GetSaveFileNameA(&ofn)) {
        return std::string(filename);
    }
#else
    // TODO: Implement for Linux/macOS
    spdlog::error("File dialog not implemented for this platform");
#endif
    return "";
}

std::vector<ScriptEditorPanel::Section> ScriptEditorPanel::ParseSections(const std::string& text) {
    std::vector<Section> sections;
    std::istringstream stream(text);
    std::string line;

    int line_num = 0;
    Section current_section;
    current_section.start_line = 0;
    bool in_section = false;

    while (std::getline(stream, line)) {
        // Check for section delimiter %%
        if (line.find("%%") != std::string::npos) {
            if (in_section) {
                // End of section
                current_section.end_line = line_num - 1;
                sections.push_back(current_section);
                current_section = Section();
                current_section.start_line = line_num + 1;
                in_section = false;
            } else {
                // Start of section
                current_section.start_line = line_num + 1;
                in_section = true;
            }
        } else if (in_section) {
            current_section.code += line + "\n";
        }

        line_num++;
    }

    // Add final section if still open
    if (in_section && !current_section.code.empty()) {
        current_section.end_line = line_num - 1;
        sections.push_back(current_section);
    }

    // If no %% markers found, treat entire file as one section
    if (sections.empty() && !text.empty()) {
        Section whole_file;
        whole_file.start_line = 0;
        whole_file.end_line = line_num - 1;
        whole_file.code = text;
        sections.push_back(whole_file);
    }

    return sections;
}

ScriptEditorPanel::Section ScriptEditorPanel::GetCurrentSection() {
    Section empty_section;

    if (active_tab_index_ < 0) {
        return empty_section;
    }

    auto& tab = tabs_[active_tab_index_];
    auto cursor_pos = tab->editor.GetCursorPosition();
    int current_line = cursor_pos.mLine;

    // Get all text and parse sections
    std::string text = tab->editor.GetText();
    std::vector<Section> sections = ParseSections(text);

    // Find section containing cursor
    for (const auto& section : sections) {
        if (current_line >= section.start_line && current_line <= section.end_line) {
            return section;
        }
    }

    return empty_section;
}

TextEditor::LanguageDefinition ScriptEditorPanel::CreatePythonLanguage() {
    TextEditor::LanguageDefinition lang;

    lang.mName = "Python";
    lang.mCaseSensitive = true;
    lang.mAutoIndentation = true;
    lang.mSingleLineComment = "#";
    lang.mCommentStart = "\"\"\"";
    lang.mCommentEnd = "\"\"\"";

    // Python keywords
    static const char* const keywords[] = {
        "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
        "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
        "while", "with", "yield", "async", "await"
    };

    for (auto& k : keywords) {
        lang.mKeywords.insert(k);
    }

    // Built-in identifiers
    static const char* const identifiers[] = {
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr",
        "classmethod", "compile", "complex", "delattr", "dict", "dir", "divmod", "enumerate",
        "eval", "exec", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
        "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter", "len",
        "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open",
        "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set", "setattr",
        "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip"
    };

    for (auto& i : identifiers) {
        TextEditor::Identifier id;
        id.mDeclaration = "Built-in function";
        lang.mIdentifiers.insert(std::make_pair(std::string(i), id));
    }

    return lang;
}

} // namespace cyxwiz
