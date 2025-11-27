#include "script_editor.h"
#include "command_window.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

namespace cyxwiz {

ScriptEditorPanel::ScriptEditorPanel()
    : Panel("Script Editor", true)
    , active_tab_index_(-1)
    , command_window_(nullptr)
    , show_editor_menu_(false)
    , request_focus_(false)
    , request_window_focus_(false)
    , close_tab_index_(-1)
    , show_output_notification_(false)
    , output_notification_time_(0.0f)
    , script_running_(false)
    , running_indicator_time_(0.0f)
{
    // Create initial empty tab
    NewFile();
}

void ScriptEditorPanel::SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) {
    scripting_engine_ = engine;
}

void ScriptEditorPanel::SetCommandWindow(CommandWindowPanel* command_window) {
    command_window_ = command_window;
}

void ScriptEditorPanel::Render() {
    if (!visible_) return;

    // Poll for pending output from async script execution
    if (scripting_engine_ && scripting_engine_->IsScriptRunning()) {
        script_running_ = true;
        running_indicator_time_ += ImGui::GetIO().DeltaTime;

        // Get any pending output and display it
        std::string pending = scripting_engine_->GetPendingOutput();
        if (!pending.empty() && command_window_) {
            command_window_->DisplayScriptOutput("Running...", pending, false);
        }
    } else if (script_running_) {
        // Script just finished - check for result
        script_running_ = false;
        running_indicator_time_ = 0.0f;

        // First, drain any remaining output from the queue (for fast-finishing scripts)
        std::string remaining_output = scripting_engine_->GetPendingOutput();
        if (!remaining_output.empty() && command_window_) {
            command_window_->DisplayScriptOutput("Output", remaining_output, false);
        }

        auto result = scripting_engine_->GetAsyncResult();
        if (result.has_value()) {
            auto& r = result.value();
            if (command_window_) {
                if (r.was_cancelled) {
                    command_window_->DisplayScriptOutput("Script", "Script cancelled by user", true);
                } else if (!r.success) {
                    command_window_->DisplayScriptOutput("Script", "Error: " + r.error_message, true);
                } else {
                    // Script completed successfully - don't print redundant "completed" message
                    // since output was already displayed above
                    spdlog::info("Script completed successfully");
                }
            }
            spdlog::info("Async script execution finished. Success: {}", r.success);
        }
    }

    ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_MenuBar);

    // Handle window focus request (bring to front)
    if (request_window_focus_) {
        ImGui::SetWindowFocus();
        request_window_focus_ = false;
    }

    // Always show menu bar
    RenderMenuBar();

    // Handle keyboard shortcuts
    HandleKeyboardShortcuts();

    // Tab bar
    RenderTabBar();

    // Editor content
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        RenderEditor();
    }

    // Status bar
    RenderStatusBar();

    // Show output notification if needed
    if (show_output_notification_) {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 10);
        ImGui::TextWrapped("%s", last_execution_output_.c_str());

        // Auto-hide after 5 seconds
        output_notification_time_ += ImGui::GetIO().DeltaTime;
        if (output_notification_time_ > 5.0f) {
            show_output_notification_ = false;
            output_notification_time_ = 0.0f;
        }
    }

    // Handle deferred tab close
    if (close_tab_index_ >= 0) {
        CloseFile(close_tab_index_);
        close_tab_index_ = -1;
    }

    ImGui::End();

    // Render modal dialogs (outside the main window)
    RenderSaveBeforeRunDialog();
    RenderSaveBeforeCloseDialog();
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
            // Show running indicator
            if (script_running_) {
                // Animated indicator
                const char* indicators[] = {"Running.", "Running..", "Running..."};
                int idx = static_cast<int>(running_indicator_time_ * 2) % 3;
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "%s", indicators[idx]);
                ImGui::Separator();
            }

            bool not_running = !script_running_;
            if (ImGui::MenuItem("Run Script", "F5", false, active_tab_index_ >= 0 && not_running)) {
                RunScript();
            }
            if (ImGui::MenuItem("Stop Script", "Shift+F5", false, script_running_)) {
                if (scripting_engine_) {
                    scripting_engine_->StopScript();
                    spdlog::info("Stop script requested");
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Run Selection", "F9", false, active_tab_index_ >= 0 && not_running)) {
                RunSelection();
            }
            if (ImGui::MenuItem("Run Section", "Ctrl+Enter", false, active_tab_index_ >= 0 && not_running)) {
                RunCurrentSection();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Debug", "F10", false, active_tab_index_ >= 0 && not_running)) {
                Debug();
            }
            ImGui::EndMenu();
        }

        // Security menu
        if (ImGui::BeginMenu("Security")) {
            bool sandbox_enabled = scripting_engine_ ? scripting_engine_->IsSandboxEnabled() : false;

            if (ImGui::MenuItem("Enable Sandbox", nullptr, &sandbox_enabled)) {
                if (scripting_engine_) {
                    scripting_engine_->EnableSandbox(sandbox_enabled);
                    spdlog::info("Sandbox {}", sandbox_enabled ? "enabled" : "disabled");
                }
            }

            ImGui::Separator();
            ImGui::Text("Sandbox Status:");
            if (sandbox_enabled) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "  Active - Scripts are sandboxed");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "  Inactive - Full Python access");
            }

            ImGui::Separator();
            ImGui::Text("Protected:");
            ImGui::BulletText("Blocks: exec, eval, open");
            ImGui::BulletText("Timeout: 60 seconds");
            ImGui::BulletText("Allowed: math, random, json");

            ImGui::EndMenu();
        }

        // View menu for theme and font settings
        if (ImGui::BeginMenu("View")) {
            // Theme submenu
            if (ImGui::BeginMenu("Theme")) {
                ImGui::TextDisabled("Popular");
                ImGui::Indent(10.0f);
                if (ImGui::MenuItem("Monokai", nullptr, current_theme_ == EditorTheme::Monokai)) {
                    current_theme_ = EditorTheme::Monokai;
                    ApplyThemeToAllTabs();
                }
                if (ImGui::MenuItem("Dracula", nullptr, current_theme_ == EditorTheme::Dracula)) {
                    current_theme_ = EditorTheme::Dracula;
                    ApplyThemeToAllTabs();
                }
                if (ImGui::MenuItem("One Dark", nullptr, current_theme_ == EditorTheme::OneDark)) {
                    current_theme_ = EditorTheme::OneDark;
                    ApplyThemeToAllTabs();
                }
                if (ImGui::MenuItem("GitHub", nullptr, current_theme_ == EditorTheme::GitHub)) {
                    current_theme_ = EditorTheme::GitHub;
                    ApplyThemeToAllTabs();
                }
                ImGui::Unindent(10.0f);
                ImGui::Separator();
                ImGui::TextDisabled("Classic");
                ImGui::Indent(10.0f);
                if (ImGui::MenuItem("Dark", nullptr, current_theme_ == EditorTheme::Dark)) {
                    current_theme_ = EditorTheme::Dark;
                    ApplyThemeToAllTabs();
                }
                if (ImGui::MenuItem("Light", nullptr, current_theme_ == EditorTheme::Light)) {
                    current_theme_ = EditorTheme::Light;
                    ApplyThemeToAllTabs();
                }
                if (ImGui::MenuItem("Retro Blue", nullptr, current_theme_ == EditorTheme::RetroBlu)) {
                    current_theme_ = EditorTheme::RetroBlu;
                    ApplyThemeToAllTabs();
                }
                ImGui::Unindent(10.0f);
                ImGui::EndMenu();
            }

            // Font Size submenu
            if (ImGui::BeginMenu("Font Size")) {
                if (ImGui::MenuItem("Small", nullptr, font_scale_ == 1.0f)) {
                    font_scale_ = 1.0f;
                }
                if (ImGui::MenuItem("Medium", nullptr, font_scale_ == 1.3f)) {
                    font_scale_ = 1.3f;
                }
                if (ImGui::MenuItem("Large", nullptr, font_scale_ == 1.6f)) {
                    font_scale_ = 1.6f;
                }
                if (ImGui::MenuItem("Extra Large", nullptr, font_scale_ == 2.0f)) {
                    font_scale_ = 2.0f;
                }
                ImGui::EndMenu();
            }

            // Tab Size submenu
            if (ImGui::BeginMenu("Tab Size")) {
                if (ImGui::MenuItem("2 Spaces", nullptr, tab_size_ == 2)) {
                    tab_size_ = 2;
                    ApplyTabSizeToAllTabs();
                }
                if (ImGui::MenuItem("4 Spaces", nullptr, tab_size_ == 4)) {
                    tab_size_ = 4;
                    ApplyTabSizeToAllTabs();
                }
                if (ImGui::MenuItem("8 Spaces", nullptr, tab_size_ == 8)) {
                    tab_size_ = 8;
                    ApplyTabSizeToAllTabs();
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            // Syntax Highlighting toggle
            if (ImGui::MenuItem("Syntax Highlighting", nullptr, &syntax_highlighting_)) {
                ApplySyntaxHighlightingToAllTabs();
            }

            // Show Whitespace toggle
            if (ImGui::MenuItem("Show Whitespace", nullptr, &show_whitespace_)) {
                // Apply to all tabs
                for (auto& tab : tabs_) {
                    tab->editor.SetShowWhitespaces(show_whitespace_);
                }
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
}

void ScriptEditorPanel::ApplyThemeToAllTabs() {
    TextEditor::Palette palette;
    const char* theme_name = "Unknown";

    switch (current_theme_) {
        case EditorTheme::Dark:
            palette = TextEditor::GetDarkPalette();
            theme_name = "Dark";
            break;
        case EditorTheme::Light:
            palette = TextEditor::GetLightPalette();
            theme_name = "Light";
            break;
        case EditorTheme::RetroBlu:
            palette = TextEditor::GetRetroBluePalette();
            theme_name = "Retro Blue";
            break;
        case EditorTheme::Monokai:
            palette = GetMonokaiPalette();
            theme_name = "Monokai";
            break;
        case EditorTheme::Dracula:
            palette = GetDraculaPalette();
            theme_name = "Dracula";
            break;
        case EditorTheme::OneDark:
            palette = GetOneDarkPalette();
            theme_name = "One Dark";
            break;
        case EditorTheme::GitHub:
            palette = GetGitHubPalette();
            theme_name = "GitHub";
            break;
    }

    for (auto& tab : tabs_) {
        tab->editor.SetPalette(palette);
    }

    spdlog::info("Applied editor theme: {}", theme_name);
}

void ScriptEditorPanel::ApplyTabSizeToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetTabSize(tab_size_);
    }
    spdlog::info("Applied tab size: {} spaces", tab_size_);
}

void ScriptEditorPanel::ApplySyntaxHighlightingToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetColorizerEnable(syntax_highlighting_);
    }
    spdlog::info("Syntax highlighting: {}", syntax_highlighting_ ? "enabled" : "disabled");
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

    // Apply font scale for editor
    if (font_scale_ != 1.0f) {
        ImGui::SetWindowFontScale(font_scale_);
    }

    // Render the TextEditor
    ImVec2 editor_size = ImVec2(0, -ImGui::GetFrameHeightWithSpacing());
    tab->editor.Render("##editor", editor_size);

    // Reset font scale
    if (font_scale_ != 1.0f) {
        ImGui::SetWindowFontScale(1.0f);
    }

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

        // Script running indicator
        if (script_running_) {
            ImGui::SameLine();
            ImGui::Text("|");
            ImGui::SameLine();
            // Animated running indicator
            const char* indicators[] = {".", "..", "..."};
            int idx = static_cast<int>(running_indicator_time_ * 2) % 3;
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "RUNNING%s (Shift+F5 to stop)", indicators[idx]);
        }

        // Sandbox indicator
        if (scripting_engine_) {
            ImGui::SameLine();
            ImGui::Text("|");
            ImGui::SameLine();
            bool sandbox_on = scripting_engine_->IsSandboxEnabled();
            if (sandbox_on) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "SANDBOX ON");
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Sandbox Off");
            }
        }

        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        ImGui::Text("%s", tab->filepath.empty() ? "Untitled" : tab->filepath.c_str());
    }
}

void ScriptEditorPanel::HandleKeyboardShortcuts() {
    ImGuiIO& io = ImGui::GetIO();

    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // File operations
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_N)) {
        NewFile();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_O)) {
        OpenFile();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && active_tab_index_ >= 0) {
        SaveFile();
    }
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && active_tab_index_ >= 0) {
        SaveFileAs();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_W) && active_tab_index_ >= 0) {
        close_tab_index_ = active_tab_index_;
    }

    // Edit operations (handled by TextEditor internally, but we can add extra handling)
    // The TextEditor component already handles Ctrl+Z, Ctrl+Y, Ctrl+X, Ctrl+C, Ctrl+V, Ctrl+A

    // Execution shortcuts
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F5) && !script_running_) {
        RunScript();
    }
    // Stop script with Shift+F5
    if (!ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F5) && script_running_) {
        if (scripting_engine_) {
            scripting_engine_->StopScript();
            spdlog::info("Stop script requested via Shift+F5");
        }
    }
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F9) && !script_running_) {
        RunSelection();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Enter) && !script_running_) {
        RunCurrentSection();
    }
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F10) && !script_running_) {
        Debug();
    }
}

void ScriptEditorPanel::NewFile() {
    auto tab = std::make_unique<EditorTab>();
    tab->filename = "Untitled" + std::to_string(tabs_.size() + 1) + ".cyx";
    tab->filepath = "";
    tab->is_new = true;
    tab->is_modified = false;

    // Configure editor with C++ language def (works for Python too - similar syntax)
    auto lang = TextEditor::LanguageDefinition::CPlusPlus();

    // Override for Python-specific keywords
    lang.mKeywords.clear();
    static const char* const py_keywords[] = {
        "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
        "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
        "while", "with", "yield", "async", "await", "print", "len", "range", "str", "int"
    };
    for (auto& k : py_keywords)
        lang.mKeywords.insert(k);

    lang.mSingleLineComment = "#";
    lang.mCommentStart = "\"\"\"";
    lang.mCommentEnd = "\"\"\"";
    lang.mName = "Python";

    tab->editor.SetLanguageDefinition(lang);
    // Apply current theme
    switch (current_theme_) {
        case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
        case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
        case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
    }
    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetTabSize(4);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);

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

    // Check if we can replace an existing empty untitled tab
    // Look through ALL tabs, not just the active one
    int empty_tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        auto& tab = tabs_[i];
        // Check if this is an empty, unmodified untitled tab
        std::string tab_text = tab->editor.GetText();
        // Trim whitespace for comparison
        tab_text.erase(0, tab_text.find_first_not_of(" \t\n\r"));
        tab_text.erase(tab_text.find_last_not_of(" \t\n\r") + 1);

        if (tab->is_new && !tab->is_modified && tab_text.empty()) {
            empty_tab_index = i;
            break;  // Found first empty untitled tab
        }
    }

    // Replace the empty untitled tab if found
    if (empty_tab_index >= 0) {
        auto& tab = tabs_[empty_tab_index];
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->editor.SetText(content);
        active_tab_index_ = empty_tab_index;  // Switch to this tab
        request_focus_ = true;
        spdlog::info("Replaced empty untitled tab at index {} with file: {}", empty_tab_index, path);
        return;
    }

    // Create new tab
    auto tab = std::make_unique<EditorTab>();
    tab->filename = std::filesystem::path(path).filename().string();
    tab->filepath = path;
    tab->is_new = false;
    tab->is_modified = false;

    // Configure editor with C++ language def (works for Python too - similar syntax)
    auto lang = TextEditor::LanguageDefinition::CPlusPlus();

    // Override for Python-specific keywords
    lang.mKeywords.clear();
    static const char* const py_keywords[] = {
        "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
        "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
        "while", "with", "yield", "async", "await", "print", "len", "range", "str", "int"
    };
    for (auto& k : py_keywords)
        lang.mKeywords.insert(k);

    lang.mSingleLineComment = "#";
    lang.mCommentStart = "\"\"\"";
    lang.mCommentEnd = "\"\"\"";
    lang.mName = "Python";

    tab->editor.SetLanguageDefinition(lang);
    // Apply current theme
    switch (current_theme_) {
        case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
        case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
        case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
    }
    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetTabSize(4);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);
    tab->editor.SetText(content);

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    request_focus_ = true;

    spdlog::info("Opened file: {}", path);
}

void ScriptEditorPanel::LoadGeneratedCode(const std::string& code, const std::string& framework_name) {
    std::string target_filename = "generated_" + framework_name + ".py";

    // Check if a tab with this filename already exists
    int existing_tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filename == target_filename) {
            existing_tab_index = i;
            break;
        }
    }

    if (existing_tab_index >= 0) {
        // Update existing tab
        auto& tab = tabs_[existing_tab_index];
        tab->editor.SetText(code);
        tab->is_modified = true;
        active_tab_index_ = existing_tab_index;
        request_focus_ = true;
        request_window_focus_ = true;
        spdlog::info("Updated existing {} code tab", framework_name);
    } else {
        // Create new tab with generated code
        auto tab = std::make_unique<EditorTab>();
        tab->filename = target_filename;
        tab->filepath = "";  // Not saved yet
        tab->is_new = true;
        tab->is_modified = true;  // Has content, mark as modified

        // Configure Python language syntax highlighting
        auto lang = TextEditor::LanguageDefinition::CPlusPlus();
        lang.mKeywords.clear();
        static const char* const py_keywords[] = {
            "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
            "while", "with", "yield", "async", "await", "print", "len", "range", "str", "int"
        };
        for (auto& k : py_keywords)
            lang.mKeywords.insert(k);

        lang.mSingleLineComment = "#";
        lang.mCommentStart = "\"\"\"";
        lang.mCommentEnd = "\"\"\"";
        lang.mName = "Python";

        tab->editor.SetLanguageDefinition(lang);
        // Apply current theme
        switch (current_theme_) {
            case EditorTheme::Dark: tab->editor.SetPalette(TextEditor::GetDarkPalette()); break;
            case EditorTheme::Light: tab->editor.SetPalette(TextEditor::GetLightPalette()); break;
            case EditorTheme::RetroBlu: tab->editor.SetPalette(TextEditor::GetRetroBluePalette()); break;
        }
        tab->editor.SetShowWhitespaces(show_whitespace_);
        tab->editor.SetTabSize(4);
        tab->editor.SetImGuiChildIgnored(false);
        tab->editor.SetReadOnly(false);
        tab->editor.SetText(code);

        tabs_.push_back(std::move(tab));
        active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
        request_focus_ = true;
        request_window_focus_ = true;

        spdlog::info("Loaded generated {} code into new tab", framework_name);
    }
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

    // Check if file has unsaved changes
    if (tabs_[tab_index]->is_modified || tabs_[tab_index]->is_new) {
        // Don't close immediately - show confirmation dialog
        pending_close_tab_index_ = tab_index;
        show_save_before_close_dialog_ = true;
        spdlog::info("File has unsaved changes, showing save dialog: {}", tabs_[tab_index]->filename);
        return;
    }

    // File is saved, close directly
    DoCloseFile(tab_index);
}

void ScriptEditorPanel::DoCloseFile(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    spdlog::info("Closing file: {}", tabs_[tab_index]->filename);
    tabs_.erase(tabs_.begin() + tab_index);

    // Adjust active tab index
    if (tabs_.empty()) {
        // Create new empty tab if all closed
        NewFile();
    } else if (active_tab_index_ >= static_cast<int>(tabs_.size())) {
        active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
    }

    // Reset pending close state
    pending_close_tab_index_ = -1;
}

void ScriptEditorPanel::RunScript() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    // Don't start if already running
    if (script_running_) {
        spdlog::warn("Script already running");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Check if file is unsaved (new or modified) - prompt to save first
    if (tab->is_new || tab->is_modified) {
        spdlog::info("Script is unsaved, prompting to save before run");
        show_save_before_run_dialog_ = true;
        return;
    }

    // File is saved, run it directly
    DoRunScript();
}

void ScriptEditorPanel::DoRunScript() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];

    spdlog::info("Running script asynchronously: {}", tab->filename);

    // Get script text - prefer file if it exists
    std::string script;
    if (!tab->filepath.empty() && std::filesystem::exists(tab->filepath)) {
        // Read file content
        std::ifstream file(tab->filepath);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            script = buffer.str();
            file.close();
        }
    }

    if (script.empty()) {
        // Use text from editor
        std::string script_text = tab->editor.GetText();

        // Strip out %% markers before executing
        std::istringstream stream(script_text);
        std::string line;
        while (std::getline(stream, line)) {
            // Skip lines containing only %% markers
            if (line.find("%%") == std::string::npos) {
                script += line + "\n";
            }
        }
    }

    // Show running indicator in command window
    if (command_window_) {
        command_window_->DisplayScriptOutput(tab->filename, "Script started...", false);
    }

    // Execute asynchronously
    scripting_engine_->ExecuteScriptAsync(script);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

void ScriptEditorPanel::RunSelection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];
    std::string selected_text = tab->editor.GetSelectedText();

    if (selected_text.empty()) {
        spdlog::warn("No text selected");
        if (command_window_) {
            command_window_->DisplayScriptOutput(tab->filename, "No text selected", true);
        } else {
            last_execution_output_ = "No text selected";
            show_output_notification_ = true;
            output_notification_time_ = 0.0f;
        }
        return;
    }

    spdlog::info("Running selection");
    auto result = scripting_engine_->ExecuteCommand(selected_text);

    // Send output to Command Window if available
    if (command_window_) {
        if (!result.success) {
            command_window_->DisplayScriptOutput(tab->filename + " (selection)", "Error: " + result.error_message, true);
        } else {
            std::string output = result.output.empty() ? "Selection executed successfully (no output)" : result.output;
            command_window_->DisplayScriptOutput(tab->filename + " (selection)", output, false);
        }
    } else {
        // Fallback to notification
        if (!result.success) {
            spdlog::error("Execution error: {}", result.error_message);
            last_execution_output_ = "Error: " + result.error_message;
            printf("[Selection Error] %s\n", result.error_message.c_str());
        } else {
            last_execution_output_ = result.output.empty() ? "Selection executed successfully" : result.output;
            if (!result.output.empty()) {
                printf("[Selection Output]\n%s\n", result.output.c_str());
            }
        }
        show_output_notification_ = true;
        output_notification_time_ = 0.0f;
    }
}

void ScriptEditorPanel::RunCurrentSection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;

    auto& tab = tabs_[active_tab_index_];
    Section section = GetCurrentSection();

    if (section.code.empty()) {
        spdlog::warn("No section found at cursor");
        if (command_window_) {
            command_window_->DisplayScriptOutput(tab->filename, "No section found at cursor", true);
        } else {
            last_execution_output_ = "No section found at cursor";
            show_output_notification_ = true;
            output_notification_time_ = 0.0f;
        }
        return;
    }

    spdlog::info("Running section (lines {}-{})", section.start_line, section.end_line);
    auto result = scripting_engine_->ExecuteScript(section.code);

    // Send output to Command Window if available
    std::string section_name = tab->filename + " (lines " +
                              std::to_string(section.start_line) + "-" +
                              std::to_string(section.end_line) + ")";

    if (command_window_) {
        if (!result.success) {
            command_window_->DisplayScriptOutput(section_name, "Error: " + result.error_message, true);
        } else {
            std::string output = result.output.empty() ? "Section executed successfully (no output)" : result.output;
            command_window_->DisplayScriptOutput(section_name, output, false);
        }
    } else {
        // Fallback to notification
        if (!result.success) {
            spdlog::error("Section execution error: {}", result.error_message);
            last_execution_output_ = "Section Error: " + result.error_message;
            printf("[Section Error] %s\n", result.error_message.c_str());
        } else {
            last_execution_output_ = "Section executed successfully (lines " +
                                    std::to_string(section.start_line) + "-" +
                                    std::to_string(section.end_line) + ")";
            if (!result.output.empty()) {
                last_execution_output_ += "\nOutput: " + result.output;
                printf("[Section Output]\n%s\n", result.output.c_str());
            }
        }
        show_output_notification_ = true;
        output_notification_time_ = 0.0f;
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

    spdlog::debug("GetCurrentSection: cursor at line {}, found {} sections", current_line, sections.size());
    for (size_t i = 0; i < sections.size(); i++) {
        spdlog::debug("  Section {}: lines {}-{}", i, sections[i].start_line, sections[i].end_line);
    }

    // Find section containing cursor
    for (const auto& section : sections) {
        if (current_line >= section.start_line && current_line <= section.end_line) {
            spdlog::debug("  -> Found section containing cursor at lines {}-{}", section.start_line, section.end_line);
            return section;
        }
    }

    // If cursor is on a %% marker line, find the nearest section
    // Check if current line contains %%
    std::istringstream stream(text);
    std::string line;
    int line_num = 0;
    while (std::getline(stream, line) && line_num <= current_line) {
        if (line_num == current_line && line.find("%%") != std::string::npos) {
            // Cursor is on a %% line, return the section after it
            for (const auto& section : sections) {
                if (section.start_line > current_line) {
                    return section;
                }
            }
        }
        line_num++;
    }

    return empty_section;
}

TextEditor::LanguageDefinition ScriptEditorPanel::CreatePythonLanguage() {
    static bool inited = false;
    static TextEditor::LanguageDefinition lang;

    if (!inited) {
        lang.mName = "Python";
        lang.mCaseSensitive = true;
        lang.mAutoIndentation = true;

        // Comment markers
        lang.mSingleLineComment = "#";
        lang.mCommentStart = "\"\"\"";
        lang.mCommentEnd = "\"\"\"";

        // Add preprocessor patterns for %% section markers
        lang.mPreprocChar = '%';

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

        inited = true;
    }

    return lang;
}

// ==================== Custom Theme Palettes ====================

TextEditor::Palette ScriptEditorPanel::GetMonokaiPalette() {
    // Monokai theme - popular dark theme with vibrant colors
    return TextEditor::Palette{{
        0xfff8f8f2, // Default (light gray)
        0xfff92672, // Keyword (pink)
        0xffae81ff, // Number (purple)
        0xffe6db74, // String (yellow)
        0xffe6db74, // Char literal (yellow)
        0xfff8f8f2, // Punctuation (light gray)
        0xffa6e22e, // Preprocessor (green)
        0xfff8f8f2, // Identifier (light gray)
        0xff66d9ef, // Known identifier (cyan)
        0xffa6e22e, // Preproc identifier (green)
        0xff75715e, // Comment (gray)
        0xff75715e, // Multi-line comment (gray)
        0xff272822, // Background (dark gray-green)
        0xffe0e0e0, // Cursor (white)
        0x80494440, // Selection (translucent)
        0xa0ff5555, // Error marker (red)
        0x80f92672, // Breakpoint (pink)
        0xff90908a, // Line number (gray)
        0x40808080, // Current line fill
        0x30808080, // Current line fill inactive
        0x40808080  // Current line edge
    }};
}

TextEditor::Palette ScriptEditorPanel::GetDraculaPalette() {
    // Dracula theme - dark theme with purple accents
    return TextEditor::Palette{{
        0xfff8f8f2, // Default (foreground)
        0xffff79c6, // Keyword (pink)
        0xffbd93f9, // Number (purple)
        0xfff1fa8c, // String (yellow)
        0xfff1fa8c, // Char literal (yellow)
        0xfff8f8f2, // Punctuation (foreground)
        0xffff79c6, // Preprocessor (pink)
        0xfff8f8f2, // Identifier (foreground)
        0xff8be9fd, // Known identifier (cyan)
        0xff50fa7b, // Preproc identifier (green)
        0xff6272a4, // Comment (comment blue-gray)
        0xff6272a4, // Multi-line comment
        0xff282a36, // Background (dark purple-gray)
        0xfff8f8f2, // Cursor (white)
        0x8044475a, // Selection (translucent)
        0xa0ff5555, // Error marker (red)
        0x80ff79c6, // Breakpoint (pink)
        0xff6272a4, // Line number (comment color)
        0x40404050, // Current line fill
        0x30404050, // Current line fill inactive
        0x40404050  // Current line edge
    }};
}

TextEditor::Palette ScriptEditorPanel::GetOneDarkPalette() {
    // One Dark theme - Atom editor inspired
    return TextEditor::Palette{{
        0xffabb2bf, // Default (light gray)
        0xffc678dd, // Keyword (purple)
        0xffd19a66, // Number (orange)
        0xff98c379, // String (green)
        0xff98c379, // Char literal (green)
        0xffabb2bf, // Punctuation (light gray)
        0xffc678dd, // Preprocessor (purple)
        0xffe06c75, // Identifier (red)
        0xff61afef, // Known identifier (blue)
        0xffe5c07b, // Preproc identifier (yellow)
        0xff5c6370, // Comment (gray)
        0xff5c6370, // Multi-line comment (gray)
        0xff282c34, // Background (dark gray)
        0xffabb2bf, // Cursor (white)
        0x803e4451, // Selection (translucent)
        0xa0e06c75, // Error marker (red)
        0x80c678dd, // Breakpoint (purple)
        0xff4b5263, // Line number (gray)
        0x20ffffff, // Current line fill
        0x15ffffff, // Current line fill inactive
        0x20ffffff  // Current line edge
    }};
}

TextEditor::Palette ScriptEditorPanel::GetGitHubPalette() {
    // GitHub Light theme - clean light theme
    return TextEditor::Palette{{
        0xff24292e, // Default (dark gray)
        0xffd73a49, // Keyword (red)
        0xff005cc5, // Number (blue)
        0xff032f62, // String (dark blue)
        0xff032f62, // Char literal (dark blue)
        0xff24292e, // Punctuation (dark gray)
        0xff6f42c1, // Preprocessor (purple)
        0xff24292e, // Identifier (dark gray)
        0xff6f42c1, // Known identifier (purple)
        0xff22863a, // Preproc identifier (green)
        0xff6a737d, // Comment (gray)
        0xff6a737d, // Multi-line comment (gray)
        0xffffffff, // Background (white)
        0xff24292e, // Cursor (dark)
        0x400366d6, // Selection (translucent blue)
        0x40cb2431, // Error marker (red)
        0x40d73a49, // Breakpoint (red)
        0xff959da5, // Line number (light gray)
        0x10000000, // Current line fill
        0x08000000, // Current line fill inactive
        0x10000000  // Current line edge
    }};
}

// ==================== Unsaved Files Check ====================

bool ScriptEditorPanel::HasUnsavedFiles() const {
    for (const auto& tab : tabs_) {
        if (tab->is_modified || tab->is_new) {
            // Check if tab has any content (not just an empty new file)
            std::string text = tab->editor.GetText();
            // Trim whitespace
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                // Has content that is unsaved
                return true;
            }
        }
    }
    return false;
}

std::vector<std::string> ScriptEditorPanel::GetUnsavedFileNames() const {
    std::vector<std::string> names;
    for (const auto& tab : tabs_) {
        if (tab->is_modified || tab->is_new) {
            // Check if tab has any content
            std::string text = tab->editor.GetText();
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                names.push_back(tab->filename);
            }
        }
    }
    return names;
}

void ScriptEditorPanel::SaveAllFiles() {
    int original_active = active_tab_index_;

    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        auto& tab = tabs_[i];
        if (tab->is_modified || tab->is_new) {
            // Check if tab has content
            std::string text = tab->editor.GetText();
            size_t start = text.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                // Has content, save it
                active_tab_index_ = i;
                SaveFile();
            }
        }
    }

    // Restore original active tab
    if (original_active >= 0 && original_active < static_cast<int>(tabs_.size())) {
        active_tab_index_ = original_active;
    }
}

// ==================== Save Confirmation Dialogs ====================

void ScriptEditorPanel::RenderSaveBeforeRunDialog() {
    if (!show_save_before_run_dialog_) return;

    ImGui::OpenPopup("Save Before Running?");

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Save Before Running?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        std::string filename = "Untitled";
        if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
            filename = tabs_[active_tab_index_]->filename;
        }

        ImGui::Text("The script '%s' has unsaved changes.", filename.c_str());
        ImGui::Text("Would you like to save before running?");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Calculate button sizes for better layout
        float button_width = 100.0f;
        float total_width = button_width * 2 + ImGui::GetStyle().ItemSpacing.x;
        float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;
        ImGui::SetCursorPosX(start_x);

        if (ImGui::Button("Save & Run", ImVec2(button_width, 0))) {
            show_save_before_run_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Save the file first
            SaveFile();

            // Check if save was successful (file is no longer new/modified after SaveFileAs completes)
            if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[active_tab_index_];
                // If user completed save (file has a path now), run the script
                if (!tab->filepath.empty()) {
                    DoRunScript();
                } else {
                    spdlog::info("Save cancelled, script not run");
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            show_save_before_run_dialog_ = false;
            ImGui::CloseCurrentPopup();
            spdlog::info("Run cancelled by user");
        }

        ImGui::EndPopup();
    }
}

void ScriptEditorPanel::RenderSaveBeforeCloseDialog() {
    if (!show_save_before_close_dialog_) return;

    ImGui::OpenPopup("Save Changes?");

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Save Changes?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        std::string filename = "Untitled";
        if (pending_close_tab_index_ >= 0 && pending_close_tab_index_ < static_cast<int>(tabs_.size())) {
            filename = tabs_[pending_close_tab_index_]->filename;
        }

        ImGui::Text("Save changes to '%s'?", filename.c_str());
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Calculate button sizes for better layout
        float button_width = 80.0f;
        float total_width = button_width * 3 + ImGui::GetStyle().ItemSpacing.x * 2;
        float start_x = (ImGui::GetWindowWidth() - total_width) * 0.5f;
        ImGui::SetCursorPosX(start_x);

        if (ImGui::Button("Yes", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Switch to the tab being closed and save it
            int original_active = active_tab_index_;
            active_tab_index_ = pending_close_tab_index_;
            SaveFile();

            // Check if save was successful
            if (pending_close_tab_index_ >= 0 && pending_close_tab_index_ < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[pending_close_tab_index_];
                if (!tab->filepath.empty()) {
                    // Save completed, close the tab
                    DoCloseFile(pending_close_tab_index_);
                } else {
                    // Save was cancelled, don't close
                    spdlog::info("Save cancelled, tab not closed");
                    active_tab_index_ = original_active;
                }
            }
            pending_close_tab_index_ = -1;
        }

        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();

            // Close without saving
            if (pending_close_tab_index_ >= 0) {
                DoCloseFile(pending_close_tab_index_);
            }
            pending_close_tab_index_ = -1;
            spdlog::info("Closed without saving");
        }

        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(button_width, 0))) {
            show_save_before_close_dialog_ = false;
            ImGui::CloseCurrentPopup();
            pending_close_tab_index_ = -1;
            spdlog::info("Close cancelled by user");
        }

        ImGui::EndPopup();
    }
}

} // namespace cyxwiz
