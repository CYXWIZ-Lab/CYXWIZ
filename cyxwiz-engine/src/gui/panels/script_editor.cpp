#include "script_editor.h"
#include "command_window.h"
#include "../../scripting/scripting_engine.h"
#include <imgui.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <regex>
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

        // View menu - quick access to editor settings (synced with Preferences)
        if (ImGui::BeginMenu("View")) {
            // Theme submenu
            if (ImGui::BeginMenu("Theme")) {
                ImGui::TextDisabled("Popular");
                ImGui::Indent(10.0f);
                if (ImGui::MenuItem("Monokai", nullptr, current_theme_ == EditorTheme::Monokai)) {
                    current_theme_ = EditorTheme::Monokai;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Dracula", nullptr, current_theme_ == EditorTheme::Dracula)) {
                    current_theme_ = EditorTheme::Dracula;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("One Dark", nullptr, current_theme_ == EditorTheme::OneDark)) {
                    current_theme_ = EditorTheme::OneDark;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("GitHub", nullptr, current_theme_ == EditorTheme::GitHub)) {
                    current_theme_ = EditorTheme::GitHub;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                ImGui::Unindent(10.0f);
                ImGui::Separator();
                ImGui::TextDisabled("Classic");
                ImGui::Indent(10.0f);
                if (ImGui::MenuItem("Dark", nullptr, current_theme_ == EditorTheme::Dark)) {
                    current_theme_ = EditorTheme::Dark;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Light", nullptr, current_theme_ == EditorTheme::Light)) {
                    current_theme_ = EditorTheme::Light;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Retro Blue", nullptr, current_theme_ == EditorTheme::RetroBlu)) {
                    current_theme_ = EditorTheme::RetroBlu;
                    ApplyThemeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                ImGui::Unindent(10.0f);
                ImGui::EndMenu();
            }

            // Font Size submenu
            if (ImGui::BeginMenu("Font Size")) {
                if (ImGui::MenuItem("Small", nullptr, font_scale_ == 1.0f)) {
                    font_scale_ = 1.0f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Medium", nullptr, font_scale_ == 1.3f)) {
                    font_scale_ = 1.3f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Large", nullptr, font_scale_ == 1.6f)) {
                    font_scale_ = 1.6f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Extra Large", nullptr, font_scale_ == 2.0f)) {
                    font_scale_ = 2.0f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                ImGui::EndMenu();
            }

            // Tab Size submenu
            if (ImGui::BeginMenu("Tab Size")) {
                if (ImGui::MenuItem("2 Spaces", nullptr, tab_size_ == 2)) {
                    tab_size_ = 2;
                    ApplyTabSizeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("4 Spaces", nullptr, tab_size_ == 4)) {
                    tab_size_ = 4;
                    ApplyTabSizeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("8 Spaces", nullptr, tab_size_ == 8)) {
                    tab_size_ = 8;
                    ApplyTabSizeToAllTabs();
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
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
                for (auto& tab : tabs_) {
                    tab->editor.SetShowWhitespaces(show_whitespace_);
                }
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }

            ImGui::Separator();
            ImGui::TextDisabled("Also in: Edit > Preferences > Editor");

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

            // Use unique ID to avoid issues with duplicate filenames
            std::string tab_id = tab_label + "##" + std::to_string(i);

            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;
            if (request_focus_ && i == active_tab_index_) {
                tab_flags |= ImGuiTabItemFlags_SetSelected;
            }

            bool open = true;
            if (ImGui::BeginTabItem(tab_id.c_str(), &open, tab_flags)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }

            // Handle tab close
            if (!open) {
                close_tab_index_ = i;
            }
        }

        // Clear request_focus_ after processing all tabs
        request_focus_ = false;

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
            request_window_focus_ = true;  // Focus the Script Editor window
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
        request_window_focus_ = true;  // Focus the Script Editor window
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
    request_window_focus_ = true;  // Focus the Script Editor window

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


// ==================== Find/Replace Operations ====================

bool ScriptEditorPanel::FindInEditor(const std::string& search_text, bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    if (search_text.empty()) {
        return false;
    }

    // Store search parameters for FindNext/FindPrevious
    last_search_text_ = search_text;
    last_case_sensitive_ = case_sensitive;
    last_whole_word_ = whole_word;
    last_use_regex_ = use_regex;

    auto& editor = tabs_[active_tab_index_]->editor;
    std::string text = editor.GetText();

    // Get current cursor position
    auto cursor = editor.GetCursorPosition();
    int start_pos = 0;

    // Convert cursor position to character offset
    auto lines = editor.GetTextLines();
    for (int i = 0; i < cursor.mLine && i < static_cast<int>(lines.size()); ++i) {
        start_pos += static_cast<int>(lines[i].length()) + 1;  // +1 for newline
    }
    start_pos += cursor.mColumn;

    // If there's a selection that matches the search text, skip past it
    if (editor.HasSelection()) {
        std::string selected = editor.GetSelectedText();
        std::string match_text = selected;
        std::string search_check = search_text;

        if (!case_sensitive) {
            std::transform(match_text.begin(), match_text.end(), match_text.begin(), ::tolower);
            std::transform(search_check.begin(), search_check.end(), search_check.begin(), ::tolower);
        }

        if (match_text == search_check) {
            // Skip past the current selection
            start_pos += static_cast<int>(selected.length());
        }
    }

    // Search from current position
    size_t found_pos = std::string::npos;
    size_t match_len = search_text.length();

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);
            std::smatch match;
            std::string search_area = text.substr(start_pos);

            if (std::regex_search(search_area, match, re)) {
                found_pos = start_pos + match.position(0);
                match_len = match.length(0);
            } else {
                // Wrap around and search from beginning
                if (std::regex_search(text, match, re)) {
                    found_pos = match.position(0);
                    match_len = match.length(0);
                }
            }
        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex: {}", e.what());
            return false;
        }
    } else {
        std::string search_text_lower = search_text;
        std::string text_lower = text;

        if (!case_sensitive) {
            std::transform(search_text_lower.begin(), search_text_lower.end(),
                           search_text_lower.begin(), ::tolower);
            std::transform(text_lower.begin(), text_lower.end(),
                           text_lower.begin(), ::tolower);
        }

        // Search from current position
        found_pos = text_lower.find(search_text_lower, start_pos);

        // Wrap around if not found
        if (found_pos == std::string::npos) {
            found_pos = text_lower.find(search_text_lower, 0);
        }

        // Check whole word boundary
        if (found_pos != std::string::npos && whole_word) {
            bool start_ok = (found_pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[found_pos - 1]));
            bool end_ok = (found_pos + search_text_lower.length() >= text_lower.length()) ||
                          !std::isalnum(static_cast<unsigned char>(text_lower[found_pos + search_text_lower.length()]));
            if (!start_ok || !end_ok) {
                found_pos = std::string::npos;
            }
        }
    }

    if (found_pos != std::string::npos) {
        // Convert character offset to line/column
        int line = 0;
        int col = 0;
        size_t pos = 0;

        for (const auto& line_text : lines) {
            if (pos + line_text.length() >= found_pos) {
                col = static_cast<int>(found_pos - pos);
                break;
            }
            pos += line_text.length() + 1;  // +1 for newline
            line++;
        }

        // Select the found text
        TextEditor::Coordinates start_coord(line, col);
        TextEditor::Coordinates end_coord(line, col + static_cast<int>(match_len));

        // Handle multi-line match
        size_t end_pos = found_pos + match_len;
        pos = 0;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (pos + lines[i].length() >= end_pos) {
                end_coord.mLine = i;
                end_coord.mColumn = static_cast<int>(end_pos - pos);
                break;
            }
            pos += lines[i].length() + 1;
        }

        editor.SetSelection(start_coord, end_coord);
        editor.SetCursorPosition(start_coord);

        spdlog::info("Found '{}' at line {}, col {}", search_text, line + 1, col + 1);
        return true;
    }

    spdlog::info("'{}' not found", search_text);
    return false;
}

bool ScriptEditorPanel::FindNext() {
    if (last_search_text_.empty()) {
        return false;
    }

    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If there's a selection (current match), move cursor past it before searching
    if (editor.HasSelection()) {
        // Get selection end and move cursor there
        auto cursor = editor.GetCursorPosition();
        auto text = editor.GetText();
        auto lines = editor.GetTextLines();

        // Calculate character offset of current position
        int current_offset = 0;
        for (int i = 0; i < cursor.mLine && i < static_cast<int>(lines.size()); ++i) {
            current_offset += static_cast<int>(lines[i].length()) + 1;
        }
        current_offset += cursor.mColumn;

        // Move cursor forward by the length of the search text to skip current match
        int new_offset = current_offset + static_cast<int>(last_search_text_.length());

        // Convert back to coordinates
        int line = 0;
        int col = 0;
        int pos = 0;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (pos + static_cast<int>(lines[i].length()) >= new_offset) {
                line = i;
                col = new_offset - pos;
                break;
            }
            pos += static_cast<int>(lines[i].length()) + 1;
            line = i + 1;
            col = 0;
        }

        // Clear selection and set cursor to end of current match
        editor.SetCursorPosition(TextEditor::Coordinates(line, col));
    }

    return FindInEditor(last_search_text_, last_case_sensitive_, last_whole_word_, last_use_regex_);
}

bool ScriptEditorPanel::FindPrevious() {
    // TODO: Implement backward search
    spdlog::info("FindPrevious not yet implemented");
    return false;
}

bool ScriptEditorPanel::Replace(const std::string& search_text, const std::string& replace_text,
                                 bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return false;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If there's a selection matching the search text, replace it
    if (editor.HasSelection()) {
        std::string selected = editor.GetSelectedText();
        std::string match_text = selected;
        std::string search_check = search_text;

        if (!case_sensitive) {
            std::transform(match_text.begin(), match_text.end(), match_text.begin(), ::tolower);
            std::transform(search_check.begin(), search_check.end(), search_check.begin(), ::tolower);
        }

        if (match_text == search_check) {
            // Replace the selection
            editor.Delete();

            auto cursor = editor.GetCursorPosition();
            editor.InsertText(replace_text);

            tabs_[active_tab_index_]->is_modified = true;

            spdlog::info("Replaced '{}' with '{}'", search_text, replace_text);

            // Find next occurrence
            FindInEditor(search_text, case_sensitive, whole_word, use_regex);
            return true;
        }
    }

    // No valid selection, find next occurrence
    return FindInEditor(search_text, case_sensitive, whole_word, use_regex);
}

int ScriptEditorPanel::ReplaceAll(const std::string& search_text, const std::string& replace_text,
                                   bool case_sensitive, bool whole_word, bool use_regex) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return 0;
    }

    if (search_text.empty()) {
        return 0;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    std::string text = editor.GetText();
    int count = 0;

    if (use_regex) {
        try {
            std::regex::flag_type flags = std::regex::ECMAScript;
            if (!case_sensitive) flags |= std::regex::icase;

            std::regex re(search_text, flags);

            // Count matches
            std::string temp = text;
            std::smatch match;
            while (std::regex_search(temp, match, re)) {
                count++;
                temp = match.suffix().str();
            }

            // Replace all
            std::string result = std::regex_replace(text, re, replace_text);
            editor.SetText(result);

        } catch (const std::regex_error& e) {
            spdlog::warn("Invalid regex: {}", e.what());
            return 0;
        }
    } else {
        std::string search_lower = search_text;
        std::string text_lower = text;

        if (!case_sensitive) {
            std::transform(search_lower.begin(), search_lower.end(), search_lower.begin(), ::tolower);
            std::transform(text_lower.begin(), text_lower.end(), text_lower.begin(), ::tolower);
        }

        // Find and replace all occurrences
        std::string result;
        size_t pos = 0;
        size_t last_pos = 0;

        while ((pos = text_lower.find(search_lower, last_pos)) != std::string::npos) {
            bool match_ok = true;

            if (whole_word) {
                bool start_ok = (pos == 0) || !std::isalnum(static_cast<unsigned char>(text_lower[pos - 1]));
                bool end_ok = (pos + search_lower.length() >= text_lower.length()) ||
                              !std::isalnum(static_cast<unsigned char>(text_lower[pos + search_lower.length()]));
                match_ok = start_ok && end_ok;
            }

            if (match_ok) {
                result += text.substr(last_pos, pos - last_pos);
                result += replace_text;
                count++;
            } else {
                result += text.substr(last_pos, pos - last_pos + search_text.length());
            }

            last_pos = pos + search_text.length();
        }

        result += text.substr(last_pos);

        if (count > 0) {
            editor.SetText(result);
        }
    }

    if (count > 0) {
        tabs_[active_tab_index_]->is_modified = true;
        spdlog::info("Replaced {} occurrences of '{}' with '{}'", count, search_text, replace_text);
    }

    return count;
}

// ==================== Comment Operations ====================

void ScriptEditorPanel::ToggleLineComment() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // Get current line
    auto cursor = editor.GetCursorPosition();
    auto lines = editor.GetTextLines();

    if (cursor.mLine >= static_cast<int>(lines.size())) {
        return;
    }

    std::string line = lines[cursor.mLine];

    // Find first non-whitespace character
    size_t first_char = line.find_first_not_of(" \t");

    if (first_char != std::string::npos && line.substr(first_char, 1) == "#") {
        // Remove comment
        std::string new_line = line.substr(0, first_char) + line.substr(first_char + 1);
        // Remove space after # if present
        if (first_char < new_line.length() && new_line[first_char] == ' ') {
            new_line = new_line.substr(0, first_char) + new_line.substr(first_char + 1);
        }

        // Replace line
        TextEditor::Coordinates start(cursor.mLine, 0);
        TextEditor::Coordinates end(cursor.mLine, static_cast<int>(line.length()));
        editor.SetSelection(start, end);
        editor.Delete();
        editor.InsertText(new_line);
    } else {
        // Add comment
        std::string new_line;
        if (first_char != std::string::npos) {
            new_line = line.substr(0, first_char) + "# " + line.substr(first_char);
        } else {
            new_line = "# " + line;
        }

        // Replace line
        TextEditor::Coordinates start(cursor.mLine, 0);
        TextEditor::Coordinates end(cursor.mLine, static_cast<int>(line.length()));
        editor.SetSelection(start, end);
        editor.Delete();
        editor.InsertText(new_line);
    }

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::ToggleBlockComment() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    if (!editor.HasSelection()) {
        spdlog::info("Select text to toggle block comment");
        return;
    }

    std::string selected = editor.GetSelectedText();

    // Check if already a block comment (Python uses triple quotes)
    if (selected.length() >= 6 &&
        selected.substr(0, 3) == "\"\"\"" &&
        selected.substr(selected.length() - 3) == "\"\"\"") {
        // Remove block comment
        selected = selected.substr(3, selected.length() - 6);
    } else {
        // Add block comment
        selected = "\"\"\"" + selected + "\"\"\"";
    }

    editor.Delete();
    editor.InsertText(selected);

    tabs_[active_tab_index_]->is_modified = true;
}

// ============================================================================
// Edit Operations
// ============================================================================

void ScriptEditorPanel::Undo() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (editor.CanUndo()) {
        editor.Undo();
        tabs_[active_tab_index_]->is_modified = true;
    }
}

void ScriptEditorPanel::Redo() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (editor.CanRedo()) {
        editor.Redo();
        tabs_[active_tab_index_]->is_modified = true;
    }
}

void ScriptEditorPanel::Cut() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If no selection, select the entire current line (including newline)
    if (!editor.HasSelection()) {
        auto cursor = editor.GetCursorPosition();
        int line = cursor.mLine;
        int totalLines = editor.GetTotalLines();

        // Select from start of current line to start of next line (or end of file)
        TextEditor::Coordinates start(line, 0);
        TextEditor::Coordinates end;

        if (line + 1 < totalLines) {
            // Select up to the start of the next line (includes the newline)
            end = TextEditor::Coordinates(line + 1, 0);
        } else {
            // Last line - get line length from text lines
            auto lines = editor.GetTextLines();
            int lineLen = (line < static_cast<int>(lines.size())) ? static_cast<int>(lines[line].size()) : 0;
            end = TextEditor::Coordinates(line, lineLen);
        }

        editor.SetSelection(start, end);
    }

    editor.Cut();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Copy() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.Copy();
}

void ScriptEditorPanel::Paste() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.Paste();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Delete() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // If no selection, select the entire current line (including newline)
    if (!editor.HasSelection()) {
        auto cursor = editor.GetCursorPosition();
        int line = cursor.mLine;
        int totalLines = editor.GetTotalLines();

        // Select from start of current line to start of next line (or end of file)
        TextEditor::Coordinates start(line, 0);
        TextEditor::Coordinates end;

        if (line + 1 < totalLines) {
            // Select up to the start of the next line (includes the newline)
            end = TextEditor::Coordinates(line + 1, 0);
        } else {
            // Last line - get line length from text lines
            auto lines = editor.GetTextLines();
            int lineLen = (line < static_cast<int>(lines.size())) ? static_cast<int>(lines[line].size()) : 0;
            end = TextEditor::Coordinates(line, lineLen);
        }

        editor.SetSelection(start, end);
    }

    editor.Delete();
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::SelectAll() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    editor.SelectAll();
}

// ========== Navigation ==========

void ScriptEditorPanel::GoToLine(int line_number) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    int total_lines = editor.GetTotalLines();

    // Clamp line number to valid range (1-based input, 0-based internal)
    if (line_number < 1) line_number = 1;
    if (line_number > total_lines) line_number = total_lines;

    // Set cursor position (0-based line number)
    TextEditor::Coordinates pos(line_number - 1, 0);
    editor.SetCursorPosition(pos);

    // Optionally select the entire line for visibility
    auto lines = editor.GetTextLines();
    int line_idx = line_number - 1;
    int line_len = (line_idx < static_cast<int>(lines.size())) ? static_cast<int>(lines[line_idx].size()) : 0;
    editor.SetSelection(pos, TextEditor::Coordinates(line_idx, line_len));
}

// ========== Line Operations ==========

void ScriptEditorPanel::DuplicateLine() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    auto lines = editor.GetTextLines();
    if (line < 0 || line >= static_cast<int>(lines.size())) {
        return;
    }

    std::string line_text = lines[line];

    // Move to end of current line
    int line_len = static_cast<int>(line_text.size());
    editor.SetCursorPosition(TextEditor::Coordinates(line, line_len));

    // Insert newline + duplicate content
    std::string to_insert = "\n" + line_text;
    editor.InsertText(to_insert);

    // Move cursor to the new duplicated line
    editor.SetCursorPosition(TextEditor::Coordinates(line + 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::MoveLineUp() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    if (line <= 0) {
        return;  // Already at the top
    }

    auto lines = editor.GetTextLines();
    if (line >= static_cast<int>(lines.size())) {
        return;
    }

    // Swap current line with line above
    std::string current_line = lines[line];
    std::string above_line = lines[line - 1];

    // Select and delete both lines
    int above_len = static_cast<int>(above_line.size());
    int current_len = static_cast<int>(current_line.size());

    editor.SetSelection(
        TextEditor::Coordinates(line - 1, 0),
        TextEditor::Coordinates(line, current_len)
    );

    // Replace with swapped content
    std::string new_text = current_line + "\n" + above_line;
    editor.Delete();
    editor.InsertText(new_text);

    // Move cursor up
    editor.SetCursorPosition(TextEditor::Coordinates(line - 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::MoveLineDown() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();
    int line = cursor.mLine;

    auto lines = editor.GetTextLines();
    if (line < 0 || line >= static_cast<int>(lines.size()) - 1) {
        return;  // Already at the bottom
    }

    // Swap current line with line below
    std::string current_line = lines[line];
    std::string below_line = lines[line + 1];

    int current_len = static_cast<int>(current_line.size());
    int below_len = static_cast<int>(below_line.size());

    editor.SetSelection(
        TextEditor::Coordinates(line, 0),
        TextEditor::Coordinates(line + 1, below_len)
    );

    // Replace with swapped content
    std::string new_text = below_line + "\n" + current_line;
    editor.Delete();
    editor.InsertText(new_text);

    // Move cursor down
    editor.SetCursorPosition(TextEditor::Coordinates(line + 1, cursor.mColumn));

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Indent() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;

    // Get selection or current line
    if (editor.HasSelection()) {
        auto sel_start = editor.GetSelectionStart();
        auto sel_end = editor.GetSelectionEnd();

        auto lines = editor.GetTextLines();
        std::string indent_str(tab_size_, ' ');

        // Build new text with indentation
        std::string new_text;
        for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
            new_text += indent_str + lines[i];
            if (i < sel_end.mLine) {
                new_text += "\n";
            }
        }

        // Select entire lines and replace
        int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
            static_cast<int>(lines[sel_end.mLine].size()) : 0;
        editor.SetSelection(
            TextEditor::Coordinates(sel_start.mLine, 0),
            TextEditor::Coordinates(sel_end.mLine, last_line_len)
        );
        editor.Delete();
        editor.InsertText(new_text);

        // Restore selection
        editor.SetSelection(
            TextEditor::Coordinates(sel_start.mLine, 0),
            TextEditor::Coordinates(sel_end.mLine, last_line_len + tab_size_)
        );
    } else {
        // Just insert tab at cursor
        std::string indent_str(tab_size_, ' ');
        editor.InsertText(indent_str);
    }

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::Outdent() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    auto cursor = editor.GetCursorPosition();

    auto lines = editor.GetTextLines();
    int start_line = cursor.mLine;
    int end_line = cursor.mLine;

    if (editor.HasSelection()) {
        start_line = editor.GetSelectionStart().mLine;
        end_line = editor.GetSelectionEnd().mLine;
    }

    // Build new text with removed indentation
    std::string new_text;
    for (int i = start_line; i <= end_line && i < static_cast<int>(lines.size()); ++i) {
        std::string line = lines[i];
        // Remove up to tab_size_ leading spaces
        int spaces_to_remove = 0;
        for (int j = 0; j < tab_size_ && j < static_cast<int>(line.size()); ++j) {
            if (line[j] == ' ') {
                spaces_to_remove++;
            } else {
                break;
            }
        }
        if (spaces_to_remove > 0) {
            line = line.substr(spaces_to_remove);
        }
        new_text += line;
        if (i < end_line) {
            new_text += "\n";
        }
    }

    // Select entire lines and replace
    int last_line_len = (end_line < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[end_line].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(start_line, 0),
        TextEditor::Coordinates(end_line, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

// ========== Text Transformation ==========

void ScriptEditorPanel::TransformToUppercase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string upper;
    upper.reserve(selected.size());
    for (char c : selected) {
        upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }

    editor.Delete();
    editor.InsertText(upper);
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::TransformToLowercase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string lower;
    lower.reserve(selected.size());
    for (char c : selected) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    editor.Delete();
    editor.InsertText(lower);
    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::TransformToTitleCase() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    std::string selected = editor.GetSelectedText();
    std::string title;
    title.reserve(selected.size());
    bool capitalize_next = true;

    for (char c : selected) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            capitalize_next = true;
            title += c;
        } else if (capitalize_next) {
            title += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalize_next = false;
        } else {
            title += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }

    editor.Delete();
    editor.InsertText(title);
    tabs_[active_tab_index_]->is_modified = true;
}

// ========== Multi-line Operations ==========

void ScriptEditorPanel::SortLinesAscending() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    // Collect lines in selection
    std::vector<std::string> selected_lines;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        selected_lines.push_back(lines[i]);
    }

    // Sort ascending
    std::sort(selected_lines.begin(), selected_lines.end());

    // Build replacement text
    std::string new_text;
    for (size_t i = 0; i < selected_lines.size(); ++i) {
        new_text += selected_lines[i];
        if (i < selected_lines.size() - 1) {
            new_text += "\n";
        }
    }

    // Replace
    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::SortLinesDescending() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    // Collect lines in selection
    std::vector<std::string> selected_lines;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        selected_lines.push_back(lines[i]);
    }

    // Sort descending
    std::sort(selected_lines.begin(), selected_lines.end(), std::greater<std::string>());

    // Build replacement text
    std::string new_text;
    for (size_t i = 0; i < selected_lines.size(); ++i) {
        new_text += selected_lines[i];
        if (i < selected_lines.size() - 1) {
            new_text += "\n";
        }
    }

    // Replace
    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(new_text);

    tabs_[active_tab_index_]->is_modified = true;
}

void ScriptEditorPanel::JoinLines() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& editor = tabs_[active_tab_index_]->editor;
    if (!editor.HasSelection()) {
        return;
    }

    auto sel_start = editor.GetSelectionStart();
    auto sel_end = editor.GetSelectionEnd();
    auto lines = editor.GetTextLines();

    // Collect lines in selection and join with spaces
    std::string joined;
    for (int i = sel_start.mLine; i <= sel_end.mLine && i < static_cast<int>(lines.size()); ++i) {
        std::string line = lines[i];
        // Trim trailing whitespace
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if (!joined.empty() && !line.empty()) {
            joined += " ";
        }
        joined += line;
    }

    // Replace
    int last_line_len = (sel_end.mLine < static_cast<int>(lines.size())) ?
        static_cast<int>(lines[sel_end.mLine].size()) : 0;
    editor.SetSelection(
        TextEditor::Coordinates(sel_start.mLine, 0),
        TextEditor::Coordinates(sel_end.mLine, last_line_len)
    );
    editor.Delete();
    editor.InsertText(joined);

    tabs_[active_tab_index_]->is_modified = true;
}

// ========== Settings ==========

void ScriptEditorPanel::SetTabSize(int size) {
    if (size >= 1 && size <= 8) {
        tab_size_ = size;
        ApplyTabSizeToAllTabs();
    }
}

void ScriptEditorPanel::SetShowWhitespace(bool show) {
    show_whitespace_ = show;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetShowWhitespaces(show);
    }
}

void ScriptEditorPanel::SetWordWrap(bool wrap) {
    word_wrap_ = wrap;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetWordWrap(wrap);
    }
}

void ScriptEditorPanel::SetAutoIndent(bool indent) {
    auto_indent_ = indent;
    // Apply to all tabs
    for (auto& tab : tabs_) {
        tab->editor.SetAutoIndent(indent);
    }
}

void ScriptEditorPanel::SetTheme(int theme_index) {
    if (theme_index >= 0 && theme_index <= 6) {
        current_theme_ = static_cast<EditorTheme>(theme_index);
        ApplyThemeToAllTabs();
    }
}

} // namespace cyxwiz
