#include "script_editor.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../editor_fonts.h"
#include "../../scripting/scripting_engine.h"
#include "../../scripting/script_output_sink.h"
#include "../../core/keyboard_shortcuts.h"
#include <imgui.h>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace cyxwiz {

ScriptEditorPanel::ScriptEditorPanel()
    : Panel("Script Editor", true)
    , active_tab_index_(-1)
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

void ScriptEditorPanel::SetScriptOutputSink(
    scripting::IScriptOutputSink* output_sink) {
    script_output_sink_ = output_sink;
}

void ScriptEditorPanel::Render() {
    if (!visible_) return;

    // Poll for pending output from async script execution
    if (scripting_engine_ && scripting_engine_->IsScriptRunning()) {
        script_running_ = true;
        running_indicator_time_ += ImGui::GetIO().DeltaTime;

        // Get any pending output and display it
        std::string pending = scripting_engine_->GetPendingOutput();
        if (!pending.empty() && script_output_sink_) {
            script_output_sink_->AppendScriptOutput("Running...", pending, false);
        }
    } else if (script_running_) {
        // Script just finished - check for result
        script_running_ = false;
        running_indicator_time_ = 0.0f;

        // First, drain any remaining output from the queue (for fast-finishing scripts)
        std::string remaining_output = scripting_engine_->GetPendingOutput();
        if (!remaining_output.empty() && script_output_sink_) {
            script_output_sink_->AppendScriptOutput("Output", remaining_output, false);
        }

        auto result = scripting_engine_->GetAsyncResult();
        if (result.has_value()) {
            auto& r = result.value();
            if (script_output_sink_) {
                if (r.was_cancelled) {
                    script_output_sink_->AppendScriptOutput("Script", "Script cancelled by user", true);
                } else if (!r.success) {
                    script_output_sink_->AppendScriptOutput("Script", "Error: " + r.error_message, true);
                } else {
                    // Script completed successfully
                    if (remaining_output.empty()) {
                        // No output was produced, show completion message
                        script_output_sink_->AppendScriptOutput("Script", "Completed successfully", false);
                    }
                    spdlog::info("Script completed successfully");
                }
            }
            spdlog::info("Async script execution finished. Success: {}", r.success);
        }
    }

    ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_MenuBar);

    // Track focus state (including child windows like TextEditor)
    is_focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);

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

    // ===== Empty Script Warning Popup =====
    if (show_empty_script_warning_) {
        ImGui::OpenPopup("Empty Script Warning");
    }

    // Center the popup
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Empty Script Warning", &show_empty_script_warning_, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_TRIANGLE_EXCLAMATION);
        ImGui::SameLine();
        ImGui::Text("Cannot Save Empty Script");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextWrapped("The script is empty. Please add some code before saving.");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float button_width = 120.0f;
        float window_width = ImGui::GetWindowWidth();
        ImGui::SetCursorPosX((window_width - button_width) * 0.5f);

        if (ImGui::Button("OK", ImVec2(button_width, 0))) {
            show_empty_script_warning_ = false;
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }
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
            if (ImGui::MenuItem("Save", "Ctrl+S", false, IsActiveTabEditable())) {
                SaveFile();
            }
            if (ImGui::MenuItem("Save As", "Ctrl+Shift+S", false, IsActiveTabEditable())) {
                SaveFileAs();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Close", "Ctrl+W", false, active_tab_index_ >= 0)) {
                close_tab_index_ = active_tab_index_;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
            bool has_active_tab = IsActiveTabEditable();
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
            if (ImGui::MenuItem("Run Script", "F5", false, IsActiveTabEditable() && not_running)) {
                RunScript();
            }
            if (ImGui::MenuItem("Stop Script", "Shift+F5", false, script_running_)) {
                if (scripting_engine_) {
                    scripting_engine_->StopScript();
                    spdlog::info("Stop script requested");
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Run Selection", "F9", false, IsActiveTabEditable() && not_running)) {
                RunSelection();
            }
            if (ImGui::MenuItem("Run Section", "Ctrl+Enter", false, IsActiveTabEditable() && not_running)) {
                RunCurrentSection();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Debug", "F10", false, IsActiveTabEditable() && not_running)) {
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
                if (ImGui::MenuItem("Small (14 px)", nullptr, font_scale_ == 1.0f)) {
                    font_scale_ = 1.0f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Medium (16 px)", nullptr, font_scale_ == 1.3f)) {
                    font_scale_ = 1.3f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Large (20 px)", nullptr, font_scale_ == 1.6f)) {
                    font_scale_ = 1.6f;
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Extra Large (24 px)", nullptr, font_scale_ == 2.0f)) {
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
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }

            // Show Whitespace toggle
            if (ImGui::MenuItem("Show Whitespace", nullptr, &show_whitespace_)) {
                for (auto& tab : tabs_) {
                    tab->editor.SetShowWhitespaces(show_whitespace_);
                }
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }

            // Minimap toggle
            if (ImGui::MenuItem("Show Minimap", nullptr, &show_minimap_)) {
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }

            ImGui::Separator();

            // Cell Mode toggle (Jupyter-like notebook mode)
            bool has_active_tab = IsActiveTabEditable();
            bool is_cell_mode = has_active_tab && tabs_[active_tab_index_]->cell_mode;
            if (ImGui::MenuItem(ICON_FA_FILE_LINES "  Notebook Mode", "Ctrl+Shift+N", is_cell_mode, has_active_tab)) {
                ToggleCellMode();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Switch between plain text and Jupyter-like cell mode");
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
        tab->cell_manager.ApplyEditorPalette(palette);
    }

    spdlog::info("Applied editor theme: {}", theme_name);
}

void ScriptEditorPanel::ApplyTabSizeToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetTabSize(tab_size_);
        tab->cell_manager.ApplyTabSize(tab_size_);
    }
    spdlog::info("Applied tab size: {} spaces", tab_size_);
}

void ScriptEditorPanel::ApplySyntaxHighlightingToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetColorizerEnable(syntax_highlighting_);
        tab->cell_manager.ApplySyntaxHighlighting(syntax_highlighting_);
    }
    spdlog::info("Syntax highlighting: {}", syntax_highlighting_ ? "enabled" : "disabled");
}

void ScriptEditorPanel::RenderTabBar() {
    if (ImGui::BeginTabBar("ScriptEditorTabs", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyScroll)) {

        // Render existing tabs
        for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
            auto& tab = tabs_[i];

            // Build tab label with loading/modified indicators
            std::string tab_label;
            if (tab->is_loading) {
                tab_label = ICON_FA_SPINNER " " + tab->filename;
            } else {
                tab_label = tab->filename;
                if (tab->is_modified) {
                    tab_label += "*";
                }
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

            // Closing a loading tab cancels its background work.
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

void ScriptEditorPanel::RenderEditorToolbar() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    const bool has_active_tab = IsActiveTabEditable();
    const bool not_running = !script_running_ && !(scripting_engine_ && scripting_engine_->IsScriptRunning());

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.13f, 0.13f, 0.15f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 5));
    ImGui::BeginChild("##script_editor_toolbar", ImVec2(0, 38), false, ImGuiWindowFlags_NoScrollbar);

    ImGui::BeginDisabled(!has_active_tab);
    if (ImGui::Button(ICON_FA_FLOPPY_DISK " Save")) {
        SaveFile();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Save current script (Ctrl+S)");
    }
    ImGui::SameLine();

    ImGui::BeginDisabled(!not_running);
    if (ImGui::Button(ICON_FA_PLAY " Run")) {
        RunScript();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run script (F5)");
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_CODE " Section")) {
        RunCurrentSection();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run current %% section (Ctrl+Enter)");
    }
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_PENCIL " Selection")) {
        RunSelection();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Run selected code (F9)");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(not_running || !scripting_engine_);
    if (ImGui::Button(ICON_FA_STOP " Stop")) {
        scripting_engine_->StopScript();
        spdlog::info("Stop script requested from editor toolbar");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Stop running script (Shift+F5)");
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    const bool was_cell_mode = tab->cell_mode;
    if (ImGui::Button(was_cell_mode ? ICON_FA_FILE_LINES " Notebook: On" : ICON_FA_FILE_CODE " Notebook: Off")) {
        ToggleCellMode();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Toggle notebook/cell mode (Ctrl+Shift+N)");
    }

    ImGui::SameLine();
    ImGui::TextDisabled("%s", tab->filepath.empty() ? tab->filename.c_str() : tab->filepath.c_str());
    ImGui::EndDisabled();

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}
void ScriptEditorPanel::RenderEditor() {
    auto& tab = tabs_[active_tab_index_];

    // Show loading indicator if tab is loading
    if (tab->is_loading) {
        if (tab->load_task_id != 0) {
            if (const auto task = AsyncTaskManager::Instance().GetTask(tab->load_task_id)) {
                const auto info = task->GetInfo();
                tab->load_progress = info.progress;
                if (!info.status_message.empty()) {
                    tab->load_status = info.status_message;
                }
            }
        }
        ImGui::Spacing();
        ImGui::Spacing();

        // Center the loading indicator
        float window_width = ImGui::GetWindowWidth();
        float text_width = ImGui::CalcTextSize(tab->load_status.c_str()).x + 50;
        ImGui::SetCursorPosX((window_width - text_width) * 0.5f);

        // Animated spinner character
        float time = static_cast<float>(ImGui::GetTime());
        const char* spinner_chars = "|/-\\";
        char spinner = spinner_chars[static_cast<int>(time * 10) % 4];

        ImGui::Text("%c %s", spinner, tab->load_status.c_str());

        ImGui::Spacing();

        // Progress bar
        ImGui::SetCursorPosX(window_width * 0.2f);
        ImGui::ProgressBar(tab->load_progress, ImVec2(window_width * 0.6f, 0.0f));

        return;
    }

    if (tab->is_large_file) {
        RenderLargeFileViewer(*tab);
        return;
    }

    // Cell-based mode (Jupyter-like notebook)
    if (tab->cell_mode) {
        RenderCellBasedEditor();
        return;
    }

    // Show debug toolbar when debugging is active (traditional mode)
    if (debug_mode_active_ && debugger_) {
        RenderDebugToolbar();
    }

    bool pushed_editor_font = false;
    if (ImFont* mono_font = gui::GetEditorMonoFont(font_scale_)) {
        ImGui::PushFont(mono_font);
        pushed_editor_font = true;
    }
    // Calculate editor size (leave room for status bar and minimap)
    float available_height = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing();
    float available_width = ImGui::GetContentRegionAvail().x;
    float gutter_width = 20.0f;  // Breakpoint gutter width

    // Hide horizontal scrollbar by making it invisible
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, ImVec4(0, 0, 0, 0));

    // Breakpoint gutter on the left
    RenderScriptBreakpointGutter(available_height);
    ImGui::SameLine();

    // Temporarily disable keyboard input if we just accepted a completion
    // This prevents Tab/Enter from being passed to the editor
    if (completion_just_accepted_) {
        tab->editor.SetHandleKeyboardInputs(false);
    }

    if (show_minimap_) {
        // Layout: Gutter | Editor | Minimap
        float editor_width = available_width - minimap_width_ - gutter_width - 8.0f;  // 8px for separators

        // Editor in the middle
        ImGui::BeginChild("##editor_region", ImVec2(editor_width, available_height), false,
                          ImGuiWindowFlags_NoScrollbar);
        tab->editor.Render("##editor", ImVec2(0, 0));
        ImGui::EndChild();

        ImGui::SameLine();

        // Minimap on the right
        RenderMinimap();
    } else {
        // Layout: Gutter | Editor
        float editor_width = available_width - gutter_width - 4.0f;
        ImVec2 editor_size = ImVec2(editor_width, available_height);
        tab->editor.Render("##editor", editor_size);
    }

    // Re-enable keyboard input and clear the flag
    if (completion_just_accepted_) {
        tab->editor.SetHandleKeyboardInputs(true);
        completion_just_accepted_ = false;
    }

    ImGui::PopStyleColor(4);

    if (pushed_editor_font) {
        ImGui::PopFont();
    }
    // Track modifications and auto-completion
    if (tab->editor.IsTextChanged()) {
        tab->is_modified = true;

        // Skip auto-trigger if popup was just opened this frame (Ctrl+Space inserts space)
        if (!completion_just_opened_) {
            // Auto-trigger completion when typing (not forced, uses trigger char check)
            UpdateAutoCompletion(false);
        }
    }

    // Clear the just-opened flag after the first frame
    completion_just_opened_ = false;

    // Render auto-completion popup (if open)
    RenderCompletionPopup();
}

void ScriptEditorPanel::RenderMinimap() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Get the text content and line count
    int total_lines = tab->editor.GetTotalLines();
    if (total_lines == 0) return;

    // Calculate available height for minimap - use full height (after horizontal scrollbar)
    float available_height = ImGui::GetContentRegionAvail().y;

    // Get theme-based colors from the current palette
    TextEditor::Palette palette;
    switch (current_theme_) {
        case EditorTheme::Monokai:   palette = GetMonokaiPalette(); break;
        case EditorTheme::Dracula:   palette = GetDraculaPalette(); break;
        case EditorTheme::OneDark:   palette = GetOneDarkPalette(); break;
        case EditorTheme::GitHub:    palette = GetGitHubPalette(); break;
        case EditorTheme::Dark:      palette = TextEditor::GetDarkPalette(); break;
        case EditorTheme::Light:     palette = TextEditor::GetLightPalette(); break;
        case EditorTheme::RetroBlu:  palette = TextEditor::GetRetroBluePalette(); break;
        default:                     palette = GetMonokaiPalette(); break;
    }

    // Extract colors from palette (palette values are ABGR format)
    // Convert to RGBA for ImGui
    auto PaletteToImU32 = [](uint32_t abgr, uint8_t alpha_override = 0) -> ImU32 {
        uint8_t a = alpha_override ? alpha_override : ((abgr >> 24) & 0xFF);
        uint8_t b = (abgr >> 16) & 0xFF;
        uint8_t g = (abgr >> 8) & 0xFF;
        uint8_t r = abgr & 0xFF;
        return IM_COL32(r, g, b, a);
    };

    // Get colors from palette with reduced alpha for minimap
    ImU32 bg_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::Background]);
    ImU32 keyword_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::Keyword], 200);
    ImU32 string_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::String], 200);
    ImU32 comment_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::Comment], 180);
    ImU32 default_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::Default], 180);

    // Darken background slightly for minimap
    uint8_t bg_r = (bg_color >> 0) & 0xFF;
    uint8_t bg_g = (bg_color >> 8) & 0xFF;
    uint8_t bg_b = (bg_color >> 16) & 0xFF;
    bg_r = static_cast<uint8_t>(bg_r * 0.85f);
    bg_g = static_cast<uint8_t>(bg_g * 0.85f);
    bg_b = static_cast<uint8_t>(bg_b * 0.85f);
    ImU32 minimap_bg_color = IM_COL32(bg_r, bg_g, bg_b, 255);

    // Begin minimap region
    ImGui::BeginChild("##minimap", ImVec2(minimap_width_, available_height), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImVec2 minimap_pos = ImGui::GetCursorScreenPos();
    ImVec2 minimap_size = ImGui::GetContentRegionAvail();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // Background color (slightly darker than editor background, theme-aware)
    draw_list->AddRectFilled(minimap_pos,
                             ImVec2(minimap_pos.x + minimap_size.x, minimap_pos.y + minimap_size.y),
                             minimap_bg_color);

    // Calculate line height in minimap (each line is 2 pixels minimum)
    float line_height = std::max(1.0f, minimap_size.y / static_cast<float>(total_lines));
    if (line_height > 3.0f) line_height = 3.0f;  // Cap at 3 pixels per line

    // Get text lines for rendering
    auto text_lines = tab->editor.GetTextLines();

    // Render minimap lines
    float y_offset = 0.0f;
    for (int i = 0; i < total_lines && y_offset < minimap_size.y; i++) {
        if (i < static_cast<int>(text_lines.size())) {
            const std::string& line = text_lines[i];

            // Determine line color based on content (simple syntax detection)
            ImU32 line_color = default_color;
            if (line.find('#') != std::string::npos) {
                line_color = comment_color;
            } else if (line.find("def ") != std::string::npos ||
                       line.find("class ") != std::string::npos ||
                       line.find("import ") != std::string::npos ||
                       line.find("from ") != std::string::npos ||
                       line.find("if ") != std::string::npos ||
                       line.find("for ") != std::string::npos ||
                       line.find("while ") != std::string::npos ||
                       line.find("return ") != std::string::npos) {
                line_color = keyword_color;
            } else if (line.find('"') != std::string::npos || line.find('\'') != std::string::npos) {
                line_color = string_color;
            }

            // Calculate line width (proportional to character count, max = minimap width - 4)
            float line_width = std::min(static_cast<float>(line.length()) * 0.8f, minimap_size.x - 4.0f);
            if (line_width > 2.0f) {
                draw_list->AddRectFilled(
                    ImVec2(minimap_pos.x + 2.0f, minimap_pos.y + y_offset),
                    ImVec2(minimap_pos.x + 2.0f + line_width, minimap_pos.y + y_offset + line_height - 1.0f),
                    line_color
                );
            }
        }
        y_offset += line_height;
    }

    // Draw viewport indicator (visible region)
    auto cursor_pos = tab->editor.GetCursorPosition();
    int visible_lines = static_cast<int>(available_height / (gui::GetEditorMonoFontPixelSize(font_scale_) + ImGui::GetStyle().ItemSpacing.y));
    int first_visible_line = std::max(0, cursor_pos.mLine - visible_lines / 2);

    float viewport_top = first_visible_line * line_height;
    float viewport_height = visible_lines * line_height;

    // Clamp viewport indicator to minimap bounds
    viewport_top = std::min(viewport_top, minimap_size.y - viewport_height);
    viewport_top = std::max(0.0f, viewport_top);

    // Get viewport colors from theme (based on selection color)
    ImU32 selection_color = PaletteToImU32(palette[(int)TextEditor::PaletteIndex::Selection]);
    uint8_t sel_r = (selection_color >> 0) & 0xFF;
    uint8_t sel_g = (selection_color >> 8) & 0xFF;
    uint8_t sel_b = (selection_color >> 16) & 0xFF;
    ImU32 viewport_fill = IM_COL32(sel_r, sel_g, sel_b, 60);
    ImU32 viewport_border = IM_COL32(
        std::min(255, sel_r + 30),
        std::min(255, sel_g + 30),
        std::min(255, sel_b + 30),
        150
    );

    // Draw viewport rectangle
    draw_list->AddRectFilled(
        ImVec2(minimap_pos.x, minimap_pos.y + viewport_top),
        ImVec2(minimap_pos.x + minimap_size.x, minimap_pos.y + viewport_top + viewport_height),
        viewport_fill
    );
    draw_list->AddRect(
        ImVec2(minimap_pos.x, minimap_pos.y + viewport_top),
        ImVec2(minimap_pos.x + minimap_size.x, minimap_pos.y + viewport_top + viewport_height),
        viewport_border
    );

    // Handle click to navigate
    if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        float relative_y = mouse_pos.y - minimap_pos.y;
        int target_line = static_cast<int>(relative_y / line_height);
        target_line = std::clamp(target_line, 0, total_lines - 1);

        // Navigate to the clicked line
        TextEditor::Coordinates new_pos;
        new_pos.mLine = target_line;
        new_pos.mColumn = 0;
        tab->editor.SetCursorPosition(new_pos);
    }

    // Handle drag for smooth scrolling
    if (ImGui::IsWindowHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        float relative_y = mouse_pos.y - minimap_pos.y;
        int target_line = static_cast<int>(relative_y / line_height);
        target_line = std::clamp(target_line, 0, total_lines - 1);

        TextEditor::Coordinates new_pos;
        new_pos.mLine = target_line;
        new_pos.mColumn = 0;
        tab->editor.SetCursorPosition(new_pos);
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::RenderStatusBar() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        auto& tab = tabs_[active_tab_index_];

        // Show loading status if loading
        if (tab->is_loading) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), ICON_FA_SPINNER " Loading: %s (%.0f%%)",
                tab->load_status.c_str(), tab->load_progress * 100.0f);
            return;
        }

        if (tab->is_large_file) {
            const auto first = tab->large_page.lines.empty()
                ? 0ULL
                : static_cast<unsigned long long>(tab->large_page.first_line + 1);
            const auto last = static_cast<unsigned long long>(
                tab->large_page.first_line + tab->large_page.lines.size());
            ImGui::Text(
                "Large file | Read only | Cached lines: %llu-%llu of %llu",
                first,
                last,
                static_cast<unsigned long long>(tab->large_index.line_count));
        } else if (tab->cell_mode) {
            const int cell_count = static_cast<int>(tab->cell_manager.GetCellCount());
            const int selected_cell = tab->selected_cell >= 0 ? tab->selected_cell + 1 : 0;
            ImGui::Text("Notebook | Cell: %d/%d | %s",
                selected_cell,
                cell_count,
                tab->is_modified ? "Modified" : "Saved");
        } else {
            auto cursor_pos = tab->editor.GetCursorPosition();
            ImGui::Text("Line: %d | Column: %d | %s | %d lines",
                cursor_pos.mLine + 1,
                cursor_pos.mColumn + 1,
                tab->is_modified ? "Modified" : "Saved",
                tab->editor.GetTotalLines());
        }

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
    // Use is_focused_ directly instead of keyboard context to avoid timing issues
    // (context is detected before Render() updates is_focused_)
    if (!is_focused_ && !show_completion_popup_) {
        return;  // Not focused and no popup, don't process shortcuts
    }

    // Handle debug shortcuts (F5, F9, F10, F11) - only when focused
    if (is_focused_ && IsActiveTabEditable()) {
        HandleDebugKeyboardShortcuts();
    }

    ImGuiIO& io = ImGui::GetIO();

    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // ========================================================================
    // COMPLETION POPUP - Highest priority when popup is open
    // ========================================================================
    if (show_completion_popup_) {
        // Escape closes completion popup
        if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            CloseCompletionPopup();
            return;  // Don't process other shortcuts
        }
        // Tab or Enter applies selected completion
        if (!ctrl && !shift && !alt && (ImGui::IsKeyPressed(ImGuiKey_Tab) || ImGui::IsKeyPressed(ImGuiKey_Enter))) {
            if (selected_completion_ >= 0 && selected_completion_ < static_cast<int>(completion_items_.size())) {
                ApplyCompletion(completion_items_[selected_completion_]);
            }
            CloseCompletionPopup();
            // Set flag to disable editor keyboard input for this frame
            completion_just_accepted_ = true;
            // Also clear Tab/Enter/Newline characters from input queue
            for (int i = io.InputQueueCharacters.Size - 1; i >= 0; --i) {
                ImWchar c = io.InputQueueCharacters[i];
                if (c == '\t' || c == '\n' || c == '\r') {
                    io.InputQueueCharacters.erase(io.InputQueueCharacters.Data + i);
                }
            }
            return;  // Don't process other shortcuts
        }
        // Let other keys pass through to editor (typing continues)
    }

    // ========================================================================
    // SCRIPT EDITOR SHORTCUTS - Only when this panel is focused
    // ========================================================================
    if (!is_focused_) {
        return;  // Not focused, don't process shortcuts
    }

    // File operations (script editor specific)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_N)) {
        NewFile();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_O)) {
        OpenFile();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && IsActiveTabEditable()) {
        SaveFile();
    }
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && IsActiveTabEditable()) {
        SaveFileAs();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_W) && active_tab_index_ >= 0) {
        close_tab_index_ = active_tab_index_;
    }

    // Toggle cell mode (Jupyter-like notebook mode)
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_N) && IsActiveTabEditable()) {
        ToggleCellMode();
    }

    // Edit operations (handled by TextEditor internally, but we can add extra handling)
    // The TextEditor component already handles Ctrl+Z, Ctrl+Y, Ctrl+X, Ctrl+C, Ctrl+V, Ctrl+A

    // Execution shortcuts
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F5) && !script_running_ && IsActiveTabEditable()) {
        RunScript();
    }
    // Stop script with Shift+F5
    if (!ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F5) && script_running_) {
        if (scripting_engine_) {
            scripting_engine_->StopScript();
            spdlog::info("Stop script requested via Shift+F5");
        }
    }
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F9) && !script_running_ && IsActiveTabEditable()) {
        RunSelection();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Enter) && !script_running_ && IsActiveTabEditable()) {
        RunCurrentSection();
    }
    if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F10) && !script_running_ && IsActiveTabEditable()) {
        Debug();
    }

    // Ctrl+Space triggers completion manually (force = true bypasses trigger char check)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Space, false) && IsActiveTabEditable()) {
        UpdateAutoCompletion(true);
    }
}


} // namespace cyxwiz
