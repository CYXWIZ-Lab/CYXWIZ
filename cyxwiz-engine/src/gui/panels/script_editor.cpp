#include "../appearance_settings.h"
#include "script_editor.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"
#include "../editor_fonts.h"
#include "../../scripting/scripting_engine.h"
#include "../../scripting/script_output_sink.h"
#include "../../core/keyboard_shortcuts.h"
#include "../../core/script_keys.h"
#include <imgui.h>
#include <imgui_internal.h>
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

float ScriptEditorPanel::GetFontScale() const { return gui::CodeFontScale(); }

void ScriptEditorPanel::SetFontScale(float scale) {
    ::gui::SetCodeTextScale(scale);
    font_scale_ = gui::CodeFontScale();
}

void ScriptEditorPanel::Render() {
    if (!visible_) return;

    // Code text size is Engine-wide; follow it when Preferences changes it.
    font_scale_ = gui::CodeFontScale();

    // Notebook tabs apply their cells' output and start queued cells here,
    // on the UI thread (TOFIX133 P0 items 8-10).
    for (auto& tab : tabs_) {
        if (tab) tab->cell_manager.Pump();
    }
    CheckFilesOnDisk();
    RenderPlotWindows();
    PollLanguageResults();  // completion, details, problems, signatures, hover (P3)
    UpdateDebugState();     // the debugger's snapshot and paused line (P6)
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size()) && tabs_[active_tab_index_])
        UpdateDiagnostics(*tabs_[active_tab_index_]);
    UpdateSignatureHelp();
    if (!deferred_open_path_.empty()) {
        const std::string path = std::move(deferred_open_path_);
        deferred_open_path_.clear();
        OpenFileAtLine(path, deferred_open_line_);
    }

    // Poll the editor's own run. A run started elsewhere (a notebook cell,
    // RL training) is not the editor's and must not print here.
    if (scripting_engine_ && script_running_ && scripting_engine_->IsScriptRunning()) {
        running_indicator_time_ += ImGui::GetIO().DeltaTime;

        // Get any pending output and display it
        std::string pending = scripting_engine_->GetPendingOutput();
        if (!pending.empty() && script_output_sink_) {
            script_output_sink_->AppendScriptOutput(running_script_name_, pending, false);
        }
    } else if (script_running_) {
        // Script just finished - check for result
        script_running_ = false;
        running_indicator_time_ = 0.0f;

        // First, drain any remaining output from the queue (for fast-finishing scripts)
        std::string remaining_output = scripting_engine_->GetPendingOutput();
        if (!remaining_output.empty() && script_output_sink_) {
            script_output_sink_->AppendScriptOutput(running_script_name_, remaining_output, false);
        }

        auto result = scripting_engine_->GetAsyncResult();
        if (result.has_value()) {
            auto& r = result.value();
            if (script_output_sink_) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - running_script_started_).count();
                if (!r.was_cancelled && !r.success && !r.error_message.empty()) {
                    script_output_sink_->AppendScriptOutput(running_script_name_, r.error_message, true);
                }
                script_output_sink_->EndScriptOutput(running_script_name_, r.success,
                                                     r.was_cancelled, seconds);
                if (r.success) {
                    spdlog::info("Script completed successfully");
                }
            }
            spdlog::info("Async script execution finished. Success: {}", r.success);
        }
    }

    // Bring to front before Begin: Begin returns false while the window is a
    // hidden dock tab, so a request made inside the body would never run. In
    // the first frames a dock node selects the tab saved in imgui.ini, so the
    // request repeats until its tab shows (at most 30 frames).
    if (request_window_focus_) {
        ImGui::SetNextWindowFocus();
        if (++window_focus_frames_ > 30) {
            request_window_focus_ = false;
            window_focus_frames_ = 0;
        }
    }

    // Collapsed or behind another dock tab: skip the body (TOFIX129 0.6).
    const bool expanded = ImGui::Begin(GetName(), &visible_, ImGuiWindowFlags_MenuBar);
    if (expanded) {
        // Track focus state (including child windows like the code view)
        is_focused_ = ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows);
        // Done once the tab shows. Not on the first frame: the window docks
        // after it, and its dock node then selects the tab saved in imgui.ini.
        if (request_window_focus_ && !ImGui::IsWindowAppearing() &&
            (!ImGui::IsWindowDocked() || ImGui::GetCurrentWindow()->DockTabIsVisible)) {
            request_window_focus_ = false;
            window_focus_frames_ = 0;
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
    }
    if (!expanded) is_focused_ = false;
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
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, has_active_tab && tabs_[active_tab_index_]->editor.Doc().CanUndo())) {
                Undo();
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, has_active_tab && tabs_[active_tab_index_]->editor.Doc().CanRedo())) {
                Redo();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Cut", "Ctrl+X", false, has_active_tab)) {
                Cut();
            }
            if (ImGui::MenuItem("Copy", "Ctrl+C", false, has_active_tab)) {
                Copy();
            }
            if (ImGui::MenuItem("Paste", "Ctrl+V", false, has_active_tab)) {
                Paste();
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
            // Font Size submenu
            if (ImGui::BeginMenu("Font Size")) {
                if (ImGui::MenuItem("Small (14 px)", nullptr, font_scale_ == 1.0f)) {
                    SetFontScale(1.0f);
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Medium (16 px)", nullptr, font_scale_ == 1.3f)) {
                    SetFontScale(1.3f);
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Large (20 px)", nullptr, font_scale_ == 1.6f)) {
                    SetFontScale(1.6f);
                    if (on_settings_changed_callback_) on_settings_changed_callback_();
                }
                if (ImGui::MenuItem("Extra Large (24 px)", nullptr, font_scale_ == 2.0f)) {
                    SetFontScale(2.0f);
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
                    tab->editor.SetShowWhitespace(show_whitespace_);
                }
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }

            if (ImGui::MenuItem("Word Wrap", nullptr, word_wrap_)) {
                SetWordWrap(!word_wrap_);
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
            const bool is_ipynb = has_active_tab && scriptfile::IsNotebookJson(tabs_[active_tab_index_]->filepath);
            if (ImGui::MenuItem(ICON_FA_FILE_LINES "  Notebook Mode", "Ctrl+Shift+M", is_cell_mode, has_active_tab && !is_ipynb)) {
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

void ScriptEditorPanel::ApplyTabSizeToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetTabSize(tab_size_);
        tab->cell_manager.ApplyTabSize(tab_size_);
    }
    spdlog::info("Applied tab size: {} spaces", tab_size_);
}

void ScriptEditorPanel::ApplySyntaxHighlightingToAllTabs() {
    for (auto& tab : tabs_) {
        tab->editor.SetColorize(syntax_highlighting_);
        tab->cell_manager.ApplySyntaxHighlighting(syntax_highlighting_);
    }
    spdlog::info("Syntax highlighting: {}", syntax_highlighting_ ? "enabled" : "disabled");
}

// Right-click menu on the code (approved board 2). Groups are separated by
// spacing, not lines.
void ScriptEditorPanel::RenderCodeContextMenu(EditorTab& tab) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    if (!ImGui::BeginPopup("##code_menu")) {
        ImGui::PopStyleVar();
        return;
    }
    editor::Document& doc = tab.editor.Doc();
    const bool engine_busy = scripting_engine_ && scripting_engine_->IsScriptRunning();
    const bool has_selection = !doc.SelectedText().empty();
    auto gap = []() { ImGui::Dummy(ImVec2(0.0f, 4.0f)); };
    // Language (board 8): at the cursor the right-click placed.
    if (ImGui::MenuItem("Go to definition", "F12")) GoToDefinition(ActiveCodeTarget(), tab.editor, doc.Primary().head);
    if (ImGui::MenuItem("Show completions", "Ctrl+Space")) UpdateAutoCompletion(true);
    if (ImGui::MenuItem("Show signature", "Ctrl+Shift+Space")) RequestSignatures(true);
    gap();
    if (ImGui::MenuItem("Run selection", "F9", false, has_selection && !engine_busy)) RunSelection();
    if (ImGui::MenuItem("Run section", "Ctrl+Enter", false, !engine_busy)) RunCurrentSection();
    gap();
    if (ImGui::MenuItem("Cut", "Ctrl+X")) Cut();
    if (ImGui::MenuItem("Copy", "Ctrl+C")) Copy();
    if (ImGui::MenuItem("Paste", "Ctrl+V", false, ImGui::GetClipboardText() != nullptr)) Paste();
    gap();
    if (ImGui::MenuItem("Toggle line comment", "Ctrl+/")) ToggleLineComment();
    const int line = doc.Primary().head.line;
    auto& folds = tab.editor.Folds();
    if (folds.CanFold(line)) {
        if (ImGui::MenuItem(folds.IsFolded(line) ? "Unfold" : "Fold", folds.IsFolded(line) ? "Ctrl+Shift+]" : "Ctrl+Shift+["))
            folds.Toggle(line);
    }
    if (ImGui::MenuItem("Fold all")) folds.FoldAll();
    if (ImGui::MenuItem("Unfold all")) folds.UnfoldAll();
    if (ImGui::MenuItem("Select every occurrence", "Ctrl+Shift+L")) doc.SelectAllOccurrences();
    gap();
    if (ImGui::MenuItem("Go to line...", "Ctrl+G", false, static_cast<bool>(go_to_line_request_))) go_to_line_request_();
    if (ImGui::MenuItem("Find...", "Ctrl+F")) OpenFind(false);
    ImGui::EndPopup();
    ImGui::PopStyleVar();
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

    // The file was not read: say so and offer to try again. The tab stays
    // read-only so Save cannot write an empty editor over it (P0 item 1).
    if (tab->load_failed) {
        ImGui::Spacing();
        ui::StatusText(ui::Status::Failed, ("Could not open " + tab->filename).c_str());
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", tab->load_status.c_str());
        ImGui::TextDisabled("%s", tab->filepath.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ui::PrimaryButton("Try again")) RetryLoad(active_tab_index_);
        ImGui::SameLine();
        if (ui::SecondaryButton("Close tab", true, nullptr, ui::ButtonSize::Regular)) close_tab_index_ = active_tab_index_;
        return;
    }

    if (tab->is_large_file) {
        RenderLargeFileViewer(*tab);
        return;
    }

    RenderDiskChangedBand(*tab);

    // Cell-based mode (Jupyter-like notebook)
    if (tab->cell_mode) {
        RenderCellBasedEditor();
        return;
    }

    // Show debug toolbar when debugging is active (traditional mode)
    if (debug_run_.active) {
        RenderDebugToolbar();
    }

    bool pushed_editor_font = false;
    if (ImFont* mono_font = gui::GetEditorMonoFont(font_scale_)) {
        ImGui::PushFont(mono_font);
        pushed_editor_font = true;
    }
    // Size: everything above the status bar. The code view draws its own
    // gutter (breakpoints, numbers, folds), scrollbars and minimap.
    RenderBreadcrumbs(*tab);
    float available_height = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing();
    CodeEditor& code = tab->editor;
    // Find: an overlay at the top right, or a full-width row when narrow.
    const float code_width = ImGui::GetContentRegionAvail().x;
    const bool narrow_find = code_width < 620.0f;
    if (find_.open && narrow_find) {
        const ImVec2 row = ImGui::GetCursorScreenPos();
        RenderFindWidget(code, row, code_width, true);
        available_height -= ImGui::GetCursorScreenPos().y - row.y;
    }
    const ImVec2 code_min = ImGui::GetCursorScreenPos();
    code.SetBreakpoints(&tab->breakpoints);
    code.on_gutter_click = [this](int line) {
        if (active_tab_index_ < 0) return;
        auto& t = tabs_[active_tab_index_];
        t->editor.Doc().SetCursor({line, 0});
        ToggleBreakpointAtCursor();
    };
    const std::string file_id = tab->filepath.empty() ? tab->filename : tab->filepath;
    // The paused line (the selected call-stack frame) when it is in this script.
    {
        int debug_line = -1;
        if (debug_.state == "paused" && debug_run_.active && debug_run_.cell_id.empty() &&
            debug_run_.document_id == tab->document_id && debug_frame_ < static_cast<int>(debug_.stack.size()) &&
            debug_.stack[debug_frame_].file == debug_.file)
            debug_line = debug_.stack[debug_frame_].line - 1;
        code.SetDebugLine(debug_line);
        (void)file_id;
    }
    code.SetShowMinimap(show_minimap_ && code_width >= 620.0f);  // no minimap in a narrow editor (board 3)
    // A completion just accepted with Tab must not also type the Tab.
    code.SetKeyboardEnabled(!completion_just_accepted_);
    if (request_focus_) {
        code.RequestFocus();
        request_focus_ = false;
    }
    // Problems: underlines, and the panel under the code when it is open (board 7).
    ApplyProblemSquiggles(*tab, code, tab->problems);
    const float problems_height = tab->show_problems ? 168.0f : 0.0f;
    const float debug_width = DebugSidebarWidth(code_width);  // board 11: beside the code while debugging
    const bool text_changed = code.Render("##code", ImVec2(debug_width > 0.0f ? code_width - debug_width : 0.0f,
                                                         available_height - problems_height));
    if (debug_width > 0.0f) {
        const ImVec2 below = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(code_min.x + code_width - debug_width, code_min.y));
        RenderDebugSidebar(debug_width, available_height - problems_height);
        ImGui::SetCursorScreenPos(below);
    }
    completion_just_accepted_ = false;
    AfterCodeRender(code, tab->problems, std::string());  // hover card, Ctrl+click
    if (find_.open && !narrow_find) {
        const ImVec2 after = ImGui::GetCursorScreenPos();
        RenderFindWidget(code, code_min, code_width, false);
        ImGui::SetCursorScreenPos(after);
    } else if (!find_.open && !find_.matches.empty()) {
        find_.matches.clear();
        code.SetMarks({});
    }

    if (pushed_editor_font) {
        ImGui::PopFont();
    }
    // Under the code, in the interface font (board 7).
    if (tab->show_problems) RenderProblemsPanel(*tab, problems_height);
    // The right-click menu uses the interface font.
    if (code.TakeContextMenuRequest()) ImGui::OpenPopup("##code_menu");
    RenderCodeContextMenu(*tab);
    // Modified follows the document (undoing back to the saved text clears it).
    tab->is_modified = tab->is_new ? !tab->editor.Doc().Text().empty() || tab->editor.Doc().Modified()
                                   : tab->editor.Doc().Modified() || tab->format_changed;
    if (text_changed) {

        // Skip auto-trigger if popup was just opened this frame (Ctrl+Space inserts space)
        if (!completion_just_opened_) {
            // Auto-trigger completion when typing (not forced, uses trigger char check)
            UpdateAutoCompletion(false);
        }
    }

    // Clear the just-opened flag after the first frame
    completion_just_opened_ = false;

    RenderCompletionPopup();
    RenderLanguageCards();
}

void ScriptEditorPanel::HandleKeyboardShortcuts() {
    // Use is_focused_ directly instead of keyboard context to avoid timing issues
    // (context is detected before Render() updates is_focused_)
    if (!is_focused_ && !show_completion_popup_) {
        return;  // Not focused and no popup, don't process shortcuts
    }

    ImGuiIO& io = ImGui::GetIO();

    bool ctrl = io.KeyCtrl;
    bool shift = io.KeyShift;
    bool alt = io.KeyAlt;

    // ========================================================================
    // COMPLETION POPUP - Highest priority when popup is open
    // ========================================================================
    // Ctrl+Space with the list open shows or hides the selected item's details.
    if (show_completion_popup_ && ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Space)) {
        completion_details_shown_ = !completion_details_shown_;
        completion_just_accepted_ = true;  // the editor must not type the space
        return;
    }
    if (show_completion_popup_ && !ctrl && !shift && !alt) {
        // TOFIX133 P0 item 6: Tab inserts, Enter types its new line, Up/Down
        // move the selection without moving the editor cursor.
        struct PopupBinding {
            ImGuiKey imgui;
            scriptkeys::PopupKey key;
        };
        static constexpr PopupBinding kPopupKeys[] = {{ImGuiKey_Tab, scriptkeys::PopupKey::Tab},
                                                      {ImGuiKey_Enter, scriptkeys::PopupKey::Enter},
                                                      {ImGuiKey_KeypadEnter, scriptkeys::PopupKey::Enter},
                                                      {ImGuiKey_Escape, scriptkeys::PopupKey::Escape},
                                                      {ImGuiKey_UpArrow, scriptkeys::PopupKey::Up},
                                                      {ImGuiKey_DownArrow, scriptkeys::PopupKey::Down}};
        const int visible = static_cast<int>(completion_entries_.size());
        for (const auto& b : kPopupKeys) {
            if (!ImGui::IsKeyPressed(b.imgui)) continue;
            switch (scriptkeys::ResolvePopupKey(b.key)) {
                case scriptkeys::PopupAction::Accept:
                    AcceptCompletion();
                    // The editor must not also insert the tab this frame.
                    completion_just_accepted_ = true;
                    for (int i = io.InputQueueCharacters.Size - 1; i >= 0; --i) {
                        if (io.InputQueueCharacters[i] == '\t') {
                            io.InputQueueCharacters.erase(io.InputQueueCharacters.Data + i);
                        }
                    }
                    return;
                case scriptkeys::PopupAction::CloseAndType:
                case scriptkeys::PopupAction::Close:
                    CloseCompletionPopup();
                    return;  // the editor still gets the key (Enter types its new line)
                case scriptkeys::PopupAction::Previous:
                case scriptkeys::PopupAction::Next:
                    selected_completion_ = scriptkeys::MoveSelection(
                        selected_completion_, b.key == scriptkeys::PopupKey::Up ? -1 : 1, visible);
                    completion_scroll_to_selected_ = true;
                    completion_just_accepted_ = true;  // keep the editor cursor where it is
                    return;
            }
        }
        // Other keys reach the editor (typing continues).
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

    // Typing, moving, selecting, clipboard and undo keys are the code view's.
    if (find_.open && !ctrl && !alt && ImGui::IsKeyPressed(ImGuiKey_F3)) FindStep(!shift);

    // Language keys (P3, boards 7-8; Preferences > Shortcuts lists them).
    if (CodeEditor* code = ActiveCodeEditor()) {
        if (!ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_F12, false)) {
            hover_ = {};
            GoToDefinition(ActiveCodeTarget(), *code, code->Doc().Primary().head);
            return;
        }
        if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
            RequestSignatures(true);
            completion_just_accepted_ = true;  // the editor must not type the space
            return;
        }
        if (signature_open_ && !ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) CloseSignatureHelp();
    }

    // Run, debug and notebook-toggle keys: one table decides, once per
    // frame (TOFIX133 P0 item 7; Preferences > Shortcuts lists the same).
    if (!IsActiveTabEditable()) return;
    scriptkeys::State key_state;
    key_state.debugging = debug_run_.active;
    key_state.paused = debug_run_.active && debug_.state == "paused";
    key_state.script_running = script_running_;
    key_state.notebook = tabs_[active_tab_index_]->cell_mode;
    struct Binding {
        ImGuiKey imgui;
        scriptkeys::Key key;
        bool repeat;
    };
    static constexpr Binding kBindings[] = {
        {ImGuiKey_F5, scriptkeys::Key::F5, false},       {ImGuiKey_F9, scriptkeys::Key::F9, false},
        {ImGuiKey_F10, scriptkeys::Key::F10, false},     {ImGuiKey_F11, scriptkeys::Key::F11, false},
        {ImGuiKey_Enter, scriptkeys::Key::Enter, false}, {ImGuiKey_Space, scriptkeys::Key::Space, false},
        {ImGuiKey_M, scriptkeys::Key::M, false}};
    for (const auto& b : kBindings) {
        if (!ImGui::IsKeyPressed(b.imgui, b.repeat)) continue;
        switch (scriptkeys::Resolve(b.key, ctrl, shift, alt, key_state)) {
            case scriptkeys::Action::RunScript: RunScript(); break;
            case scriptkeys::Action::StopScript:
                if (scripting_engine_) {
                    scripting_engine_->StopScript();
                    spdlog::info("Stop script requested via Shift+F5");
                }
                break;
            case scriptkeys::Action::RunSelection: RunSelection(); break;
            case scriptkeys::Action::RunSection: RunCurrentSection(); break;
            case scriptkeys::Action::StartDebug: Debug(); break;
            case scriptkeys::Action::Continue: DebugCommand("continue"); break;
            case scriptkeys::Action::StopDebug: StopDebugging(); break;
            case scriptkeys::Action::StepOver: DebugCommand("over"); break;
            case scriptkeys::Action::StepInto: DebugCommand("into"); break;
            case scriptkeys::Action::StepOut: DebugCommand("out"); break;
            case scriptkeys::Action::ToggleBreakpoint: ToggleBreakpointAtCursor(); break;
            case scriptkeys::Action::ToggleNotebook: ToggleCellMode(); break;
            case scriptkeys::Action::Completion: UpdateAutoCompletion(true); break;
            case scriptkeys::Action::None: continue;
        }
        return;  // one action per key press
    }
}


} // namespace cyxwiz
