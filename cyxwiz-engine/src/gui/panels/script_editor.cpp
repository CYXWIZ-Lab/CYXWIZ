#include "script_editor.h"
#include "command_window.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../../scripting/scripting_engine.h"
#include "../../scripting/debugger.h"
#include "../../core/file_dialogs.h"
#include "../../core/keyboard_shortcuts.h"
#include <imgui.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdio>
#include <regex>
#include <spdlog/spdlog.h>

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
                    // Script completed successfully
                    if (remaining_output.empty()) {
                        // No output was produced, show completion message
                        command_window_->DisplayScriptOutput("Script", "Completed successfully", false);
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
            bool has_active_tab = active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size());
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

            // Handle tab close (don't allow closing while loading)
            if (!open && !tab->is_loading) {
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

    // Show loading indicator if tab is loading
    if (tab->is_loading) {
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

    // Cell-based mode (Jupyter-like notebook)
    if (tab->cell_mode) {
        RenderCellBasedEditor();
        return;
    }

    // Show debug toolbar when debugging is active (traditional mode)
    if (debug_mode_active_ && debugger_) {
        RenderDebugToolbar();
    }

    // Apply font scale for editor
    if (font_scale_ != 1.0f) {
        ImGui::SetWindowFontScale(font_scale_);
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

    // Reset font scale
    if (font_scale_ != 1.0f) {
        ImGui::SetWindowFontScale(1.0f);
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
    int visible_lines = static_cast<int>(available_height / (16.0f * font_scale_));  // Approximate visible lines
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
    // Use is_focused_ directly instead of keyboard context to avoid timing issues
    // (context is detected before Render() updates is_focused_)
    if (!is_focused_ && !show_completion_popup_) {
        return;  // Not focused and no popup, don't process shortcuts
    }

    // Handle debug shortcuts (F5, F9, F10, F11) - only when focused
    if (is_focused_) {
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
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && active_tab_index_ >= 0) {
        SaveFile();
    }
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_S) && active_tab_index_ >= 0) {
        SaveFileAs();
    }
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_W) && active_tab_index_ >= 0) {
        close_tab_index_ = active_tab_index_;
    }

    // Toggle cell mode (Jupyter-like notebook mode)
    if (ctrl && shift && !alt && ImGui::IsKeyPressed(ImGuiKey_N) && active_tab_index_ >= 0) {
        ToggleCellMode();
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

    // Ctrl+Space triggers completion manually (force = true bypasses trigger char check)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
        UpdateAutoCompletion(true);
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

    // Check if we can replace an existing empty untitled tab
    int empty_tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        auto& tab = tabs_[i];
        std::string tab_text = tab->editor.GetText();
        tab_text.erase(0, tab_text.find_first_not_of(" \t\n\r"));
        tab_text.erase(tab_text.find_last_not_of(" \t\n\r") + 1);

        if (tab->is_new && !tab->is_modified && tab_text.empty()) {
            empty_tab_index = i;
            break;
        }
    }

    // Use the empty tab or create a new one
    int tab_index;
    if (empty_tab_index >= 0) {
        tab_index = empty_tab_index;
        auto& tab = tabs_[tab_index];
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = "Loading...";
    } else {
        // Create new tab with loading state
        auto tab = std::make_unique<EditorTab>();
        tab->filename = std::filesystem::path(path).filename().string();
        tab->filepath = path;
        tab->is_new = false;
        tab->is_modified = false;
        tab->is_loading = true;
        tab->load_progress = 0.0f;
        tab->load_status = "Loading...";
        tabs_.push_back(std::move(tab));
        tab_index = static_cast<int>(tabs_.size()) - 1;
    }

    active_tab_index_ = tab_index;
    request_focus_ = true;
    request_window_focus_ = true;

    // Load file asynchronously
    OpenFileAsync(path);
}

void ScriptEditorPanel::OpenFileAsync(const std::string& filepath) {
    std::string path = filepath;
    std::string filename = std::filesystem::path(path).filename().string();

    // Find the tab that's loading this file
    int tab_index = -1;
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == path && tabs_[i]->is_loading) {
            tab_index = i;
            break;
        }
    }

    if (tab_index < 0) {
        spdlog::error("OpenFileAsync: Could not find loading tab for {}", path);
        return;
    }

    spdlog::info("Starting async load of script: {}", filename);

    AsyncTaskManager::Instance().RunAsync(
        "Loading: " + filename,
        [this, tab_index, path](LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            // Read file content in background thread
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                task.MarkFailed("Could not open file");
                return;
            }

            task.ReportProgress(0.3f, "Reading content...");

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::string content;
            content.resize(static_cast<size_t>(size));

            if (!file.read(&content[0], size)) {
                task.MarkFailed("Failed to read file content");
                return;
            }

            task.ReportProgress(0.8f, "Finalizing...");

            // Store content for main thread to finalize
            if (tab_index < static_cast<int>(tabs_.size())) {
                tabs_[tab_index]->pending_content = std::move(content);
            }

            task.ReportProgress(1.0f, "Complete");
            task.MarkCompleted();
        },
        [this, tab_index](float progress, const std::string& status) {
            // Progress callback - update tab
            if (tab_index < static_cast<int>(tabs_.size())) {
                tabs_[tab_index]->load_progress = progress;
                tabs_[tab_index]->load_status = status;
            }
        },
        [this, tab_index, path](bool success, const std::string& error) {
            // Completion callback
            if (tab_index < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[tab_index];
                if (success) {
                    // Finalize on main thread
                    FinalizeAsyncLoad(tab_index);
                    spdlog::info("Async script load completed: {}", path);
                } else {
                    tab->is_loading = false;
                    tab->load_status = "Failed: " + error;
                    spdlog::error("Async script load failed: {} - {}", path, error);
                }
            }
        }
    );
}

void ScriptEditorPanel::FinalizeAsyncLoad(int tab_index) {
    if (tab_index < 0 || tab_index >= static_cast<int>(tabs_.size())) return;

    auto& tab = tabs_[tab_index];
    if (!tab->is_loading) return;

    // Configure editor with Python language definition
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
        case EditorTheme::Monokai: tab->editor.SetPalette(GetMonokaiPalette()); break;
        case EditorTheme::Dracula: tab->editor.SetPalette(GetDraculaPalette()); break;
        case EditorTheme::OneDark: tab->editor.SetPalette(GetOneDarkPalette()); break;
        case EditorTheme::GitHub: tab->editor.SetPalette(GetGitHubPalette()); break;
    }

    tab->editor.SetShowWhitespaces(show_whitespace_);
    tab->editor.SetTabSize(tab_size_);
    tab->editor.SetImGuiChildIgnored(false);
    tab->editor.SetReadOnly(false);
    tab->editor.SetText(tab->pending_content);

    // Clear loading state
    tab->is_loading = false;
    tab->load_progress = 1.0f;
    tab->load_status.clear();
    tab->pending_content.clear();

    request_focus_ = true;
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

    // Check if script is empty before showing save dialog
    std::string content = tab->editor.GetText();
    // Trim whitespace to check if truly empty
    bool is_empty = content.empty() ||
                    content.find_first_not_of(" \t\n\r") == std::string::npos;
    if (is_empty) {
        show_empty_script_warning_ = true;
        spdlog::warn("Cannot save empty script - no content present");
        return;
    }

    // Show save dialog
    std::string path = SaveFileDialog();
    if (path.empty()) return;  // User cancelled

    // Ensure .cyx extension
    std::filesystem::path fspath(path);
    if (fspath.extension() != ".cyx") {
        path += ".cyx";
    }

    // Save content (already have content from empty check above)
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
    std::string script_text;
    if (!tab->filepath.empty() && std::filesystem::exists(tab->filepath)) {
        // Read file content
        std::ifstream file(tab->filepath);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            script_text = buffer.str();
            file.close();
        }
    }

    if (script_text.empty()) {
        // Use text from editor
        script_text = tab->editor.GetText();
    }

    // Strip out %% section markers before executing (always, regardless of source)
    std::string script;
    std::istringstream stream(script_text);
    std::string line;
    while (std::getline(stream, line)) {
        // Only skip lines that are ONLY a %% marker (with optional whitespace)
        std::string trimmed = line;
        size_t start = trimmed.find_first_not_of(" \t");
        size_t end = trimmed.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end != std::string::npos) {
            trimmed = trimmed.substr(start, end - start + 1);
        } else {
            trimmed = "";
        }
        // Keep the line unless it's exactly "%%"
        if (trimmed != "%%") {
            script += line + "\n";
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

std::string ScriptEditorPanel::DedentCode(const std::string& code) {
    std::vector<std::string> lines;
    std::istringstream stream(code);
    std::string line;

    // Split into lines
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    if (lines.empty()) return code;

    // Find minimum indentation (ignoring empty lines and whitespace-only lines)
    size_t min_indent = std::string::npos;
    for (const auto& l : lines) {
        if (l.empty()) continue;
        size_t first_non_space = l.find_first_not_of(" \t");
        if (first_non_space == std::string::npos) continue;  // Whitespace-only line
        if (first_non_space < min_indent) min_indent = first_non_space;
    }

    // No dedent needed
    if (min_indent == 0 || min_indent == std::string::npos) return code;

    // Remove common indentation
    std::string result;
    for (const auto& l : lines) {
        if (l.empty()) {
            result += "\n";
        } else {
            size_t first_non_space = l.find_first_not_of(" \t");
            if (first_non_space == std::string::npos) {
                // Whitespace-only line, keep it empty
                result += "\n";
            } else {
                result += l.substr(min_indent) + "\n";
            }
        }
    }

    // Remove trailing newline if original didn't have one
    if (!code.empty() && code.back() != '\n' && !result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    return result;
}

void ScriptEditorPanel::RunSelection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;
    if (script_running_) return;  // Already running

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

    spdlog::info("Running selection asynchronously");
    if (command_window_) {
        command_window_->DisplayScriptOutput(tab->filename + " (selection)", "Running...", false);
    }

    // Dedent and execute asynchronously for plot capture support
    std::string dedented = DedentCode(selected_text);
    spdlog::debug("Dedented selection:\n{}", dedented);
    scripting_engine_->ExecuteScriptAsync(dedented);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

void ScriptEditorPanel::RunCurrentSection() {
    if (active_tab_index_ < 0 || !scripting_engine_) return;
    if (script_running_) return;  // Already running

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

    std::string section_name = tab->filename + " (lines " +
                              std::to_string(section.start_line) + "-" +
                              std::to_string(section.end_line) + ")";

    spdlog::info("Running section {} asynchronously", section_name);
    if (command_window_) {
        command_window_->DisplayScriptOutput(section_name, "Running...", false);
    }

    // Dedent and execute asynchronously for plot capture support
    std::string dedented = DedentCode(section.code);
    spdlog::debug("Dedented section:\n{}", dedented);
    scripting_engine_->ExecuteScriptAsync(dedented);
    script_running_ = true;
    running_indicator_time_ = 0.0f;
}

void ScriptEditorPanel::Debug() {
    if (tabs_.empty() || active_tab_index_ < 0) {
        spdlog::warn("No script to debug");
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Initialize debugger if not already done
    if (!debugger_) {
        debugger_ = std::make_unique<scripting::DebuggerManager>();
        if (scripting_engine_ && scripting_engine_->IsInitialized()) {
            // Get the raw ScriptingEngine pointer from the shared_ptr
            debugger_->Initialize(scripting_engine_.get());

            // Set up callbacks
            debugger_->SetBreakpointHitCallback([this](const std::string& cell_id, int line) {
                debug_mode_active_ = true;
                debug_current_cell_ = cell_id;
                debug_current_line_ = line;
                spdlog::info("Breakpoint hit at {}:{}", cell_id, line);
            });

            debugger_->SetStateChangedCallback([this](scripting::DebugState state) {
                if (state == scripting::DebugState::Disconnected) {
                    debug_mode_active_ = false;
                    debug_current_line_ = -1;
                    debug_current_cell_.clear();
                } else if (state == scripting::DebugState::Running) {
                    debug_mode_active_ = true;
                }
            });

            spdlog::info("Debugger initialized");
        } else {
            spdlog::error("Cannot initialize debugger: scripting engine not ready");
            debugger_.reset();
            return;
        }
    }

    // Get current cell content to debug
    if (tab->cell_mode && tab->selected_cell >= 0 &&
        tab->selected_cell < static_cast<int>(tab->cell_manager.GetCellCount())) {
        Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
        if (cell.type == CellType::Code) {
            cell.SyncSourceFromEditor();

            // Execute with debugging enabled
            debugger_->ExecuteWithDebug(cell.source, cell.id);
            debug_mode_active_ = true;
            spdlog::info("Started debugging cell {}", cell.id);
        }
    } else {
        // Debug whole script (traditional mode)
        std::string script = tab->editor.GetText();
        debugger_->ExecuteWithDebug(script, tab->filepath.empty() ? tab->filename : tab->filepath);
        debug_mode_active_ = true;
        spdlog::info("Started debugging script");
    }
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
    auto result = FileDialogs::OpenScript();
    return result.value_or("");
}

std::string ScriptEditorPanel::SaveFileDialog() {
    auto result = FileDialogs::SaveScript();
    return result.value_or("");
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
        // Each %% both ENDS the previous section AND STARTS the next one (MATLAB-style)
        if (line.find("%%") != std::string::npos) {
            if (in_section && !current_section.code.empty()) {
                // End current section (only if it has content)
                current_section.end_line = line_num - 1;
                sections.push_back(current_section);
            }
            // Start new section after this %%
            current_section = Section();
            current_section.start_line = line_num + 1;
            in_section = true;
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

bool ScriptEditorPanel::HasEmptyNewTab() const {
    for (const auto& tab : tabs_) {
        if (tab->is_new && !tab->is_modified) {
            return true;
        }
    }
    return false;
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

void ScriptEditorPanel::SetSyntaxHighlighting(bool enabled) {
    syntax_highlighting_ = enabled;
    ApplySyntaxHighlightingToAllTabs();
}

void ScriptEditorPanel::SetTheme(int theme_index) {
    if (theme_index >= 0 && theme_index <= 6) {
        current_theme_ = static_cast<EditorTheme>(theme_index);
        ApplyThemeToAllTabs();
    }
}

std::vector<std::string> ScriptEditorPanel::GetOpenFilePaths() const {
    std::vector<std::string> result;
    result.reserve(tabs_.size());
    for (const auto& tab : tabs_) {
        if (tab && !tab->filepath.empty()) {
            result.push_back(tab->filepath);
        }
    }
    return result;
}

void ScriptEditorPanel::SetActiveTabIndex(int index) {
    if (index < 0 || index >= static_cast<int>(tabs_.size())) {
        return;
    }
    active_tab_index_ = index;
    request_window_focus_ = true;
}

// ==================== Cell-Based Editor (Jupyter-like) ====================

void ScriptEditorPanel::ToggleCellMode() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    tab->cell_mode = !tab->cell_mode;

    if (tab->cell_mode) {
        // Entering cell mode - parse the text into cells
        std::string content = tab->editor.GetText();
        tab->cell_manager.SetScriptingEngine(scripting_engine_);
        tab->cell_manager.ParseFromCyx(content);

        // If no cells were created, add an empty code cell
        if (tab->cell_manager.GetCellCount() == 0) {
            tab->cell_manager.AddCell(CellType::Code);
        }

        tab->selected_cell = 0;
        tab->editing_cell = -1;  // Start in command mode
        tab->last_editing_cell = -1;
        spdlog::info("Entered cell mode with {} cells", tab->cell_manager.GetCellCount());
    } else {
        // Exiting cell mode - serialize cells back to text
        std::string content = tab->cell_manager.SerializeToCyx();
        tab->editor.SetText(content);
        tab->is_modified = true;
        spdlog::info("Exited cell mode");
    }
}

void ScriptEditorPanel::RenderCellBasedEditor() {
    auto& tab = tabs_[active_tab_index_];

    // Don't apply font scaling in notebook mode - use default font for cleaner look

    // Handle keyboard shortcuts in cell mode
    HandleCellKeyboardShortcuts();

    // Calculate available size
    float available_height = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing();
    float available_width = ImGui::GetContentRegionAvail().x;

    // Jupyter-style toolbar at top with subtle background
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.16f, 0.16f, 0.18f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
    ImGui::BeginChild("##cell_toolbar", ImVec2(available_width, 36), false);
    {
        // Style toolbar buttons
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.25f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.35f, 0.38f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);

        // Add cell buttons
        if (ImGui::Button(ICON_FA_PLUS " Code")) {
            int pos = tab->selected_cell >= 0 ? tab->selected_cell + 1 : -1;
            int new_idx = tab->cell_manager.AddCell(CellType::Code, pos);
            tab->selected_cell = new_idx;
            tab->editing_cell = new_idx;
            tab->is_modified = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_PLUS " Markdown")) {
            int pos = tab->selected_cell >= 0 ? tab->selected_cell + 1 : -1;
            int new_idx = tab->cell_manager.AddCell(CellType::Markdown, pos);
            tab->selected_cell = new_idx;
            tab->editing_cell = new_idx;
            tab->is_modified = true;
        }

        ImGui::SameLine();
        ImGui::SameLine(0, 15);
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
        ImGui::SameLine(0, 15);

        // Run buttons with accent color
        bool can_run = scripting_engine_ && !scripting_engine_->IsScriptRunning();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.55f, 0.30f, 1.0f));
        ImGui::BeginDisabled(!can_run || tab->selected_cell < 0);
        if (ImGui::Button(ICON_FA_PLAY " Run")) {
            if (tab->selected_cell >= 0) {
                tab->cell_manager.RunCell(tab->selected_cell);
            }
        }
        ImGui::EndDisabled();
        ImGui::PopStyleColor(2);

        ImGui::SameLine();

        ImGui::BeginDisabled(!can_run);
        if (ImGui::Button(ICON_FA_FORWARD " Run All")) {
            tab->cell_manager.RunAllCells();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::SameLine(0, 15);
        ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "|");
        ImGui::SameLine(0, 15);

        // Clear outputs
        if (ImGui::Button(ICON_FA_ERASER " Clear")) {
            tab->cell_manager.ClearAllOutputs();
        }

        // Right-aligned cell count
        float right_text_width = ImGui::CalcTextSize("Cells: 999").x + 20;
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - right_text_width);
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Cells: %d", tab->cell_manager.GetCellCount());

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // Show debug toolbar when debugging is active
    if (debug_mode_active_ && debugger_) {
        RenderDebugToolbar();
    }

    // Jupyter-style cells container with scroll and subtle background
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.11f, 0.11f, 0.13f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 15));
    ImGui::BeginChild("##cells_container", ImVec2(available_width, available_height - 40), false,
                      ImGuiWindowFlags_AlwaysVerticalScrollbar);
    {
        // Restore scroll position
        if (tab->cell_scroll_y >= 0.0f) {
            // Only restore scroll on first frame after mode switch
            static bool first_render = true;
            if (first_render) {
                ImGui::SetScrollY(tab->cell_scroll_y);
                first_render = false;
            }
        }

        // Render each cell
        for (int i = 0; i < static_cast<int>(tab->cell_manager.GetCellCount()); ++i) {
            Cell& cell = tab->cell_manager.GetCell(i);
            RenderCell(cell, i);
        }

        // Save scroll position
        tab->cell_scroll_y = ImGui::GetScrollY();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void ScriptEditorPanel::RenderCell(Cell& cell, int index) {
    auto& tab = tabs_[active_tab_index_];
    bool is_selected = (tab->selected_cell == index);

    ImGui::PushID(index);

    // Cell container - Jupyter-style
    float available_width = ImGui::GetContentRegionAvail().x;

    // Jupyter-style: left border indicator for selected cell
    ImVec4 left_border_color;
    if (cell.state == CellState::Running) {
        left_border_color = ImVec4(0.0f, 0.7f, 0.4f, 1.0f);  // Green while running
    } else if (cell.state == CellState::Error) {
        left_border_color = ImVec4(0.9f, 0.3f, 0.3f, 1.0f);  // Red on error
    } else if (is_selected) {
        left_border_color = ImVec4(0.3f, 0.5f, 0.9f, 1.0f);  // Blue for selected
    } else {
        left_border_color = ImVec4(0.2f, 0.2f, 0.22f, 1.0f);  // Subtle gray
    }

    // Draw left border indicator (Jupyter-style)
    ImVec2 cell_start_pos = ImGui::GetCursorScreenPos();

    // Cell background color
    ImVec4 cell_bg = is_selected ? ImVec4(0.14f, 0.14f, 0.16f, 1.0f) : ImVec4(0.12f, 0.12f, 0.14f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, cell_bg);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.2f, 0.22f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 10));

    // Start cell region
    ImGui::BeginChild(("##cell_" + std::to_string(index)).c_str(), ImVec2(available_width, 0),
                      ImGuiChildFlags_Border | ImGuiChildFlags_AutoResizeY);

    // Draw left accent border after BeginChild (overlay)
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cell_min = ImGui::GetWindowPos();
    ImVec2 cell_max = ImVec2(cell_min.x + 4.0f, cell_min.y + ImGui::GetWindowHeight());
    draw_list->AddRectFilled(cell_min, cell_max, ImGui::ColorConvertFloat4ToU32(left_border_color));

    // Minimal toolbar - just show cell type and essential buttons inline
    RenderCellToolbar(index);

    // Cell content (skip if collapsed)
    if (!cell.collapsed) {
        if (cell.type == CellType::Code) {
            RenderCodeCell(cell, index);
        } else if (cell.type == CellType::Markdown) {
            RenderMarkdownCell(cell, index);
        } else {
            // Raw cell - just text
            ImGui::TextWrapped("%s", cell.source.c_str());
        }

        // Cell outputs (only when not collapsed)
        // Cell outputs - Jupyter-style with Out[n]: label
        if (!cell.outputs.empty() && !cell.output_collapsed) {
            ImGui::Spacing();
            ImGui::Spacing();

            // Jupyter-style Out[n]: label
            std::string out_label = "Out[" + (cell.execution_count > 0 ? std::to_string(cell.execution_count) : " ") + "]:";
            ImGui::TextColored(ImVec4(0.7f, 0.4f, 0.4f, 1.0f), "%s", out_label.c_str());
            ImGui::SameLine(0, 10);

            // Output area with subtle background
            float output_width = ImGui::GetContentRegionAvail().x;
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 1.0f));
            ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 3.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
            ImGui::BeginChild("##output_view", ImVec2(output_width, 0), ImGuiChildFlags_AutoResizeY);

            // Render outputs
            for (const auto& output : cell.outputs) {
                RenderCellOutput(output);
            }

            ImGui::EndChild();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleColor();

            // Clear outputs button (subtle, right-aligned)
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.2f, 0.2f, 0.5f));
            ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20);
            if (ImGui::SmallButton(ICON_FA_XMARK "##clear_output")) {
                cell.ClearOutputs();
            }
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Clear output");
            }
        } else if (!cell.outputs.empty() && cell.output_collapsed) {
            // Show collapsed output indicator
            ImGui::Spacing();
            std::string out_label = "Out[" + (cell.execution_count > 0 ? std::to_string(cell.execution_count) : " ") + "]:";
            ImGui::TextColored(ImVec4(0.5f, 0.3f, 0.3f, 1.0f), "%s", out_label.c_str());
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "(output hidden - %zu items)", cell.outputs.size());
        }
    } else {
        // Collapsed indicator
        int line_count = 0;
        for (char c : cell.source) {
            if (c == '\n') line_count++;
        }
        line_count++;  // Count last line even without trailing newline

        ImGui::TextDisabled("... (%d lines collapsed)", line_count);
    }

    ImGui::EndChild();

    ImGui::PopStyleVar(3);  // ChildBorderSize, ChildRounding, WindowPadding
    ImGui::PopStyleColor(2);  // ChildBg, Border

    // Handle cell selection
    if (ImGui::IsItemClicked() && !is_selected) {
        tab->selected_cell = index;
        tab->editing_cell = -1;  // Exit edit mode when clicking another cell
        tab->last_editing_cell = -1;
    }

    // Double-click to edit
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
        tab->selected_cell = index;
        tab->editing_cell = index;
    }

    ImGui::PopID();
    ImGui::Spacing();
    ImGui::Spacing();  // Extra spacing between cells like Jupyter
}

void ScriptEditorPanel::RenderCodeCell(Cell& cell, int index) {
    auto& tab = tabs_[active_tab_index_];
    bool is_editing = (tab->editing_cell == index);

    // Jupyter-style In [n]: label on the left
    ImGui::BeginGroup();
    {
        // Execution count label - Jupyter style
        std::string exec_label;
        ImVec4 label_color;
        if (cell.state == CellState::Running) {
            exec_label = "In [*]:";
            label_color = ImVec4(0.0f, 0.7f, 0.4f, 1.0f);  // Green while running
        } else if (cell.execution_count > 0) {
            exec_label = "In [" + std::to_string(cell.execution_count) + "]:";
            label_color = ImVec4(0.4f, 0.5f, 0.7f, 1.0f);  // Blue-gray for executed
        } else {
            exec_label = "In [ ]:";
            label_color = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);  // Gray for not executed
        }

        ImGui::TextColored(label_color, "%s", exec_label.c_str());
    }
    ImGui::EndGroup();

    ImGui::SameLine(0, 10);

    // Code content area
    float code_width = ImGui::GetContentRegionAvail().x;
    float min_height = 50.0f;

    if (is_editing) {
        // Edit mode - show TextEditor
        // Only sync editor from source when ENTERING edit mode, not every frame
        if (tab->last_editing_cell != index) {
            cell.SyncEditorFromSource();
            tab->last_editing_cell = index;
        }

        // Calculate height based on content
        int line_count = cell.editor.GetTotalLines();
        float line_height = ImGui::GetTextLineHeightWithSpacing();
        float content_height = std::max(min_height, (line_count + 1) * line_height);
        content_height = std::min(content_height, 400.0f);  // Cap height

        ImGui::PushID("code_editor");

        // Temporarily disable keyboard input if we just accepted a completion
        if (completion_just_accepted_) {
            cell.editor.SetHandleKeyboardInputs(false);
        }

        cell.editor.Render("##code", ImVec2(code_width, content_height));

        // Re-enable keyboard input and clear the flag
        if (completion_just_accepted_) {
            cell.editor.SetHandleKeyboardInputs(true);
            completion_just_accepted_ = false;
        }

        // Sync changes back
        cell.SyncSourceFromEditor();

        // Mark modified if text changed
        if (cell.editor.IsTextChanged()) {
            tab->is_modified = true;
        }

        ImGui::PopID();
    } else {
        // View mode - display code with subtle background
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.09f, 0.09f, 0.10f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 3.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
        ImGui::BeginChild("##code_view", ImVec2(code_width, 0), ImGuiChildFlags_AutoResizeY);

        // Code display with line numbers
        std::istringstream stream(cell.source);
        std::string line;
        int line_num = 1;
        while (std::getline(stream, line)) {
            // Line number in subtle color
            ImGui::TextColored(ImVec4(0.35f, 0.35f, 0.38f, 1.0f), "%3d ", line_num++);
            ImGui::SameLine(0, 0);
            // Code in bright color
            ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "%s", line.c_str());
        }

        // Handle empty cell
        if (cell.source.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "# Empty cell - double-click to edit");
        }

        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
}

void ScriptEditorPanel::RenderMarkdownCell(Cell& cell, int index) {
    auto& tab = tabs_[active_tab_index_];
    bool is_editing = (tab->editing_cell == index);

    float content_width = ImGui::GetContentRegionAvail().x;

    if (is_editing) {
        // Edit mode - show TextEditor for markdown directly (no extra container)
        // Only sync editor from source when ENTERING edit mode, not every frame
        if (tab->last_editing_cell != index) {
            cell.SyncEditorFromSource();
            tab->last_editing_cell = index;
        }

        int line_count = cell.editor.GetTotalLines();
        float line_height = ImGui::GetTextLineHeightWithSpacing();
        float content_height = std::max(80.0f, (line_count + 1) * line_height);
        content_height = std::min(content_height, 300.0f);

        // Temporarily disable keyboard input if we just accepted a completion
        if (completion_just_accepted_) {
            cell.editor.SetHandleKeyboardInputs(false);
        }

        // Render editor directly
        cell.editor.Render("##markdown_edit", ImVec2(content_width, content_height));

        // Re-enable keyboard input and clear the flag
        if (completion_just_accepted_) {
            cell.editor.SetHandleKeyboardInputs(true);
            completion_just_accepted_ = false;
        }

        cell.SyncSourceFromEditor();

        if (cell.editor.IsTextChanged()) {
            tab->is_modified = true;
        }
    } else {
        // View mode - render markdown with background
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.09f, 0.09f, 0.10f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 3.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 8));
        ImGui::BeginChild("##md_view", ImVec2(content_width, 0), ImGuiChildFlags_AutoResizeY);

        if (cell.source.empty()) {
            ImGui::TextColored(ImVec4(0.4f, 0.4f, 0.4f, 1.0f), "Empty markdown cell - double-click to edit");
        } else {
            OutputRenderer::RenderMarkdown(cell.source);
        }

        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
    }
}

void ScriptEditorPanel::RenderCellOutput(const CellOutput& output) {
    OutputRenderer::RenderCellOutput(output);
}

void ScriptEditorPanel::RenderCellToolbar(int index) {
    auto& tab = tabs_[active_tab_index_];
    if (index < 0 || index >= static_cast<int>(tab->cell_manager.GetCellCount())) return;
    Cell& cell = tab->cell_manager.GetCell(index);

    bool is_editing = (tab->editing_cell == index);

    // Compact Jupyter-style toolbar with subtle styling
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));  // Transparent buttons
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));

    // Cell type badge
    const char* type_label = (cell.type == CellType::Code) ? "Code" :
                             (cell.type == CellType::Markdown) ? "Markdown" : "Raw";
    ImVec4 badge_color = (cell.type == CellType::Code) ? ImVec4(0.3f, 0.4f, 0.6f, 1.0f) :
                         (cell.type == CellType::Markdown) ? ImVec4(0.4f, 0.5f, 0.3f, 1.0f) :
                         ImVec4(0.5f, 0.4f, 0.3f, 1.0f);
    ImGui::TextColored(badge_color, "%s", type_label);
    ImGui::SameLine(0, 15);

    // Run button (for code cells) with play icon
    if (cell.type == CellType::Code) {
        bool can_run = scripting_engine_ && !scripting_engine_->IsScriptRunning();
        ImGui::BeginDisabled(!can_run);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 0.5f, 1.0f));  // Green
        if (ImGui::SmallButton(ICON_FA_PLAY)) {
            tab->cell_manager.RunCell(index);
        }
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Run cell (Shift+Enter)");
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
    }

    // Spacer to push remaining buttons to the right
    float right_buttons_width = 120.0f;
    float available = ImGui::GetContentRegionAvail().x;
    if (available > right_buttons_width) {
        ImGui::Dummy(ImVec2(available - right_buttons_width, 0));
        ImGui::SameLine();
    }

    // Edit/View toggle
    if (is_editing) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.8f, 0.5f, 1.0f));  // Green check
        if (ImGui::SmallButton(ICON_FA_CHECK)) {
            tab->editing_cell = -1;  // Exit edit mode
            tab->last_editing_cell = -1;
        }
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Done (Escape)");
        }
    } else {
        if (ImGui::SmallButton(ICON_FA_PEN)) {
            tab->editing_cell = index;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Edit (Enter)");
        }
    }
    ImGui::SameLine();

    // Move up/down
    ImGui::BeginDisabled(index == 0);
    if (ImGui::SmallButton(ICON_FA_ARROW_UP)) {
        if (tab->cell_manager.MoveCell(index, index - 1)) {
            tab->selected_cell = index - 1;
            tab->is_modified = true;
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, 2);

    ImGui::BeginDisabled(index >= tab->cell_manager.GetCellCount() - 1);
    if (ImGui::SmallButton(ICON_FA_ARROW_DOWN)) {
        if (tab->cell_manager.MoveCell(index, index + 1)) {
            tab->selected_cell = index + 1;
            tab->is_modified = true;
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();

    // Delete with confirmation color on hover
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.6f, 0.2f, 0.2f, 0.8f));
    if (ImGui::SmallButton(ICON_FA_TRASH)) {
        if (tab->cell_manager.DeleteCell(index)) {
            if (tab->selected_cell >= tab->cell_manager.GetCellCount()) {
                tab->selected_cell = tab->cell_manager.GetCellCount() - 1;
            }
            tab->editing_cell = -1;
            tab->last_editing_cell = -1;
            tab->is_modified = true;
        }
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Delete cell");
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);

    ImGui::Spacing();
}

void ScriptEditorPanel::HandleCellKeyboardShortcuts() {
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        return;
    }

    // Handle debug shortcuts first (F5, F9, F10, F11)
    HandleDebugKeyboardShortcuts();

    auto& tab = tabs_[active_tab_index_];
    bool ctrl = ImGui::GetIO().KeyCtrl;
    bool shift = ImGui::GetIO().KeyShift;
    bool is_editing = (tab->editing_cell >= 0);

    // Toggle cell mode: Ctrl+Shift+N
    if (ctrl && shift && ImGui::IsKeyPressed(ImGuiKey_N)) {
        ToggleCellMode();
        return;
    }

    // Escape - exit edit mode
    if (ImGui::IsKeyPressed(ImGuiKey_Escape) && is_editing) {
        // Sync changes before exiting
        if (tab->editing_cell >= 0 && tab->editing_cell < tab->cell_manager.GetCellCount()) {
            Cell& cell = tab->cell_manager.GetCell(tab->editing_cell);
            cell.SyncSourceFromEditor();
        }
        tab->editing_cell = -1;
        tab->last_editing_cell = -1;
        return;
    }

    // Enter - enter edit mode (when not editing)
    if (ImGui::IsKeyPressed(ImGuiKey_Enter) && !is_editing && tab->selected_cell >= 0 && !shift) {
        tab->editing_cell = tab->selected_cell;
        return;
    }

    // Shift+Enter - run cell and move to next
    if (shift && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (tab->selected_cell >= 0 && scripting_engine_ && !scripting_engine_->IsScriptRunning()) {
            // Sync changes if editing
            if (is_editing && tab->editing_cell >= 0 && tab->editing_cell < tab->cell_manager.GetCellCount()) {
                Cell& cell = tab->cell_manager.GetCell(tab->editing_cell);
                cell.SyncSourceFromEditor();
            }
            tab->cell_manager.RunCell(tab->selected_cell);

            // Move to next cell or create new one
            if (tab->selected_cell < tab->cell_manager.GetCellCount() - 1) {
                tab->selected_cell++;
            } else {
                // Create new cell at end
                int new_idx = tab->cell_manager.AddCell(CellType::Code);
                tab->selected_cell = new_idx;
                tab->is_modified = true;
            }
            tab->editing_cell = -1;  // Exit edit mode
            tab->last_editing_cell = -1;
        }
        return;
    }

    // Arrow keys for navigation (when not editing)
    if (!is_editing) {
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            if (tab->selected_cell > 0) {
                tab->selected_cell--;
            }
            return;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            if (tab->selected_cell < tab->cell_manager.GetCellCount() - 1) {
                tab->selected_cell++;
            }
            return;
        }

        // A - add cell above
        if (ImGui::IsKeyPressed(ImGuiKey_A)) {
            int pos = tab->selected_cell >= 0 ? tab->selected_cell : 0;
            int new_idx = tab->cell_manager.AddCell(CellType::Code, pos);
            tab->selected_cell = new_idx;
            tab->editing_cell = new_idx;
            tab->is_modified = true;
            return;
        }

        // B - add cell below
        if (ImGui::IsKeyPressed(ImGuiKey_B)) {
            int pos = tab->selected_cell >= 0 ? tab->selected_cell + 1 : -1;
            int new_idx = tab->cell_manager.AddCell(CellType::Code, pos);
            tab->selected_cell = new_idx;
            tab->editing_cell = new_idx;
            tab->is_modified = true;
            return;
        }

        // D,D - delete cell (double tap)
        static float last_d_press = -10.0f;
        if (ImGui::IsKeyPressed(ImGuiKey_D)) {
            float current_time = static_cast<float>(ImGui::GetTime());
            if (current_time - last_d_press < 0.3f) {
                if (tab->selected_cell >= 0) {
                    tab->cell_manager.DeleteCell(tab->selected_cell);
                    if (tab->selected_cell >= tab->cell_manager.GetCellCount()) {
                        tab->selected_cell = tab->cell_manager.GetCellCount() - 1;
                    }
                    tab->is_modified = true;
                }
                last_d_press = -10.0f;
            } else {
                last_d_press = current_time;
            }
            return;
        }

        // M - convert to markdown
        if (ImGui::IsKeyPressed(ImGuiKey_M)) {
            if (tab->selected_cell >= 0 && tab->selected_cell < tab->cell_manager.GetCellCount()) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                if (cell.type == CellType::Code) {
                    cell.type = CellType::Markdown;
                    tab->is_modified = true;
                }
            }
            return;
        }

        // Y - convert to code
        if (ImGui::IsKeyPressed(ImGuiKey_Y)) {
            if (tab->selected_cell >= 0 && tab->selected_cell < tab->cell_manager.GetCellCount()) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                if (cell.type == CellType::Markdown) {
                    cell.type = CellType::Code;
                    cell.SetupCodeEditor();  // Restore Python syntax highlighting
                    tab->is_modified = true;
                }
            }
            return;
        }

        // C - toggle cell collapse
        if (ImGui::IsKeyPressed(ImGuiKey_C)) {
            if (tab->selected_cell >= 0 && tab->selected_cell < tab->cell_manager.GetCellCount()) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                cell.collapsed = !cell.collapsed;
                tab->is_modified = true;
            }
            return;
        }

        // O - toggle output collapse
        if (ImGui::IsKeyPressed(ImGuiKey_O)) {
            if (tab->selected_cell >= 0 && tab->selected_cell < tab->cell_manager.GetCellCount()) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                if (!cell.outputs.empty()) {
                    cell.output_collapsed = !cell.output_collapsed;
                }
            }
            return;
        }
    }
}

// ============================================================================
// Debugger UI Functions
// ============================================================================

void ScriptEditorPanel::RenderDebugToolbar() {
    if (!debugger_) return;

    auto state = debugger_->GetState();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15f, 0.15f, 0.2f, 1.0f));
    ImGui::BeginChild("##debug_toolbar", ImVec2(0, 35), ImGuiChildFlags_Border);

    // Debug status indicator
    if (state == scripting::DebugState::Paused) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), ICON_FA_PAUSE " Paused at line %d", debug_current_line_);
    } else if (state == scripting::DebugState::Running) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), ICON_FA_PLAY " Running...");
    } else if (state == scripting::DebugState::Stepping) {
        ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), ICON_FA_FORWARD_STEP " Stepping...");
    } else {
        ImGui::TextDisabled(ICON_FA_BUG " Debugger Disconnected");
    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 280);

    // Continue/Pause button
    if (state == scripting::DebugState::Paused) {
        if (ImGui::Button(ICON_FA_PLAY " Continue")) {
            debugger_->Continue();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Continue execution (F5)");
        }
    } else if (state == scripting::DebugState::Running || state == scripting::DebugState::Stepping) {
        if (ImGui::Button(ICON_FA_PAUSE " Pause")) {
            debugger_->Pause();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Pause execution (F5)");
        }
    }

    ImGui::SameLine();

    // Step controls (only enabled when paused)
    ImGui::BeginDisabled(state != scripting::DebugState::Paused);

    if (ImGui::Button(ICON_FA_ARROW_DOWN " Over")) {
        debugger_->StepOver();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Over (F10)");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROW_RIGHT " Into")) {
        debugger_->StepInto();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Into (F11)");
    }

    ImGui::SameLine();

    if (ImGui::Button(ICON_FA_ARROW_UP " Out")) {
        debugger_->StepOut();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Step Out (Shift+F11)");
    }

    ImGui::EndDisabled();

    ImGui::SameLine();

    // Stop button
    if (ImGui::Button(ICON_FA_STOP " Stop")) {
        debugger_->Stop();
        debug_mode_active_ = false;
        debug_current_line_ = -1;
        debug_current_cell_.clear();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Stop debugging (Shift+F5)");
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void ScriptEditorPanel::RenderBreakpointGutter(Cell& cell, int /*cell_index*/) {
    if (cell.type != CellType::Code) return;

    int line_count = cell.editor.GetTotalLines();
    if (line_count == 0) line_count = 1;

    float line_height = ImGui::GetTextLineHeightWithSpacing();
    float gutter_width = 20.0f;

    ImGui::BeginChild("##bp_gutter", ImVec2(gutter_width, line_count * line_height), ImGuiChildFlags_None);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    for (int line = 0; line < line_count; ++line) {
        ImVec2 line_pos = ImVec2(cursor_start.x, cursor_start.y + line * line_height);
        ImVec2 center = ImVec2(line_pos.x + gutter_width * 0.5f, line_pos.y + line_height * 0.5f);
        float radius = 5.0f;

        // Check if this line has a breakpoint
        bool has_breakpoint = std::find(cell.breakpoints.begin(), cell.breakpoints.end(), line + 1) != cell.breakpoints.end();

        // Check if this is the current debug line
        bool is_current_line = debug_mode_active_ &&
                               debug_current_cell_ == cell.id &&
                               debug_current_line_ == line + 1;

        // Draw hover indicator
        ImGui::SetCursorScreenPos(line_pos);
        ImGui::InvisibleButton(("##bp_line_" + std::to_string(line)).c_str(), ImVec2(gutter_width, line_height));

        if (ImGui::IsItemHovered()) {
            draw_list->AddCircle(center, radius, IM_COL32(150, 150, 150, 150), 12, 1.0f);

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                // Toggle breakpoint
                if (has_breakpoint) {
                    cell.breakpoints.erase(
                        std::remove(cell.breakpoints.begin(), cell.breakpoints.end(), line + 1),
                        cell.breakpoints.end()
                    );

                    // Notify debugger
                    if (debugger_) {
                        auto breakpoints = debugger_->GetBreakpointsForCell(cell.id);
                        for (const auto& bp : breakpoints) {
                            if (bp.line == line + 1) {
                                debugger_->RemoveBreakpoint(bp.id);
                                break;
                            }
                        }
                    }
                } else {
                    cell.breakpoints.push_back(line + 1);

                    // Notify debugger
                    if (debugger_) {
                        debugger_->AddBreakpoint(cell.id, line + 1);
                    }
                }
            }
        }

        // Draw breakpoint indicator
        if (has_breakpoint) {
            draw_list->AddCircleFilled(center, radius, IM_COL32(200, 50, 50, 255), 12);
        }

        // Draw current line indicator (arrow)
        if (is_current_line) {
            ImVec2 arrow_points[3] = {
                ImVec2(center.x - 4, center.y - 4),
                ImVec2(center.x + 4, center.y),
                ImVec2(center.x - 4, center.y + 4)
            };
            draw_list->AddTriangleFilled(arrow_points[0], arrow_points[1], arrow_points[2],
                                         IM_COL32(255, 255, 0, 255));
        }
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::RenderScriptBreakpointGutter(float height) {
    if (tabs_.empty() || active_tab_index_ < 0) return;

    auto& tab = tabs_[active_tab_index_];
    int line_count = tab->editor.GetTotalLines();
    if (line_count == 0) line_count = 1;

    float line_height = ImGui::GetTextLineHeightWithSpacing();
    float gutter_width = 20.0f;

    // Get editor scroll position to sync gutter scrolling
    // Note: TextEditor doesn't expose scroll position directly, so we estimate
    auto coords = tab->editor.GetCursorPosition();

    ImGui::BeginChild("##script_bp_gutter", ImVec2(gutter_width, height), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 cursor_start = ImGui::GetCursorScreenPos();

    // Calculate visible lines based on height
    int visible_lines = static_cast<int>(height / line_height) + 1;
    int start_line = 0;  // Would need TextEditor scroll position for accurate sync

    for (int line = start_line; line < std::min(start_line + visible_lines + 5, line_count); ++line) {
        ImVec2 line_pos = ImVec2(cursor_start.x, cursor_start.y + (line - start_line) * line_height);
        ImVec2 center = ImVec2(line_pos.x + gutter_width * 0.5f, line_pos.y + line_height * 0.5f);
        float radius = 5.0f;

        int line_number = line + 1;  // 1-based line numbers

        // Check if this line has a breakpoint
        bool has_breakpoint = std::find(tab->breakpoints.begin(), tab->breakpoints.end(), line_number) != tab->breakpoints.end();

        // Check if this is the current debug line
        std::string file_id = tab->filepath.empty() ? tab->filename : tab->filepath;
        bool is_current_line = debug_mode_active_ &&
                               debug_current_cell_ == file_id &&
                               debug_current_line_ == line_number;

        // Draw hover indicator and handle clicks
        ImGui::SetCursorScreenPos(line_pos);
        ImGui::InvisibleButton(("##script_bp_line_" + std::to_string(line)).c_str(), ImVec2(gutter_width, line_height));

        if (ImGui::IsItemHovered()) {
            draw_list->AddCircle(center, radius, IM_COL32(150, 150, 150, 150), 12, 1.0f);

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                // Toggle breakpoint
                if (has_breakpoint) {
                    tab->breakpoints.erase(
                        std::remove(tab->breakpoints.begin(), tab->breakpoints.end(), line_number),
                        tab->breakpoints.end()
                    );

                    // Notify debugger
                    if (debugger_) {
                        auto breakpoints = debugger_->GetBreakpointsForCell(file_id);
                        for (const auto& bp : breakpoints) {
                            if (bp.line == line_number) {
                                debugger_->RemoveBreakpoint(bp.id);
                                break;
                            }
                        }
                    }
                } else {
                    tab->breakpoints.push_back(line_number);

                    // Notify debugger
                    if (debugger_) {
                        debugger_->AddBreakpoint(file_id, line_number);
                    }
                }
            }
        }

        // Draw breakpoint indicator
        if (has_breakpoint) {
            draw_list->AddCircleFilled(center, radius, IM_COL32(200, 50, 50, 255), 12);
        }

        // Draw current line indicator (arrow)
        if (is_current_line) {
            ImVec2 arrow_points[3] = {
                ImVec2(center.x - 4, center.y - 4),
                ImVec2(center.x + 4, center.y),
                ImVec2(center.x - 4, center.y + 4)
            };
            draw_list->AddTriangleFilled(arrow_points[0], arrow_points[1], arrow_points[2],
                                         IM_COL32(255, 255, 0, 255));
        }
    }

    ImGui::EndChild();
}

void ScriptEditorPanel::HandleDebugKeyboardShortcuts() {
    if (!debugger_) return;

    auto state = debugger_->GetState();

    // F5 - Continue / Start Debug
    if (ImGui::IsKeyPressed(ImGuiKey_F5)) {
        if (ImGui::GetIO().KeyShift) {
            // Shift+F5 - Stop debugging
            debugger_->Stop();
            debug_mode_active_ = false;
            debug_current_line_ = -1;
            debug_current_cell_.clear();
        } else if (state == scripting::DebugState::Paused) {
            debugger_->Continue();
        } else if (state == scripting::DebugState::Disconnected) {
            Debug(); // Start debugging
        }
    }

    // F10 - Step Over
    if (ImGui::IsKeyPressed(ImGuiKey_F10)) {
        if (state == scripting::DebugState::Paused) {
            debugger_->StepOver();
        }
    }

    // F11 - Step Into / Shift+F11 - Step Out
    if (ImGui::IsKeyPressed(ImGuiKey_F11)) {
        if (state == scripting::DebugState::Paused) {
            if (ImGui::GetIO().KeyShift) {
                debugger_->StepOut();
            } else {
                debugger_->StepInto();
            }
        }
    }

    // F9 - Toggle breakpoint at current line
    if (ImGui::IsKeyPressed(ImGuiKey_F9)) {
        if (tabs_.empty() || active_tab_index_ < 0) return;

        auto& tab = tabs_[active_tab_index_];

        if (tab->cell_mode) {
            // Cell mode - toggle breakpoint in selected cell
            if (tab->selected_cell >= 0 &&
                tab->selected_cell < static_cast<int>(tab->cell_manager.GetCellCount())) {
                Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
                if (cell.type == CellType::Code) {
                    // Get current cursor line from editor
                    auto coords = cell.editor.GetCursorPosition();
                    int line = coords.mLine + 1; // 1-based

                    // Toggle breakpoint
                    auto it = std::find(cell.breakpoints.begin(), cell.breakpoints.end(), line);
                    if (it != cell.breakpoints.end()) {
                        cell.breakpoints.erase(it);
                        if (debugger_) {
                            auto breakpoints = debugger_->GetBreakpointsForCell(cell.id);
                            for (const auto& bp : breakpoints) {
                                if (bp.line == line) {
                                    debugger_->RemoveBreakpoint(bp.id);
                                    break;
                                }
                            }
                        }
                    } else {
                        cell.breakpoints.push_back(line);
                        if (debugger_) {
                            debugger_->AddBreakpoint(cell.id, line);
                        }
                    }
                }
            }
        } else {
            // Traditional mode - toggle breakpoint in script
            auto coords = tab->editor.GetCursorPosition();
            int line = coords.mLine + 1; // 1-based

            std::string file_id = tab->filepath.empty() ? tab->filename : tab->filepath;

            // Toggle breakpoint
            auto it = std::find(tab->breakpoints.begin(), tab->breakpoints.end(), line);
            if (it != tab->breakpoints.end()) {
                tab->breakpoints.erase(it);
                if (debugger_) {
                    auto breakpoints = debugger_->GetBreakpointsForCell(file_id);
                    for (const auto& bp : breakpoints) {
                        if (bp.line == line) {
                            debugger_->RemoveBreakpoint(bp.id);
                            break;
                        }
                    }
                }
            } else {
                tab->breakpoints.push_back(line);
                if (debugger_) {
                    debugger_->AddBreakpoint(file_id, line);
                }
            }
        }
    }
}

// ============================================================================
// Auto-Completion Implementation
// ============================================================================

void ScriptEditorPanel::UpdateAutoCompletion(bool force) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        CloseCompletionPopup();
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    if (tab->cell_mode || tab->is_loading) {
        CloseCompletionPopup();
        return;
    }

    // Get current cursor position and line
    auto cursor_pos = tab->editor.GetCursorPosition();
    std::string current_line = tab->editor.GetCurrentLineText();
    int col = cursor_pos.mColumn;

    // Check if we should show completions (allow empty line/col=0 for force mode)
    if (!force && (col <= 0 || current_line.empty())) {
        CloseCompletionPopup();
        return;
    }

    // Get the character just typed
    char last_char = (col > 0 && col <= static_cast<int>(current_line.length()))
                     ? current_line[col - 1] : '\0';

    // Check if we should trigger completion (skip check if forced via Ctrl+Space)
    if (!force && !script_manager_.ShouldTriggerCompletion(last_char)) {
        // Only close if we're not in an identifier
        std::string word = scripting::ScriptManager::GetWordAtCursor(current_line, col);
        if (word.empty() && !show_completion_popup_) {
            return;
        }
        if (word.empty() && show_completion_popup_) {
            CloseCompletionPopup();
            return;
        }
    }

    // Get completions
    std::string code = tab->editor.GetText();
    size_t cursor_offset = 0;
    auto lines = tab->editor.GetTextLines();
    for (int i = 0; i < cursor_pos.mLine && i < static_cast<int>(lines.size()); ++i) {
        cursor_offset += lines[i].length() + 1; // +1 for newline
    }
    cursor_offset += col;

    completion_items_ = script_manager_.GetCompletions(code, cursor_offset, current_line, col);

    if (completion_items_.empty()) {
        CloseCompletionPopup();
        return;
    }

    // Get prefix and start position
    completion_prefix_ = scripting::ScriptManager::GetWordAtCursor(current_line, col);
    completion_start_pos_ = cursor_pos;
    completion_start_pos_.mColumn = col - static_cast<int>(completion_prefix_.length());

    show_completion_popup_ = true;
    completion_just_opened_ = true;  // Prevent immediate close from Ctrl+Space inserting space
    selected_completion_ = 0;
}

void ScriptEditorPanel::RenderCompletionPopup() {
    if (!show_completion_popup_ || completion_items_.empty()) {
        return;
    }

    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // NO keyboard handling here - it interferes with the text editor!
    // Keyboard shortcuts are handled in HandleKeyboardShortcuts() instead

    // Get the window position where the editor is rendered
    ImVec2 window_pos = ImGui::GetWindowPos();
    ImVec2 content_region_min = ImGui::GetWindowContentRegionMin();

    // Calculate popup position based on cursor position in editor
    auto cursor_pos = tab->editor.GetCursorPosition();

    // Estimate character dimensions using monospace font
    float char_width = ImGui::CalcTextSize("M").x * font_scale_;
    float line_height = ImGui::GetTextLineHeightWithSpacing() * font_scale_;

    // Account for editor offset (gutter, margins, etc.)
    float editor_left_offset = 45.0f;  // Approximate gutter + padding
    float editor_top_offset = 80.0f;   // Approximate tab bar + menu bar height

    // Position popup below the cursor
    float popup_x = window_pos.x + content_region_min.x + editor_left_offset +
                    (completion_start_pos_.mColumn * char_width);
    float popup_y = window_pos.y + content_region_min.y + editor_top_offset +
                    ((cursor_pos.mLine + 1) * line_height);

    // Clamp to screen bounds
    ImVec2 display_size = ImGui::GetIO().DisplaySize;
    popup_x = std::min(popup_x, display_size.x - 320.0f);
    popup_y = std::min(popup_y, display_size.y - 250.0f);

    ImGui::SetNextWindowPos(ImVec2(popup_x, popup_y), ImGuiCond_Always);

    // Popup flags - NO focus stealing!
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(6, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 2));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.15f, 0.15f, 0.18f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.4f, 0.4f, 0.5f, 0.8f));

    if (ImGui::Begin("##completion_popup", nullptr, flags)) {
        // Header with hint
        ImGui::TextDisabled("Tab: insert | Esc: close | Ctrl+Space: trigger");
        ImGui::Separator();

        // Render completion list
        for (int i = 0; i < static_cast<int>(completion_items_.size()) && i < 10; ++i) {
            const auto& item = completion_items_[i];
            bool is_selected = (i == selected_completion_);

            // Kind icon
            const char* icon = scripting::GetCompletionKindIcon(item.kind);

            ImGui::PushID(i);

            // Highlight selected item
            if (is_selected) {
                ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.8f, 0.7f));
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.3f, 0.5f, 0.8f, 0.8f));
            }

            if (ImGui::Selectable("##item", is_selected, ImGuiSelectableFlags_None, ImVec2(280, 0))) {
                ApplyCompletion(item);
                CloseCompletionPopup();
            }

            if (is_selected) {
                ImGui::PopStyleColor(2);
            }

            ImGui::SameLine(0, 0);
            ImGui::SetCursorPosX(8);

            // Icon with color based on kind
            ImVec4 kind_color;
            switch (item.kind) {
                case scripting::CompletionItem::Kind::Keyword:  kind_color = ImVec4(0.8f, 0.4f, 0.8f, 1.0f); break;
                case scripting::CompletionItem::Kind::Builtin:  kind_color = ImVec4(0.4f, 0.8f, 0.8f, 1.0f); break;
                case scripting::CompletionItem::Kind::Module:   kind_color = ImVec4(0.8f, 0.6f, 0.2f, 1.0f); break;
                case scripting::CompletionItem::Kind::Function: kind_color = ImVec4(0.4f, 0.7f, 1.0f, 1.0f); break;
                default: kind_color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f); break;
            }
            ImGui::TextColored(kind_color, "[%s]", icon);
            ImGui::SameLine();

            // Label
            ImGui::Text("%s", item.label.c_str());

            // Detail (if any)
            if (!item.detail.empty() && item.detail != "keyword" && item.detail != "builtin") {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", item.detail.c_str());
            }

            ImGui::PopID();
        }

        // Show more items indicator
        if (completion_items_.size() > 10) {
            ImGui::Separator();
            ImGui::TextDisabled("... and %zu more", completion_items_.size() - 10);
        }
    }
    ImGui::End();

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
}

void ScriptEditorPanel::ApplyCompletion(const scripting::CompletionItem& item) {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];

    // Get the text to insert
    std::string text_to_insert = item.insert_text.empty() ? item.label : item.insert_text;

    // Select the prefix text (from completion_start_pos_ to current cursor)
    auto cursor_pos = tab->editor.GetCursorPosition();
    tab->editor.SetSelection(completion_start_pos_, cursor_pos);

    // Delete the selected prefix, then insert completion
    if (tab->editor.HasSelection()) {
        tab->editor.Delete();  // Deletes selected text
    }
    tab->editor.InsertText(text_to_insert);
}

void ScriptEditorPanel::CloseCompletionPopup() {
    show_completion_popup_ = false;
    completion_items_.clear();
    selected_completion_ = 0;
}

} // namespace cyxwiz
