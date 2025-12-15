#include "script_editor.h"
#include "command_window.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../../scripting/scripting_engine.h"
#include "../../scripting/debugger.h"
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

    ImGui::PopStyleColor(4);

    // Reset font scale
    if (font_scale_ != 1.0f) {
        ImGui::SetWindowFontScale(1.0f);
    }

    // Track modifications
    if (tab->editor.IsTextChanged()) {
        tab->is_modified = true;
    }
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
    // Handle debug shortcuts (F5, F9, F10, F11)
    HandleDebugKeyboardShortcuts();

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
        cell.editor.Render("##code", ImVec2(code_width, content_height));

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

    if (is_editing) {
        // Edit mode - simple text input
        ImGui::PushItemWidth(-1);

        // Use TextEditor for markdown editing too
        // Only sync editor from source when ENTERING edit mode, not every frame
        if (tab->last_editing_cell != index) {
            cell.SyncEditorFromSource();
            tab->last_editing_cell = index;
        }

        int line_count = cell.editor.GetTotalLines();
        float line_height = ImGui::GetTextLineHeightWithSpacing();
        float content_height = std::max(60.0f, (line_count + 1) * line_height);
        content_height = std::min(content_height, 300.0f);

        cell.editor.Render("##markdown_edit", ImVec2(-1, content_height));
        cell.SyncSourceFromEditor();

        if (cell.editor.IsTextChanged()) {
            tab->is_modified = true;
        }

        ImGui::PopItemWidth();
    } else {
        // View mode - render markdown
        OutputRenderer::RenderMarkdown(cell.source);
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

void ScriptEditorPanel::RenderBreakpointGutter(Cell& cell, int cell_index) {
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

} // namespace cyxwiz
