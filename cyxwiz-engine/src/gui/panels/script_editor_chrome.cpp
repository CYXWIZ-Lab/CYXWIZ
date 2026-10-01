// Script Editor chrome around the code (TOFIX133 P2, approved boards 1 and
// 3): file tabs with the unsaved dot and the run actions, breadcrumbs
// (folder > file > class > function), and the status bar. No outlines; a
// narrow editor keeps every item, folding the rest behind "···".

#include "script_editor.h"

#include "../../core/editor/outline.h"
#include "../../core/notebook_presentation.h"
#include "../../core/project_manager.h"
#include "../../scripting/scripting_engine.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <chrono>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {
constexpr float kNarrow = 620.0f;  // below this the editor uses its narrow layout (board 3)

const char* FileIcon(const std::string& name) {
    const std::string ext = std::filesystem::path(name).extension().string();
    if (ext == ".cyx" || ext == ".ipynb") return ICON_FA_LAYER_GROUP;
    if (ext == ".py") return ICON_FA_FILE_CODE;
    return ICON_FA_FILE_LINES;
}

// A status bar item that can be clicked: text with a hover tint, no outline.
bool StatusItem(const char* id, const std::string& text, bool clickable, const char* tooltip = nullptr) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, clickable ? t.hover : ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, clickable ? t.hover : ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::PushID(id);
    const bool clicked = ImGui::Selectable(text.c_str(), false, ImGuiSelectableFlags_None,
                                           ImVec2(ImGui::CalcTextSize(text.c_str()).x, 0.0f));
    ImGui::PopID();
    ImGui::PopStyleColor(4);
    if (tooltip && ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", tooltip);
    return clicked && clickable;
}
}  // namespace

void ScriptEditorPanel::RenderTabBar() {
    const ui::Tokens& t = ui::CurrentTokens();
    const float avail = ImGui::GetContentRegionAvail().x;
    const bool narrow = avail < kNarrow;
    const bool has_tab = IsActiveTabEditable();
    const bool text_mode = IsActiveTabTextMode();
    const bool engine_busy = scripting_engine_ && scripting_engine_->IsScriptRunning();
    const bool notebook = has_tab && tabs_[active_tab_index_]->cell_mode;

    // Run actions at the right; the tabs take the rest (scroll arrows when crowded).
    const float gap = t.space_sm;
    float actions_w = ui::ButtonWidth(ICON_FA_PLAY "  Run", ui::ButtonSize::Small);
    if (notebook) actions_w = ui::ButtonWidth("Debug cell", ui::ButtonSize::Small);
    else if (narrow) actions_w += gap + ui::ButtonWidth("\xC2\xB7\xC2\xB7\xC2\xB7", ui::ButtonSize::Small);
    else actions_w += 2.0f * gap + ui::ButtonWidth("Run selection", ui::ButtonSize::Small) + ui::ButtonWidth("Debug", ui::ButtonSize::Small);

    ImGui::BeginChild("##tab_strip", ImVec2(std::max(80.0f, avail - actions_w - t.space_md), ImGui::GetFrameHeight() + 2.0f),
                      ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (ImGui::BeginTabBar("ScriptEditorTabs", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_AutoSelectNewTabs |
                                                   ImGuiTabBarFlags_FittingPolicyScroll)) {
        for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
            auto& tab = tabs_[i];
            // The unsaved dot (it turns into the close button on hover) replaces the old "*".
            const std::string label = std::string(tab->is_loading ? ICON_FA_SPINNER : FileIcon(tab->filename)) + "  " +
                                      tab->filename + "###tab" + std::to_string(tab->document_id);
            ImGuiTabItemFlags flags = ImGuiTabItemFlags_None;
            if (tab->is_modified) flags |= ImGuiTabItemFlags_UnsavedDocument;
            if (request_focus_ && i == active_tab_index_) flags |= ImGuiTabItemFlags_SetSelected;
            bool open = true;
            if (ImGui::BeginTabItem(label.c_str(), &open, flags)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
                ImGui::SetTooltip("%s", tab->filepath.empty() ? "Not saved yet" : tab->filepath.c_str());
            }
            if (!open) close_tab_index_ = i;  // closing a loading tab cancels its work
        }
        if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing | ImGuiTabItemFlags_NoTooltip)) NewFile();
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::SameLine(0.0f, t.space_md);
    const char* why = !has_tab ? "Open or write a script first" : "Another script is running";
    if (notebook) {
        // The notebook toolbar runs cells (board 4); here only the debugger.
        const bool has_cell = tabs_[active_tab_index_]->selected_cell >= 0;
        if (ui::SecondaryButton("Debug cell", has_cell && !engine_busy, has_cell ? why : "Select a code cell first")) Debug();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && has_cell) ImGui::SetTooltip("Debug the selected cell (F10)");
        return;
    }
    if (script_running_) {
        if (ui::DangerButton(ICON_FA_STOP "  Stop", true, nullptr, ui::ButtonSize::Small)) StopScript();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Stop the script (Shift+F5)");
    } else {
        if (ui::PrimaryButton(ICON_FA_PLAY "  Run", has_tab && !engine_busy, why, ui::ButtonSize::Small)) RunScript();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && has_tab && !engine_busy) ImGui::SetTooltip("Run the script (F5)");
    }
    if (!narrow) {
        ImGui::SameLine(0.0f, gap);
        if (ui::SecondaryButton("Run selection", text_mode && !engine_busy,
                                text_mode ? "Another script is running" : "Not in notebook mode")) RunSelection();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && text_mode) ImGui::SetTooltip("Run the selected lines (F9)");
        ImGui::SameLine(0.0f, gap);
        if (ui::SecondaryButton("Debug", has_tab && !engine_busy, why)) Debug();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && has_tab) ImGui::SetTooltip("Debug the script (F10)");
    } else {
        ImGui::SameLine(0.0f, gap);
        if (ui::SecondaryButton("\xC2\xB7\xC2\xB7\xC2\xB7", true, nullptr)) ImGui::OpenPopup("##run_more");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("More run actions");
        if (ImGui::BeginPopup("##run_more")) {
            if (ImGui::MenuItem("Run selection", "F9", false, text_mode && !engine_busy)) RunSelection();
            if (ImGui::MenuItem("Run section", "Ctrl+Enter", false, text_mode && !engine_busy)) RunCurrentSection();
            if (ImGui::MenuItem("Debug", "F10", false, has_tab && !engine_busy)) Debug();
            ImGui::EndPopup();
        }
    }
}

void ScriptEditorPanel::RenderBreadcrumbs(EditorTab& tab) {
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    const bool narrow = width < kNarrow;
    const auto scopes = editor::EnclosingScopes(tab.editor.Doc(), tab.editor.Doc().Primary().head.line);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + t.space_md);
    auto sep = [&]() {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "\xE2\x80\xBA");  // ›
        ImGui::SameLine();
    };
    bool first = true;
    if (!narrow) {
        const std::filesystem::path path(tab.filepath);
        const std::string folder = tab.filepath.empty() ? std::string("Not saved") : path.parent_path().filename().string();
        ImGui::TextUnformatted(folder.c_str());
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && !tab.filepath.empty())
            ImGui::SetTooltip("%s", path.parent_path().string().c_str());
        sep();
        ImGui::TextUnformatted(tab.filename.c_str());
        first = false;
    } else {
        ImGui::TextUnformatted("\xE2\x80\xA6");  // …
        first = false;
    }
    // In a narrow editor only the innermost scope stays.
    const size_t from = narrow && scopes.size() > 1 ? scopes.size() - 1 : 0;
    for (size_t i = from; i < scopes.size(); ++i) {
        if (!first) sep();
        const auto& s = scopes[i];
        const std::string label = s.kind == "def" ? s.name + "()" : s.name;
        ImGui::PushStyleColor(ImGuiCol_Text, i + 1 == scopes.size() ? t.text : t.text_dim);
        ImGui::PushID(static_cast<int>(i));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_None, ImVec2(ImGui::CalcTextSize(label.c_str()).x, 0.0f)))
            tab.editor.GoToLine(s.line);
        ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Go to line %d", s.line + 1);
        ImGui::PopID();
        ImGui::PopStyleColor();
        first = false;
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}

void ScriptEditorPanel::RenderStatusBar() {
    if (active_tab_index_ < 0 || active_tab_index_ >= static_cast<int>(tabs_.size())) return;
    auto& tab = tabs_[active_tab_index_];
    const ui::Tokens& t = ui::CurrentTokens();
    const float width = ImGui::GetContentRegionAvail().x;
    const bool narrow = width < kNarrow;

    // The bar is a tone, not a line.
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight();
    ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + width, p.y + h), ui::ToU32(t.bg_bar));
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + t.space_md);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(16.0f, 0.0f));

    // State: a dot and a word.
    ImVec4 dot = t.success;
    std::string state = "Ready";
    if (tab->is_loading) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Loading %.0f%%", tab->load_progress * 100.0f);
        state = buf;
        dot = t.running;
    } else if (script_running_) {
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - running_script_started_).count();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "Running %s  %.1f s", running_script_name_.c_str(), seconds);
        state = buf;
        dot = t.running;
    } else if (tab->load_failed) {
        state = "Not opened";
        dot = t.error;
    } else if (tab->cell_mode) {
        const CellManager& cells = tab->cell_manager;
        nbview::RunFacts facts;
        facts.running = cells.IsRunning();
        facts.run_position = cells.BatchPosition();
        facts.run_total = cells.BatchTotal();
        facts.stopped_at_count = cells.StoppedAtCount();
        facts.stopped_by_interrupt = cells.StoppedByInterrupt();
        facts.restarting = cells.IsRestarting();
        const nbview::RunStatus status = nbview::RunStatusFor(facts);
        state = status.text;
        dot = NotebookToneColour(static_cast<int>(status.tone));
    }
    {
        const ImVec2 c = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(c.x + 4.0f, c.y + h * 0.5f), 3.5f, ui::ToU32(dot));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 12.0f);
        ImGui::TextColored(t.text_dim, "%s", state.c_str());
    }
    ImGui::SameLine();
    ImGui::TextColored(t.text_dim, "%s", tab->is_modified ? "Modified" : "Saved");

    std::string sandbox = scripting_engine_ && scripting_engine_->IsSandboxEnabled() ? "Sandbox on" : "Sandbox off";
    if (tab->is_large_file) {
        const auto first = tab->large_page.lines.empty() ? 0ULL : static_cast<unsigned long long>(tab->large_page.first_line + 1);
        const auto last = static_cast<unsigned long long>(tab->large_page.first_line + tab->large_page.lines.size());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "Large file, read only  Lines %llu-%llu of %llu", first, last,
                           static_cast<unsigned long long>(tab->large_index.line_count));
    } else if (tab->cell_mode) {
        const int cells = static_cast<int>(tab->cell_manager.GetCellCount());
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "Notebook \xC2\xB7 Cell %d of %d", tab->selected_cell >= 0 ? tab->selected_cell + 1 : 0, cells);
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", tab->editing_cell >= 0 ? "Edit mode" : "Command mode");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", nbview::NotebookKeysHint());
    } else {
        const editor::Document& doc = tab->editor.Doc();
        const editor::Pos head = doc.Primary().head;
        const int column = editor::VisualColumn(doc.Line(head.line), head.col, doc.Settings().tab_size) + 1;
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "Ln %d, Col %d", head.line + 1, column);
        if (doc.CursorCount() > 1) {
            ImGui::SameLine();
            ImGui::TextColored(t.text, "%d cursors", doc.CursorCount());
        }
        if (!narrow) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%d lines", doc.LineCount());
        }
    }

    // Right side: indentation, encoding, line endings, Python, sandbox.
    RefreshPythonStatus();
    const bool text_file = !tab->is_large_file && !tab->cell_mode;
    char spaces[32];
    std::snprintf(spaces, sizeof(spaces), "Spaces: %d", tab_size_);
    const std::string encoding = tab->format.bom ? "UTF-8 with BOM" : "UTF-8";
    const std::string eol = tab->format.eol == scriptfile::LineEnding::CRLF ? "CRLF" : "LF";
    std::vector<std::string> right;
    if (!narrow) {
        if (text_file) right = {spaces, encoding, eol};
        right.push_back(python_status_);
        right.push_back(sandbox);
    } else {
        right = {python_status_, "\xC2\xB7\xC2\xB7\xC2\xB7"};
    }
    float right_w = 0.0f;
    for (const auto& s : right) right_w += ImGui::CalcTextSize(s.c_str()).x + 16.0f;
    ImGui::SameLine();
    // Nothing may be pushed out of sight: when even the narrow set does not
    // fit, only "···" stays and the Python line moves into its menu.
    bool python_in_menu = false;
    if (narrow && ImGui::GetCursorPosX() + right_w > ImGui::GetWindowContentRegionMax().x) {
        python_in_menu = true;
        right_w = ImGui::CalcTextSize("\xC2\xB7\xC2\xB7\xC2\xB7").x + 16.0f;
    }
    const float x = ImGui::GetWindowContentRegionMax().x - right_w;
    if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);

    auto spaces_menu = [&]() {
        for (int n : {2, 4, 8}) {
            char label[24];
            std::snprintf(label, sizeof(label), "%d spaces", n);
            if (ImGui::MenuItem(label, nullptr, tab_size_ == n)) {
                SetTabSize(n);
                if (on_settings_changed_callback_) on_settings_changed_callback_();
            }
        }
    };
    auto set_eol = [&](scriptfile::LineEnding e) {
        if (tab->format.eol != e) {
            tab->format.eol = e;
            tab->format_changed = true;  // the file is written with the new endings on save
        }
    };
    bool first_item = true;
    auto next = [&]() {
        if (!first_item) ImGui::SameLine();
        first_item = false;
    };
    if (!narrow) {
        if (text_file) {
            next();
            if (StatusItem("spaces", spaces, true, "Indentation: spaces per level")) ImGui::OpenPopup("##status_spaces");
            next();
            if (StatusItem("encoding", encoding, true, "Click to save with or without a byte-order mark")) {
                tab->format.bom = !tab->format.bom;
                tab->format_changed = true;
            }
            next();
            if (StatusItem("eol", eol, true, "Line endings: click to switch between LF and CRLF"))
                set_eol(tab->format.eol == scriptfile::LineEnding::CRLF ? scriptfile::LineEnding::LF : scriptfile::LineEnding::CRLF);
        }
        next();
        StatusItem("python", python_status_, false, python_tooltip_.c_str());
        next();
        StatusItem("sandbox", sandbox, false, "Security > Enable Sandbox");
    } else {
        if (!python_in_menu) {
            next();
            StatusItem("python", python_status_, false, python_tooltip_.c_str());
        }
        next();
        if (StatusItem("more", "\xC2\xB7\xC2\xB7\xC2\xB7", true, "More: lines, indentation, encoding, line endings, sandbox"))
            ImGui::OpenPopup("##status_more");
    }
    if (ImGui::BeginPopup("##status_spaces")) {
        spaces_menu();
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("##status_more")) {
        if (text_file) {
            ImGui::TextDisabled("%d lines", tab->editor.Doc().LineCount());
            if (ImGui::BeginMenu(spaces)) {
                spaces_menu();
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("Byte-order mark", nullptr, tab->format.bom)) {
                tab->format.bom = !tab->format.bom;
                tab->format_changed = true;
            }
            if (ImGui::MenuItem("CRLF line endings", nullptr, tab->format.eol == scriptfile::LineEnding::CRLF))
                set_eol(tab->format.eol == scriptfile::LineEnding::CRLF ? scriptfile::LineEnding::LF : scriptfile::LineEnding::CRLF);
        }
        ImGui::TextDisabled("%s", sandbox.c_str());
        if (python_in_menu) ImGui::TextDisabled("%s", python_status_.c_str());
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
}

void ScriptEditorPanel::RefreshPythonStatus() {
    const double now = ImGui::GetTime();
    if (now - python_status_time_ < 2.0 && !python_status_.empty()) return;
    python_status_time_ = now;
    if (!scripting_engine_) {
        python_status_ = "Python unavailable";
        python_tooltip_ = "This build has no Python scripting.";
        return;
    }
    const auto info = scripting_engine_->GetInterpreterInfo();
    python_started_ = info.initialized;
    python_version_ = info.version;
    python_environment_ = info.source == "system" ? "system Python"
                          : ProjectManager::Instance().HasActiveProject() ? ProjectManager::Instance().GetProjectName()
                                                                          : "project environment";
    if (info.initialized) {
        const std::string where = info.source == "project" ? "project environment" : "system Python";
        python_status_ = "Python " + (info.version.empty() ? std::string("") : info.version + " ") + "\xC2\xB7 " + where;
        python_tooltip_ = info.interpreter_path + (info.mismatch.empty() ? "" : "\nRestart needed: " + info.mismatch);
    } else {
        python_status_ = "Python \xC2\xB7 starts on first run";
        python_tooltip_ = info.interpreter_path.empty() ? std::string("No interpreter configured.") : info.interpreter_path;
    }
}

}  // namespace cyxwiz
