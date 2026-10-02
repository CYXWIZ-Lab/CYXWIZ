// Script Editor debugger UI (TOFIX133 P6, approved boards 11-12). The run
// is debugged on the script worker by python_tools/cyxwiz_debug.py; this
// file starts it, draws its toolbar, sends the commands and follows the
// paused line.

#include "script_editor.h"

#include "../../scripting/script_output_sink.h"
#include "../../scripting/scripting_engine.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {
// "%%" section markers are not Python: blank them, keeping the line numbers
// the breakpoints and the paused line refer to.
std::string WithoutSectionMarkers(const std::string& text) {
    std::string out;
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(start, end - start);
        const size_t a = line.find_first_not_of(" \t\r");
        const size_t b = line.find_last_not_of(" \t\r");
        if (a != std::string::npos && line.substr(a, b - a + 1) == "%%") line.clear();
        out += line;
        if (end == text.size()) break;
        out += '\n';
        start = end + 1;
    }
    return out;
}
}  // namespace

std::vector<scripting::DebugBreakpoint> ScriptEditorPanel::DebugBreakpointsFor(const EditorTab& tab,
                                                                                                const Cell* cell) const {
    std::vector<scripting::DebugBreakpoint> out;
    for (int line : cell ? cell->breakpoints : tab.breakpoints) {
        scripting::DebugBreakpoint bp;
        bp.line = line;
        out.push_back(bp);
    }
    return out;
}

void ScriptEditorPanel::SyncDebugBreakpoints() {
    if (!debug_run_.active || !scripting_engine_) return;
    const int index = FindTabIndex(debug_run_.document_id);
    if (index < 0) return;
    const EditorTab& tab = *tabs_[index];
    const Cell* cell = nullptr;
    if (!debug_run_.cell_id.empty())
        for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i)
            if (tab.cell_manager.GetCell(i).id == debug_run_.cell_id) cell = &tab.cell_manager.GetCell(i);
    scripting_engine_->DebugSetBreakpoints(DebugBreakpointsFor(tab, cell));
}

CodeEditor* ScriptEditorPanel::DebugEditorFor(const std::string& file, EditorTab** tab_out) {
    if (!debug_run_.active || file.empty() || file != debug_.file) return nullptr;  // other files are not debugged
    const int index = FindTabIndex(debug_run_.document_id);
    if (index < 0) return nullptr;
    EditorTab& tab = *tabs_[index];
    if (tab_out) *tab_out = &tab;
    if (debug_run_.cell_id.empty()) return tab.cell_mode ? nullptr : &tab.editor;
    for (int i = 0; i < tab.cell_manager.GetCellCount(); ++i) {
        Cell& cell = tab.cell_manager.GetCell(i);
        if (cell.id == debug_run_.cell_id) return &cell.editor;
    }
    return nullptr;
}

// Once per frame: the engine's snapshot, the paused line, the end of the run.
void ScriptEditorPanel::UpdateDebugState() {
    if (!scripting_engine_) return;
    debug_ = scripting_engine_->GetDebugSnapshot();
    if (!debug_run_.active) return;
    if (debug_.version != debug_seen_version_) {
        debug_seen_version_ = debug_.version;
        if (debug_.state == "paused") {
            debug_frame_ = 0;
            // Bring the paused line into view (and its tab or cell).
            if (!debug_.stack.empty()) {
                EditorTab* tab = nullptr;
                if (CodeEditor* code = DebugEditorFor(debug_.stack.front().file, &tab)) {
                    const int index = FindTabIndex(tab->document_id);
                    if (index >= 0 && index != active_tab_index_) active_tab_index_ = index;
                    code->GoToLine(std::max(0, debug_.stack.front().line - 1));
                    if (!debug_run_.cell_id.empty())
                        for (int i = 0; i < tab->cell_manager.GetCellCount(); ++i)
                            if (tab->cell_manager.GetCell(i).id == debug_run_.cell_id) tab->scroll_to_cell = i;
                }
            }
        }
    }
    // The run ended (finished, stopped, failed): leave debugging.
    const bool cell_run = !debug_run_.cell_id.empty();
    if (debug_.state == "idle" && !scripting_engine_->IsScriptRunning() && (cell_run || !script_running_)) {
        debug_run_ = DebugRun{};
        debug_frame_ = 0;
    }
}

void ScriptEditorPanel::DebugCommand(const char* command) {
    if (scripting_engine_ && scripting_engine_->IsDebugPaused()) scripting_engine_->DebugCommand(command);
}

void ScriptEditorPanel::StopDebugging() {
    if (!scripting_engine_ || !debug_run_.active) return;
    scripting_engine_->StopScript();  // a paused run ends at once, a running one at its next line
    spdlog::info("Debugging stopped");
}

// Board 11: state, Continue (or Pause while running), steps, Stop.
void ScriptEditorPanel::RenderDebugToolbar() {
    if (!debug_run_.active) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const bool paused = debug_.state == "paused";
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight() + 10.0f;
    ImGui::Dummy(ImVec2(0.0f, 5.0f));
    ImGui::SetCursorScreenPos(ImVec2(p.x + 8.0f, p.y + 5.0f));

    // State pill.
    std::string state = "Running...";
    ImVec4 dot = t.running;
    if (paused) {
        const auto& top = debug_.stack.empty() ? scripting::DebugFrame{} : debug_.stack.front();
        const char* why = debug_.reason == "breakpoint" ? "Paused at a breakpoint"
                          : debug_.reason == "error"    ? "Stopped on an error"
                          : debug_.reason == "pause"    ? "Paused"
                                                        : "Paused after a step";
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s \xC2\xB7 %s, line %d", why, top.name.c_str(), top.line);
        state = buf;
        dot = debug_.reason == "error" ? t.error : t.warning;
    }
    ui::StatusPill("##debug_state", state.c_str(), dot);
    ImGui::SameLine(0.0f, 14.0f);

    if (paused) {
        if (ui::PrimaryButton(ICON_FA_PLAY " Continue", true, nullptr, ui::ButtonSize::Small)) DebugCommand("continue");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Continue (F5)");
    } else {
        if (ui::SecondaryButton(ICON_FA_PAUSE " Pause", true, nullptr, ui::ButtonSize::Small) && scripting_engine_)
            scripting_engine_->DebugPause();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Pause at the next line");
    }
    struct Step {
        const char* label;
        const char* command;
        const char* key;
    };
    static constexpr Step kSteps[] = {{ICON_FA_ROTATE_RIGHT " Step over", "over", "F10"},
                                      {ICON_FA_ARROW_DOWN " Step into", "into", "F11"},
                                      {ICON_FA_ARROW_UP " Step out", "out", "Shift+F11"}};
    for (const auto& s : kSteps) {
        ImGui::SameLine(0.0f, 6.0f);
        if (ui::GhostButton(s.label, paused, "Only while paused")) DebugCommand(s.command);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s (%s)", s.label + 4, s.key);
    }
    const float stop_w = ui::ButtonWidth(ICON_FA_STOP " Stop", ui::ButtonSize::Small);
    ui::SameLineRight(stop_w + 8.0f);
    if (ui::DangerButton(ICON_FA_STOP " Stop", true, nullptr, ui::ButtonSize::Small)) StopDebugging();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Stop debugging (Shift+F5)");
    if (paused && !debug_.error.empty()) {
        ImGui::SetCursorScreenPos(ImVec2(p.x + 8.0f, p.y + h));
        ImGui::TextColored(debug_.reason == "error" ? t.error : t.warning, "%s", debug_.error.c_str());
    }
    ImGui::SetCursorScreenPos(ImVec2(p.x, ImGui::GetCursorScreenPos().y + 4.0f));
}

// F9 (TOFIX133 P0 item 7: dispatched once, from HandleKeyboardShortcuts,
// through core/script_keys) and gutter clicks.
void ScriptEditorPanel::ToggleBreakpointAtCursor() {
    if (tabs_.empty() || active_tab_index_ < 0) return;
    auto& tab = tabs_[active_tab_index_];
    std::vector<int>* lines = nullptr;
    int line = 0;
    if (tab->cell_mode) {
        if (tab->selected_cell < 0 || tab->selected_cell >= tab->cell_manager.GetCellCount()) return;
        Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
        if (cell.type != CellType::Code) return;
        lines = &cell.breakpoints;
        line = cell.editor.Doc().Primary().head.line + 1;
    } else {
        lines = &tab->breakpoints;
        line = tab->editor.Doc().Primary().head.line + 1;
    }
    auto it = std::find(lines->begin(), lines->end(), line);
    if (it != lines->end()) lines->erase(it);
    else lines->push_back(line);
    SyncDebugBreakpoints();  // a run being debugged picks the change up at once
}

void ScriptEditorPanel::Debug() {
    if (!IsActiveTabEditable() || !scripting_engine_ || debug_run_.active) return;
    if (scripting_engine_->IsScriptRunning()) {
        spdlog::warn("Debug: another run holds the interpreter");
        return;
    }
    auto& tab = tabs_[active_tab_index_];
    if (tab->cell_mode) {
        // Debug cell: the selected code cell, in this notebook's variables.
        if (tab->selected_cell < 0 || tab->selected_cell >= tab->cell_manager.GetCellCount()) return;
        Cell& cell = tab->cell_manager.GetCell(tab->selected_cell);
        if (cell.type != CellType::Code) return;
        cell.SyncSourceFromEditor();
        debug_run_ = DebugRun{true, tab->document_id, cell.id};
        debug_seen_version_ = scripting_engine_->GetDebugSnapshot().version;
        tab->cell_manager.DebugCell(tab->selected_cell, DebugBreakpointsFor(*tab, &cell), debug_stop_on_error_);
        spdlog::info("Debugging cell {}", cell.id);
        return;
    }
    // The script, from the editor's text (saved or not), in the session.
    scripting::ScriptingEngine::RunCallbacks callbacks;
    callbacks.debug = true;
    callbacks.stop_on_error = debug_stop_on_error_;
    callbacks.script_filename = tab->filepath.empty() ? tab->filename : tab->filepath;
    callbacks.breakpoints = DebugBreakpointsFor(*tab, nullptr);
    running_script_name_ = tab->filename;
    running_script_started_ = std::chrono::steady_clock::now();
    if (script_output_sink_) script_output_sink_->AppendScriptOutput(running_script_name_, "", false);
    running_indicator_time_ = 0.0f;
    debug_seen_version_ = scripting_engine_->GetDebugSnapshot().version;
    script_running_ = scripting_engine_->ExecuteScriptAsync(WithoutSectionMarkers(tab->editor.GetText()), std::move(callbacks));
    if (script_running_) {
        debug_run_ = DebugRun{true, tab->document_id, {}};
        spdlog::info("Debugging {}", tab->filename);
    }
}

}  // namespace cyxwiz
