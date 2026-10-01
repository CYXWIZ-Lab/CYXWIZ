// Script Editor cell-mode rendering and keyboard handling.

#include "script_editor.h"

#include <cmath>
#include "output_renderer.h"
#include "../icons.h"
#include "../editor_fonts.h"
#include "../ui_tokens.h"
#include "../../scripting/scripting_engine.h"

#include <algorithm>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

// ==================== Cell-Based Editor (Jupyter-like) ====================

void ScriptEditorPanel::ToggleCellMode() {
    if (!IsActiveTabEditable()) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    if (tab->cell_mode && scriptfile::IsNotebookJson(tab->filepath)) {
        spdlog::info("A Jupyter notebook always opens as a notebook");
        return;
    }
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

        tab->restore_cell_scroll = true;  // back where the notebook view was
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

    if (tab->cell_manager.TakeOutputsChanged() && scriptfile::IsNotebookJson(tab->filepath)) tab->is_modified = true;
    RenderNotebookToolbar(*tab);
    const float available_width = ImGui::GetContentRegionAvail().x;

    // Show debug toolbar when debugging is active
    if (debug_mode_active_ && debugger_) {
        RenderDebugToolbar();
    }

    // Cells on the window tone (board 4); the scrollbar in the surface tone.
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec4 ink = t.light ? ImVec4(0, 0, 0, 1) : ImVec4(1, 1, 1, 1);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_window);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, t.bg_window);
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ui::Mix(t.bg_window, ink, 0.08f));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, ui::Mix(t.bg_window, ink, 0.14f));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, ui::Mix(t.bg_window, ink, 0.14f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12, 16));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 4.0f));
    // The status bar below takes one frame row.
    const float cells_height = std::max(60.0f, ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing());
    ImGui::BeginChild("##cells_container", ImVec2(available_width, cells_height), false);
    {
        // Restore scroll position
        if (tab->restore_cell_scroll) {
            ImGui::SetScrollY(tab->cell_scroll_y);
            tab->restore_cell_scroll = false;
        }

        // Render each cell
        for (int i = 0; i < static_cast<int>(tab->cell_manager.GetCellCount()); ++i) {
            Cell& cell = tab->cell_manager.GetCell(i);
            RenderCell(cell, i);
            if (i < static_cast<int>(tab->cell_manager.GetCellCount())) RenderInsertBar(i);
        }

        // Save scroll position
        tab->cell_scroll_y = ImGui::GetScrollY();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(5);
}


void ScriptEditorPanel::HandleCellKeyboardShortcuts() {
    if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        return;
    }

    auto& tab = tabs_[active_tab_index_];
    bool ctrl = ImGui::GetIO().KeyCtrl;
    bool shift = ImGui::GetIO().KeyShift;
    bool is_editing = (tab->editing_cell >= 0);

    // Ctrl+Shift+M (notebook on/off) and the run/debug keys are dispatched
    // once in HandleKeyboardShortcuts. Command-mode letters below take no
    // modifier, so Ctrl+Shift+M no longer also turns the cell into markdown.
    const bool alt = ImGui::GetIO().KeyAlt;
    const bool bare = !ctrl && !shift && !alt;

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

    // Ctrl+Enter - run the cell and stay on it (command mode)
    if (ctrl && !shift && !alt && ImGui::IsKeyPressed(ImGuiKey_Enter)) {
        if (tab->selected_cell >= 0) {
            if (is_editing && tab->editing_cell < tab->cell_manager.GetCellCount())
                tab->cell_manager.GetCell(tab->editing_cell).SyncSourceFromEditor();
            tab->cell_manager.RunCell(tab->selected_cell);
            tab->editing_cell = -1;
            tab->last_editing_cell = -1;
        }
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

    // Arrow keys and letters (command mode, no modifiers)
    if (!is_editing && bare) {
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
} // namespace cyxwiz
