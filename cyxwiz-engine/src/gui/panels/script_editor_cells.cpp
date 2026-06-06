// Script Editor cell-mode rendering and keyboard handling.

#include "script_editor.h"
#include "output_renderer.h"
#include "../icons.h"
#include "../../scripting/scripting_engine.h"

#include <algorithm>
#include <sstream>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

namespace cyxwiz {

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
} // namespace cyxwiz
