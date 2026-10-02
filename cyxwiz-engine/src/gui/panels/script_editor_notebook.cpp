// Script Editor notebook (TOFIX133 P4 step 4.3, approved board 4): the
// toolbar with kernel controls and the kernel chip, and the cells. Wording
// comes from core/notebook_presentation.

#include "script_editor.h"

#include "../../core/notebook_presentation.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../markdown_view.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {
ImVec4 ToneColour(nbview::Tone tone) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (tone) {
        case nbview::Tone::Success: return t.success;
        case nbview::Tone::Error: return t.error;
        case nbview::Tone::Running: return t.accent_text;
        case nbview::Tone::Warning: return t.warning;
        case nbview::Tone::Muted: break;
    }
    return t.pending;
}
}  // namespace

ImVec4 ScriptEditorPanel::NotebookToneColour(int tone) { return ToneColour(static_cast<nbview::Tone>(tone)); }

void ScriptEditorPanel::RenderNotebookToolbar(EditorTab& tab) {
    const ui::Tokens& t = ui::CurrentTokens();
    CellManager& cells = tab.cell_manager;
    const bool running = cells.IsRunning();
    const bool other_busy = !running && scripting_engine_ && scripting_engine_->IsScriptRunning();
    const bool has_selection = tab.selected_cell >= 0;

    auto add_cell = [&](CellType type) {
        const int pos = tab.selected_cell >= 0 ? tab.selected_cell + 1 : -1;
        const int index = cells.AddCell(type, pos);
        tab.selected_cell = index;
        tab.editing_cell = index;
        tab.is_modified = true;
    };

    // Kernel chip: Python version, environment, state.
    RefreshPythonStatus();
    nbview::KernelFacts facts;
    facts.python_version = python_version_;
    facts.environment = python_environment_;
    facts.started = python_started_;
    facts.busy = running;
    facts.other_busy = other_busy;
    facts.restarting = cells.IsRestarting();
    const nbview::KernelChip chip = nbview::KernelChipFor(facts);

    const ImVec2 bar_min = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float bar_h = ImGui::GetFrameHeight() + 2.0f * t.space_md;
    ImGui::GetWindowDrawList()->AddRectFilled(bar_min, ImVec2(bar_min.x + width, bar_min.y + bar_h), ui::ToU32(t.bg_window));
    ImGui::SetCursorScreenPos(ImVec2(bar_min.x + t.space_lg, bar_min.y + t.space_md));

    const char* run_all = ICON_FA_FORWARD "  Run All";
    const char* interrupt = ICON_FA_STOP "  Interrupt";
    const float gap = t.space_xs;
    const float group = 18.0f;
    auto w = [](const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); };
    const float full = w("+ Code") + w("+ Markdown") + group + w(run_all) + w("Run Above") + w("Run Below") + group +
                       w(interrupt) + w("Restart") + w("Clear outputs") + w("Outline") + w("Variables") + 11.0f * gap +
                       2.0f * group + ui::StatusPillWidth(chip.text.c_str()) + 2.0f * t.space_lg;
    const bool compact = full > width;
    const char* why_busy = "Another script is running in the Engine's Python";

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, 0.0f));
    if (ui::GhostButton("+ Code")) add_cell(CellType::Code);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Add a code cell below the selected one (B)");
    if (!compact) {
        ImGui::SameLine();
        if (ui::GhostButton("+ Markdown")) add_cell(CellType::Markdown);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Add a text cell below the selected one");
    }
    ImGui::SameLine(0.0f, compact ? gap : group);
    if (ui::PrimaryButton(run_all, !other_busy, why_busy, ui::ButtonSize::Small)) cells.RunAllCells();
    if (!compact) {
        ImGui::SameLine();
        if (ui::GhostButton("Run Above", has_selection && !other_busy, has_selection ? why_busy : "Select a cell first"))
            cells.RunCellsAbove(tab.selected_cell);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Run the cells from the top through the selected one");
        ImGui::SameLine();
        if (ui::GhostButton("Run Below", has_selection && !other_busy, has_selection ? why_busy : "Select a cell first"))
            cells.RunCellsBelow(tab.selected_cell);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Run the selected cell and every cell below it");
    }
    ImGui::SameLine(0.0f, compact ? gap : group);
    if (ui::DangerButton(interrupt, running, "Nothing is running")) cells.InterruptExecution();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) && running)
        ImGui::SetTooltip("Stop the running cell at its next Python line; queued cells are not run");
    if (!compact) {
        ImGui::SameLine();
        if (ui::GhostButton("Restart", !cells.IsRestarting(), "Restarting")) cells.Restart();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Clear this notebook's variables and start the [n] count again; outputs stay");
        ImGui::SameLine();
        if (ui::GhostButton("Clear outputs")) {
            cells.ClearAllOutputs();
            tab.is_modified = true;
        }
    } else {
        // Nothing hides: the rest is in a menu.
        ImGui::SameLine();
        if (ui::GhostButton("\xC2\xB7\xC2\xB7\xC2\xB7")) ImGui::OpenPopup("##notebook_more");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("More notebook actions");
        if (ImGui::BeginPopup("##notebook_more")) {
            if (ImGui::MenuItem("+ Markdown")) add_cell(CellType::Markdown);
            ImGui::Separator();
            if (ImGui::MenuItem("Run Above", nullptr, false, has_selection && !other_busy)) cells.RunCellsAbove(tab.selected_cell);
            if (ImGui::MenuItem("Run Below", nullptr, false, has_selection && !other_busy)) cells.RunCellsBelow(tab.selected_cell);
            ImGui::Separator();
            if (ImGui::MenuItem("Restart", nullptr, false, !cells.IsRestarting())) cells.Restart();
            if (ImGui::MenuItem("Clear outputs")) {
                cells.ClearAllOutputs();
                tab.is_modified = true;
            }
            ImGui::Separator();
            ImGui::MenuItem("Outline", nullptr, &tab.show_outline);
            ImGui::MenuItem("Variables", nullptr, &tab.show_variables);
            ImGui::Separator();
            ImGui::TextDisabled("%s", chip.text.c_str());
            ImGui::EndPopup();
        }
    }

    // Outline and Variables (board 4), then the kernel chip at the right edge
    // (in the menu above when it does not fit).
    const float chip_w = ui::StatusPillWidth(chip.text.c_str());
    if (!compact) {
        const float toggles_w = w("Outline") + w("Variables") + gap + 10.0f;
        ImGui::SameLine();
        const float toggles_x = bar_min.x + width - t.space_lg - chip_w - toggles_w;
        if (toggles_x > ImGui::GetCursorScreenPos().x)
            ImGui::SetCursorScreenPos(ImVec2(toggles_x, ImGui::GetCursorScreenPos().y));
        if (ui::GhostButton("Outline", true, nullptr, tab.show_outline)) tab.show_outline = !tab.show_outline;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("Headings and code cells; click to go there");
        ImGui::SameLine();
        if (ui::GhostButton("Variables", true, nullptr, tab.show_variables)) tab.show_variables = !tab.show_variables;
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("This notebook's variables");
    }
    ImGui::SameLine();
    const float chip_x = bar_min.x + width - t.space_lg - chip_w;
    if (chip_x > ImGui::GetCursorScreenPos().x) {
        ImGui::SetCursorScreenPos(ImVec2(chip_x, ImGui::GetCursorScreenPos().y));
        if (ui::StatusPill("##kernel_chip", chip.text.c_str(), ToneColour(chip.tone))) ImGui::OpenPopup("##kernel_menu");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", chip.tooltip.c_str());
    } else {
        ImGui::NewLine();
    }
    if (ImGui::BeginPopup("##kernel_menu")) {
        ImGui::TextDisabled("%s", python_status_.c_str());
        if (!python_tooltip_.empty()) ImGui::TextDisabled("%s", python_tooltip_.c_str());
        ImGui::Separator();
        if (ImGui::MenuItem("Interrupt", nullptr, false, running)) cells.InterruptExecution();
        if (ImGui::MenuItem("Restart", nullptr, false, !cells.IsRestarting())) cells.Restart();
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
    ImGui::SetCursorScreenPos(ImVec2(bar_min.x, bar_min.y + bar_h));
    ImGui::Dummy(ImVec2(width, 0.0f));
}


// ---------------------------------------------------------------------------
// Cells (board 4): [n] and time in a gutter, code on a raised block, the
// selected cell tinted with an accent bar and a floating toolbar, rendered
// text cells, an insert bar between cells, "Not run" notes.

namespace {
constexpr float kGutterWidth = 78.0f;

nbview::CellRun RunOf(CellState state) {
    switch (state) {
        case CellState::Queued: return nbview::CellRun::Queued;
        case CellState::Running: return nbview::CellRun::Running;
        case CellState::Success: return nbview::CellRun::Success;
        case CellState::Error: return nbview::CellRun::Error;
        case CellState::NotRun: return nbview::CellRun::NotRun;
        case CellState::Idle: break;
    }
    return nbview::CellRun::Idle;
}

ImVec4 BlockColour() {
    const ui::Tokens& t = ui::CurrentTokens();
    return t.light ? ui::Mix(t.bg_window, ImVec4(0, 0, 0, 1), 0.035f) : ui::Mix(t.bg_window, ImVec4(1, 1, 1, 1), 0.035f);
}

int SourceLines(const std::string& s) {
    int n = 1;
    for (char c : s) n += c == '\n';
    return n;
}
}  // namespace

// The code (or text being edited) on a raised block that fits its rows.
bool ScriptEditorPanel::RenderCellEditorBlock(Cell& cell, int index, float width, bool editing, bool python) {
    auto& tab = *tabs_[active_tab_index_];
    const ui::Tokens& t = ui::CurrentTokens();
    ImFont* code_font = gui::GetEditorMonoFont(font_scale_);
    const float font_size = code_font ? code_font->FontSize : ImGui::GetFontSize();
    const float line_h = CodeEditor::LineHeightFor(font_size);
    const float pad_y = 8.0f;

    if (editing && tab.last_editing_cell != index) {
        cell.SyncEditorFromSource();
        cell.editor.RequestFocus();
        tab.last_editing_cell = index;
    } else if (!editing && cell.editor.GetText() != cell.source) {
        cell.SyncEditorFromSource();
    }
    cell.editor.SetLanguageIsPython(python);
    cell.editor.SetShowLineNumbers(false);
    cell.editor.SetWordWrap(true);
    cell.editor.SetScrollPastEnd(false);
    cell.editor.SetBackground(ui::ToU32(BlockColour()));
    cell.editor.SetReadOnly(!editing);
    cell.editor.SetKeyboardEnabled(editing && !completion_just_accepted_);
    ApplyProblemSquiggles(tab, cell.editor, cell.problems);

    const int rows = std::max(1, cell.editor.RowCount() > 0 ? cell.editor.RowCount() : SourceLines(cell.source));
    const float view_h = rows * line_h;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 block_max(p.x + width, p.y + view_h + 2.0f * pad_y);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, block_max, ui::ToU32(BlockColour()), 6.0f);
    if (cell.state == CellState::Running)
        dl->AddRectFilled(ImVec2(p.x + 4.0f, block_max.y - 2.0f), ImVec2(block_max.x - 4.0f, block_max.y),
                          ui::ToU32(ui::WithAlpha(t.accent_text, 0.6f)));

    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + pad_y));
    if (code_font) ImGui::PushFont(code_font);
    const bool changed = cell.editor.Render("##cell_code", ImVec2(width - 8.0f, view_h));
    if (code_font) ImGui::PopFont();
    if (python) AfterCodeRender(cell.editor, cell.problems, cell.id);  // hover card, Ctrl+click
    if (editing) {
        const bool accepted = completion_just_accepted_;
        completion_just_accepted_ = false;
        cell.SyncSourceFromEditor();
        if (changed) {
            tab.is_modified = true;
            if (python && !accepted && !completion_just_opened_) UpdateAutoCompletion(false);
        }
    }

    // A click in the block edits the cell (the click also placed the cursor).
    const bool clicked = ImGui::IsMouseHoveringRect(p, block_max) && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                         ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    if (clicked && !editing) {
        tab.selected_cell = index;
        tab.editing_cell = index;
        tab.last_editing_cell = index;  // keep the cursor where the click put it
        cell.editor.RequestFocus();
    }
    ImGui::SetCursorScreenPos(ImVec2(p.x, block_max.y));
    ImGui::Dummy(ImVec2(width, 0.0f));
    return clicked;
}

void ScriptEditorPanel::RenderCell(Cell& cell, int index) {
    auto& tab = *tabs_[active_tab_index_];
    const ui::Tokens& t = ui::CurrentTokens();
    const bool selected = tab.selected_cell == index;
    const bool editing = tab.editing_cell == index;
    const bool code = cell.type == CellType::Code;
    ImGui::PushID(cell.id.c_str());

    const ImVec2 row_min = ImGui::GetCursorScreenPos();
    const float row_w = ImGui::GetContentRegionAvail().x;
    if (tab.scroll_to_cell == index) {
        ImGui::SetScrollHereY(0.1f);
        tab.scroll_to_cell = -1;
    }
    const float content_x = row_min.x + kGutterWidth;
    const float content_w = std::max(120.0f, row_w - kGutterWidth - 12.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Selection: a tint over the whole row and an accent bar at its left,
    // drawn first with the height the row had last frame.
    const auto known = tab.cell_heights.find(cell.id);
    const float last_h = known != tab.cell_heights.end() ? known->second : 0.0f;
    if (selected && last_h > 0.0f) {
        dl->AddRectFilled(row_min, ImVec2(row_min.x + row_w, row_min.y + last_h), ui::ToU32(ui::WithAlpha(t.accent, 0.07f)), 6.0f);
        dl->AddRectFilled(row_min, ImVec2(row_min.x + 2.0f, row_min.y + last_h), ui::ToU32(t.accent));
    }
    const bool dim = cell.state == CellState::Queued || cell.state == CellState::NotRun;
    if (dim) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.85f);

    // Gutter: [n] and how the last run went.
    if (code) {
        const nbview::Gutter g = nbview::GutterFor(RunOf(cell.state), cell.execution_count, cell.duration_seconds,
                                                   cell.state == CellState::Running ? tab.cell_manager.RunningSeconds() : 0.0);
        const ImVec4 tone = g.tone == nbview::Tone::Muted ? t.text_dim : NotebookToneColour(static_cast<int>(g.tone));
        ImFont* mono = gui::GetEditorMonoFont(font_scale_);
        ImFont* font = mono ? mono : ImGui::GetFont();
        const float size = std::max(11.0f, ImGui::GetFontSize() - 1.0f);
        const float right = row_min.x + kGutterWidth - 10.0f;
        const float lw = font->CalcTextSizeA(size, FLT_MAX, 0.0f, g.label.c_str()).x;
        const ImVec4 label_colour = g.tone == nbview::Tone::Error || g.tone == nbview::Tone::Running ? tone : t.text_dim;
        dl->AddText(font, size, ImVec2(right - lw, row_min.y + 12.0f), ui::ToU32(label_colour), g.label.c_str());
        if (!g.detail.empty()) {
            // The interface font carries the icon glyphs (Font Awesome merged in).
            std::string detail = g.detail;
            if (g.mark == nbview::Gutter::Mark::Check) detail = std::string(ICON_FA_CHECK " ") + detail;
            if (g.mark == nbview::Gutter::Mark::Cross) detail = std::string(ICON_FA_XMARK " ") + detail;
            ImFont* small = ImGui::GetFont();
            const float ss = std::max(10.0f, ImGui::GetFontSize() - 2.0f);
            const float dw = small->CalcTextSizeA(ss, FLT_MAX, 0.0f, detail.c_str()).x;
            dl->AddText(small, ss, ImVec2(right - dw, row_min.y + 14.0f + size), ui::ToU32(tone), detail.c_str());
        }
    }

    ImGui::SetCursorScreenPos(ImVec2(content_x, row_min.y + 4.0f));
    ImGui::BeginGroup();
    bool block_clicked = false;
    if (cell.collapsed) {
        // Folded (C): the first line and how much is hidden.
        const std::string first = cell.source.substr(0, cell.source.find('\n'));
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "%s", first.empty() ? "(empty)" : first.c_str());
        ImGui::SameLine();
        char more[64];
        std::snprintf(more, sizeof(more), "%d lines folded##unfold", SourceLines(cell.source));
        if (ui::LinkButton(more)) cell.collapsed = false;
    } else if (code) {
        block_clicked = RenderCellEditorBlock(cell, index, content_w, editing, true);
    } else if (cell.type == CellType::Markdown && !editing) {
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        ImGui::Indent(14.0f);
        ImGui::PushTextWrapPos(content_x + content_w - 14.0f);
        const ImVec2 md_min = ImGui::GetCursorScreenPos();
        ImGui::BeginGroup();
        if (cell.source.find_first_not_of(" \t\r\n") == std::string::npos)
            ImGui::TextColored(t.text_faint, "Empty text cell. Double-click to write.");
        else
            ui::MarkdownView(cell.source);
        ImGui::EndGroup();
        ImGui::PopTextWrapPos();
        ImGui::Unindent(14.0f);
        ImGui::Dummy(ImVec2(content_w, 6.0f));
        if (ImGui::IsMouseHoveringRect(md_min, ImVec2(content_x + content_w, ImGui::GetItemRectMax().y)) &&
            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && ImGui::IsWindowHovered()) {
            tab.selected_cell = index;
            tab.editing_cell = index;
        }
    } else if (cell.type == CellType::Raw && !editing) {
        ImGui::PushTextWrapPos(content_x + content_w);
        ImGui::TextColored(t.text_dim, "%s", cell.source.c_str());
        ImGui::PopTextWrapPos();
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            tab.selected_cell = index;
            tab.editing_cell = index;
        }
    } else {
        block_clicked = RenderCellEditorBlock(cell, index, content_w, editing, false);
    }

    // Outputs (step 4.3c restyles them to board 5).
    if (!cell.collapsed && !cell.outputs.empty()) {
        if (cell.output_collapsed) {
            char shown[64];
            std::snprintf(shown, sizeof(shown), "Output hidden (%zu)  Show##show_out", cell.outputs.size());
            ImGui::Indent(12.0f);
            if (ui::LinkButton(shown)) cell.output_collapsed = false;
            ImGui::Unindent(12.0f);
        } else {
            ImGui::Dummy(ImVec2(0.0f, 2.0f));
            ImGui::Indent(12.0f);
            ImGui::PushTextWrapPos(content_x + content_w);
            RenderNotebookOutputs(cell, index, content_w - 12.0f);
            ImGui::PopTextWrapPos();
            ImGui::Unindent(12.0f);
        }
    }
    if (cell.state == CellState::NotRun) {
        ImGui::Indent(12.0f);
        ImGui::AlignTextToFramePadding();
        const std::string note = nbview::NotRunMessage(tab.cell_manager.StoppedAtCount(), tab.cell_manager.StoppedByInterrupt());
        ImGui::TextColored(t.text_dim, "%s", note.c_str());
        ImGui::SameLine();
        if (ui::LinkButton("Run from here")) tab.cell_manager.RunCellsBelow(index);
        ImGui::Unindent(12.0f);
    }
    ImGui::EndGroup();
    if (dim) ImGui::PopStyleVar();

    const float row_bottom = std::max(ImGui::GetItemRectMax().y + 6.0f, row_min.y + (code ? 56.0f : 0.0f));
    tab.cell_heights[cell.id] = row_bottom - row_min.y;
    const ImVec2 row_max(row_min.x + row_w, row_bottom);
    const bool hovered = ImGui::IsMouseHoveringRect(row_min, row_max) && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !block_clicked) {
        if (tab.selected_cell != index || editing) {
            if (editing) cell.SyncSourceFromEditor();
            tab.selected_cell = index;
            if (!editing || tab.editing_cell == index) {
                tab.editing_cell = -1;
                tab.last_editing_cell = -1;
            }
        }
    }
    ImGui::SetCursorScreenPos(ImVec2(row_min.x, row_bottom));
    ImGui::Dummy(ImVec2(row_w, 0.0f));

    if (selected || hovered) {
        const ImVec2 after = ImGui::GetCursorScreenPos();
        RenderCellActions(cell, index, ImVec2(content_x + content_w - 4.0f, index > 0 ? row_min.y - 14.0f : row_min.y + 2.0f));
        ImGui::SetCursorScreenPos(after);
    }
    ImGui::PopID();
}

// The selected (or hovered) cell's toolbar, floating at its top right.
void ScriptEditorPanel::RenderCellActions(Cell& cell, int index, const ImVec2& top_right) {
    auto& tab = *tabs_[active_tab_index_];
    const std::string cell_id = cell.id;  // the cell may be deleted below
    const ui::Tokens& t = ui::CurrentTokens();
    CellManager& cells = tab.cell_manager;
    const bool code = cell.type == CellType::Code;
    const bool other_busy = !cells.IsRunning() && scripting_engine_ && scripting_engine_->IsScriptRunning();
    const char* type_label = code ? "Code " ICON_FA_CARET_DOWN : (cell.type == CellType::Markdown ? "Text " ICON_FA_CARET_DOWN : "Raw " ICON_FA_CARET_DOWN);
    auto bw = [](const char* l) { return ui::ButtonWidth(l, ui::ButtonSize::Small); };
    const float gap = 2.0f;
    float w = bw(type_label) + bw(ICON_FA_ARROW_UP) + bw(ICON_FA_ARROW_DOWN) + bw("\xC2\xB7\xC2\xB7\xC2\xB7") + bw(ICON_FA_TRASH) +
              8.0f + 5.0f * gap;
    if (code) w += bw(ICON_FA_PLAY) + gap;
    const float h = ImGui::GetFrameHeight() + 6.0f;

    ImGui::SetCursorScreenPos(ImVec2(top_right.x - w, top_right.y));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, t.bg_panel);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(3.0f, 3.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, 0.0f));
    ImGui::BeginChild("##cell_actions", ImVec2(w, h), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    auto tip = [](const char* text) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", text);
    };
    auto stop_editing = [&]() {
        if (tab.editing_cell == index) cell.SyncSourceFromEditor();
        tab.editing_cell = -1;
        tab.last_editing_cell = -1;
    };
    if (code) {
        if (ui::GhostButton(ICON_FA_PLAY "##run", !other_busy, "Another script is running", false, ui::ButtonSize::Small,
                            &t.success)) {
            tab.selected_cell = index;
            cells.RunCell(index);
        }
        tip("Run cell (Ctrl+Enter)");
        ImGui::SameLine();
    }
    if (ui::GhostButton(type_label)) ImGui::OpenPopup("##cell_type");
    tip("Cell type (Y code, M text)");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_ARROW_UP "##up", index > 0) && cells.MoveCell(index, index - 1)) {
        tab.selected_cell = index - 1;
        tab.is_modified = true;
    }
    tip("Move up");
    ImGui::SameLine();
    if (ui::GhostButton(ICON_FA_ARROW_DOWN "##down", index + 1 < cells.GetCellCount()) && cells.MoveCell(index, index + 1)) {
        tab.selected_cell = index + 1;
        tab.is_modified = true;
    }
    tip("Move down");
    ImGui::SameLine();
    if (ui::GhostButton("\xC2\xB7\xC2\xB7\xC2\xB7##more")) ImGui::OpenPopup("##cell_more");
    tip("More: edit, duplicate, split, merge, copy, cut, paste, fold, output");
    ImGui::SameLine();
    bool deleted = false;
    if (ui::DangerButton(ICON_FA_TRASH "##delete")) deleted = true;
    tip("Delete cell (D, D)");

    if (ImGui::BeginPopup("##cell_type")) {
        const struct { const char* label; CellType type; } kinds[] = {
            {"Code", CellType::Code}, {"Text (markdown)", CellType::Markdown}, {"Raw", CellType::Raw}};
        for (const auto& k : kinds) {
            if (ImGui::MenuItem(k.label, nullptr, cell.type == k.type) && cell.type != k.type) {
                stop_editing();
                if (cells.ChangeCellType(index, k.type)) tab.is_modified = true;
            }
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("##cell_more")) {
        const bool editing = tab.editing_cell == index;
        if (!editing && ImGui::MenuItem("Edit", "Enter")) {
            tab.selected_cell = index;
            tab.editing_cell = index;
        }
        if (editing && ImGui::MenuItem("Stop editing", "Esc")) stop_editing();
        if (code && ImGui::MenuItem("Run from here", nullptr, false, !other_busy)) cells.RunCellsBelow(index);
        ImGui::Separator();
        if (ImGui::MenuItem("Duplicate")) {
            stop_editing();
            const int copy = cells.DuplicateCell(index);
            if (copy >= 0) {
                tab.selected_cell = copy;
                tab.is_modified = true;
            }
        }
        const int cursor_line = cell.editor.Doc().Primary().head.line;
        if (ImGui::MenuItem("Split at cursor", nullptr, false, editing && cursor_line > 0)) {
            cell.SyncSourceFromEditor();
            const int part = cells.SplitCell(index, cursor_line);
            if (part >= 0) {
                tab.editing_cell = -1;
                tab.last_editing_cell = -1;
                tab.selected_cell = part;
                tab.is_modified = true;
            }
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !editing)
            ImGui::SetTooltip("Edit the cell and put the cursor where the second part starts");
        if (ImGui::MenuItem("Merge with the cell below", nullptr, false, index + 1 < cells.GetCellCount())) {
            stop_editing();
            if (cells.MergeCells(index, index + 1)) tab.is_modified = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Copy cell")) {
            if (tab.editing_cell == index) cell.SyncSourceFromEditor();
            cell_clipboard_ = {true, cell.type, cell.source};
        }
        if (ImGui::MenuItem("Cut cell")) {
            stop_editing();
            cell_clipboard_ = {true, cell.type, cell.source};
            deleted = true;
        }
        if (ImGui::MenuItem("Paste cell below", nullptr, false, cell_clipboard_.has)) {
            const int pasted = cells.AddCell(cell_clipboard_.type, index + 1);
            Cell& p = cells.GetCell(pasted);
            p.source = cell_clipboard_.source;
            p.SyncEditorFromSource();
            tab.selected_cell = pasted;
            tab.is_modified = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem(cell.collapsed ? "Unfold cell" : "Fold cell", "C")) cell.collapsed = !cell.collapsed;
        if (code) {
            if (ImGui::MenuItem(cell.output_collapsed ? "Show output" : "Hide output", "O", false, !cell.outputs.empty()))
                cell.output_collapsed = !cell.output_collapsed;
            if (ImGui::MenuItem("Clear output", nullptr, false, !cell.outputs.empty())) {
                cell.ClearOutputs();
                tab.is_modified = true;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();

    if (deleted && cells.DeleteCell(index)) {
        tab.cell_heights.erase(cell_id);
        tab.editing_cell = -1;
        tab.last_editing_cell = -1;
        if (tab.selected_cell >= cells.GetCellCount()) tab.selected_cell = cells.GetCellCount() - 1;
        tab.is_modified = true;
    }
}

// Between cells: "+ Code  + Text" on hover, and always after the selected cell.
void ScriptEditorPanel::RenderInsertBar(int after_index) {
    auto& tab = *tabs_[active_tab_index_];
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float w = ImGui::GetContentRegionAvail().x;
    const float h = 18.0f;
    const bool hovered = ImGui::IsMouseHoveringRect(p, ImVec2(p.x + w, p.y + h)) && ImGui::IsWindowHovered();
    if (!hovered && tab.selected_cell != after_index) {
        ImGui::Dummy(ImVec2(w, h));
        return;
    }
    ImGui::PushID(after_index);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const char* code_label = "+ Code";
    const char* text_label = "+ Text";
    ImFont* font = ImGui::GetFont();
    const float size = std::max(10.0f, ImGui::GetFontSize() - 2.0f);
    const float cw = font->CalcTextSizeA(size, FLT_MAX, 0.0f, code_label).x;
    const float tw = font->CalcTextSizeA(size, FLT_MAX, 0.0f, text_label).x;
    const float mid = p.x + kGutterWidth + (w - kGutterWidth) * 0.5f;
    const float gap = 14.0f;
    const float x0 = mid - (cw + gap + tw) * 0.5f;
    const float y = p.y + (h - size) * 0.5f;
    const ImU32 line = ui::ToU32(ui::WithAlpha(t.text, 0.06f));
    dl->AddLine(ImVec2(p.x + kGutterWidth + 12.0f, p.y + h * 0.5f), ImVec2(x0 - 10.0f, p.y + h * 0.5f), line);
    dl->AddLine(ImVec2(x0 + cw + gap + tw + 10.0f, p.y + h * 0.5f), ImVec2(p.x + w - 12.0f, p.y + h * 0.5f), line);
    auto text_button = [&](const char* id, const char* label, float x, float lw) {
        ImGui::SetCursorScreenPos(ImVec2(x, p.y));
        const bool clicked = ImGui::InvisibleButton(id, ImVec2(lw, h));
        const bool hot = ImGui::IsItemHovered();
        if (hot) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        dl->AddText(font, size, ImVec2(x, y), ui::ToU32(hot ? t.text_bright : t.text_dim), label);
        return clicked;
    };
    auto insert = [&](CellType type) {
        const int index = tab.cell_manager.AddCell(type, after_index + 1);
        tab.selected_cell = index;
        tab.editing_cell = index;
        tab.is_modified = true;
    };
    if (text_button("##add_code", code_label, x0, cw)) insert(CellType::Code);
    if (text_button("##add_text", text_label, x0 + cw + gap, tw)) insert(CellType::Markdown);
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h));
    ImGui::Dummy(ImVec2(w, 0.0f));
    ImGui::PopID();
}

}  // namespace cyxwiz
