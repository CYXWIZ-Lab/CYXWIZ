// Script Editor notebook chrome (TOFIX133 P4 step 4.3, approved board 4):
// the toolbar with kernel controls and the kernel chip. Wording comes from
// core/notebook_presentation.

#include "script_editor.h"

#include "../../core/notebook_presentation.h"
#include "../../scripting/scripting_engine.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <imgui.h>

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
                       w(interrupt) + w("Restart") + w("Clear outputs") + 9.0f * gap + group +
                       ui::StatusPillWidth(chip.text.c_str()) + 2.0f * t.space_lg;
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
            ImGui::TextDisabled("%s", chip.text.c_str());
            ImGui::EndPopup();
        }
    }

    // Kernel chip at the right edge (in the menu above when it does not fit).
    const float chip_w = ui::StatusPillWidth(chip.text.c_str());
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

}  // namespace cyxwiz
