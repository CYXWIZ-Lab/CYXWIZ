// Script Editor notebook side panels (TOFIX133 P4 step 4.3d, board 4): the
// notebook's Variables (its own namespace, decision D4; since P5 the shared
// Variables view of board 9) under the cells, and an Outline of headings and
// code cells beside them.

#include "script_editor.h"

#include "../../core/markdown_blocks.h"
#include "../../data/data_table.h"
#include "../../scripting/scripting_engine.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <string>

namespace cyxwiz {

void ScriptEditorPanel::RenderNotebookVariables(EditorTab& tab, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!tab.variables_view) {
        tab.variables_view = std::make_unique<VariablesView>();
        tab.variables_view->SetTitle("Variables");
        tab.variables_view->on_open_table = [this](const std::string& name, const VariablesView::Scope& scope,
                                                    const scripting::VariablesService::Result& result) {
            if (!result.table) return;
            result.table->SetName(name + " \xC2\xB7 " + scope.label);
            if (open_table_callback_) open_table_callback_(result.table);
        };
        tab.variables_view->on_insert_name = [this](const std::string& name) { InsertTextAtCursor(name); };
    }
    tab.variables_view->SetEngine(scripting_engine_.get());
    tab.variables_view->SetScope({tab.cell_manager.NamespaceKey(), "this notebook", ""});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.bg_panel, 0.6f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 6.0f));
    ImGui::BeginChild("##notebook_variables", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    tab.variables_view->Render(0.0f);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void ScriptEditorPanel::RestartNotebook(EditorTab& tab) {
    tab.cell_manager.Restart();
    if (tab.variables_view) tab.variables_view->Forget("after Restart");
}

std::vector<VariablesView::Scope> ScriptEditorPanel::NotebookScopes() const {
    std::vector<VariablesView::Scope> out;
    for (const auto& tab : tabs_)
        if (tab && tab->cell_mode) out.push_back({tab->cell_manager.NamespaceKey(), tab->filename, "notebook"});
    return out;
}

void ScriptEditorPanel::InsertTextAtCursor(const std::string& text) {
    CodeEditor* code = ActiveCodeEditor();
    if (!code) return;
    code->Doc().Paste(text);
    code->RequestFocus();
    request_window_focus_ = true;
}

void ScriptEditorPanel::RenderNotebookOutline(EditorTab& tab, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.bg_panel, 0.6f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));
    ImGui::BeginChild("##notebook_outline", ImVec2(width, height), ImGuiChildFlags_AlwaysUseWindowPadding);
    {
        ui::FontScope bold(ui::Font::Bold);
        ImGui::TextUnformatted("Outline");
    }
    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
    CellManager& cells = tab.cell_manager;
    ImFont* mono = gui::GetCodeFont();
    for (int i = 0; i < cells.GetCellCount(); ++i) {
        const Cell& c = cells.GetCell(i);
        ImGui::PushID(i);
        bool clicked = false;
        const bool selected = tab.selected_cell == i;
        if (c.type == CellType::Markdown) {
            // Headings, indented by level; a text cell without one shows its first words.
            bool any = false;
            for (const auto& b : md::Parse(c.source)) {
                if (b.kind != md::Block::Kind::Heading) continue;
                std::string title;
                for (const auto& r : b.runs) title += r.text;
                ImGui::Indent(10.0f * static_cast<float>(b.level - 1));
                clicked = ImGui::Selectable((title + "##h" + std::to_string(any)).c_str(), selected) || clicked;
                ImGui::Unindent(10.0f * static_cast<float>(b.level - 1));
                any = true;
            }
            if (!any) {
                std::string first = c.source.substr(0, c.source.find('\n'));
                if (first.size() > 40) first = first.substr(0, 40) + "...";
                ImGui::PushStyleColor(ImGuiCol_Text, t.text_dim);
                clicked = ImGui::Selectable(first.empty() ? "(empty text)" : first.c_str(), selected);
                ImGui::PopStyleColor();
            }
        } else if (c.type == CellType::Code) {
            std::string first = c.source.substr(0, c.source.find('\n'));
            if (first.size() > 36) first = first.substr(0, 36) + "...";
            const std::string label = (c.execution_count > 0 ? "[" + std::to_string(c.execution_count) + "] " : "[ ] ") +
                                      (first.empty() ? "(empty)" : first);
            if (mono) ImGui::PushFont(mono);
            ImGui::PushStyleColor(ImGuiCol_Text, c.state == CellState::Error ? t.error : t.text_dim);
            clicked = ImGui::Selectable(label.c_str(), selected);
            ImGui::PopStyleColor();
            if (mono) ImGui::PopFont();
        }
        if (clicked) {
            tab.selected_cell = i;
            tab.editing_cell = -1;
            tab.last_editing_cell = -1;
            tab.scroll_to_cell = i;
        }
        ImGui::PopID();
    }
    ImGui::PopStyleColor(3);
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

}  // namespace cyxwiz
