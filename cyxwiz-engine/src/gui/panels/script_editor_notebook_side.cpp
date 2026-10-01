// Script Editor notebook side panels (TOFIX133 P4 step 4.3d, board 4): the
// notebook's Variables (its own namespace, decision D4) under the cells, and
// an Outline of headings and code cells beside them.

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
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

namespace cyxwiz {

namespace {
bool ContainsCI(const std::string& text, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(text.begin(), text.end(), needle.begin(), needle.end(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    });
    return it != text.end();
}
}  // namespace

void ScriptEditorPanel::RefreshNotebookVariables(EditorTab& tab) {
    if (!scripting_engine_) return;
    std::string json;
    if (!scripting_engine_->NotebookVariablesJson(tab.cell_manager.NamespaceKey(), &json)) return;  // busy: next frame
    tab.variables.clear();
    try {
        for (const auto& item : nlohmann::json::parse(json)) {
            NotebookVariable v;
            v.name = item.value("name", "");
            v.type = item.value("type", "");
            v.size = item.value("size", "");
            v.value = item.value("value", "");
            v.table = item.value("table", false);
            tab.variables.push_back(std::move(v));
        }
    } catch (const std::exception& e) {
        spdlog::warn("Notebook variables could not be read: {}", e.what());
    }
    tab.variables_generation = tab.cell_manager.RunGeneration();
}

void ScriptEditorPanel::RenderNotebookVariables(EditorTab& tab, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (tab.variables_generation != tab.cell_manager.RunGeneration() && !tab.cell_manager.IsRunning())
        RefreshNotebookVariables(tab);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ui::Mix(t.bg_window, t.bg_panel, 0.6f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 6.0f));
    ImGui::BeginChild("##notebook_variables", ImVec2(0.0f, height), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar);
    // Header: title, count, filter.
    int shown = 0;
    for (const auto& v : tab.variables)
        if (ContainsCI(v.name, tab.variables_filter) || ContainsCI(v.type, tab.variables_filter)) ++shown;
    ImGui::AlignTextToFramePadding();
    {
        ui::FontScope bold(ui::Font::Bold);
        ImGui::TextUnformatted("Variables");
    }
    ImGui::SameLine(0.0f, 12.0f);
    ImGui::TextColored(t.text_dim, "this notebook \xC2\xB7 %d", static_cast<int>(tab.variables.size()));
    const float filter_w = 220.0f;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 12.0f, ImGui::GetContentRegionMax().x - filter_w));
    ui::SearchField("##variables_filter", tab.variables_filter, sizeof(tab.variables_filter), "Filter by name or type", filter_w);

    if (tab.variables.empty()) {
        ImGui::Dummy(ImVec2(0.0f, 4.0f));
        ImGui::TextColored(t.text_dim, "%s", tab.cell_manager.GetExecutionCount() > 0
                                                 ? "No variables yet in this notebook."
                                                 : "Run a cell: the variables it makes show here.");
    } else {
        const ImGuiTableFlags flags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoBordersInBody;
        ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ui::WithAlpha(t.text, 0.025f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_TableBorderLight, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, ImVec4(0, 0, 0, 0));
        // Rows highlight in the selection tone, not the theme's header colour.
        ImGui::PushStyleColor(ImGuiCol_Header, t.selection);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.hover);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.selection);
        if (ImGui::BeginTable("##vars", 4, flags, ImVec2(0.0f, ImGui::GetContentRegionAvail().y))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 160.0f);
            ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 140.0f);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 140.0f);
            ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            for (int c = 0; c < 4; ++c) {
                ImGui::TableSetColumnIndex(c);
                ImGui::TextColored(t.text_dim, "%s", ImGui::TableGetColumnName(c));
            }
            ImFont* mono = gui::GetCodeFont();
            for (size_t i = 0; i < tab.variables.size(); ++i) {
                const NotebookVariable& v = tab.variables[i];
                if (!ContainsCI(v.name, tab.variables_filter) && !ContainsCI(v.type, tab.variables_filter)) continue;
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (mono) ImGui::PushFont(mono);
                const bool picked = ImGui::Selectable(v.name.c_str(), false,
                                                      ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
                if (mono) ImGui::PopFont();
                if (picked && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && v.table) OpenVariableInTableViewer(tab, v.name);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
                    ImGui::SetTooltip("%s%s", v.value.c_str(), v.table ? "\n\nDouble-click: open in the Table Viewer" : "");
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(t.info, "%s", v.type.c_str());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextColored(t.text_dim, "%s", v.size.c_str());
                ImGui::TableSetColumnIndex(3);
                if (mono) ImGui::PushFont(mono);
                ImGui::TextColored(t.text_dim, "%s", v.value.c_str());
                if (mono) ImGui::PopFont();
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleColor(8);
        if (shown == 0) ImGui::TextColored(t.text_dim, "No variable matches \"%s\".", tab.variables_filter);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
}

void ScriptEditorPanel::OpenVariableInTableViewer(EditorTab& tab, const std::string& name) {
    if (!scripting_engine_) return;
    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec) / "cyxwiz_notebook_tables";
    std::filesystem::create_directories(dir, ec);
    const auto file = dir / (tab.cell_manager.NamespaceKey() + "_" + name + ".csv");
    std::string why;
    if (!scripting_engine_->ExportNotebookVariableToCsv(tab.cell_manager.NamespaceKey(), name, file.string(), &why)) {
        spdlog::warn("Could not open {} in the Table Viewer: {}", name, why);
        return;
    }
    auto table = std::make_shared<DataTable>();
    if (!table->LoadFromCSV(file.string())) return;
    table->SetName(std::filesystem::path(tab.filename).stem().string() + " " + name);
    if (open_table_callback_) open_table_callback_(table);
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
