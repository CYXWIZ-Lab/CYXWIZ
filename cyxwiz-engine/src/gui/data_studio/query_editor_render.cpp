#include "query_editor.h"

#include "../../core/arrow_data_table.h"
#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/session_query_service.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace cyxwiz {

namespace {

std::string Thousands(long long n) {
    std::string s = std::to_string(n < 0 ? -n : n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<size_t>(i), ",");
    return n < 0 ? "-" + s : s;
}

std::shared_ptr<arrow::Schema> SchemaOf(const std::string& shown) {
    const std::string name = DatasetCatalog::Instance().NameFor(shown);
    auto& registry = DataRegistry::Instance();
    if (auto ds = registry.GetArrowDataset(name)) return ds->GetSchema();
    if (auto ds = registry.GetParquetBackedDataset(name)) return ds->GetSchema();
    return nullptr;
}

}  // namespace

void QueryEditor::Render() {
    RenderEditor();
    ImGui::Spacing();
    RenderResult();
}

void QueryEditor::RenderEditor() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImFont* mono = gui::GetCodeFont();
    if (mono) ImGui::PushFont(mono);
    ImGui::InputTextMultiline("##query", query_buffer_, sizeof(query_buffer_), ImVec2(-1, 150), ImGuiInputTextFlags_AllowTabInput);
    if (mono) ImGui::PopFont();
    // Ctrl+Enter runs while the editor has focus (Preferences > Shortcuts, Data Studio).
    const bool editor_focused = ImGui::IsItemActive() || ImGui::IsItemFocused();
    if (editor_focused && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Enter, false) && !IsRunning()) ExecuteQuery();

    // The tables a query can name.
    const auto names = SessionQueryService::Instance().QueryableNames();
    std::string tables;
    for (size_t i = 0; i < names.size() && i < 6; ++i) tables += (i ? ", " : "") + SessionQueryEngine::QuoteIdentifier(names[i]);
    if (names.size() > 6) tables += ", +" + std::to_string(names.size() - 6) + " more";
    ImGui::PushTextWrapPos(0.0f);
    if (names.empty()) ImGui::TextColored(t.text_dim, "No tables to query yet: load a Data Input (or run a node's result) and name it here.");
    else ImGui::TextColored(t.text_dim, "Tables you can name: %s. Reading only; no files or network.", tables.c_str());
    ImGui::PopTextWrapPos();

    const bool running = IsRunning();
    if (ui::PrimaryButton(ICON_FA_PLAY " Run", !running, running ? "A query is running" : nullptr, ui::ButtonSize::Small)) ExecuteQuery();
    ImGui::SameLine();
    if (ui::SecondaryButton("Cancel", running, "Nothing is running")) CancelQuery();
    ImGui::SameLine();
    if (ui::SecondaryButton("Examples")) ImGui::OpenPopup("##query_examples");
    RenderExamples();
    ImGui::SameLine();
    if (ui::GhostButton("Clear")) query_buffer_[0] = '\0';
    ImGui::SameLine();
    if (running) {
        ImGui::TextColored(t.info, ICON_FA_SPINNER " Running... %.1f s (Task View shows it)", ImGui::GetTime() - started_at_);
    } else {
        ImGui::TextColored(t.text_faint, "Ctrl+Enter");
    }
    if (!last_error_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(last_error_ == "Cancelled." ? t.text_dim : t.error, "%s", last_error_.c_str());
        ImGui::PopTextWrapPos();
    }
}

void QueryEditor::RenderExamples() {
    if (!ImGui::BeginPopup("##query_examples")) return;
    const std::string table = ExampleTable();
    const std::string q = SessionQueryEngine::QuoteIdentifier(table);
    // Real columns of the table when it is loaded.
    std::string text_col = "column_name", num_col = "column_name";
    if (auto schema = SchemaOf(table)) {
        bool have_text = false, have_num = false;
        for (const auto& f : schema->fields()) {
            const bool number = arrow::is_integer(f->type()->id()) || arrow::is_floating(f->type()->id());
            if (number && !have_num) {
                num_col = SessionQueryEngine::QuoteIdentifier(f->name());
                have_num = true;
            }
            if (!number && !have_text) {
                text_col = SessionQueryEngine::QuoteIdentifier(f->name());
                have_text = true;
            }
        }
    }
    const auto pick = [&](const char* label, const std::string& sql) {
        if (ImGui::Selectable(label)) std::snprintf(query_buffer_, sizeof(query_buffer_), "%s", sql.c_str());
    };
    pick("First 100 rows", "SELECT * FROM " + q + " LIMIT 100");
    pick("Count rows", "SELECT count(*) AS rows FROM " + q);
    pick("Rows per value of a column", "SELECT " + text_col + ", count(*) AS rows FROM " + q + " GROUP BY 1 ORDER BY rows DESC");
    pick("Average of a number per value", "SELECT " + text_col + ", round(avg(" + num_col + "), 2) AS mean FROM " + q + " GROUP BY 1 ORDER BY mean DESC");
    pick("Filter rows", "SELECT * FROM " + q + " WHERE " + num_col + " > 0");
    pick("Columns and their types", "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '" + table + "'");
    ImGui::EndPopup();
}

void QueryEditor::RenderResult() {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto& r = last_result_;
    if (!r.table) {
        ImGui::TextColored(t.text_dim, "No results yet. Run a query to see them here.");
        return;
    }
    const long long rows = r.table->num_rows();
    // exact / first N rows, what was read, how long.
    const std::string pill = r.truncated ? "first " + Thousands(rows) + " rows" : "exact";
    ui::StatusPill("##query_label", pill.c_str(), r.truncated ? t.warning : t.success);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(r.truncated ? "The result has more rows than are shown here; Save as Dataset keeps them all."
                                      : "Every row of the result is shown.");
    ImGui::SameLine();
    std::string read;
    for (size_t i = 0; i < r.inputs.size(); ++i) read += (i ? ", " : "") + r.inputs[i];
    ImGui::TextColored(t.text_dim, "%s rows \xC2\xB7 %d columns%s \xC2\xB7 %.0f ms", Thousands(rows).c_str(), r.table->num_columns(),
                       read.empty() ? "" : (" \xC2\xB7 read " + read + " (" + Thousands(static_cast<long long>(r.rows_in_inputs)) + " rows)").c_str(),
                       r.elapsed_ms);
    ImGui::SameLine();
    if (on_open_table && ui::LinkButton("Open in Table Viewer")) on_open_table(DataTableFromArrow(*r.table, "Query result"));
    ImGui::SameLine();
    if (ui::LinkButton("Plot")) OpenPlot();
    ImGui::SameLine();
    if (ui::LinkButton("Save as Dataset")) ImGui::OpenPopup("Save query result");
    if (ImGui::BeginPopupModal("Save query result", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Save the whole result as a new dataset");
        ImGui::SetNextItemWidth(320);
        ImGui::InputText("##save_name", save_name_, sizeof(save_name_));
        if (ui::PrimaryButton("Save", save_name_[0] != '\0', "Give it a name", ui::ButtonSize::Small)) {
            SaveResultAsDataset(save_name_);
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ui::SecondaryButton("Cancel")) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    if (!save_note_.empty()) ImGui::TextColored(t.text_dim, "%s", save_note_.c_str());

    const int cols = r.table->num_columns();
    if (cols <= 0 || cols > 512) {
        ImGui::TextColored(t.text_dim, cols > 512 ? "Too many columns to show here: open it in the Table Viewer." : "No columns.");
        return;
    }
    // On the window background; no grid lines (rows tinted alternately).
    if (ImGui::BeginTable("##query_results", cols,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                              ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoBordersInBody)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        for (const auto& f : r.table->schema()->fields()) ImGui::TableSetupColumn(f->name().c_str());
        ImGui::TableHeadersRow();
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(std::min<long long>(rows, std::numeric_limits<int>::max())));
        while (clipper.Step())
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                ImGui::TableNextRow();
                for (int c = 0; c < cols; ++c) {
                    ImGui::TableSetColumnIndex(c);
                    const std::string cell = ArrowCellText(*r.table->column(c), row);
                    if (r.table->column(c)->null_count() > 0 && cell.empty()) ImGui::TextColored(t.text_faint, "null");
                    else ImGui::TextUnformatted(cell.c_str());
                }
            }
        ImGui::EndTable();
    }
}

}  // namespace cyxwiz
