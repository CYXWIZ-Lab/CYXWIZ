#include "query_editor.h"

#include "../../core/arrow_dataset.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_catalog.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/session_query_service.h"
#include "../dashboard/dashboard_links.h"
#include "../editor_fonts.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <algorithm>
#include <cstdio>
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
    // Completion offers the picked dataset's columns.
    if (columns_for_ != current_dataset_) {
        columns_for_ = current_dataset_;
        std::vector<std::string> columns;
        if (auto schema = SchemaOf(current_dataset_))
            for (const auto& f : schema->fields()) columns.push_back(f->name());
        pane_.SetCompletionColumns(std::move(columns));
    }
    pane_.Refresh();
    pane_.Render(std::clamp(ImGui::GetContentRegionAvail().y * 0.35f, 120.0f, 260.0f));
    // Ctrl+Enter runs while the editor has focus (Preferences > Shortcuts, Data Studio).
    if (pane_.Editor().IsFocused() && ImGui::GetIO().KeyCtrl &&
        (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) && !IsRunning())
        ExecuteQuery();

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
    // The row wraps in a narrow panel: an item that does not fit starts a line.
    const float right_edge = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    const auto next = [right_edge](float width) {
        ImGui::SameLine();
        if (ImGui::GetCursorScreenPos().x + width > right_edge) ImGui::NewLine();
    };
    const auto button_w = [](const char* label) { return ui::ButtonWidth(label, ui::ButtonSize::Small); };
    if (ui::PrimaryButton(ICON_FA_PLAY " Run", !running, running ? "A query is running" : nullptr, ui::ButtonSize::Small)) ExecuteQuery();
    next(button_w("Cancel"));
    if (ui::SecondaryButton("Cancel", running, "Nothing is running")) CancelQuery();
    next(button_w("Turn into SQL step"));
    // Turn into SQL step: the last result's query as a step of the graph.
    {
        std::string why = last_result_.table ? std::string() : "Run the query first";
        if (why.empty() && step_blocker) why = step_blocker(last_result_.inputs);
        else if (why.empty()) why = "No graph is open";
        if (ui::SecondaryButton("Turn into SQL step", why.empty(), why.c_str()) && on_turn_into_step)
            step_note_ = on_turn_into_step(result_sql_, last_result_.inputs);
        if (why.empty() && ImGui::IsItemHovered())
            ImGui::SetTooltip("Adds an SQL step running this query after the step whose result it reads (in its recipe)");
    }
    next(button_w("Examples"));
    if (ui::SecondaryButton("Examples")) ImGui::OpenPopup("##query_examples");
    RenderExamples();
    next(button_w("Format"));
    if (ui::SecondaryButton("Format")) pane_.Format();
    next(button_w("Clear"));
    if (ui::GhostButton("Clear")) SetQuery("");
    next(ImGui::CalcTextSize(running ? ICON_FA_SPINNER " Running... 00.0 s (Task View shows it)" : "Ctrl+Enter").x);
    if (running) {
        ImGui::TextColored(t.info, ICON_FA_SPINNER " Running... %.1f s (Task View shows it)", ImGui::GetTime() - started_at_);
    } else {
        ImGui::TextColored(t.text_faint, "Ctrl+Enter");
    }
    if (!step_note_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "%s", step_note_.c_str());
        ImGui::PopTextWrapPos();
    }
    if (!last_error_.empty()) {
        // DuckDB's errors can list every candidate function: the first line
        // here, the rest under Details.
        const size_t eol = last_error_.find('\n');
        const bool more = eol != std::string::npos && last_error_.find_first_not_of(" \r\n", eol) != std::string::npos;
        ImGui::PushTextWrapPos(0.0f);
        const ImVec4 colour = last_error_ == "Cancelled." ? t.text_dim : t.error;
        ImGui::TextColored(colour, "%s", (error_details_ || !more ? last_error_ : last_error_.substr(0, eol)).c_str());
        ImGui::PopTextWrapPos();
        if (more && ui::LinkButton(error_details_ ? "Hide details##qerr" : "Details##qerr")) error_details_ = !error_details_;
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
        if (ImGui::Selectable(label)) SetQuery(sql);
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
    if (on_open_table && ui::LinkButton("Open in Table Viewer")) on_open_table(r.table, "Query result");
    ImGui::SameLine();
    if (ui::LinkButton("Plot")) OpenPlot();
    ImGui::SameLine();
    // Add to Dashboard: a query widget on the picked dataset (board 15).
    {
        std::string why;
        const dashboard::WidgetSpec widget = dashboard::QueryResultWidget(result_sql_, current_dataset_, r.table->schema(), &why);
        if (ui::LinkButton("Add to Dashboard")) ImGui::OpenPopup("##query_add_to_dashboard");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", why.empty() ? "Adds this query as a widget: it reads the dashboard's rows, so its filters apply."
                                                : why.c_str());
        if (ImGui::BeginPopup("##query_add_to_dashboard")) {
            if (!why.empty()) {
                ImGui::TextColored(t.warning, "%s", why.c_str());
            } else {
                ImGui::TextColored(t.text_dim, "Add \"%s\" (%s) to", widget.title.c_str(), plot::Info(widget.plot.kind).label);
                if (dashboard::DrawAddToDashboardItems(active_registry_, widget)) save_note_ = "Added to the dashboard as a query widget.";
            }
            ImGui::EndPopup();
        }
    }
    ImGui::SameLine();
    if (ui::LinkButton("Save as Dataset")) ImGui::OpenPopup("Save query result");
    if (ImGui::BeginPopupModal("Save query result", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Save the whole result as a new dataset");
        // Honest about where it goes (TOFIX129 D7a): a table for looking, not a node.
        ImGui::PushTextWrapPos(360.0f);
        ImGui::TextColored(t.text_dim, "Kept for Explore and dashboards only. Training reads the graph's steps, not saved tables: "
                                       "use Turn into step for that.");
        ImGui::PopTextWrapPos();
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

    // The Table Viewer's grid on the window background: sort, select, copy.
    // It fills the space left, and keeps about ten rows when little is left
    // (a short Data Studio column): the job area then scrolls to it.
    const float min_grid_h = ImGui::GetTextLineHeightWithSpacing() * 12.0f;
    const float grid_h = ImGui::GetContentRegionAvail().y < min_grid_h ? min_grid_h : 0.0f;
    result_grid_.Render("##query_result_grid", ImVec2(0.0f, grid_h));
}

}  // namespace cyxwiz
