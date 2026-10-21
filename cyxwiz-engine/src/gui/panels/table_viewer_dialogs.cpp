// Table Viewer: Export (a menu: CSV, TSV, JSON, Parquet to a file, or CSV to
// the clipboard) and Find (a window over the grid's filter and its options).

#include "table_viewer.h"
#include "../../core/async_task_manager.h"
#include "../../core/file_dialogs.h"
#include "../../data/arrow_data_table.h"
#include "../../data/table_text_export.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"

#include <arrow/io/file.h>
#include <imgui.h>
#include <parquet/arrow/writer.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <fstream>
#include <numeric>
#include <string>

namespace cyxwiz {

void TableViewerPanel::RenderExportMenu(TableTab* tab) {
    if (!ImGui::BeginPopup("##export")) return;
    const size_t all = tab->GetRowCount();
    const size_t shown = tab->grid.DisplayIndices().size();
    ImGui::MenuItem("Only the shown rows", nullptr, &export_shown_rows_only_);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%zu of %zu rows, in the grid's order (sorted, filtered)", shown, all);
    ImGui::MenuItem("Column names first (CSV, TSV)", nullptr, &export_headers_);
    ImGui::Separator();
    if (ImGui::MenuItem("CSV...")) ExportTable(tab, 0);
    if (ImGui::MenuItem("TSV...")) ExportTable(tab, 1);
    if (ImGui::MenuItem("JSON...")) ExportTable(tab, 2);
    if (ImGui::MenuItem("Parquet...")) ExportTable(tab, 3);
    ImGui::Separator();
    if (ImGui::MenuItem(ICON_FA_COPY " Copy as CSV")) ExportTable(tab, 4);
    ImGui::EndPopup();
}

void TableViewerPanel::ExportTable(TableTab* tab, int format) {
    if (!tab || !tab->grid.table) return;
    // What the task reads and writes; released when it is done.
    struct Job {
        std::shared_ptr<DataTable> table;
        std::vector<size_t> rows;
        std::shared_ptr<void> hold;  // no cell edits while the table is read
        std::string text;            // Copy as CSV: the text for the clipboard
    };
    auto job = std::make_shared<Job>();
    job->table = tab->grid.table;
    if (export_shown_rows_only_) {
        job->rows = tab->grid.DisplayIndices();
    } else {
        job->rows.resize(job->table->GetRowCount());
        std::iota(job->rows.begin(), job->rows.end(), size_t(0));
    }
    const std::string count = std::to_string(job->rows.size()) + " rows";
    std::string path;
    if (format != 4) {
        static const char* kExt[] = {"csv", "tsv", "json", "parquet"};
        static const char* kName[] = {"CSV", "TSV", "JSON", "Parquet"};
        const FileDialogs::FilterList filters = {FileDialogs::Filter{kName[format], kExt[format]}};
        const std::string suggested = (tab->filename.empty() ? std::string("table") : tab->filename) + "." + kExt[format];
        const auto chosen = FileDialogs::SaveFile("Export table", filters, nullptr, suggested.c_str());
        if (!chosen) return;
        path = *chosen;
    }
    job->hold = tab->grid.HoldTable();
    const bool headers = export_headers_;
    const std::string name = tab->filename.empty() ? std::string("the table") : tab->filename;
    status_note_error_ = false;
    status_note_ = (format == 4 ? "Copying " : "Exporting ") + count + "...";
    std::weak_ptr<int> alive = alive_;
    AsyncTaskManager::Instance().RunAsync(
        format == 4 ? "Copy " + name + " as CSV" : "Export " + name + " to " + path,
        [job, format, path, headers](LambdaTask& task) {
            task.ReportProgress(0.1f, "Writing " + std::to_string(job->rows.size()) + " rows");
            const DataTable& table = *job->table;
            bool ok = false;
            if (format == 4) {
                job->text = TableText(table, job->rows, TableTextFormat::Csv, headers);
                ok = true;
            } else if (format == 3) {
                const auto arrow_table = ArrowFromDataTable(*TableRows(table, job->rows));
                auto out = arrow::io::FileOutputStream::Open(path);
                ok = arrow_table && out.ok() &&
                     parquet::arrow::WriteTable(*arrow_table, arrow::default_memory_pool(), *out, 64 * 1024).ok() &&
                     (*out)->Close().ok();
            } else {
                const auto kind =
                    format == 0 ? TableTextFormat::Csv : format == 1 ? TableTextFormat::Tsv : TableTextFormat::Json;
                std::ofstream file(path, std::ios::binary);
                file << TableText(table, job->rows, kind, headers);
                ok = static_cast<bool>(file);
            }
            job->table.reset();
            job->hold.reset();
            if (ok) task.MarkCompleted(std::to_string(job->rows.size()) + " rows");
            else task.MarkFailed("Could not write " + path);
        },
        nullptr,
        [this, alive, job, format, path, count](bool success, const std::string&) {
            job->table.reset();  // a task stopped before it ran
            job->hold.reset();
            if (alive.expired()) return;
            status_note_error_ = !success;
            if (format == 4) {
                if (success) ImGui::SetClipboardText(job->text.c_str());
                status_note_ = success ? "Copied " + count + " as CSV" : "Could not copy the table";
                job->text.clear();
                return;
            }
            status_note_ = success ? "Exported " + count + " to " + path : "Could not write " + path;
            if (success) spdlog::info("Table Viewer: exported {} to {}", count, path);
            else spdlog::error("Table Viewer: could not write {}", path);
        },
        alive_);
}

void TableViewerPanel::RenderFindDialog() {
    // Only the grid Find is open over counts its matches (every cell read).
    TableTab* active = show_find_dialog_ ? GetActiveTab() : nullptr;
    for (auto& tab : tabs_) {
        const bool count = tab.get() == active;
        if (tab->grid.count_matches == count) continue;
        tab->grid.count_matches = count;
        if (count && !tab->grid.filter_text.empty()) tab->grid.RefreshFilter();
    }
    if (!show_find_dialog_) return;

    ImGui::SetNextWindowSize(ImVec2(420, 170), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS " Find in Table", &show_find_dialog_)) {
        TableTab* tab = active;
        if (!tab || !tab->grid.table) {
            ImGui::TextDisabled("No table loaded");
            ImGui::End();
            return;
        }
        TableGrid& grid = tab->grid;
        const ui::Tokens& t = ui::CurrentTokens();

        ImGui::SetNextItemWidth(-1);
        // The grid's filter, the same text as the toolbar's: matches are
        // highlighted (Hide in the toolbar shows only their rows); the count
        // is of every cell.
        if (ImGui::InputTextWithHint("##FindInput", "Text to find", grid.filter_buffer, sizeof(grid.filter_buffer)))
            grid.SetFilter(grid.filter_buffer);
        bool options = ImGui::Checkbox("Case sensitive", &grid.filter_case_sensitive);
        ImGui::SameLine();
        options |= ImGui::Checkbox("Whole cell", &grid.filter_whole_cell);
        if (options) grid.RefreshFilter();

        ImGui::Spacing();
        if (ui::SecondaryButton(ICON_FA_XMARK " Clear", !grid.filter_text.empty(), "Nothing to clear"))
            grid.SetFilter("");
        if (grid.Busy()) {
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_dim, "Counting... (the progress is over the grid)");
        } else if (grid.match_cells >= 0) {
            const int64_t cells = grid.match_cells, rows = grid.match_rows;
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(cells ? t.text : t.text_dim, "%lld %s in %lld %s match", static_cast<long long>(cells),
                               cells == 1 ? "cell" : "cells", static_cast<long long>(rows), rows == 1 ? "row" : "rows");
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Matches are highlighted; tick Hide in the toolbar to show only their rows.");
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
}

}  // namespace cyxwiz
