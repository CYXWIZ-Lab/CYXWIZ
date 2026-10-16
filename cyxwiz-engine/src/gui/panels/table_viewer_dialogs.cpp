// Table Viewer: Export (a menu: CSV, TSV, JSON, Parquet to a file, or CSV to
// the clipboard) and Find (a window over the grid's filter and its options).

#include "table_viewer.h"
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
    const DataTable& table = *tab->grid.table;
    std::vector<size_t> rows;
    if (export_shown_rows_only_) {
        rows = tab->grid.DisplayIndices();
    } else {
        rows.resize(table.GetRowCount());
        std::iota(rows.begin(), rows.end(), size_t(0));
    }
    const std::string count = std::to_string(rows.size()) + " rows";
    status_note_error_ = false;
    if (format == 4) {
        ImGui::SetClipboardText(TableText(table, rows, TableTextFormat::Csv, export_headers_).c_str());
        status_note_ = "Copied " + count + " as CSV";
        return;
    }
    static const char* kExt[] = {"csv", "tsv", "json", "parquet"};
    static const char* kName[] = {"CSV", "TSV", "JSON", "Parquet"};
    const FileDialogs::FilterList filters = {FileDialogs::Filter{kName[format], kExt[format]}};
    const std::string suggested = (tab->filename.empty() ? std::string("table") : tab->filename) + "." + kExt[format];
    const auto path = FileDialogs::SaveFile("Export table", filters, nullptr, suggested.c_str());
    if (!path) return;
    bool ok = false;
    if (format == 3) {
        const auto arrow_table = ArrowFromDataTable(*TableRows(table, rows));
        auto out = arrow::io::FileOutputStream::Open(*path);
        ok = arrow_table && out.ok() &&
             parquet::arrow::WriteTable(*arrow_table, arrow::default_memory_pool(), *out, 64 * 1024).ok() &&
             (*out)->Close().ok();
    } else {
        const auto kind = format == 0 ? TableTextFormat::Csv : format == 1 ? TableTextFormat::Tsv : TableTextFormat::Json;
        std::ofstream file(*path, std::ios::binary);
        file << TableText(table, rows, kind, export_headers_);
        ok = static_cast<bool>(file);
    }
    status_note_error_ = !ok;
    status_note_ = ok ? "Exported " + count + " to " + *path : "Could not write " + *path;
    if (ok) spdlog::info("Table Viewer: exported {} to {}", count, *path);
    else spdlog::error("Table Viewer: could not write {}", *path);
}

void TableViewerPanel::RenderFindDialog() {
    if (!show_find_dialog_) return;

    ImGui::SetNextWindowSize(ImVec2(420, 170), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS " Find in Table", &show_find_dialog_)) {
        TableTab* tab = GetActiveTab();
        if (!tab || !tab->grid.table) {
            ImGui::TextDisabled("No table loaded");
            ImGui::End();
            return;
        }
        TableGrid& grid = tab->grid;
        const ui::Tokens& t = ui::CurrentTokens();

        ImGui::SetNextItemWidth(-1);
        bool changed = ImGui::InputTextWithHint("##FindInput", "Text to find", find_buffer_, sizeof(find_buffer_));
        changed |= ImGui::Checkbox("Case sensitive", &grid.filter_case_sensitive);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Whole cell", &grid.filter_whole_cell);

        // The grid's filter: matches are highlighted (Hide in the toolbar
        // shows only their rows); the count is of every cell.
        if (changed) {
            grid.SetFilter(find_buffer_);
            find_cells_ = -1;
            find_rows_ = 0;
            if (find_buffer_[0] != '\0') {
                find_cells_ = 0;
                for (size_t r = 0; r < grid.table->GetRowCount(); r++) {
                    bool row_hit = false;
                    for (size_t c = 0; c < grid.table->GetColumnCount(); c++) {
                        if (grid.Matches(grid.CellText(r, c))) {
                            ++find_cells_;
                            row_hit = true;
                        }
                    }
                    if (row_hit) ++find_rows_;
                }
            }
        }

        ImGui::Spacing();
        if (ui::SecondaryButton(ICON_FA_XMARK " Clear", find_buffer_[0] != '\0', "Nothing to clear")) {
            std::memset(find_buffer_, 0, sizeof(find_buffer_));
            grid.SetFilter("");
            find_cells_ = -1;
        }
        if (find_cells_ >= 0) {
            ImGui::SameLine(0.0f, 14.0f);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(find_cells_ ? t.text : t.text_dim, "%d %s in %zu %s match", find_cells_,
                               find_cells_ == 1 ? "cell" : "cells", find_rows_, find_rows_ == 1 ? "row" : "rows");
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_dim, "Matches are highlighted; tick Hide in the toolbar to show only their rows.");
        ImGui::PopTextWrapPos();
    }
    ImGui::End();
}

}  // namespace cyxwiz
