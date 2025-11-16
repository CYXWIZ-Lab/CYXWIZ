#include "table_viewer.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <filesystem>

namespace cyxwiz {

TableViewerPanel::TableViewerPanel()
    : Panel("Table Viewer", true)
    , show_line_numbers_(true)
    , rows_per_page_(100)
    , current_page_(0)
{
    std::memset(filter_buffer_, 0, sizeof(filter_buffer_));
}

void TableViewerPanel::Render() {
    if (!visible_) return;

    ImGui::Begin(GetName(), &visible_);

    // Toolbar
    RenderToolbar();

    ImGui::Separator();

    // Table display
    if (current_table_) {
        RenderTable();
    } else {
        ImGui::TextWrapped("No table loaded. Use File → Open Data File to load CSV, HDF5, or Excel files.");
    }

    ImGui::Separator();

    // Status bar
    RenderStatusBar();

    ImGui::End();
}

void TableViewerPanel::RenderToolbar() {
    // Table selection dropdown
    ImGui::Text("Table:");
    ImGui::SameLine();

    auto table_names = DataTableRegistry::Instance().GetTableNames();
    if (table_names.empty()) {
        ImGui::TextDisabled("(none loaded)");
    } else {
        if (ImGui::BeginCombo("##TableSelect", current_table_name_.c_str())) {
            for (const auto& name : table_names) {
                bool is_selected = (name == current_table_name_);
                if (ImGui::Selectable(name.c_str(), is_selected)) {
                    SetTableByName(name);
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
    }

    ImGui::SameLine();

    // Options
    ImGui::Checkbox("Line Numbers", &show_line_numbers_);

    ImGui::SameLine();

    // Filter
    ImGui::Text("Filter:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::InputText("##Filter", filter_buffer_, sizeof(filter_buffer_))) {
        filter_text_ = filter_buffer_;
    }

    // Export button
    ImGui::SameLine();
    if (ImGui::Button("Export CSV")) {
        if (current_table_) {
            std::string export_path = "export_" + current_table_name_ + ".csv";
            if (current_table_->SaveToCSV(export_path)) {
                spdlog::info("Table exported to: {}", export_path);
            }
        }
    }
}

void TableViewerPanel::RenderTable() {
    if (!current_table_) return;

    size_t row_count = current_table_->GetRowCount();
    size_t col_count = current_table_->GetColumnCount();

    if (row_count == 0 || col_count == 0) {
        ImGui::Text("Table is empty");
        return;
    }

    // Calculate pagination
    size_t total_pages = (row_count + rows_per_page_ - 1) / rows_per_page_;
    size_t start_row = current_page_ * rows_per_page_;
    size_t end_row = std::min(start_row + rows_per_page_, row_count);

    // ImGui table
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
                           ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                           ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable;

    int column_count = static_cast<int>(col_count);
    if (show_line_numbers_) {
        column_count++;
    }

    if (ImGui::BeginTable("DataTable", column_count, flags)) {
        // Setup columns
        if (show_line_numbers_) {
            ImGui::TableSetupColumn("Row", ImGuiTableColumnFlags_WidthFixed, 60.0f);
        }

        const auto& headers = current_table_->GetHeaders();
        for (size_t i = 0; i < col_count; i++) {
            ImGui::TableSetupColumn(headers[i].c_str(), ImGuiTableColumnFlags_WidthStretch);
        }

        ImGui::TableHeadersRow();

        // Render rows
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(end_row - start_row));

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                size_t actual_row = start_row + i;

                ImGui::TableNextRow();

                // Line number column
                int col_idx = 0;
                if (show_line_numbers_) {
                    ImGui::TableSetColumnIndex(col_idx++);
                    ImGui::Text("%zu", actual_row + 1);
                }

                // Data columns
                for (size_t c = 0; c < col_count; c++) {
                    ImGui::TableSetColumnIndex(col_idx++);

                    std::string cell_text = current_table_->GetCellAsString(actual_row, c);

                    // Apply filter
                    if (!filter_text_.empty() && cell_text.find(filter_text_) == std::string::npos) {
                        ImGui::TextDisabled("%s", cell_text.c_str());
                    } else {
                        ImGui::Text("%s", cell_text.c_str());
                    }
                }
            }
        }

        ImGui::EndTable();
    }

    // Pagination controls
    if (total_pages > 1) {
        ImGui::Separator();
        ImGui::Text("Page:");
        ImGui::SameLine();

        if (ImGui::Button("<<")) {
            current_page_ = 0;
        }
        ImGui::SameLine();

        if (ImGui::Button("<")) {
            if (current_page_ > 0) current_page_--;
        }
        ImGui::SameLine();

        ImGui::Text("%zu / %zu", current_page_ + 1, total_pages);
        ImGui::SameLine();

        if (ImGui::Button(">")) {
            if (current_page_ < static_cast<int>(total_pages) - 1) current_page_++;
        }
        ImGui::SameLine();

        if (ImGui::Button(">>")) {
            current_page_ = static_cast<int>(total_pages) - 1;
        }

        ImGui::SameLine();
        ImGui::Text("(showing rows %zu - %zu)", start_row + 1, end_row);
    }
}

void TableViewerPanel::RenderStatusBar() {
    if (current_table_) {
        ImGui::Text("Rows: %zu | Columns: %zu | Name: %s",
                   current_table_->GetRowCount(),
                   current_table_->GetColumnCount(),
                   current_table_name_.c_str());
    } else {
        ImGui::TextDisabled("No table loaded");
    }
}

bool TableViewerPanel::LoadCSV(const std::string& filepath) {
    auto table = std::make_shared<DataTable>();

    if (!table->LoadFromCSV(filepath)) {
        spdlog::error("Failed to load CSV: {}", filepath);
        return false;
    }

    // Extract filename for table name
    std::filesystem::path path(filepath);
    std::string table_name = path.stem().string();
    table->SetName(table_name);

    // Add to registry
    DataTableRegistry::Instance().AddTable(table_name, table);

    // Set as current table
    SetTableByName(table_name);

    spdlog::info("Loaded CSV table: {}", table_name);
    return true;
}

bool TableViewerPanel::LoadHDF5(const std::string& filepath, const std::string& dataset_name) {
    auto table = std::make_shared<DataTable>();

    if (!table->LoadFromHDF5(filepath, dataset_name)) {
        spdlog::error("Failed to load HDF5: {}", filepath);
        return false;
    }

    // Extract filename for table name
    std::filesystem::path path(filepath);
    std::string table_name = path.stem().string() + "_" + dataset_name;
    table->SetName(table_name);

    // Add to registry
    DataTableRegistry::Instance().AddTable(table_name, table);

    // Set as current table
    SetTableByName(table_name);

    spdlog::info("Loaded HDF5 table: {}", table_name);
    return true;
}

bool TableViewerPanel::LoadExcel(const std::string& filepath, const std::string& sheet_name) {
    auto table = std::make_shared<DataTable>();

    if (!table->LoadFromExcel(filepath, sheet_name)) {
        spdlog::error("Failed to load Excel: {}", filepath);
        return false;
    }

    // Extract filename for table name
    std::filesystem::path path(filepath);
    std::string table_name = path.stem().string();
    if (!sheet_name.empty()) {
        table_name += "_" + sheet_name;
    }
    table->SetName(table_name);

    // Add to registry
    DataTableRegistry::Instance().AddTable(table_name, table);

    // Set as current table
    SetTableByName(table_name);

    spdlog::info("Loaded Excel table: {}", table_name);
    return true;
}

void TableViewerPanel::SetTable(std::shared_ptr<DataTable> table) {
    current_table_ = table;
    current_page_ = 0;
    filter_text_.clear();
    std::memset(filter_buffer_, 0, sizeof(filter_buffer_));
}

void TableViewerPanel::SetTableByName(const std::string& name) {
    auto table = DataTableRegistry::Instance().GetTable(name);
    if (table) {
        current_table_ = table;
        current_table_name_ = name;
        current_page_ = 0;
        filter_text_.clear();
        std::memset(filter_buffer_, 0, sizeof(filter_buffer_));
    } else {
        spdlog::warn("Table not found in registry: {}", name);
    }
}

void TableViewerPanel::Clear() {
    current_table_.reset();
    current_table_name_.clear();
    current_page_ = 0;
    filter_text_.clear();
    std::memset(filter_buffer_, 0, sizeof(filter_buffer_));
}

} // namespace cyxwiz
