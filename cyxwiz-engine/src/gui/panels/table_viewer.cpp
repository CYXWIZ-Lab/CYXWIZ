#include "table_viewer.h"
#include "../icons.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz {

TableViewerPanel::TableViewerPanel()
    : Panel("Table Viewer", false)  // Start hidden
{
}

void TableViewerPanel::Render() {
    if (!visible_) return;

    // Handle deferred tab close
    if (close_tab_index_ >= 0 && close_tab_index_ < static_cast<int>(tabs_.size())) {
        tabs_.erase(tabs_.begin() + close_tab_index_);
        if (active_tab_index_ >= static_cast<int>(tabs_.size())) {
            active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
        }
        close_tab_index_ = -1;
    }

    ImGui::Begin(GetName(), &visible_);

    // Tab bar at top
    RenderTabBar();

    ImGui::Separator();

    // Toolbar
    RenderToolbar();

    ImGui::Separator();

    // Table display or loading indicator
    TableTab* active_tab = GetActiveTab();
    if (active_tab) {
        if (active_tab->is_loading) {
            RenderLoadingIndicator();
        } else if (active_tab->table) {
            RenderTable();
        } else {
            ImGui::TextWrapped("Failed to load table.");
        }
    } else {
        ImGui::TextWrapped("No table loaded. Right-click on a data file in Asset Browser and select 'View in Table'.");
    }

    ImGui::Separator();

    // Status bar
    RenderStatusBar();

    ImGui::End();
}

void TableViewerPanel::RenderTabBar() {
    if (tabs_.empty()) {
        ImGui::TextDisabled("No tables open");
        return;
    }

    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_Reorderable |
                                      ImGuiTabBarFlags_AutoSelectNewTabs |
                                      ImGuiTabBarFlags_TabListPopupButton |
                                      ImGuiTabBarFlags_FittingPolicyScroll;

    if (ImGui::BeginTabBar("TableViewerTabs", tab_bar_flags)) {
        for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
            auto& tab = tabs_[i];

            // Tab name with loading indicator
            std::string tab_name = tab->filename;
            if (tab->is_loading) {
                tab_name = ICON_FA_SPINNER " " + tab_name;
            } else {
                tab_name = ICON_FA_TABLE " " + tab_name;
            }

            // Make tab closable
            bool tab_open = true;
            ImGuiTabItemFlags tab_flags = ImGuiTabItemFlags_None;

            if (ImGui::BeginTabItem(tab_name.c_str(), &tab_open, tab_flags)) {
                active_tab_index_ = i;
                ImGui::EndTabItem();
            }

            // Handle tab close
            if (!tab_open) {
                close_tab_index_ = i;
            }
        }
        ImGui::EndTabBar();
    }
}

void TableViewerPanel::RenderToolbar() {
    TableTab* active_tab = GetActiveTab();
    if (!active_tab) return;

    // Options
    ImGui::Checkbox("Line Numbers", &show_line_numbers_);

    ImGui::SameLine();

    // Filter
    ImGui::Text("Filter:");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(200);
    if (ImGui::InputText("##Filter", active_tab->filter_buffer, sizeof(active_tab->filter_buffer))) {
        active_tab->filter_text = active_tab->filter_buffer;
    }

    // Export button
    if (active_tab->table && !active_tab->is_loading) {
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_DOWNLOAD " Export CSV")) {
            std::string export_path = "export_" + active_tab->filename + ".csv";
            if (active_tab->table->SaveToCSV(export_path)) {
                spdlog::info("Table exported to: {}", export_path);
            }
        }
    }

    // Close tab button
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_XMARK " Close")) {
        close_tab_index_ = active_tab_index_;
    }
}

void TableViewerPanel::RenderTable() {
    TableTab* active_tab = GetActiveTab();
    if (!active_tab || !active_tab->table) return;

    auto& table = active_tab->table;
    size_t row_count = table->GetRowCount();
    size_t col_count = table->GetColumnCount();

    if (row_count == 0 || col_count == 0) {
        ImGui::Text("Table is empty");
        return;
    }

    // Calculate pagination
    size_t total_pages = (row_count + rows_per_page_ - 1) / rows_per_page_;
    size_t start_row = active_tab->current_page * rows_per_page_;
    size_t end_row = std::min(start_row + rows_per_page_, row_count);

    // ImGui table flags
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

        const auto& headers = table->GetHeaders();
        for (size_t i = 0; i < col_count; i++) {
            ImGui::TableSetupColumn(headers[i].c_str(), ImGuiTableColumnFlags_WidthStretch);
        }

        ImGui::TableHeadersRow();

        // Render rows with clipper for performance
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
                    ImGui::TextDisabled("%zu", actual_row + 1);
                }

                // Data columns
                for (size_t c = 0; c < col_count; c++) {
                    ImGui::TableSetColumnIndex(col_idx++);

                    std::string cell_text = table->GetCellAsString(actual_row, c);

                    // Apply filter highlighting
                    if (!active_tab->filter_text.empty() &&
                        cell_text.find(active_tab->filter_text) != std::string::npos) {
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "%s", cell_text.c_str());
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

        if (ImGui::Button(ICON_FA_ANGLES_LEFT "##First")) {
            active_tab->current_page = 0;
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CHEVRON_LEFT "##Prev")) {
            if (active_tab->current_page > 0) active_tab->current_page--;
        }
        ImGui::SameLine();

        ImGui::Text("%d / %zu", active_tab->current_page + 1, total_pages);
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_CHEVRON_RIGHT "##Next")) {
            if (active_tab->current_page < static_cast<int>(total_pages) - 1) {
                active_tab->current_page++;
            }
        }
        ImGui::SameLine();

        if (ImGui::Button(ICON_FA_ANGLES_RIGHT "##Last")) {
            active_tab->current_page = static_cast<int>(total_pages) - 1;
        }

        ImGui::SameLine();
        ImGui::Text("(rows %zu - %zu of %zu)", start_row + 1, end_row, row_count);
    }
}

void TableViewerPanel::RenderStatusBar() {
    TableTab* active_tab = GetActiveTab();
    if (active_tab && active_tab->table) {
        ImGui::Text(ICON_FA_TABLE " Rows: %zu | Columns: %zu | File: %s",
                   active_tab->table->GetRowCount(),
                   active_tab->table->GetColumnCount(),
                   active_tab->filename.c_str());
    } else if (active_tab && active_tab->is_loading) {
        ImGui::Text(ICON_FA_SPINNER " Loading: %s", active_tab->filename.c_str());
    } else {
        ImGui::TextDisabled("No table loaded");
    }
}

void TableViewerPanel::RenderLoadingIndicator() {
    TableTab* active_tab = GetActiveTab();
    if (!active_tab) return;

    ImGui::Spacing();
    ImGui::Spacing();

    // Center the loading indicator
    float window_width = ImGui::GetWindowWidth();
    float text_width = ImGui::CalcTextSize(active_tab->load_status.c_str()).x;
    ImGui::SetCursorPosX((window_width - text_width) * 0.5f);

    // Animated spinner
    float time = static_cast<float>(ImGui::GetTime());
    const char* spinner_chars = "|/-\\";
    char spinner = spinner_chars[static_cast<int>(time * 10) % 4];

    ImGui::Text("%c %s", spinner, active_tab->load_status.c_str());

    ImGui::Spacing();

    // Progress bar
    ImGui::SetCursorPosX(window_width * 0.2f);
    ImGui::ProgressBar(active_tab->load_progress, ImVec2(window_width * 0.6f, 0.0f));
}

bool TableViewerPanel::LoadCSV(const std::string& filepath) {
    // Check if already open
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "csv");
    return true;
}

bool TableViewerPanel::LoadTXT(const std::string& filepath, char delimiter) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "txt", delimiter);
    return true;
}

bool TableViewerPanel::LoadHDF5(const std::string& filepath, const std::string& dataset_name) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    // For HDF5, we pass dataset name as part of type
    LoadFileAsync(filepath, "hdf5:" + dataset_name);
    return true;
}

bool TableViewerPanel::LoadExcel(const std::string& filepath, const std::string& sheet_name) {
    if (IsFileOpen(filepath)) {
        FocusTab(filepath);
        return true;
    }

    LoadFileAsync(filepath, "excel:" + sheet_name);
    return true;
}

void TableViewerPanel::LoadFileAsync(const std::string& filepath, const std::string& type, char delimiter) {
    // Create new tab
    auto tab = std::make_unique<TableTab>();
    tab->filepath = filepath;
    tab->filename = fs::path(filepath).filename().string();
    tab->is_loading = true;
    tab->load_progress = 0.0f;
    tab->load_status = "Loading...";

    int tab_index = static_cast<int>(tabs_.size());
    tabs_.push_back(std::move(tab));
    active_tab_index_ = tab_index;

    spdlog::info("Starting async load of: {}", filepath);

    // Capture values for lambda
    std::string path = filepath;
    std::string file_type = type;

    AsyncTaskManager::Instance().RunAsync(
        "Loading: " + fs::path(filepath).filename().string(),
        [this, tab_index, path, file_type, delimiter](LambdaTask& task) {
            task.ReportProgress(0.1f, "Opening file...");

            auto table = std::make_shared<DataTable>();
            bool success = false;

            // Parse type and optional parameter
            std::string main_type = file_type;
            std::string type_param;
            size_t colon_pos = file_type.find(':');
            if (colon_pos != std::string::npos) {
                main_type = file_type.substr(0, colon_pos);
                type_param = file_type.substr(colon_pos + 1);
            }

            task.ReportProgress(0.3f, "Parsing data...");

            if (main_type == "csv") {
                success = table->LoadFromCSV(path);
            } else if (main_type == "txt") {
                success = table->LoadFromTXT(path, delimiter);
            } else if (main_type == "hdf5") {
                success = table->LoadFromHDF5(path, type_param.empty() ? "data" : type_param);
            } else if (main_type == "excel") {
                success = table->LoadFromExcel(path, type_param);
            } else {
                // Default to CSV
                success = table->LoadFromCSV(path);
            }

            task.ReportProgress(0.9f, "Finalizing...");

            if (success) {
                table->SetName(fs::path(path).stem().string());
                task.MarkCompleted();
            } else {
                task.MarkFailed("Failed to parse file");
            }

            // Update tab with result (thread-safe)
            if (tab_index < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[tab_index];
                tab->table = success ? table : nullptr;
                tab->is_loading = false;
                tab->load_progress = 1.0f;
                tab->load_status = success ? "Complete" : "Failed";
            }
        },
        [this, tab_index](float progress, const std::string& status) {
            // Progress callback - update tab
            if (tab_index < static_cast<int>(tabs_.size())) {
                auto& tab = tabs_[tab_index];
                tab->load_progress = progress;
                tab->load_status = status;
            }
        },
        [this, tab_index, path](bool success, const std::string& error) {
            if (success) {
                spdlog::info("Async load completed: {}", path);
            } else {
                spdlog::error("Async load failed: {} - {}", path, error);
            }
        }
    );
}

void TableViewerPanel::SetTable(std::shared_ptr<DataTable> table) {
    if (!table) return;

    auto tab = std::make_unique<TableTab>();
    tab->filename = table->GetName();
    tab->filepath = "";  // In-memory table
    tab->table = table;
    tab->is_loading = false;

    tabs_.push_back(std::move(tab));
    active_tab_index_ = static_cast<int>(tabs_.size()) - 1;
}

void TableViewerPanel::SetTableByName(const std::string& name) {
    auto table = DataTableRegistry::Instance().GetTable(name);
    if (table) {
        SetTable(table);
    } else {
        spdlog::warn("Table not found in registry: {}", name);
    }
}

void TableViewerPanel::CloseCurrentTab() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        CloseTab(active_tab_index_);
    }
}

void TableViewerPanel::CloseTab(int index) {
    if (index >= 0 && index < static_cast<int>(tabs_.size())) {
        close_tab_index_ = index;
    }
}

void TableViewerPanel::CloseAllTabs() {
    tabs_.clear();
    active_tab_index_ = -1;
}

bool TableViewerPanel::IsFileOpen(const std::string& filepath) const {
    return FindTabByPath(filepath) >= 0;
}

void TableViewerPanel::FocusTab(const std::string& filepath) {
    int index = FindTabByPath(filepath);
    if (index >= 0) {
        active_tab_index_ = index;
    }
}

int TableViewerPanel::FindTabByPath(const std::string& filepath) const {
    for (int i = 0; i < static_cast<int>(tabs_.size()); i++) {
        if (tabs_[i]->filepath == filepath) {
            return i;
        }
    }
    return -1;
}

TableViewerPanel::TableTab* TableViewerPanel::GetActiveTab() {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        return tabs_[active_tab_index_].get();
    }
    return nullptr;
}

const TableViewerPanel::TableTab* TableViewerPanel::GetActiveTab() const {
    if (active_tab_index_ >= 0 && active_tab_index_ < static_cast<int>(tabs_.size())) {
        return tabs_[active_tab_index_].get();
    }
    return nullptr;
}

void TableViewerPanel::Clear() {
    CloseAllTabs();
}

} // namespace cyxwiz
