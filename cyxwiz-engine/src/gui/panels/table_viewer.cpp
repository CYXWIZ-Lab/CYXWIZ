#include "table_viewer.h"
#include "../icons.h"
#include <imgui.h>
#include <implot.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>

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
            // 3-pane layout: sidebar + splitter + main table
            if (show_stats_sidebar_) {
                RenderStatsSidebar(active_tab);
                ImGui::SameLine();

                // Draggable splitter
                ImGui::Button("##vsplitter", ImVec2(4.0f, -1));
                if (ImGui::IsItemActive()) {
                    stats_sidebar_width_ += ImGui::GetIO().MouseDelta.x;
                    stats_sidebar_width_ = std::clamp(stats_sidebar_width_, 120.0f, 300.0f);
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                }
                ImGui::SameLine();
            }

            // Main table area
            ImGui::BeginChild("TableContent", ImVec2(0, -30));
            RenderTable();
            ImGui::EndChild();
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

    // Stats sidebar toggle
    if (ImGui::Button(show_stats_sidebar_ ? ICON_FA_CHART_BAR " Stats" : ICON_FA_CHART_BAR)) {
        show_stats_sidebar_ = !show_stats_sidebar_;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Statistics Sidebar");

    ImGui::SameLine();
    ImGui::Checkbox("Data Bars", &show_data_bars_);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show data bars in numeric columns");

    ImGui::SameLine();
    ImGui::Checkbox("Line #", &show_line_numbers_);

    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    if (ImGui::InputInt("##RowsPerPage", &rows_per_page_, 0, 0)) {
        rows_per_page_ = std::clamp(rows_per_page_, 10, 1000);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rows per page");

    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Filter
    ImGui::Text(ICON_FA_FILTER);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(150);
    bool filter_changed = ImGui::InputText("##Filter", active_tab->filter_buffer, sizeof(active_tab->filter_buffer),
        ImGuiInputTextFlags_EnterReturnsTrue);
    if (filter_changed) {
        active_tab->filter_text = active_tab->filter_buffer;
        if (active_tab->filter_mode_hide) {
            ApplyFilter(active_tab);
        }
    }

    // Filter mode toggle
    ImGui::SameLine();
    if (ImGui::Checkbox("Hide", &active_tab->filter_mode_hide)) {
        if (active_tab->filter_mode_hide && !active_tab->filter_text.empty()) {
            ApplyFilter(active_tab);
        } else {
            active_tab->filtered_indices.clear();
        }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hide non-matching rows instead of highlighting");

    // Clear filter button
    if (!active_tab->filter_text.empty()) {
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_XMARK "##ClearFilter")) {
            ClearFilter(active_tab);
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Clear filter");
    }

    // Export button
    if (active_tab->table && !active_tab->is_loading) {
        ImGui::SameLine();
        if (ImGui::Button(ICON_FA_DOWNLOAD " Export")) {
            std::string export_path = "export_" + active_tab->filename + ".csv";
            if (active_tab->table->SaveToCSV(export_path)) {
                spdlog::info("Table exported to: {}", export_path);
            }
        }
    }

    // Close tab button
    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_XMARK)) {
        close_tab_index_ = active_tab_index_;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Close Tab");
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

    // Compute column stats if not done
    if (active_tab->column_stats.empty()) {
        ComputeColumnStats(active_tab);
    }

    // Initialize sorted indices if empty
    if (active_tab->sorted_indices.empty()) {
        active_tab->sorted_indices.resize(row_count);
        std::iota(active_tab->sorted_indices.begin(), active_tab->sorted_indices.end(), size_t(0));
    }

    // Determine which indices to display (sorted or filtered)
    const std::vector<size_t>& display_indices = (active_tab->filter_mode_hide && !active_tab->filtered_indices.empty())
        ? active_tab->filtered_indices
        : active_tab->sorted_indices;

    size_t display_count = display_indices.size();

    // Calculate pagination based on display count
    size_t total_pages = (display_count + rows_per_page_ - 1) / rows_per_page_;
    if (total_pages == 0) total_pages = 1;
    size_t start_row = active_tab->current_page * rows_per_page_;
    size_t end_row = std::min(start_row + rows_per_page_, display_count);

    // ImGui table flags
    ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                           ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX |
                           ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
                           ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable |
                           ImGuiTableFlags_SizingFixedFit;

    int column_count = static_cast<int>(col_count);
    if (show_line_numbers_) {
        column_count++;
    }

    if (ImGui::BeginTable("DataTable", column_count, flags)) {
        // Setup columns with type indicators and sort arrows
        if (show_line_numbers_) {
            ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_DefaultSort, 50.0f);
        }

        const auto& headers = table->GetHeaders();
        for (size_t i = 0; i < col_count; i++) {
            std::string header = headers[i];

            // Add type indicator
            if (i < active_tab->column_stats.size()) {
                auto& stats = active_tab->column_stats[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            ImGui::TableSetupColumn(header.c_str(), ImGuiTableColumnFlags_DefaultSort);
        }

        // ═══════════════════════════════════════════════════════════
        // Manual header row with context menu support
        // ═══════════════════════════════════════════════════════════
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

        // Line number column header (if enabled)
        if (show_line_numbers_) {
            ImGui::TableSetColumnIndex(0);
            ImGui::TableHeader("#");
        }

        // Data column headers with context menus
        for (size_t i = 0; i < col_count; i++) {
            int table_col = show_line_numbers_ ? static_cast<int>(i) + 1 : static_cast<int>(i);
            ImGui::TableSetColumnIndex(table_col);

            // Build header text with type indicator
            std::string header = headers[i];
            if (i < active_tab->column_stats.size()) {
                auto& stats = active_tab->column_stats[i];
                if (stats.type == "Numeric") {
                    header = ICON_FA_HASHTAG " " + header;
                } else {
                    header = ICON_FA_FONT " " + header;
                }
            }

            // Render clickable header
            ImGui::TableHeader(header.c_str());

            // Right-click context menu on header
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                context_menu_column_ = static_cast<int>(i);
            }
            if (ImGui::BeginPopupContextItem(("ColumnContextMenu_" + std::to_string(i)).c_str())) {
                RenderColumnContextMenu(active_tab, static_cast<int>(i));
                ImGui::EndPopup();
            }
        }

        // Handle ImGui's built-in sorting
        if (ImGuiTableSortSpecs* sort_specs = ImGui::TableGetSortSpecs()) {
            if (sort_specs->SpecsDirty && sort_specs->SpecsCount > 0) {
                const ImGuiTableColumnSortSpecs& spec = sort_specs->Specs[0];
                int sort_col = spec.ColumnIndex;
                if (show_line_numbers_) sort_col--;  // Adjust for line number column

                if (sort_col < 0) {
                    // Sorting by line number column - reset to original order
                    active_tab->sort_column = -1;
                    active_tab->sorted_indices.resize(row_count);
                    if (spec.SortDirection == ImGuiSortDirection_Ascending) {
                        std::iota(active_tab->sorted_indices.begin(), active_tab->sorted_indices.end(), size_t(0));
                    } else {
                        // Reverse order
                        for (size_t i = 0; i < row_count; i++) {
                            active_tab->sorted_indices[i] = row_count - 1 - i;
                        }
                    }
                    spdlog::info("Reset to original order ({})",
                        spec.SortDirection == ImGuiSortDirection_Ascending ? "ascending" : "descending");
                } else {
                    active_tab->sort_column = sort_col;
                    active_tab->sort_ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);
                    SortByColumn(active_tab, sort_col);
                }
                sort_specs->SpecsDirty = false;
            }
        }

        // Auto-select first column if none selected
        if (active_tab->selected_column < 0 && !active_tab->column_stats.empty()) {
            active_tab->selected_column = 0;
        }

        // Render rows with clipper for performance using display indices
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(end_row - start_row));

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                size_t display_idx = start_row + i;
                size_t actual_row = display_indices[display_idx];

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

                    // Check for colormap on this column
                    bool has_colormap = (c < active_tab->column_colormaps.size() &&
                                        active_tab->column_colormaps[c].type != ColorMapType::None);

                    // Data bar or colormap for numeric columns
                    if ((show_data_bars_ || has_colormap) && c < active_tab->column_stats.size()) {
                        auto& stats = active_tab->column_stats[c];
                        if (stats.type == "Numeric" && stats.max_val > stats.min_val) {
                            auto cell = table->GetCell(actual_row, c);
                            double val = 0;
                            if (std::holds_alternative<double>(cell)) {
                                val = std::get<double>(cell);
                            } else if (std::holds_alternative<int64_t>(cell)) {
                                val = static_cast<double>(std::get<int64_t>(cell));
                            }

                            float norm = static_cast<float>((val - stats.min_val) / (stats.max_val - stats.min_val));
                            norm = std::clamp(norm, 0.0f, 1.0f);

                            ImVec2 pos = ImGui::GetCursorScreenPos();
                            float cell_width = ImGui::GetContentRegionAvail().x;
                            float cell_height = ImGui::GetTextLineHeight();

                            if (has_colormap) {
                                // Use colormap for full cell background
                                auto& cmap = active_tab->column_colormaps[c];
                                ImVec4 color = GetColorMapColor(norm, cmap.type);
                                color.w = 0.4f;  // Semi-transparent background
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + cell_width, pos.y + cell_height),
                                    ImGui::GetColorU32(color));
                            } else if (show_data_bars_) {
                                // Default data bar
                                float bar_width = cell_width * norm;
                                ImU32 bar_color = ImGui::GetColorU32(ImVec4(0.2f, 0.5f, 0.8f, data_bar_alpha_));
                                ImGui::GetWindowDrawList()->AddRectFilled(
                                    pos, ImVec2(pos.x + bar_width, pos.y + cell_height),
                                    bar_color);
                            }
                        }
                    }

                    // Use Selectable for clickable cells with unique ID
                    bool is_selected = (active_tab->selected_row == static_cast<int>(actual_row) &&
                                       active_tab->selected_col == static_cast<int>(c));

                    // Apply filter highlighting color
                    bool has_filter_match = !active_tab->filter_text.empty() &&
                        cell_text.find(active_tab->filter_text) != std::string::npos;

                    if (has_filter_match) {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
                    }

                    // Push unique ID for each cell to avoid conflicts
                    ImGui::PushID(static_cast<int>(actual_row * col_count + c));

                    // Selectable cell - this makes it clickable (no SpanAllColumns for individual cell selection)
                    if (ImGui::Selectable(cell_text.c_str(), is_selected, ImGuiSelectableFlags_None)) {
                        active_tab->selected_row = static_cast<int>(actual_row);
                        active_tab->selected_col = static_cast<int>(c);
                        active_tab->selected_column = static_cast<int>(c);  // Update stats sidebar
                        spdlog::info("Selected cell: Row {}, Col {}", actual_row + 1, c + 1);
                    }

                    // Cell context menu (right-click on cell) - using BeginPopupContextItem
                    if (ImGui::BeginPopupContextItem()) {
                        RenderCellContextMenu(active_tab, static_cast<int>(actual_row), static_cast<int>(c));
                        ImGui::EndPopup();
                    }

                    ImGui::PopID();

                    if (has_filter_match) {
                        ImGui::PopStyleColor();
                    }
                }
            }
        }

        ImGui::EndTable();
    }

    // Render quick plot popup
    if (show_plot_popup_) {
        RenderQuickPlot();
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
        if (active_tab->filter_mode_hide && !active_tab->filtered_indices.empty()) {
            ImGui::Text("(rows %zu - %zu of %zu | %zu filtered)",
                start_row + 1, end_row, display_count, row_count - display_count);
        } else {
            ImGui::Text("(rows %zu - %zu of %zu)", start_row + 1, end_row, display_count);
        }
    }
}

void TableViewerPanel::RenderStatusBar() {
    TableTab* tab = GetActiveTab();
    if (!tab) {
        ImGui::TextDisabled("No table loaded");
        return;
    }

    if (tab->is_loading) {
        ImGui::Text(ICON_FA_SPINNER " Loading: %s", tab->filename.c_str());
        return;
    }

    if (!tab->table) {
        ImGui::TextDisabled("Failed to load table");
        return;
    }

    auto& table = tab->table;

    // Left: Table info
    ImGui::Text(ICON_FA_TABLE " %zu rows x %zu cols",
        table->GetRowCount(), table->GetColumnCount());

    // Middle: Selection info
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text("Cell: Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }

    // Right: Memory estimate
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();
    size_t mem_bytes = table->GetRowCount() * table->GetColumnCount() * sizeof(double);
    if (mem_bytes > 1024 * 1024) {
        ImGui::Text(ICON_FA_MEMORY " %.1f MB", mem_bytes / (1024.0 * 1024.0));
    } else {
        ImGui::Text(ICON_FA_MEMORY " %.1f KB", mem_bytes / 1024.0);
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

void TableViewerPanel::ComputeColumnStats(TableTab* tab) {
    if (!tab || !tab->table) return;
    auto& table = tab->table;
    size_t cols = table->GetColumnCount();
    size_t rows = table->GetRowCount();

    tab->column_stats.resize(cols);

    for (size_t c = 0; c < cols; c++) {
        auto& stats = tab->column_stats[c];
        stats.count = rows;
        stats.null_count = 0;

        bool has_numeric = false;
        double min_v = std::numeric_limits<double>::max();
        double max_v = std::numeric_limits<double>::lowest();
        double sum_v = 0;
        std::vector<double> numeric_values;  // For std dev, median, histogram
        std::map<std::string, int> value_counts;

        numeric_values.reserve(rows);

        for (size_t r = 0; r < rows; r++) {
            auto cell = table->GetCell(r, c);
            if (std::holds_alternative<std::monostate>(cell)) {
                stats.null_count++;
            } else if (std::holds_alternative<double>(cell)) {
                double v = std::get<double>(cell);
                if (!std::isnan(v) && !std::isinf(v)) {
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    sum_v += v;
                    numeric_values.push_back(v);
                    has_numeric = true;
                } else {
                    stats.null_count++;
                }
            } else if (std::holds_alternative<int64_t>(cell)) {
                double v = static_cast<double>(std::get<int64_t>(cell));
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                sum_v += v;
                numeric_values.push_back(v);
                has_numeric = true;
            } else {
                std::string s = table->GetCellAsString(r, c);
                value_counts[s]++;
            }
        }

        size_t numeric_count = numeric_values.size();
        if (has_numeric && numeric_count > 0) {
            stats.type = "Numeric";
            stats.min_val = min_v;
            stats.max_val = max_v;
            stats.sum = sum_v;
            stats.avg = sum_v / numeric_count;

            // Compute standard deviation
            double sq_sum = 0;
            for (double v : numeric_values) {
                sq_sum += (v - stats.avg) * (v - stats.avg);
            }
            stats.std_dev = std::sqrt(sq_sum / numeric_count);

            // Compute median and quartiles (sort a copy)
            std::vector<double> sorted_vals = numeric_values;
            std::sort(sorted_vals.begin(), sorted_vals.end());
            size_t n = sorted_vals.size();
            stats.median = sorted_vals[n / 2];
            stats.q1 = sorted_vals[n / 4];
            stats.q3 = sorted_vals[3 * n / 4];

            // Compute histogram bins (16 bins)
            constexpr int NUM_BINS = 16;
            stats.histogram_bins.resize(NUM_BINS, 0);
            double range = max_v - min_v;
            if (range > 0) {
                for (double v : numeric_values) {
                    int bin = static_cast<int>((v - min_v) / range * (NUM_BINS - 1));
                    bin = std::clamp(bin, 0, NUM_BINS - 1);
                    stats.histogram_bins[bin]++;
                }
            } else {
                // All values are the same
                stats.histogram_bins[NUM_BINS / 2] = static_cast<int>(numeric_count);
            }
        } else {
            stats.type = "Text";
            // Get top 5 values
            std::vector<std::pair<std::string, int>> sorted_values(
                value_counts.begin(), value_counts.end());
            std::sort(sorted_values.begin(), sorted_values.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
            size_t top_n = std::min(size_t(5), sorted_values.size());
            stats.top_values.assign(sorted_values.begin(), sorted_values.begin() + top_n);
        }
        stats.computed = true;
    }

    spdlog::info("Computed column stats for {} columns", cols);
}

void TableViewerPanel::SortByColumn(TableTab* tab, int column) {
    if (!tab || !tab->table || column < 0) return;

    // sort_column and sort_ascending are set by the caller (from ImGui sort specs)
    auto& table = tab->table;
    size_t rows = table->GetRowCount();

    tab->sorted_indices.resize(rows);
    std::iota(tab->sorted_indices.begin(), tab->sorted_indices.end(), size_t(0));

    bool is_numeric = (column < static_cast<int>(tab->column_stats.size()) &&
                       tab->column_stats[column].type == "Numeric");
    bool ascending = tab->sort_ascending;

    std::sort(tab->sorted_indices.begin(), tab->sorted_indices.end(),
        [&](size_t a, size_t b) {
            if (is_numeric) {
                auto get_val = [&](size_t r) -> double {
                    auto cell = table->GetCell(r, column);
                    if (std::holds_alternative<double>(cell))
                        return std::get<double>(cell);
                    if (std::holds_alternative<int64_t>(cell))
                        return static_cast<double>(std::get<int64_t>(cell));
                    return ascending ? std::numeric_limits<double>::max()
                                    : std::numeric_limits<double>::lowest();
                };
                double va = get_val(a), vb = get_val(b);
                return ascending ? va < vb : va > vb;
            } else {
                std::string sa = table->GetCellAsString(a, column);
                std::string sb = table->GetCellAsString(b, column);
                return ascending ? sa < sb : sa > sb;
            }
        });

    spdlog::info("Sorted by column {} ({})", column, ascending ? "ascending" : "descending");
}

void TableViewerPanel::RenderStatsSidebar(TableTab* tab) {
    ImGui::BeginChild("StatsSidebar", ImVec2(stats_sidebar_width_, 0), true);

    // Cell selection info at top
    ImGui::TextDisabled("Selection");
    if (tab->selected_row >= 0 && tab->selected_col >= 0) {
        ImGui::Text(ICON_FA_CROSSHAIRS " Row %d, Col %d", tab->selected_row + 1, tab->selected_col + 1);
    } else {
        ImGui::TextDisabled("No cell selected");
    }
    ImGui::Separator();
    ImGui::Spacing();

    if (tab->selected_column < 0 || tab->selected_column >= static_cast<int>(tab->column_stats.size())) {
        ImGui::TextDisabled("Click a cell to view");
        ImGui::TextDisabled("column statistics");
        ImGui::EndChild();
        return;
    }

    auto& stats = tab->column_stats[tab->selected_column];
    auto& headers = tab->table->GetHeaders();

    // Column name with icon
    ImGui::Text(ICON_FA_CHART_COLUMN " %s", headers[tab->selected_column].c_str());
    ImGui::Separator();

    // Type badge
    ImVec4 type_color = (stats.type == "Numeric")
        ? ImVec4(0.2f, 0.6f, 0.9f, 1.0f)  // Blue
        : ImVec4(0.9f, 0.6f, 0.2f, 1.0f);  // Orange
    ImGui::TextColored(type_color, "%s %s",
        stats.type == "Numeric" ? ICON_FA_HASHTAG : ICON_FA_FONT,
        stats.type.c_str());

    ImGui::Spacing();

    // Count info
    ImGui::TextDisabled("Count");
    ImGui::SameLine(90);
    ImGui::Text("%zu", stats.count);

    ImGui::TextDisabled("Nulls");
    ImGui::SameLine(90);
    ImGui::Text("%zu", stats.null_count);

    ImGui::Separator();

    if (stats.type == "Numeric") {
        ImGui::TextDisabled("Min");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.min_val);

        ImGui::TextDisabled("Max");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.max_val);

        ImGui::TextDisabled("Mean");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.avg);

        ImGui::TextDisabled("Std Dev");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.std_dev);

        ImGui::TextDisabled("Median");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.median);

        ImGui::Separator();

        // Percentiles
        ImGui::TextDisabled("Percentiles:");
        ImGui::TextDisabled("  25%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.q1);
        ImGui::TextDisabled("  50%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.median);
        ImGui::TextDisabled("  75%%");
        ImGui::SameLine(90);
        ImGui::Text("%.4g", stats.q3);

        ImGui::Separator();

        // Mini histogram
        ImGui::TextDisabled("Distribution:");
        RenderMiniHistogram(tab, tab->selected_column);

        ImGui::Separator();

        // Quick plot button
        if (ImGui::Button(ICON_FA_CHART_BAR " Plot", ImVec2(-1, 0))) {
            plot_popup_.type = QuickPlotType::Histogram;
            plot_popup_.x_column = tab->selected_column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Histogram of " + headers[tab->selected_column];
            plot_popup_.x_data = GetColumnAsDoubles(tab, tab->selected_column);
            show_plot_popup_ = true;
        }
    } else {
        ImGui::TextDisabled("Top Values:");
        for (auto& [val, cnt] : stats.top_values) {
            std::string display = val.length() > 15 ? val.substr(0, 15) + "..." : val;
            ImGui::BulletText("%s (%d)", display.c_str(), cnt);
        }
    }

    ImGui::EndChild();
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

// ============================================================================
// Context Menus
// ============================================================================

void TableViewerPanel::RenderColumnContextMenu(TableTab* tab, int column) {
    if (!tab || column < 0) return;

    // Note: This function is called from within BeginPopupContextItem, so no BeginPopup needed
    const auto& headers = tab->table->GetHeaders();
    std::string col_name = (column < static_cast<int>(headers.size())) ? headers[column] : "Column";

    ImGui::TextDisabled("%s %s", ICON_FA_TABLE_COLUMNS, col_name.c_str());
    ImGui::Separator();

    // Sorting
    if (ImGui::MenuItem(ICON_FA_SORT_UP " Sort Ascending")) {
        tab->sort_column = column;
        tab->sort_ascending = true;
        SortByColumn(tab, column);
    }
    if (ImGui::MenuItem(ICON_FA_SORT_DOWN " Sort Descending")) {
        tab->sort_column = column;
        tab->sort_ascending = false;
        SortByColumn(tab, column);
    }

    ImGui::Separator();

    // Filtering
    if (ImGui::MenuItem(ICON_FA_FILTER " Filter by Value...")) {
        // Focus the filter input
        tab->selected_column = column;
    }
    if (ImGui::MenuItem(ICON_FA_CIRCLE_XMARK " Clear Filter", nullptr, false, !tab->filter_text.empty())) {
        ClearFilter(tab);
    }

    ImGui::Separator();

    // Colormaps submenu
    if (ImGui::BeginMenu(ICON_FA_PALETTE " Apply Colormap")) {
        if (ImGui::MenuItem("Viridis", nullptr,
            column < static_cast<int>(tab->column_colormaps.size()) &&
            tab->column_colormaps[column].type == ColorMapType::Viridis)) {
            ApplyColorMap(tab, column, ColorMapType::Viridis);
        }
        if (ImGui::MenuItem("Inferno", nullptr,
            column < static_cast<int>(tab->column_colormaps.size()) &&
            tab->column_colormaps[column].type == ColorMapType::Inferno)) {
            ApplyColorMap(tab, column, ColorMapType::Inferno);
        }
        if (ImGui::MenuItem("RdYlGn (Red-Yellow-Green)", nullptr,
            column < static_cast<int>(tab->column_colormaps.size()) &&
            tab->column_colormaps[column].type == ColorMapType::RdYlGn)) {
            ApplyColorMap(tab, column, ColorMapType::RdYlGn);
        }
        if (ImGui::MenuItem("Blues", nullptr,
            column < static_cast<int>(tab->column_colormaps.size()) &&
            tab->column_colormaps[column].type == ColorMapType::Blues)) {
            ApplyColorMap(tab, column, ColorMapType::Blues);
        }
        if (ImGui::MenuItem("Plasma", nullptr,
            column < static_cast<int>(tab->column_colormaps.size()) &&
            tab->column_colormaps[column].type == ColorMapType::Plasma)) {
            ApplyColorMap(tab, column, ColorMapType::Plasma);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_XMARK " Clear Colormap")) {
            ClearColorMap(tab, column);
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    // Plotting
    if (ImGui::BeginMenu(ICON_FA_CHART_SIMPLE " Plot")) {
        if (ImGui::MenuItem(ICON_FA_CHART_BAR " Histogram")) {
            plot_popup_.type = QuickPlotType::Histogram;
            plot_popup_.x_column = column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Histogram of " + col_name;
            plot_popup_.x_data = GetColumnAsDoubles(tab, column);
            show_plot_popup_ = true;
        }
        if (ImGui::MenuItem(ICON_FA_CHART_LINE " Line Chart")) {
            plot_popup_.type = QuickPlotType::Line;
            plot_popup_.x_column = column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Line Chart of " + col_name;
            plot_popup_.x_data = GetColumnAsDoubles(tab, column);
            show_plot_popup_ = true;
        }
        if (ImGui::MenuItem(ICON_FA_CHART_COLUMN " Bar Chart")) {
            plot_popup_.type = QuickPlotType::Bar;
            plot_popup_.x_column = column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Bar Chart of " + col_name;
            plot_popup_.x_data = GetColumnAsDoubles(tab, column);
            show_plot_popup_ = true;
        }
        if (ImGui::MenuItem(ICON_FA_CUBE " Box Plot")) {
            plot_popup_.type = QuickPlotType::Box;
            plot_popup_.x_column = column;
            plot_popup_.y_column = -1;
            plot_popup_.title = "Box Plot of " + col_name;
            plot_popup_.x_data = GetColumnAsDoubles(tab, column);
            show_plot_popup_ = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_CHART_SCATTER " Use as X-axis for Scatter")) {
            plot_popup_.type = QuickPlotType::Scatter;
            plot_popup_.x_column = column;
            // Y column will be selected in popup
            plot_popup_.title = "Scatter Plot";
            plot_popup_.x_data = GetColumnAsDoubles(tab, column);
            show_plot_popup_ = true;
        }
        ImGui::EndMenu();
    }

    ImGui::Separator();

    // Copy
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Column")) {
        CopyColumnToClipboard(tab, column);
    }
}

void TableViewerPanel::RenderCellContextMenu(TableTab* tab, int row, int col) {
    if (!tab || row < 0 || col < 0) return;

    // Note: This function is called from within BeginPopupContextItem, so no BeginPopup needed
    std::string cell_value = tab->table->GetCellAsString(row, col);
    std::string preview = cell_value.length() > 20 ? cell_value.substr(0, 20) + "..." : cell_value;

    ImGui::TextDisabled("%s [%d, %d] = %s", ICON_FA_TABLE_CELLS, row + 1, col + 1, preview.c_str());
    ImGui::Separator();

    // Copy operations
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Cell")) {
        CopyCellToClipboard(tab, row, col);
    }
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Row")) {
        CopyRowToClipboard(tab, row);
    }
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Selection", nullptr, false, !tab->selections.empty())) {
        CopySelectionToClipboard(tab);
    }

    ImGui::Separator();

    // Selection
    if (ImGui::MenuItem(ICON_FA_BARS " Select Entire Row")) {
        tab->selected_row = row;
        tab->selected_col = -1;  // Entire row
    }
    if (ImGui::MenuItem(ICON_FA_TABLE_COLUMNS " Select Entire Column")) {
        tab->selected_column = col;
        tab->selected_col = col;
        tab->selected_row = -1;  // Entire column
    }

    ImGui::Separator();

    // Filtering
    if (ImGui::MenuItem(ICON_FA_FILTER " Filter by This Value")) {
        tab->filter_text = cell_value;
        std::strncpy(tab->filter_buffer, cell_value.c_str(), sizeof(tab->filter_buffer) - 1);
        ApplyFilter(tab);
    }

    // Column stats
    if (ImGui::MenuItem(ICON_FA_CHART_SIMPLE " View Column Statistics")) {
        tab->selected_column = col;
    }
}

// ============================================================================
// Colormaps
// ============================================================================

void TableViewerPanel::ApplyColorMap(TableTab* tab, int column, ColorMapType type) {
    if (!tab || column < 0) return;

    // Ensure vector is sized
    if (tab->column_colormaps.size() <= static_cast<size_t>(column)) {
        tab->column_colormaps.resize(column + 1);
    }

    auto& cmap = tab->column_colormaps[column];
    cmap.type = type;
    cmap.auto_range = true;

    // Get column min/max for normalization
    if (column < static_cast<int>(tab->column_stats.size())) {
        cmap.min_val = static_cast<float>(tab->column_stats[column].min_val);
        cmap.max_val = static_cast<float>(tab->column_stats[column].max_val);
    }

    spdlog::info("Applied colormap {} to column {}", static_cast<int>(type), column);
}

void TableViewerPanel::ClearColorMap(TableTab* tab, int column) {
    if (!tab || column < 0) return;

    if (static_cast<size_t>(column) < tab->column_colormaps.size()) {
        tab->column_colormaps[column].type = ColorMapType::None;
    }
}

ImVec4 TableViewerPanel::GetColorMapColor(double normalized, ColorMapType type) const {
    // Clamp to [0, 1]
    float t = std::clamp(static_cast<float>(normalized), 0.0f, 1.0f);

    switch (type) {
        case ColorMapType::Viridis: {
            // Purple → Blue → Green → Yellow
            if (t < 0.25f) {
                float s = t / 0.25f;
                return ImVec4(
                    0.267f + s * (0.192f - 0.267f),
                    0.004f + s * (0.408f - 0.004f),
                    0.329f + s * (0.557f - 0.329f),
                    1.0f);
            } else if (t < 0.50f) {
                float s = (t - 0.25f) / 0.25f;
                return ImVec4(
                    0.192f + s * (0.208f - 0.192f),
                    0.408f + s * (0.718f - 0.408f),
                    0.557f + s * (0.475f - 0.557f),
                    1.0f);
            } else if (t < 0.75f) {
                float s = (t - 0.50f) / 0.25f;
                return ImVec4(
                    0.208f + s * (0.565f - 0.208f),
                    0.718f + s * (0.820f - 0.718f),
                    0.475f + s * (0.251f - 0.475f),
                    1.0f);
            } else {
                float s = (t - 0.75f) / 0.25f;
                return ImVec4(
                    0.565f + s * (0.992f - 0.565f),
                    0.820f + s * (0.906f - 0.820f),
                    0.251f + s * (0.145f - 0.251f),
                    1.0f);
            }
        }

        case ColorMapType::Inferno: {
            // Black → Magenta → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(s * 0.737f, s * 0.216f, s * 0.329f, 1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.737f + s * (0.976f - 0.737f),
                    0.216f + s * (0.557f - 0.216f),
                    0.329f + s * (0.035f - 0.329f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.976f + s * (0.988f - 0.976f),
                    0.557f + s * (0.996f - 0.557f),
                    0.035f + s * (0.643f - 0.035f),
                    1.0f);
            }
        }

        case ColorMapType::RdYlGn: {
            // Red → Yellow → Green (diverging)
            if (t < 0.5f) {
                float s = t / 0.5f;
                return ImVec4(
                    0.843f + s * (0.996f - 0.843f),
                    0.188f + s * (0.878f - 0.188f),
                    0.153f + s * (0.545f - 0.153f),
                    1.0f);
            } else {
                float s = (t - 0.5f) / 0.5f;
                return ImVec4(
                    0.996f + s * (0.102f - 0.996f),
                    0.878f + s * (0.596f - 0.878f),
                    0.545f + s * (0.314f - 0.545f),
                    1.0f);
            }
        }

        case ColorMapType::Blues: {
            // Light blue → Dark blue
            return ImVec4(
                0.031f + (1.0f - t) * 0.937f,
                0.188f + (1.0f - t) * 0.757f,
                0.420f + (1.0f - t) * 0.525f,
                1.0f);
        }

        case ColorMapType::Plasma: {
            // Purple → Pink → Orange → Yellow
            if (t < 0.33f) {
                float s = t / 0.33f;
                return ImVec4(
                    0.050f + s * (0.798f - 0.050f),
                    0.030f + s * (0.280f - 0.030f),
                    0.528f + s * (0.469f - 0.528f),
                    1.0f);
            } else if (t < 0.66f) {
                float s = (t - 0.33f) / 0.33f;
                return ImVec4(
                    0.798f + s * (0.973f - 0.798f),
                    0.280f + s * (0.580f - 0.280f),
                    0.469f + s * (0.254f - 0.469f),
                    1.0f);
            } else {
                float s = (t - 0.66f) / 0.34f;
                return ImVec4(
                    0.973f + s * (0.940f - 0.973f),
                    0.580f + s * (0.975f - 0.580f),
                    0.254f + s * (0.131f - 0.254f),
                    1.0f);
            }
        }

        default:
            return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

// ============================================================================
// Filtering
// ============================================================================

void TableViewerPanel::ApplyFilter(TableTab* tab) {
    if (!tab || !tab->table) return;

    tab->filtered_indices.clear();

    if (tab->filter_text.empty()) {
        // No filter - show all
        tab->filtered_indices = tab->sorted_indices;
        return;
    }

    // Filter matching rows
    for (size_t idx : tab->sorted_indices) {
        bool match = false;
        for (size_t c = 0; c < tab->table->GetColumnCount(); c++) {
            std::string cell = tab->table->GetCellAsString(idx, c);
            if (cell.find(tab->filter_text) != std::string::npos) {
                match = true;
                break;
            }
        }
        if (match) {
            tab->filtered_indices.push_back(idx);
        }
    }

    spdlog::info("Filter applied: {} of {} rows match", tab->filtered_indices.size(), tab->sorted_indices.size());
}

void TableViewerPanel::ClearFilter(TableTab* tab) {
    if (!tab) return;

    tab->filter_text.clear();
    std::memset(tab->filter_buffer, 0, sizeof(tab->filter_buffer));
    tab->filtered_indices.clear();
    tab->filter_mode_hide = false;
}

// ============================================================================
// Quick Plot
// ============================================================================

void TableViewerPanel::ShowQuickPlotPopup(TableTab* tab) {
    if (!tab) return;
    show_plot_popup_ = true;
}

void TableViewerPanel::RenderQuickPlot() {
    if (!show_plot_popup_) return;

    ImGui::SetNextWindowSize(ImVec2(700, 550), ImGuiCond_FirstUseEver);

    if (ImGui::Begin(ICON_FA_CHART_SIMPLE " Quick Plot", &show_plot_popup_)) {
        TableTab* tab = GetActiveTab();

        // Chart type selector
        ImGui::Text("Chart Type:");
        ImGui::SameLine();

        if (ImGui::RadioButton("Histogram", plot_popup_.type == QuickPlotType::Histogram)) {
            plot_popup_.type = QuickPlotType::Histogram;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Bar", plot_popup_.type == QuickPlotType::Bar)) {
            plot_popup_.type = QuickPlotType::Bar;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Line", plot_popup_.type == QuickPlotType::Line)) {
            plot_popup_.type = QuickPlotType::Line;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scatter", plot_popup_.type == QuickPlotType::Scatter)) {
            plot_popup_.type = QuickPlotType::Scatter;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Box", plot_popup_.type == QuickPlotType::Box)) {
            plot_popup_.type = QuickPlotType::Box;
        }

        // Column selector(s)
        if (tab && tab->table) {
            const auto& headers = tab->table->GetHeaders();

            ImGui::Text("X Column:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(200);
            if (ImGui::BeginCombo("##XColumn",
                (plot_popup_.x_column >= 0 && plot_popup_.x_column < static_cast<int>(headers.size()))
                    ? headers[plot_popup_.x_column].c_str() : "Select...")) {
                for (int i = 0; i < static_cast<int>(headers.size()); i++) {
                    if (ImGui::Selectable(headers[i].c_str(), plot_popup_.x_column == i)) {
                        plot_popup_.x_column = i;
                        plot_popup_.x_data = GetColumnAsDoubles(tab, i);
                    }
                }
                ImGui::EndCombo();
            }

            // Y column for scatter
            if (plot_popup_.type == QuickPlotType::Scatter) {
                ImGui::SameLine();
                ImGui::Text("Y Column:");
                ImGui::SameLine();
                ImGui::SetNextItemWidth(200);
                if (ImGui::BeginCombo("##YColumn",
                    (plot_popup_.y_column >= 0 && plot_popup_.y_column < static_cast<int>(headers.size()))
                        ? headers[plot_popup_.y_column].c_str() : "Select...")) {
                    for (int i = 0; i < static_cast<int>(headers.size()); i++) {
                        if (ImGui::Selectable(headers[i].c_str(), plot_popup_.y_column == i)) {
                            plot_popup_.y_column = i;
                            plot_popup_.y_data = GetColumnAsDoubles(tab, i);
                        }
                    }
                    ImGui::EndCombo();
                }
            }
        }

        ImGui::Separator();

        // Plot area
        float plot_height = ImGui::GetContentRegionAvail().y - 40;
        if (plot_height < 200) plot_height = 200;

        if (!plot_popup_.x_data.empty()) {
            switch (plot_popup_.type) {
                case QuickPlotType::Histogram: {
                    if (ImPlot::BeginPlot("##Histogram", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Value", "Frequency", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.5f, 0.9f, 0.7f));
                        ImPlot::PlotHistogram("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()), 30);
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Bar: {
                    if (ImPlot::BeginPlot("##Bar", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::SetNextFillStyle(ImVec4(0.4f, 0.7f, 0.4f, 0.8f));
                        int n = std::min(static_cast<int>(plot_popup_.x_data.size()), 100);
                        ImPlot::PlotBars("Data", plot_popup_.x_data.data(), n, 0.67);
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Line: {
                    if (ImPlot::BeginPlot("##Line", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("Index", "Value", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::PlotLine("Data", plot_popup_.x_data.data(),
                            static_cast<int>(plot_popup_.x_data.size()));
                        ImPlot::EndPlot();
                    }
                    break;
                }

                case QuickPlotType::Scatter: {
                    if (!plot_popup_.y_data.empty()) {
                        if (ImPlot::BeginPlot("##Scatter", ImVec2(-1, plot_height))) {
                            ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                            int n = static_cast<int>(std::min(plot_popup_.x_data.size(), plot_popup_.y_data.size()));
                            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 4, ImVec4(0.2f, 0.6f, 1.0f, 0.8f));
                            ImPlot::PlotScatter("Data", plot_popup_.x_data.data(), plot_popup_.y_data.data(), n);
                            ImPlot::EndPlot();
                        }
                    } else {
                        ImGui::TextDisabled("Select Y column for scatter plot");
                    }
                    break;
                }

                case QuickPlotType::Box: {
                    if (ImPlot::BeginPlot("##Box", ImVec2(-1, plot_height))) {
                        ImPlot::SetupAxes("", "Value", ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_AutoFit);

                        // Compute box plot statistics
                        std::vector<double> sorted_data = plot_popup_.x_data;
                        std::sort(sorted_data.begin(), sorted_data.end());
                        size_t n = sorted_data.size();

                        if (n > 0) {
                            double q1 = sorted_data[n / 4];
                            double median = sorted_data[n / 2];
                            double q3 = sorted_data[3 * n / 4];
                            double iqr = q3 - q1;
                            double lower = std::max(sorted_data.front(), q1 - 1.5 * iqr);
                            double upper = std::min(sorted_data.back(), q3 + 1.5 * iqr);

                            // Draw box
                            double box_x[] = {-0.3, 0.3, 0.3, -0.3, -0.3};
                            double box_y[] = {q1, q1, q3, q3, q1};
                            ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
                            ImPlot::PlotLine("##Box", box_x, box_y, 5);
                            ImPlot::PopStyleColor();

                            // Draw median
                            double med_x[] = {-0.3, 0.3};
                            double med_y[] = {median, median};
                            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), 3.0f);
                            ImPlot::PlotLine("Median", med_x, med_y, 2);

                            // Draw whiskers
                            double whisker_x[] = {0, 0};
                            double whisker_lo[] = {lower, q1};
                            double whisker_hi[] = {q3, upper};
                            ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 2.0f);
                            ImPlot::PlotLine("##LowerWhisker", whisker_x, whisker_lo, 2);
                            ImPlot::PlotLine("##UpperWhisker", whisker_x, whisker_hi, 2);
                        }

                        ImPlot::EndPlot();
                    }
                    break;
                }

                default:
                    ImGui::TextDisabled("Select a chart type");
                    break;
            }
        } else {
            ImGui::TextDisabled("No data to plot. Select a column.");
        }

        // Stats footer
        if (!plot_popup_.x_data.empty()) {
            ImGui::Separator();
            double sum = std::accumulate(plot_popup_.x_data.begin(), plot_popup_.x_data.end(), 0.0);
            double mean = sum / plot_popup_.x_data.size();
            double min_val = *std::min_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
            double max_val = *std::max_element(plot_popup_.x_data.begin(), plot_popup_.x_data.end());
            ImGui::Text("Points: %zu | Min: %.4g | Max: %.4g | Mean: %.4g",
                plot_popup_.x_data.size(), min_val, max_val, mean);
        }
    }
    ImGui::End();
}

void TableViewerPanel::RenderMiniHistogram(TableTab* tab, int column) {
    if (!tab || column < 0 || column >= static_cast<int>(tab->column_stats.size())) return;

    auto& stats = tab->column_stats[column];
    if (stats.type != "Numeric" || stats.histogram_bins.empty()) return;

    // Small ImPlot histogram
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(2, 2));

    if (ImPlot::BeginPlot("##MiniHist", ImVec2(stats_sidebar_width_ - 20, 60),
        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText |
        ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect)) {

        ImPlot::SetupAxes(nullptr, nullptr,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines,
            ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_AutoFit);

        std::vector<double> bins_d(stats.histogram_bins.begin(), stats.histogram_bins.end());
        ImPlot::SetNextFillStyle(ImVec4(0.3f, 0.5f, 0.9f, 0.7f));
        ImPlot::PlotBars("##bins", bins_d.data(), static_cast<int>(bins_d.size()), 0.8);

        ImPlot::EndPlot();
    }

    ImPlot::PopStyleVar();

    // Min/max labels
    ImGui::TextDisabled("%.2g", stats.min_val);
    ImGui::SameLine(stats_sidebar_width_ - 60);
    ImGui::TextDisabled("%.2g", stats.max_val);
}

std::vector<double> TableViewerPanel::GetColumnAsDoubles(TableTab* tab, int column) const {
    std::vector<double> result;
    if (!tab || !tab->table || column < 0) return result;

    size_t row_count = tab->table->GetRowCount();
    result.reserve(row_count);

    for (size_t r = 0; r < row_count; r++) {
        auto cell = tab->table->GetCell(r, column);
        if (std::holds_alternative<double>(cell)) {
            double val = std::get<double>(cell);
            if (!std::isnan(val) && !std::isinf(val)) {
                result.push_back(val);
            }
        } else if (std::holds_alternative<int64_t>(cell)) {
            result.push_back(static_cast<double>(std::get<int64_t>(cell)));
        }
    }

    return result;
}

// ============================================================================
// Clipboard Operations
// ============================================================================

void TableViewerPanel::CopyToClipboard(const std::string& text) {
    ImGui::SetClipboardText(text.c_str());
    spdlog::info("Copied to clipboard: {} chars", text.length());
}

void TableViewerPanel::CopyCellToClipboard(TableTab* tab, int row, int col) {
    if (!tab || !tab->table) return;

    std::string value = tab->table->GetCellAsString(row, col);
    CopyToClipboard(value);
}

void TableViewerPanel::CopyRowToClipboard(TableTab* tab, int row) {
    if (!tab || !tab->table) return;

    std::string result;
    size_t col_count = tab->table->GetColumnCount();
    for (size_t c = 0; c < col_count; c++) {
        if (c > 0) result += "\t";
        result += tab->table->GetCellAsString(row, c);
    }
    CopyToClipboard(result);
}

void TableViewerPanel::CopyColumnToClipboard(TableTab* tab, int col) {
    if (!tab || !tab->table) return;

    std::string result;
    size_t row_count = tab->table->GetRowCount();
    for (size_t r = 0; r < row_count; r++) {
        if (r > 0) result += "\n";
        result += tab->table->GetCellAsString(r, col);
    }
    CopyToClipboard(result);
}

void TableViewerPanel::CopySelectionToClipboard(TableTab* tab) {
    if (!tab || !tab->table || tab->selections.empty()) return;

    // For now, just copy the first selection range
    auto& sel = tab->selections[0];
    std::string result;

    for (int r = sel.row_start; r <= sel.row_end; r++) {
        if (r > sel.row_start) result += "\n";
        for (int c = sel.col_start; c <= sel.col_end; c++) {
            if (c > sel.col_start) result += "\t";
            result += tab->table->GetCellAsString(r, c);
        }
    }
    CopyToClipboard(result);
}

} // namespace cyxwiz
