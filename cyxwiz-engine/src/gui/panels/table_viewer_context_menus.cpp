#include "table_viewer.h"
#include "../icons.h"
#include <imgui.h>
#include <algorithm>
#include <cstring>

namespace cyxwiz {

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

    // Format Numbers submenu (only for numeric columns)
    if (column < static_cast<int>(tab->column_stats.size()) &&
        tab->column_stats[column].type == "Numeric") {
        if (ImGui::BeginMenu(ICON_FA_HASHTAG " Format Numbers")) {
            // Ensure column_formats is sized correctly
            if (tab->column_formats.size() <= static_cast<size_t>(column)) {
                tab->column_formats.resize(column + 1);
            }
            auto& fmt = tab->column_formats[column];

            // Decimal places
            ImGui::Text("Decimal places:");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            int decimals = fmt.decimal_places;
            if (ImGui::InputInt("##Decimals", &decimals, 1, 1)) {
                fmt.decimal_places = std::clamp(decimals, -1, 10);  // -1 = auto
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("-1 = auto");

            ImGui::Separator();

            // Formatting options
            if (ImGui::MenuItem("Thousands separator", nullptr, fmt.thousands_separator)) {
                fmt.thousands_separator = !fmt.thousands_separator;
            }
            if (ImGui::MenuItem("As percentage (%)", nullptr, fmt.as_percentage)) {
                fmt.as_percentage = !fmt.as_percentage;
            }

            ImGui::Separator();

            // Prefix/Suffix
            ImGui::Text("Prefix:");
            ImGui::SameLine();
            char prefix_buf[32];
            strncpy(prefix_buf, fmt.prefix.c_str(), sizeof(prefix_buf) - 1);
            prefix_buf[sizeof(prefix_buf) - 1] = '\0';
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputText("##Prefix", prefix_buf, sizeof(prefix_buf))) {
                fmt.prefix = prefix_buf;
            }

            ImGui::Text("Suffix:");
            ImGui::SameLine();
            char suffix_buf[32];
            strncpy(suffix_buf, fmt.suffix.c_str(), sizeof(suffix_buf) - 1);
            suffix_buf[sizeof(suffix_buf) - 1] = '\0';
            ImGui::SetNextItemWidth(60);
            if (ImGui::InputText("##Suffix", suffix_buf, sizeof(suffix_buf))) {
                fmt.suffix = suffix_buf;
            }

            ImGui::Separator();

            // Quick presets
            if (ImGui::MenuItem(ICON_FA_DOLLAR_SIGN " Currency ($)")) {
                fmt.prefix = "$";
                fmt.suffix = "";
                fmt.decimal_places = 2;
                fmt.thousands_separator = true;
                fmt.as_percentage = false;
            }
            if (ImGui::MenuItem(ICON_FA_PERCENT " Percentage")) {
                fmt.prefix = "";
                fmt.suffix = "";
                fmt.as_percentage = true;
                fmt.decimal_places = 1;
            }
            if (ImGui::MenuItem(ICON_FA_XMARK " Clear Formatting")) {
                fmt = ColumnFormat();  // Reset to defaults
            }

            ImGui::EndMenu();
        }
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

}  // namespace cyxwiz

