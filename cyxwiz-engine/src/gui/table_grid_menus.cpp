// TableGrid (TOFIX148): the column and cell menus, copy and number formats.

#include "table_grid.h"

#include "icons.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace cyxwiz {

namespace {
void CopyToClipboard(const std::string& text) {
    ImGui::SetClipboardText(text.c_str());
    spdlog::info("Copied to clipboard: {} chars", text.length());
}
}  // namespace

// ============================================================================
// Context Menus
// ============================================================================

void TableGrid::RenderColumnContextMenu(int column) {
    if (column < 0) return;

    // Note: This function is called from within BeginPopupContextItem, so no BeginPopup needed
    const auto& headers = table->GetHeaders();
    std::string col_name = (column < static_cast<int>(headers.size())) ? headers[column] : "Column";

    ImGui::TextDisabled("%s %s", ICON_FA_TABLE_COLUMNS, col_name.c_str());
    ImGui::Separator();

    // Sorting
    // (the header's arrow follows on the next frame)
    if (ImGui::MenuItem(ICON_FA_SORT_UP " Sort Ascending")) {
        SetOrder(column, true);
        sync_sort_header_ = true;
    }
    if (ImGui::MenuItem(ICON_FA_SORT_DOWN " Sort Descending")) {
        SetOrder(column, false);
        sync_sort_header_ = true;
    }

    ImGui::Separator();

    // Filtering
    if (ImGui::MenuItem(ICON_FA_CIRCLE_XMARK " Clear Filter", nullptr, false, !filter_text.empty())) {
        ClearFilter();
    }

    ImGui::Separator();

    // Colormaps submenu
    if (ImGui::BeginMenu(ICON_FA_PALETTE " Apply Colormap")) {
        if (ImGui::MenuItem("Viridis", nullptr,
            column < static_cast<int>(column_colormaps.size()) &&
            column_colormaps[column].type == ColorMapType::Viridis)) {
            ApplyColorMap(column, ColorMapType::Viridis);
        }
        if (ImGui::MenuItem("Inferno", nullptr,
            column < static_cast<int>(column_colormaps.size()) &&
            column_colormaps[column].type == ColorMapType::Inferno)) {
            ApplyColorMap(column, ColorMapType::Inferno);
        }
        if (ImGui::MenuItem("RdYlGn (Red-Yellow-Green)", nullptr,
            column < static_cast<int>(column_colormaps.size()) &&
            column_colormaps[column].type == ColorMapType::RdYlGn)) {
            ApplyColorMap(column, ColorMapType::RdYlGn);
        }
        if (ImGui::MenuItem("Blues", nullptr,
            column < static_cast<int>(column_colormaps.size()) &&
            column_colormaps[column].type == ColorMapType::Blues)) {
            ApplyColorMap(column, ColorMapType::Blues);
        }
        if (ImGui::MenuItem("Plasma", nullptr,
            column < static_cast<int>(column_colormaps.size()) &&
            column_colormaps[column].type == ColorMapType::Plasma)) {
            ApplyColorMap(column, ColorMapType::Plasma);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(ICON_FA_XMARK " Clear Colormap")) {
            ClearColorMap(column);
        }
        ImGui::EndMenu();
    }

    // Format Numbers submenu (only for numeric columns)
    if (column < static_cast<int>(column_kinds.size()) &&
        column_kinds[column].type == "Numeric") {
        if (ImGui::BeginMenu(ICON_FA_HASHTAG " Format Numbers")) {
            // Ensure column_formats is sized correctly
            if (column_formats.size() <= static_cast<size_t>(column)) {
                column_formats.resize(column + 1);
            }
            auto& fmt = column_formats[column];

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
    if (on_plot && ImGui::BeginMenu(ICON_FA_CHART_SIMPLE " Plot")) {
        // The Plot window opens on the type; other types are one click there.
        if (ImGui::MenuItem(ICON_FA_CHART_BAR " Histogram")) on_plot(column, plot::Kind::Histogram);
        if (ImGui::MenuItem(ICON_FA_CHART_LINE " Line Chart")) on_plot(column, plot::Kind::Line);
        if (ImGui::MenuItem(ICON_FA_CHART_COLUMN " Bar Chart")) on_plot(column, plot::Kind::Bar);
        if (ImGui::MenuItem(ICON_FA_CUBE " Box Plot")) on_plot(column, plot::Kind::Box);
        ImGui::Separator();
        // The Y column is chosen in the Plot window.
        if (ImGui::MenuItem(ICON_FA_CHART_SCATTER " Use as X-axis for Scatter")) on_plot(column, plot::Kind::Scatter);
        ImGui::EndMenu();
    }
    if (on_plot) ImGui::Separator();

    // Copy
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Column")) {
        CopyColumnToClipboard(column);
    }
}

void TableGrid::RenderCellContextMenu(int row, int col) {
    if (row < 0 || col < 0) return;

    // Note: This function is called from within BeginPopupContextItem, so no BeginPopup needed
    std::string cell_value = CellText(row, col);
    std::string preview = cell_value.length() > 20 ? cell_value.substr(0, 20) + "..." : cell_value;

    ImGui::TextDisabled("%s [%d, %d] = %s", ICON_FA_TABLE_CELLS, row + 1, col + 1, preview.c_str());
    ImGui::Separator();

    // Copy operations
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Cell")) {
        CopyCellToClipboard(row, col);
    }
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Row")) {
        CopyRowToClipboard(row);
    }
    if (ImGui::MenuItem(ICON_FA_COPY " Copy Selection", nullptr, false, !selections.empty())) {
        CopySelectionToClipboard();
    }

    ImGui::Separator();

    // Selection
    if (ImGui::MenuItem(ICON_FA_BARS " Select Entire Row")) {
        selected_row = row;
        selected_col = -1;  // Entire row
    }
    if (ImGui::MenuItem(ICON_FA_TABLE_COLUMNS " Select Entire Column")) {
        selected_column = col;
        selected_col = col;
        selected_row = -1;  // Entire column
    }

    ImGui::Separator();

    // Filtering
    if (ImGui::MenuItem(ICON_FA_FILTER " Filter by This Value")) {
        SetFilter(cell_value);
    }

    // Column stats
    if (options.column_stats && ImGui::MenuItem(ICON_FA_CHART_SIMPLE " View Column Statistics")) {
        selected_column = col;
    }
}

// ============================================================================
// Clipboard Operations
// ============================================================================

void TableGrid::CopyCellToClipboard(int row, int col) const {
    if (!table) return;

    std::string value = CellText(row, col);
    CopyToClipboard(value);
}

void TableGrid::CopyRowToClipboard(int row) const {
    if (!table) return;

    std::string result;
    size_t col_count = table->GetColumnCount();
    for (size_t c = 0; c < col_count; c++) {
        if (c > 0) result += "\t";
        result += CellText(row, c);
    }
    CopyToClipboard(result);
}

void TableGrid::CopyColumnToClipboard(int col) const {
    if (!table) return;

    std::string result;
    size_t row_count = table->GetRowCount();
    for (size_t r = 0; r < row_count; r++) {
        if (r > 0) result += "\n";
        result += CellText(r, col);
    }
    CopyToClipboard(result);
}

void TableGrid::CopySelectionToClipboard() const {
    if (!table || selections.empty()) return;

    // For now, just copy the first selection range
    const auto& sel = selections[0];
    std::string result;

    for (int r = sel.row_start; r <= sel.row_end; r++) {
        if (r > sel.row_start) result += "\n";
        for (int c = sel.col_start; c <= sel.col_end; c++) {
            if (c > sel.col_start) result += "\t";
            result += CellText(r, c);
        }
    }
    CopyToClipboard(result);
}

// ============================================================================
// Number Formatting
// ============================================================================

std::string TableGrid::FormatNumber(double value, const ColumnFormat& fmt) {
    std::ostringstream ss;

    // Apply percentage conversion
    double display_val = fmt.as_percentage ? value * 100.0 : value;

    // Determine decimal places
    int decimals = fmt.decimal_places;
    if (decimals < 0) {
        // Auto: use 2 decimals for floats, 0 for integers
        if (std::abs(display_val - std::round(display_val)) < 0.0001) {
            decimals = 0;
        } else {
            decimals = 2;
        }
    }

    // Format the number
    ss << std::fixed << std::setprecision(decimals) << display_val;
    std::string num_str = ss.str();

    // Add thousands separator
    if (fmt.thousands_separator) {
        // Find decimal point
        size_t decimal_pos = num_str.find('.');
        std::string integer_part = (decimal_pos != std::string::npos) ?
            num_str.substr(0, decimal_pos) : num_str;
        std::string decimal_part = (decimal_pos != std::string::npos) ?
            num_str.substr(decimal_pos) : "";

        // Insert commas
        std::string formatted_int;
        int count = 0;
        for (int i = static_cast<int>(integer_part.length()) - 1; i >= 0; i--) {
            if (count > 0 && count % 3 == 0 && integer_part[i] != '-') {
                formatted_int = ',' + formatted_int;
            }
            formatted_int = integer_part[i] + formatted_int;
            count++;
        }
        num_str = formatted_int + decimal_part;
    }

    // Add prefix and suffix
    std::string result = fmt.prefix + num_str;
    if (fmt.as_percentage) {
        result += "%";
    } else {
        result += fmt.suffix;
    }

    return result;
}

}  // namespace cyxwiz
