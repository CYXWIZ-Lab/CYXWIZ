#include "table_viewer.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace cyxwiz {
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

// ============================================================================
// Number Formatting
// ============================================================================

std::string TableViewerPanel::FormatNumber(double value, const ColumnFormat& fmt) {
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

