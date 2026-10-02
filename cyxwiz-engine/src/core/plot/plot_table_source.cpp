#include "plot_table_source.h"

#include "../../data/data_table.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <variant>

namespace cyxwiz::plot {

namespace {

// The number in a cell; NaN for empty or text. `is_text` says a non-empty
// cell was not a number.
double CellNumber(const DataTable::CellValue& cell, bool* is_text) {
    *is_text = false;
    if (std::holds_alternative<double>(cell)) return std::get<double>(cell);
    if (std::holds_alternative<int64_t>(cell)) return static_cast<double>(std::get<int64_t>(cell));
    if (std::holds_alternative<std::string>(cell)) {
        const std::string& s = std::get<std::string>(cell);
        if (s.empty()) return NAN;
        char* end = nullptr;
        errno = 0;
        const double v = std::strtod(s.c_str(), &end);
        while (end && *end == ' ') ++end;
        if (end && *end == '\0' && end != s.c_str() && errno != ERANGE) return v;
        *is_text = true;
    }
    return NAN;
}

}  // namespace

std::vector<bool> NumericColumns(const DataTable& table) {
    const size_t cols = table.GetColumnCount(), rows = table.GetRowCount();
    std::vector<bool> numeric(cols, false);
    for (size_t c = 0; c < cols; ++c) {
        bool any = false, text = false;
        for (size_t r = 0; r < rows && !text; ++r) {
            bool is_text = false;
            const double v = CellNumber(table.GetCell(r, c), &is_text);
            text = is_text;
            any = any || !std::isnan(v);
        }
        numeric[c] = any && !text;
    }
    return numeric;
}

Source SourceFromTable(const DataTable& table, const std::vector<std::string>& columns,
                       const std::vector<bool>& numeric) {
    Source src;
    const auto& headers = table.GetHeaders();
    const size_t rows = table.GetRowCount();
    for (const auto& name : columns) {
        bool seen = false;
        for (const auto& c : src.columns) seen = seen || c.name == name;
        if (seen || name.empty()) continue;
        for (size_t c = 0; c < headers.size(); ++c) {
            if (headers[c] != name) continue;
            SourceColumn col;
            col.name = name;
            col.numeric = c < numeric.size() && numeric[c];
            if (col.numeric) {
                col.numbers.reserve(rows);
                for (size_t r = 0; r < rows; ++r) {
                    bool is_text = false;
                    col.numbers.push_back(CellNumber(table.GetCell(r, c), &is_text));
                }
            } else {
                col.text.reserve(rows);
                for (size_t r = 0; r < rows; ++r) col.text.push_back(table.GetCellAsString(r, c));
            }
            src.columns.push_back(std::move(col));
            break;
        }
    }
    return src;
}

}  // namespace cyxwiz::plot
