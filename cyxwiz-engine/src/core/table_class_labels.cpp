#include "table_class_labels.h"

#include "data_analyzer.h"
#include "../data/data_table.h"

#include <cmath>
#include <map>
#include <variant>

namespace cyxwiz {

namespace {
// Empty, an empty string, or NaN: no label in this row.
bool IsEmptyCell(const DataTable::CellValue& cell) {
    if (std::holds_alternative<std::monostate>(cell)) return true;
    if (const auto* text = std::get_if<std::string>(&cell)) return text->empty();
    if (const auto* value = std::get_if<double>(&cell)) return std::isnan(*value);
    return false;
}
}  // namespace

std::string TableClassLabels::Name(int id) const {
    if (id >= 0 && static_cast<size_t>(id) < names.size()) return names[id];
    return std::to_string(id);
}

TableClassLabels ReadTableClassLabels(const DataTable& table, const std::vector<int>& columns) {
    const size_t rows = table.GetRowCount();
    const auto in_table = [&](int col) { return col >= 0 && static_cast<size_t>(col) < table.GetColumnCount(); };

    bool numeric = true;
    for (int col : columns) {
        if (!in_table(col)) continue;
        for (size_t row = 0; row < rows && numeric; ++row) {
            const auto cell = table.GetCell(row, col);
            if (!IsEmptyCell(cell) && !DataAnalyzer::ToDouble(cell).has_value()) {
                numeric = false;
            }
        }
    }

    TableClassLabels labels;
    labels.columns.assign(columns.size(), std::vector<std::optional<int>>(rows));
    std::map<std::string, int> ids;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t c = 0; c < columns.size(); ++c) {
            if (!in_table(columns[c])) continue;
            if (numeric) {
                const auto value = DataAnalyzer::ToDouble(table.GetCell(row, columns[c]));
                if (value) labels.columns[c][row] = static_cast<int>(*value);
                continue;
            }
            if (IsEmptyCell(table.GetCell(row, columns[c]))) continue;
            const std::string text = table.GetCellAsString(row, columns[c]);
            const auto [it, added] = ids.emplace(text, static_cast<int>(labels.names.size()));
            if (added) labels.names.push_back(text);
            labels.columns[c][row] = it->second;
        }
    }
    return labels;
}

}  // namespace cyxwiz
