#include "data_input_preview.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>

namespace gui::data_input {

PreviewTable LoadDelimitedPreview(
    const std::string& path,
    bool has_header,
    char delimiter,
    int detected_type,
    int max_lines) {
    PreviewTable table;
    if (path.empty()) {
        return table;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        table.error = "Cannot open file";
        return table;
    }

    char delim = delimiter;
    if (delim == '\0') {
        delim = ',';
    }
    if (detected_type == 2) {
        delim = '\t';
    }

    std::string line;
    int line_count = 0;
    while (std::getline(file, line) && line_count < max_lines) {
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, delim)) {
            const std::size_t start = cell.find_first_not_of(" \t\r\n\"");
            const std::size_t end = cell.find_last_not_of(" \t\r\n\"");
            if (start != std::string::npos && end != std::string::npos) {
                cell = cell.substr(start, end - start + 1);
            }
            cells.push_back(cell);
        }

        if (line_count == 0 && has_header) {
            table.columns = cells;
        } else {
            table.rows.push_back(cells);
            if (!has_header && line_count == 0) {
                for (std::size_t i = 0; i < cells.size(); ++i) {
                    table.columns.push_back("Column" + std::to_string(i + 1));
                }
            }
        }
        ++line_count;
    }

    return table;
}

LabelDistribution ComputeLabelDistribution(
    const std::vector<std::string>& columns,
    const std::vector<std::vector<std::string>>& rows,
    const std::string& label_column) {
    LabelDistribution distribution;
    if (label_column.empty() || columns.empty() || rows.empty()) {
        return distribution;
    }

    int label_idx = -1;
    for (int i = 0; i < static_cast<int>(columns.size()); ++i) {
        if (columns[i] == label_column) {
            label_idx = i;
            break;
        }
    }
    if (label_idx < 0) {
        return distribution;
    }

    std::map<std::string, std::size_t> counts;
    for (const auto& row : rows) {
        if (label_idx >= static_cast<int>(row.size())) {
            continue;
        }
        std::string value = row[label_idx];
        if (value.empty()) {
            value = "(empty)";
        }
        counts[value]++;
        distribution.total++;
    }

    distribution.values.assign(counts.begin(), counts.end());
    std::sort(distribution.values.begin(), distribution.values.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) {
                      return a.second > b.second;
                  }
                  return a.first < b.first;
              });
    distribution.column = label_column;
    return distribution;
}

} // namespace gui::data_input
