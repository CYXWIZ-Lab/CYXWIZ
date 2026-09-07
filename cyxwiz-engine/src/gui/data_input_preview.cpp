#include "data_input_preview.h"
#include "data_input_capabilities.h"
#include "../core/data_input_parameters.h"
#include "loaders/text_csv_preflight.h"
#include <algorithm>
#include <fstream>
#include <map>

namespace gui::data_input {

bool IsDelimitedPreviewSource(const std::string& path, int detected_type) {
    const int effective_type = detected_type == 0
        ? DetectFileTypeForPath(path, nullptr) : detected_type;
    return effective_type == 1 || effective_type == 2;
}

bool MatchesAppliedTabularPreview(
    const std::map<std::string, std::string>& parameters,
    const TabularPreviewSource& source) {
    std::string format;
    std::string error;
    if (!cyxwiz::ResolveDataInputFormatAliases(parameters, format, error)) {
        return false;
    }
    const int applied_type = format == "auto"
        ? DetectFileTypeForPath(source.path, nullptr)
        : FileTypeFromParam(format, -1);
    const int current_type = source.detected_type == 0
        ? DetectFileTypeForPath(source.path, nullptr) : source.detected_type;
    if (applied_type <= 0 || applied_type != current_type) return false;

    const auto matches = [&parameters](const char* key, const std::string& value) {
        const auto it = parameters.find(key);
        return it != parameters.end() && it->second == value;
    };
    return matches("file_path", source.path) &&
        matches("has_header", source.has_header ? "true" : "false") &&
        matches("delimiter", source.delimiter) &&
        matches("decimal_point", std::string(1, source.decimal_point)) &&
        matches("missing_value_tokens", source.missing_value_tokens) &&
        matches("skip_rows", std::to_string(source.skip_rows)) &&
        matches("max_rows", std::to_string(source.max_rows));
}

PreviewTable LoadDelimitedPreview(
    const std::string& path,
    bool has_header,
    char delimiter,
    int detected_type,
    int skip_rows,
    int max_lines) {
    PreviewTable table;
    if (path.empty()) {
        return table;
    }

    if (!IsDelimitedPreviewSource(path, detected_type)) {
        table.error = "Apply this source first, then refresh Preview to browse the loaded dataset. "
                      "Only CSV/TSV supports a pre-load source sample.";
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
    if (detected_type == 2 ||
        (detected_type == 0 && DetectFileTypeForPath(path, nullptr) == 2)) {
        delim = '\t';
    }

    std::string line;
    const int rows_to_skip = std::max(0, skip_rows);
    int rows_skipped = 0;
    while (rows_skipped < rows_to_skip && std::getline(file, line)) {
        ++rows_skipped;
    }
    if (rows_skipped < rows_to_skip ||
        (rows_to_skip > 0 &&
         file.peek() == std::ifstream::traits_type::eof())) {
        table.error = "No tabular rows remain after skipping " +
            std::to_string(rows_to_skip) + " source rows";
        return table;
    }

    int line_count = 0;
    while (line_count < max_lines) {
        std::vector<std::string> cells;
        std::string error;
        if (!cyxwiz::loaders::ReadTextCsvRow(file, delim, cells, error)) {
            if (!error.empty()) {
                table.error = "Preview row " + std::to_string(line_count + 1) + ": " + error;
            }
            break;
        }
        if (cells.size() == 1 && cells.front().empty()) continue;
        if (line_count > 0 && cells.size() != table.columns.size()) {
            table.error = "Preview row " + std::to_string(line_count + 1) +
                " has a different field count from the header; check delimiter and quoting";
            break;
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
