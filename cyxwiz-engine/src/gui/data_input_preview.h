#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace gui::data_input {

struct PreviewTable {
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string error;
};

struct LabelDistribution {
    std::vector<std::pair<std::string, std::size_t>> values;
    std::string column;
    std::size_t total = 0;
};

PreviewTable LoadDelimitedPreview(
    const std::string& path,
    bool has_header,
    char delimiter,
    int detected_type,
    int skip_rows = 0,
    int max_lines = 25);

LabelDistribution ComputeLabelDistribution(
    const std::vector<std::string>& columns,
    const std::vector<std::vector<std::string>>& rows,
    const std::string& label_column);

} // namespace gui::data_input
