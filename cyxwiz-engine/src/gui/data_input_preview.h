#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace gui::data_input {

// Current dialog source settings; registered preview is valid only for the
// applied source contract. Registry identity/lifetime stays with the caller.
struct TabularPreviewSource {
    std::string path;
    int detected_type = 0;
    bool has_header = true;
    std::string delimiter;
    char decimal_point = '.';
    std::string missing_value_tokens;
    int skip_rows = 0;
    int max_rows = 0;
};

bool MatchesAppliedTabularPreview(
    const std::map<std::string, std::string>& parameters,
    const TabularPreviewSource& source);

bool IsDelimitedPreviewSource(const std::string& path, int detected_type);

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
