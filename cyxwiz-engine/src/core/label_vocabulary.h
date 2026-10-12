#pragma once

// Class labels held as text (TOFIX145). A tabular label column of strings
// ("delta", "east", ...) becomes int32 class indices over one vocabulary: the
// distinct values sorted, as scikit-learn's LabelEncoder and pandas order
// categories, built once over every partition (train, dev, test) so they
// share the numbering. The batchers read only numbers; before this, every
// text label trained as class 0.

#include <arrow/api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

struct LabelVocabulary {
    std::vector<std::string> names;                 // index -> class name, sorted
    std::unordered_map<std::string, int32_t> index;  // class name -> index

    int32_t IndexOf(const std::string& name) const;  // -1 when unknown
    bool empty() const { return names.empty(); }
};

// utf8 / large_utf8.
bool IsTextLabelType(const arrow::DataType& type);

// The distinct non-null values of text label columns, sorted.
LabelVocabulary BuildLabelVocabulary(const std::vector<std::shared_ptr<arrow::ChunkedArray>>& columns);

// The column as int32 class indices; nulls stay null; a value missing from the
// vocabulary is an error naming it.
arrow::Result<std::shared_ptr<arrow::ChunkedArray>> EncodeLabels(const arrow::ChunkedArray& column,
                                                                 const LabelVocabulary& vocabulary);

// The table with `label_column` replaced by its encoded column (same name and
// position, int32).
arrow::Result<std::shared_ptr<arrow::Table>> EncodeLabelColumn(const std::shared_ptr<arrow::Table>& table,
                                                                const std::string& label_column,
                                                                const LabelVocabulary& vocabulary);

// The dataset's text label column, or nullptr when the column is missing or
// not text.
std::shared_ptr<arrow::ChunkedArray> TextLabelColumn(const std::shared_ptr<arrow::Table>& table,
                                                     const std::string& label_column);

}  // namespace cyxwiz
