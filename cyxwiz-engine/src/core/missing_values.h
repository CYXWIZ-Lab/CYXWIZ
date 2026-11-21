#pragma once

// Training refuses missing values (TOFIX223 0.3, owner decision D4 of
// TOFIX222): a feature or label cell that is empty, or NaN in a float
// column, is never trained on as 0 or class 0. The user resolves it with a
// step (Missing Value for features, a Row Filter for missing labels) or in
// the file. One message for every batcher.

#include "missing_columns.h"

#include <arrow/api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// NaN cells of a float32 / float64 column (0 for any other type).
inline int64_t NanCells(const arrow::ChunkedArray& column) {
    const auto id = column.type()->id();
    if (id != arrow::Type::FLOAT && id != arrow::Type::DOUBLE) return 0;
    int64_t nans = 0;
    for (const auto& chunk : column.chunks()) {
        const int64_t length = chunk->length();
        if (id == arrow::Type::FLOAT) {
            const auto& values = static_cast<const arrow::FloatArray&>(*chunk);
            for (int64_t i = 0; i < length; ++i)
                if (values.IsValid(i) && std::isnan(values.Value(i))) ++nans;
        } else {
            const auto& values = static_cast<const arrow::DoubleArray&>(*chunk);
            for (int64_t i = 0; i < length; ++i)
                if (values.IsValid(i) && std::isnan(values.Value(i))) ++nans;
        }
    }
    return nans;
}

// Empty cells, plus NaN in a float32 / float64 column.
inline int64_t MissingCells(const arrow::ChunkedArray& column) { return column.null_count() + NanCells(column); }

inline bool IsTextArrowType(arrow::Type::type id) {
    return id == arrow::Type::STRING || id == arrow::Type::LARGE_STRING ||
           id == arrow::Type::DICTIONARY;
}

// One column's missing cells as a refusal names them.
inline MissingColumn CountMissingColumn(const std::string& name, const arrow::ChunkedArray& column, bool label) {
    MissingColumn c;
    c.name = name;
    c.nan_rows = NanCells(column);
    c.rows = column.null_count() + c.nan_rows;
    c.label = label;
    c.text = IsTextArrowType(column.type()->id());
    return c;
}


// Throws the refusal when any column has missing cells.
// `total_rows`: the rows the counts are out of (-1 when not known).
inline void ThrowIfMissingValues(const char* batcher, const std::vector<MissingColumn>& columns,
                                 int64_t total_rows = -1) {
    std::string features, labels;
    size_t listed = 0, more = 0;
    for (const auto& c : columns) {
        if (c.rows <= 0) continue;
        if (listed == 6) {
            ++more;
            continue;
        }
        ++listed;
        std::string& list = c.label ? labels : features;
        list += (list.empty() ? "" : ", ") + ("'" + c.name + "' " + (c.at_least ? "at least " : "") +
                                              std::to_string(c.rows) +
                                              (c.rows == 1 ? " row" : " rows"));
    }
    if (listed == 0) return;
    std::string message = std::string(batcher) + ": the data has missing values (empty cells or NaN), which training does not use: ";
    if (!features.empty()) message += "feature " + features;
    if (!labels.empty()) message += std::string(features.empty() ? "" : "; ") + "label " + labels;
    if (more > 0) message += " and " + std::to_string(more) + " more column(s)";
    message += ". Fill feature columns with a Missing Value step, remove rows without a label with a Row "
               "Filter (condition: <label> IS NOT EMPTY), or fix the file.";
    std::vector<MissingColumn> missing;
    for (const auto& c : columns)
        if (c.rows > 0) missing.push_back(c);
    throw MissingValuesError(message, batcher, std::move(missing), total_rows);
}

}  // namespace cyxwiz
