#pragma once

// Training refuses missing values (TOFIX223 0.3, owner decision D4 of
// TOFIX222): a feature or label cell that is empty, or NaN in a float
// column, is never trained on as 0 or class 0. The user resolves it with a
// step (Missing Value for features, a Row Filter for missing labels) or in
// the file. One message for every batcher.

#include <arrow/api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Empty cells, plus NaN in a float32 / float64 column.
inline int64_t MissingCells(const arrow::ChunkedArray& column) {
    int64_t missing = column.null_count();
    const auto id = column.type()->id();
    if (id != arrow::Type::FLOAT && id != arrow::Type::DOUBLE) return missing;
    for (const auto& chunk : column.chunks()) {
        const int64_t length = chunk->length();
        if (id == arrow::Type::FLOAT) {
            const auto& values = static_cast<const arrow::FloatArray&>(*chunk);
            for (int64_t i = 0; i < length; ++i)
                if (values.IsValid(i) && std::isnan(values.Value(i))) ++missing;
        } else {
            const auto& values = static_cast<const arrow::DoubleArray&>(*chunk);
            for (int64_t i = 0; i < length; ++i)
                if (values.IsValid(i) && std::isnan(values.Value(i))) ++missing;
        }
    }
    return missing;
}

struct MissingColumn {
    std::string name;
    int64_t rows = 0;
    bool label = false;
    // Counted in the rows read so far, not the whole column (NaN on the
    // Parquet path, found as a batch is read).
    bool at_least = false;
};

// Throws the refusal when any column has missing cells.
inline void ThrowIfMissingValues(const char* batcher, const std::vector<MissingColumn>& columns) {
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
    throw std::runtime_error(message);
}

}  // namespace cyxwiz
