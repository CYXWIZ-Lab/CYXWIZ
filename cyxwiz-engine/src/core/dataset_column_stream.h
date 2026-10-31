#pragma once

// The chosen columns of a table the Engine holds, every row, as Arrow
// batches (TOFIX181): from memory, or one row group at a time from a table
// kept on disk, so a large table never has to fit in memory. Explore >
// Plugins hands this stream to a plugin's analytic.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class RecordBatchReader;
}

namespace cyxwiz {

struct ColumnStreamOptions {
    std::function<bool()> should_cancel;      // true: the next read stops with "Cancelled"
    std::function<void(int64_t rows)> on_rows;  // rows handed out so far, after each batch
};

// Null with `error` set when the table is not one held in memory or on disk,
// or a column is missing. `total_rows` gets the table's row count.
std::shared_ptr<arrow::RecordBatchReader> OpenDatasetColumns(const std::string& dataset,
                                                             const std::vector<std::string>& columns,
                                                             const ColumnStreamOptions& options, std::string& error,
                                                             int64_t* total_rows = nullptr);

}  // namespace cyxwiz
