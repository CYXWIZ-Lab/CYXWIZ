#pragma once

// The columns a missing-values refusal names, kept as data so the Engine can
// offer the fix (fill the feature columns, drop the rows without a label)
// instead of only showing the message. No Arrow here: the training metrics
// and the GUI include it.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

struct MissingColumn {
    std::string name;
    int64_t rows = 0;  // empty cells plus NaN
    bool label = false;
    // Counted in the rows read so far, not the whole column (NaN on the
    // Parquet path, found as a batch is read).
    bool at_least = false;
    int64_t nan_rows = 0;  // of `rows`, NaN in a number column
    bool text = false;     // a text column: it fills with its most frequent value
};

// The refusal training throws for missing values: its text (what()) names
// the columns and what to do; Columns() holds them as data.
class MissingValuesError : public std::runtime_error {
public:
    MissingValuesError(const std::string& message, std::string source, std::vector<MissingColumn> columns,
                       int64_t total_rows)
        : std::runtime_error(message),
          source_(std::move(source)),
          columns_(std::move(columns)),
          total_rows_(total_rows) {}

    const std::string& Source() const { return source_; }
    const std::vector<MissingColumn>& Columns() const { return columns_; }
    int64_t TotalRows() const { return total_rows_; }  // -1 when not known

    // "Validation" or "Test" when the refusing table is a supplied input read
    // from its own Data Input (set by the executor as the error passes): a
    // step after the training input does not reach it. Empty for Train.
    const std::string& SuppliedRole() const { return supplied_role_; }
    void SetSuppliedRole(std::string role) { supplied_role_ = std::move(role); }

private:
    std::string source_;
    std::string supplied_role_;
    std::vector<MissingColumn> columns_;
    int64_t total_rows_ = -1;
};

}  // namespace cyxwiz
