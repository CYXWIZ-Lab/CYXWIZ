#pragma once

// Results > Test > By column value (TOFIX157; in the Test Results window until TOFIX129 D4):
// a test's error sliced by the values of one column of the tested table. The
// test's rows (its per-row table) are joined back to that table by row_id; the
// chosen column is read by a task (a table on disk a row group at a time). The
// model is core/data_studio_results (RowErrorsFromPredictions, ColumnGatherer,
// SliceErrors).

#include "../../core/data_studio_results.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

class ErrorsByValueView {
public:
    ~ErrorsByValueView();
    // The test's per-row table (nullptr: none). A different table starts over.
    void SetRows(const std::shared_ptr<arrow::Table>& rows);
    void Render();
    bool IsRunning() const { return task_ != 0; }

private:
    void SliceBy(const std::string& column);
    void CancelTask();

    std::shared_ptr<arrow::Table> rows_;
    studio::TestRowErrors row_errors_;
    std::vector<std::string> columns_;  // the tested table's columns that can be sliced
    std::string column_;
    studio::ErrorSlices slices_;
    std::string error_;
    uint64_t task_ = 0;
    std::shared_ptr<int> alive_ = std::make_shared<int>(0);
};

}  // namespace cyxwiz
