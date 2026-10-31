#pragma once

// Data Studio > Results (TOFIX148 job 7, approved mockup v8): a model's
// results next to the data. Language models' generation previews are read by
// core/generation_preview_presentation (TOFIX159 A4). This file is the test
// run side: a test's per-row table (kTestPredictionsDataset, or a kept copy)
// joined back to the tested table by its row_id, so the error can be sliced
// by the values of one of its columns (TOFIX157 B2). The run's own figures
// are the TestingMetrics the table reads back into
// (test_predictions_metrics.h), shown by Data Studio > Results > Test.
//
// Pure data in and out.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class ChunkedArray;
class Table;
}

namespace cyxwiz::studio {

// ---- Errors by column value --------------------------------------------------
//
// Joins a test run's rows back to the tested table (row_id) and slices the
// error by the values of one of its columns.

// Per test row: its row in the tested table and its error (a classifier:
// 1 wrong, 0 right; regression: |error| averaged over the targets).
struct TestRowErrors {
    bool ok = false;
    std::string error;
    bool classifier = true;
    std::string source_dataset;     // the tested table's name (schema metadata)
    std::vector<int64_t> row_ids;
    std::vector<double> errors;
};
TestRowErrors RowErrorsFromPredictions(const std::shared_ptr<arrow::Table>& table);

// Reads one column of the tested table at the test rows, a piece at a time in
// table order (the whole column in memory, or one row group of a file).
struct GatheredColumn {
    bool numeric = false;
    std::vector<double> numbers;      // numeric: per test row
    std::vector<std::string> labels;  // otherwise: per test row
    std::vector<uint8_t> valid;       // 0 missing
};
class ColumnGatherer {
public:
    explicit ColumnGatherer(const std::vector<int64_t>& row_ids);
    // The next piece of the column; false (with `error`) on a type that cannot
    // be sliced or a row id past the table's end at Finish.
    bool Add(const std::shared_ptr<arrow::ChunkedArray>& piece, std::string& error);
    bool Finish(GatheredColumn& out, std::string& error);

private:
    std::vector<int64_t> row_ids_;
    std::vector<size_t> order_;  // test positions sorted by row id
    size_t cursor_ = 0;          // into order_
    int64_t offset_ = 0;         // rows seen
    GatheredColumn column_;
    bool typed_ = false;
};

struct ValueSlice {
    std::string value;  // the value, a range "a to b" for numbers, "(missing)", or "(other values)"
    int64_t rows = 0;
    double error = 0.0;  // a classifier: the share wrong; regression: mean |error|
};
struct ErrorSlices {
    bool binned = false;        // numbers grouped in equal-count ranges
    double overall = 0.0;       // the same measure over every test row
    std::vector<ValueSlice> slices;  // highest error first
};
// At most `max_values` distinct values (the most common; the rest as
// "(other values)"); a numeric column with more distinct values than that is
// cut into `bins` ranges of about equal rows.
ErrorSlices SliceErrors(const GatheredColumn& column, const std::vector<double>& errors, size_t max_values = 20,
                        size_t bins = 5);

}  // namespace cyxwiz::studio
