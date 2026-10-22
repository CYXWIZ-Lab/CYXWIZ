#pragma once

// A test run as its per-row table and back (TOFIX157). A finished test
// publishes the table as "test_predictions"; Keep as node saves a copy to a
// file. Data Studio > Results > Test and the Test Evaluation
// node read a table back into the TestingMetrics a run produces, so one
// calculation (FinishClassifierMetrics; MAE / RMSE here) gives every view its
// numbers.

#include "test_executor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// The run's per-row table (test_predictions_table.h): a classifier's or a
// regression's (a metric-learning test's 1-NN classes too); nullptr when the
// run kept no rows (a language model, or too many values to keep).
std::shared_ptr<arrow::Table> PredictionsTableFromMetrics(const TestingMetrics& m);

// The run back from its table: rows, row ids, source dataset, class names,
// loss, time and quick-test flag from the metadata, then the classifier's
// confusion matrix, per-class figures, averages and curves, or the regression's
// MAE / RMSE. False with `error` when the table is not a test's.
bool MetricsFromPredictionsTable(const arrow::Table& table, TestingMetrics& out, std::string& error);

// Any table read as a test (TOFIX129 D4, Results > Test > A table...): a
// true column and a predicted column, and for curves the scores. Classes: the
// labels (text or numbers) of both columns, sorted as the label encoder sorts
// them; `scores` is one column per class in that order, or one column for
// two classes (the score of `positive`, the second class when empty).
// Numbers (regression): true and predicted values. Rows missing a true or
// predicted value are left out (counted in `left_out`). The table goes through
// the test_predictions format and MetricsFromPredictionsTable, so it gets the
// same figures as a test run; its rows keep their row numbers in `dataset`.
struct PredictionColumns {
    std::string truth;
    std::string predicted;
    std::vector<std::string> scores;
    std::string positive;
    bool numbers = false;
    std::string dataset;
};
bool MetricsFromColumns(const arrow::Table& table, const PredictionColumns& columns, TestingMetrics& out,
                        int64_t& left_out, std::string& error);

// A first guess at the columns (by name: true / actual / label / target, and
// predicted / prediction; score_<class> per class, or score / probability),
// with `numbers` set when both columns hold numbers with more than 20
// distinct values or fractions.
PredictionColumns GuessPredictionColumns(const arrow::Table& table);

}  // namespace cyxwiz
