#pragma once

// A test run as its per-row table and back (TOFIX157). A finished test
// publishes the table as "test_predictions"; Keep as node saves a copy to a
// file. The Test Results panel, Data Studio > Results and the Test Evaluation
// node read a table back into the TestingMetrics a run produces, so one
// calculation (FinishClassifierMetrics; MAE / RMSE here) gives every view its
// numbers.

#include "test_executor.h"

#include <memory>
#include <string>

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

}  // namespace cyxwiz
