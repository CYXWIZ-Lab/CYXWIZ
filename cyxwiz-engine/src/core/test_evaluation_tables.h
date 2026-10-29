#pragma once

// The Test Evaluation node's outputs (TOFIX157, Keep as node): a test run's
// results as four tables, in Results > Test's words and order
// (test_report::BuildView), so a kept test reads the same on the canvas, in
// Data Studio and in the panel.
//
//   Metrics      metric, value      accuracy, macro precision / recall / F1,
//                                   weighted F1, AUC, loss, rows (a classifier);
//                                   MAE, RMSE, bias, loss, rows (regression)
//   Confusion    true, predicted, count   (a classifier; empty for regression)
//   Per class    class, rows, precision, recall, f1   (a classifier)
//   Worst rows   dataset_row, true, predicted, confidence   the most confident
//                mistakes; regression: dataset_row, true, predicted, error,
//                the largest errors (the first target)

#include "test_executor.h"

#include <memory>
#include <string>

namespace arrow {
class Table;
}

namespace cyxwiz {

struct TestEvaluationTables {
    std::shared_ptr<arrow::Table> metrics, confusion, per_class, worst_rows;
};

// False with `error` when a table cannot be built.
bool BuildTestEvaluationTables(const TestingMetrics& m, size_t worst_rows, TestEvaluationTables& out, std::string& error);

}  // namespace cyxwiz
