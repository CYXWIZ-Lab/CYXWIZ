#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// The dataset name the latest test run's per-sample results are published as
// (TOFIX140): ROC / PR nodes and panels can read it like any table.
inline constexpr const char* kTestPredictionsDataset = "test_predictions";

// The schema metadata key naming the table the rows were tested from; the
// row_id column counts rows in it (0-based).
inline constexpr const char* kTestPredictionsSourceKey = "source_dataset";

// More schema metadata (TOFIX157), so a table read back gives the whole run:
// the class names (one per line, a classifier's), the test loss and the
// seconds it took (as text), and "partial" = "true" for a quick test.
inline constexpr const char* kTestPredictionsClassNamesKey = "class_names";
inline constexpr const char* kTestPredictionsLossKey = "test_loss";
inline constexpr const char* kTestPredictionsSecondsKey = "seconds";
inline constexpr const char* kTestPredictionsPartialKey = "partial";

// Where the test rows come from: a row_id column is written when `row_ids`
// has one entry per row, and `dataset` goes in the schema metadata; the
// run's loss, time and quick-test flag go there too when set.
struct TestPredictionsSource {
    std::vector<int64_t> row_ids;
    std::string dataset;
    std::optional<double> loss;
    double seconds = 0.0;
    bool partial = false;
    // Each row's file (images); a "file" column after row_id when one per row.
    std::vector<std::string> files;
};

// One row per test sample: row_id (when known), file (images), true_class, predicted_class,
// confidence, then the scores (the class names also go in the metadata) - binary (two classes): "score", the positive
// class's probability; multi-class: "score_<class name>" per class (the class
// index when names are missing or repeat). `class_probabilities` is row-major
// [samples][classes]; an empty one (scores not kept) leaves out the score
// columns.
std::shared_ptr<arrow::Table> BuildTestPredictionsTable(const std::vector<int>& ground_truth,
                                                        const std::vector<int>& predictions,
                                                        const std::vector<float>& confidences,
                                                        const std::vector<float>& class_probabilities,
                                                        int probability_classes,
                                                        const std::vector<std::string>& class_names,
                                                        const TestPredictionsSource& source = {});

// A regression test: row_id (when known), then per target "true_value",
// "predicted_value" and "error" (predicted - true); with several targets
// "true_<i>", "predicted_<i>", "error_<i>". `truth` and `predicted` are
// row-major [samples][width].
std::shared_ptr<arrow::Table> BuildRegressionPredictionsTable(const std::vector<float>& truth,
                                                              const std::vector<float>& predicted, int width,
                                                              const TestPredictionsSource& source = {});

}  // namespace cyxwiz
