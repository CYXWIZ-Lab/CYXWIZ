#pragma once

#include <memory>
#include <string>
#include <vector>

namespace arrow {
class Table;
}

namespace cyxwiz {

// The dataset name the latest test run's per-sample results are published as
// (TOFIX140): ROC / PR nodes and panels can read it like any table.
inline constexpr const char* kTestPredictionsDataset = "test_predictions";

// One row per test sample: true_class, predicted_class, confidence, then the
// scores - binary (two classes): "score", the positive class's probability;
// multi-class: "score_<class name>" per class (the class index when names are
// missing or repeat). `class_probabilities` is row-major [samples][classes];
// an empty one (scores not kept) leaves out the score columns.
std::shared_ptr<arrow::Table> BuildTestPredictionsTable(const std::vector<int>& ground_truth,
                                                        const std::vector<int>& predictions,
                                                        const std::vector<float>& confidences,
                                                        const std::vector<float>& class_probabilities,
                                                        int probability_classes,
                                                        const std::vector<std::string>& class_names);

}  // namespace cyxwiz
