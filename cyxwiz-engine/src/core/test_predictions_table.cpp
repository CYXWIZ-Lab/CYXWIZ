#include "test_predictions_table.h"

#include <arrow/api.h>

#include <set>

namespace cyxwiz {

namespace {

std::shared_ptr<arrow::Array> Finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    if (!builder.Finish(&array).ok()) return nullptr;
    return array;
}

}  // namespace

std::shared_ptr<arrow::Table> BuildTestPredictionsTable(const std::vector<int>& ground_truth,
                                                        const std::vector<int>& predictions,
                                                        const std::vector<float>& confidences,
                                                        const std::vector<float>& class_probabilities,
                                                        int probability_classes,
                                                        const std::vector<std::string>& class_names) {
    const size_t rows = ground_truth.size();
    if (rows == 0 || predictions.size() != rows || confidences.size() != rows) return nullptr;
    const size_t classes = probability_classes > 1 ? static_cast<size_t>(probability_classes) : 0;
    const bool with_scores = classes > 0 && class_probabilities.size() == rows * classes;

    arrow::Int64Builder truth;
    arrow::Int64Builder predicted;
    arrow::DoubleBuilder confidence;
    for (size_t i = 0; i < rows; ++i) {
        if (!truth.Append(ground_truth[i]).ok() || !predicted.Append(predictions[i]).ok() ||
            !confidence.Append(confidences[i]).ok()) {
            return nullptr;
        }
    }
    std::vector<std::shared_ptr<arrow::Field>> fields = {arrow::field("true_class", arrow::int64()),
                                                         arrow::field("predicted_class", arrow::int64()),
                                                         arrow::field("confidence", arrow::float64())};
    std::vector<std::shared_ptr<arrow::Array>> columns = {Finish(truth), Finish(predicted), Finish(confidence)};

    if (with_scores) {
        // Binary: the positive class's probability; multi-class: one per class.
        const size_t first = classes == 2 ? 1 : 0;
        std::set<std::string> used = {"true_class", "predicted_class", "confidence"};
        for (size_t c = first; c < classes; ++c) {
            std::string name = classes == 2 ? "score" : "score_" + std::to_string(c);
            if (classes > 2 && c < class_names.size() && !class_names[c].empty() &&
                !used.count("score_" + class_names[c])) {
                name = "score_" + class_names[c];
            }
            used.insert(name);
            arrow::DoubleBuilder score;
            for (size_t i = 0; i < rows; ++i) {
                if (!score.Append(class_probabilities[i * classes + c]).ok()) return nullptr;
            }
            fields.push_back(arrow::field(name, arrow::float64()));
            columns.push_back(Finish(score));
        }
    }
    for (const auto& column : columns) {
        if (!column) return nullptr;
    }
    return arrow::Table::Make(arrow::schema(fields), columns);
}

}  // namespace cyxwiz
