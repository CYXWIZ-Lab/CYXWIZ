#include "test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

#include <set>

namespace cyxwiz {

namespace {

std::shared_ptr<arrow::Array> Finish(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    if (!builder.Finish(&array).ok()) return nullptr;
    return array;
}

// The row_id column first, when the source knows one row per sample.
bool AddRowIds(const TestPredictionsSource& source, size_t rows, std::vector<std::shared_ptr<arrow::Field>>& fields,
               std::vector<std::shared_ptr<arrow::Array>>& columns) {
    if (source.row_ids.size() != rows) return true;
    arrow::Int64Builder ids;
    if (!ids.AppendValues(source.row_ids).ok()) return false;
    fields.push_back(arrow::field("row_id", arrow::int64()));
    columns.push_back(Finish(ids));
    return columns.back() != nullptr;
}

std::shared_ptr<arrow::Table> MakeTable(const TestPredictionsSource& source, const std::vector<std::string>& class_names,
                                        std::vector<std::shared_ptr<arrow::Field>> fields,
                                        const std::vector<std::shared_ptr<arrow::Array>>& columns) {
    for (const auto& column : columns) {
        if (!column) return nullptr;
    }
    std::vector<std::string> keys, values;
    const auto put = [&](const char* key, std::string value) {
        keys.emplace_back(key);
        values.push_back(std::move(value));
    };
    if (!source.dataset.empty()) put(kTestPredictionsSourceKey, source.dataset);
    if (!class_names.empty()) {
        std::string joined;
        for (size_t i = 0; i < class_names.size(); ++i) joined += (i ? "\n" : "") + class_names[i];
        put(kTestPredictionsClassNamesKey, joined);
    }
    if (source.loss) put(kTestPredictionsLossKey, std::to_string(*source.loss));
    if (source.seconds > 0.0) put(kTestPredictionsSecondsKey, std::to_string(source.seconds));
    if (source.partial) put(kTestPredictionsPartialKey, "true");
    std::shared_ptr<arrow::KeyValueMetadata> metadata;
    if (!keys.empty()) metadata = arrow::key_value_metadata(keys, values);
    return arrow::Table::Make(arrow::schema(std::move(fields), metadata), columns);
}

}  // namespace

std::shared_ptr<arrow::Table> BuildTestPredictionsTable(const std::vector<int>& ground_truth,
                                                        const std::vector<int>& predictions,
                                                        const std::vector<float>& confidences,
                                                        const std::vector<float>& class_probabilities,
                                                        int probability_classes,
                                                        const std::vector<std::string>& class_names,
                                                        const TestPredictionsSource& source) {
    const size_t rows = ground_truth.size();
    if (rows == 0 || predictions.size() != rows || confidences.size() != rows) return nullptr;
    const size_t classes = probability_classes > 1 ? static_cast<size_t>(probability_classes) : 0;
    const bool with_scores = classes > 0 && class_probabilities.size() == rows * classes;

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    if (!AddRowIds(source, rows, fields, columns)) return nullptr;

    arrow::Int64Builder truth;
    arrow::Int64Builder predicted;
    arrow::DoubleBuilder confidence;
    for (size_t i = 0; i < rows; ++i) {
        if (!truth.Append(ground_truth[i]).ok() || !predicted.Append(predictions[i]).ok() ||
            !confidence.Append(confidences[i]).ok()) {
            return nullptr;
        }
    }
    fields.push_back(arrow::field("true_class", arrow::int64()));
    fields.push_back(arrow::field("predicted_class", arrow::int64()));
    fields.push_back(arrow::field("confidence", arrow::float64()));
    columns.push_back(Finish(truth));
    columns.push_back(Finish(predicted));
    columns.push_back(Finish(confidence));

    if (with_scores) {
        // Binary: the positive class's probability; multi-class: one per class.
        const size_t first = classes == 2 ? 1 : 0;
        std::set<std::string> used = {"row_id", "true_class", "predicted_class", "confidence"};
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
    return MakeTable(source, class_names, std::move(fields), columns);
}

std::shared_ptr<arrow::Table> BuildRegressionPredictionsTable(const std::vector<float>& truth,
                                                              const std::vector<float>& predicted, int width,
                                                              const TestPredictionsSource& source) {
    if (width < 1 || truth.empty() || truth.size() != predicted.size() || truth.size() % static_cast<size_t>(width))
        return nullptr;
    const size_t targets = static_cast<size_t>(width);
    const size_t rows = truth.size() / targets;

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    if (!AddRowIds(source, rows, fields, columns)) return nullptr;
    for (size_t t = 0; t < targets; ++t) {
        const std::string suffix = targets == 1 ? "_value" : "_" + std::to_string(t);
        arrow::DoubleBuilder true_values;
        arrow::DoubleBuilder predicted_values;
        arrow::DoubleBuilder errors;
        for (size_t i = 0; i < rows; ++i) {
            const double y = truth[i * targets + t];
            const double p = predicted[i * targets + t];
            if (!true_values.Append(y).ok() || !predicted_values.Append(p).ok() || !errors.Append(p - y).ok())
                return nullptr;
        }
        fields.push_back(arrow::field("true" + suffix, arrow::float64()));
        fields.push_back(arrow::field("predicted" + suffix, arrow::float64()));
        fields.push_back(arrow::field(targets == 1 ? "error" : "error_" + std::to_string(t), arrow::float64()));
        columns.push_back(Finish(true_values));
        columns.push_back(Finish(predicted_values));
        columns.push_back(Finish(errors));
    }
    return MakeTable(source, {}, std::move(fields), columns);
}

}  // namespace cyxwiz
