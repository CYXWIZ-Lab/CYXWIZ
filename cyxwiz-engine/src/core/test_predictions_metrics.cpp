#include "test_predictions_metrics.h"

#include "arrow_number.h"
#include "test_predictions_table.h"

#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace cyxwiz {

std::shared_ptr<arrow::Table> PredictionsTableFromMetrics(const TestingMetrics& m) {
    if (m.causal_lm_mode) return nullptr;
    TestPredictionsSource source{m.row_ids, m.source_dataset};
    source.loss = m.test_loss;
    source.seconds = m.total_time_seconds;
    source.partial = m.partial;
    if (m.regression_mode)
        return BuildRegressionPredictionsTable(m.regression_truth, m.regression_predictions, m.regression_width, source);
    if (m.ground_truth.empty()) return nullptr;
    std::vector<std::string> class_names;
    for (const auto& c : m.per_class_metrics) class_names.push_back(c.class_name);
    return BuildTestPredictionsTable(m.ground_truth, m.predictions, m.confidences, m.class_probabilities,
                                     m.probability_classes, class_names, source);
}

namespace {

std::string MetadataValue(const arrow::Table& table, const char* key) {
    const auto metadata = table.schema()->metadata();
    if (!metadata) return {};
    const int at = metadata->FindKey(key);
    return at >= 0 ? metadata->value(at) : std::string();
}

std::vector<std::string> SplitLines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        lines.push_back(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return lines;
}

// A column of whole numbers (a class index, a row id); false when it is not a
// number column, has a missing value or a fraction.
bool Whole(const arrow::Table& table, const std::string& name, std::vector<int64_t>& out) {
    const std::vector<double> values = ArrowNumbers(table.GetColumnByName(name));
    if (values.size() != static_cast<size_t>(table.num_rows())) return false;
    out.clear();
    out.reserve(values.size());
    for (double v : values) {
        if (std::isnan(v) || v != std::floor(v)) return false;
        out.push_back(static_cast<int64_t>(v));
    }
    return true;
}

bool Real(const arrow::Table& table, const std::string& name, std::vector<double>& out) {
    out = ArrowNumbers(table.GetColumnByName(name));
    return out.size() == static_cast<size_t>(table.num_rows());
}

bool ReadClassifier(const arrow::Table& table, TestingMetrics& m, std::string& error) {
    const size_t rows = static_cast<size_t>(table.num_rows());
    std::vector<int64_t> truth, predicted;
    if (!Whole(table, "true_class", truth) || !Whole(table, "predicted_class", predicted)) {
        error = "the true_class / predicted_class columns are missing or not whole numbers";
        return false;
    }
    std::vector<double> confidence;
    if (!Real(table, "confidence", confidence)) confidence.assign(rows, 1.0);

    // The scores: binary "score" (the positive class), multi-class one
    // "score_..." column per class, in order.
    std::vector<std::vector<double>> scores;
    if (table.GetColumnByName("score")) {
        std::vector<double> positive;
        if (Real(table, "score", positive)) {
            std::vector<double> negative(rows);
            for (size_t i = 0; i < rows; ++i) negative[i] = 1.0 - positive[i];
            scores = {std::move(negative), std::move(positive)};
        }
    } else {
        for (const auto& field : table.schema()->fields()) {
            if (field->name().rfind("score_", 0) != 0) continue;
            std::vector<double> column;
            if (!Real(table, field->name(), column)) {
                scores.clear();
                break;
            }
            scores.push_back(std::move(column));
        }
    }

    std::vector<std::string> names;
    const std::string joined = MetadataValue(table, kTestPredictionsClassNamesKey);
    if (!joined.empty()) names = SplitLines(joined);
    int64_t highest = -1;
    for (size_t i = 0; i < rows; ++i) {
        if (truth[i] < 0 || predicted[i] < 0) {
            error = "a class index is negative";
            return false;
        }
        highest = std::max({highest, truth[i], predicted[i]});
    }
    const int classes = static_cast<int>(std::max<int64_t>({highest + 1, static_cast<int64_t>(names.size()),
                                                            static_cast<int64_t>(scores.size())}));

    m.confusion_matrix.Resize(classes);
    m.confusion_matrix.class_names = names;
    m.per_class_metrics.assign(static_cast<size_t>(classes), ClassMetrics{});
    for (int c = 0; c < classes; ++c)
        m.per_class_metrics[static_cast<size_t>(c)].class_name =
            static_cast<size_t>(c) < names.size() && !names[static_cast<size_t>(c)].empty() ? names[static_cast<size_t>(c)]
                                                                                            : "Class " + std::to_string(c);
    m.ground_truth.reserve(rows);
    m.predictions.reserve(rows);
    m.confidences.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        const int t = static_cast<int>(truth[i]);
        const int p = static_cast<int>(predicted[i]);
        m.ground_truth.push_back(t);
        m.predictions.push_back(p);
        m.confidences.push_back(static_cast<float>(confidence[i]));
        m.confusion_matrix.Add(t, p);
        if (t == p) ++m.correct_predictions;
    }
    if (scores.size() >= 2 && static_cast<int>(scores.size()) == classes) {
        m.probability_classes = classes;
        m.class_probabilities.reserve(rows * scores.size());
        for (size_t i = 0; i < rows; ++i)
            for (const auto& column : scores) m.class_probabilities.push_back(static_cast<float>(column[i]));
    }
    FinishClassifierMetrics(m);
    return true;
}

bool ReadRegression(const arrow::Table& table, TestingMetrics& m, std::string& error) {
    const size_t rows = static_cast<size_t>(table.num_rows());
    std::vector<std::pair<std::string, std::string>> targets;  // true, predicted
    if (table.GetColumnByName("true_value")) {
        targets.emplace_back("true_value", "predicted_value");
    } else {
        for (int t = 0; table.GetColumnByName("true_" + std::to_string(t)); ++t)
            targets.emplace_back("true_" + std::to_string(t), "predicted_" + std::to_string(t));
    }
    if (targets.empty()) {
        error = "the table has neither class columns (true_class) nor target columns (true_value)";
        return false;
    }
    const size_t width = targets.size();
    std::vector<std::vector<double>> truth(width), predicted(width);
    for (size_t t = 0; t < width; ++t) {
        if (!Real(table, targets[t].first, truth[t]) || !Real(table, targets[t].second, predicted[t])) {
            error = "'" + targets[t].first + "' / '" + targets[t].second + "' are missing or not numbers";
            return false;
        }
    }
    m.regression_mode = true;
    m.regression_width = static_cast<int>(width);
    m.regression_truth.reserve(rows * width);
    m.regression_predictions.reserve(rows * width);
    double absolute = 0.0, squared = 0.0;
    size_t counted = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t t = 0; t < width; ++t) {
            m.regression_truth.push_back(static_cast<float>(truth[t][i]));
            m.regression_predictions.push_back(static_cast<float>(predicted[t][i]));
            const double e = predicted[t][i] - truth[t][i];
            if (std::isnan(e)) continue;
            absolute += std::abs(e);
            squared += e * e;
            ++counted;
        }
    }
    m.total_target_values = rows * width;
    m.test_mae = counted ? static_cast<float>(absolute / static_cast<double>(counted)) : 0.0f;
    m.test_rmse = counted ? static_cast<float>(std::sqrt(squared / static_cast<double>(counted))) : 0.0f;
    return true;
}

}  // namespace

bool MetricsFromPredictionsTable(const arrow::Table& table, TestingMetrics& out, std::string& error) {
    out = TestingMetrics{};
    if (table.num_rows() == 0) {
        error = "the table has no rows";
        return false;
    }
    const bool classifier = table.GetColumnByName("true_class") != nullptr;
    if (!(classifier ? ReadClassifier(table, out, error) : ReadRegression(table, out, error))) return false;

    out.total_samples = static_cast<int>(table.num_rows());
    out.source_dataset = MetadataValue(table, kTestPredictionsSourceKey);
    if (table.GetColumnByName("row_id") && !Whole(table, "row_id", out.row_ids)) out.row_ids.clear();
    const std::string loss = MetadataValue(table, kTestPredictionsLossKey);
    if (!loss.empty()) out.test_loss = std::strtof(loss.c_str(), nullptr);
    const std::string seconds = MetadataValue(table, kTestPredictionsSecondsKey);
    if (!seconds.empty()) out.total_time_seconds = std::strtof(seconds.c_str(), nullptr);
    if (out.total_time_seconds > 0.0f) out.samples_per_second = static_cast<float>(out.total_samples) / out.total_time_seconds;
    out.partial = MetadataValue(table, kTestPredictionsPartialKey) == "true";
    out.is_complete = true;
    out.status_message = "Testing complete";
    return true;
}

}  // namespace cyxwiz
