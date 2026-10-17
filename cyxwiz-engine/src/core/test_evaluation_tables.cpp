#include "test_evaluation_tables.h"

#include "test_results_presentation.h"

#include <arrow/api.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

template <typename Builder>
std::shared_ptr<arrow::Array> Finish(Builder& builder) {
    std::shared_ptr<arrow::Array> array;
    return builder.Finish(&array).ok() ? array : nullptr;
}

std::shared_ptr<arrow::Table> Make(std::vector<std::shared_ptr<arrow::Field>> fields,
                                   const std::vector<std::shared_ptr<arrow::Array>>& columns, std::string& error) {
    for (const auto& column : columns)
        if (!column) {
            error = "a result column could not be built";
            return nullptr;
        }
    return arrow::Table::Make(arrow::schema(std::move(fields)), columns);
}

// One test row's dataset row, counted from 1 as the Table Viewer counts.
void AppendDatasetRow(arrow::Int64Builder& rows, const TestingMetrics& m, size_t i, size_t samples) {
    if (m.row_ids.size() == samples) (void)rows.Append(m.row_ids[i] + 1);
    else (void)rows.AppendNull();
}

}  // namespace

bool BuildTestEvaluationTables(const TestingMetrics& m, size_t worst_rows, TestEvaluationTables& out, std::string& error) {
    out = {};
    const test_report::View view = test_report::BuildView(m, nullptr);

    arrow::StringBuilder metric;
    arrow::DoubleBuilder value;
    const auto add = [&](const char* name, double v) {
        (void)metric.Append(name);
        (void)value.Append(v);
    };
    if (m.regression_mode) {
        add("mae", m.test_mae);
        add("rmse", m.test_rmse);
        const size_t n = m.regression_truth.size();
        if (n && n == m.regression_predictions.size()) {
            double bias = 0.0;
            for (size_t i = 0; i < n; ++i) bias += static_cast<double>(m.regression_predictions[i]) - m.regression_truth[i];
            add("bias", bias / static_cast<double>(n));
        }
    } else {
        add("accuracy", m.test_accuracy);
        add("macro_precision", m.macro_precision);
        add("macro_recall", m.macro_recall);
        add("macro_f1", m.macro_f1);
        add("weighted_f1", m.weighted_f1);
        if (m.roc && m.roc->success) add("auc", m.roc->auc);
    }
    add("loss", m.test_loss);
    add("rows", m.total_samples);
    out.metrics = Make({arrow::field("metric", arrow::utf8()), arrow::field("value", arrow::float64())},
                       {Finish(metric), Finish(value)}, error);
    if (!out.metrics) return false;

    // Confusion and per class: a classifier's; regression gets empty tables.
    arrow::StringBuilder truth_names, predicted_names, class_names;
    arrow::Int64Builder counts, class_rows;
    arrow::DoubleBuilder precision, recall, f1;
    if (!m.regression_mode) {
        const auto& matrix = m.confusion_matrix.matrix;
        for (size_t a = 0; a < matrix.size(); ++a)
            for (size_t b = 0; b < matrix[a].size(); ++b) {
                if (matrix[a][b] == 0) continue;
                (void)truth_names.Append(test_report::ClassName(m, static_cast<int>(a)));
                (void)predicted_names.Append(test_report::ClassName(m, static_cast<int>(b)));
                (void)counts.Append(matrix[a][b]);
            }
        for (size_t c = 0; c < m.per_class_metrics.size(); ++c) {
            const ClassMetrics& cm = m.per_class_metrics[c];
            (void)class_names.Append(test_report::ClassName(m, static_cast<int>(c)));
            (void)class_rows.Append(cm.support);
            (void)precision.Append(cm.precision);
            (void)recall.Append(cm.recall);
            (void)f1.Append(cm.f1_score);
        }
    }
    out.confusion = Make({arrow::field("true", arrow::utf8()), arrow::field("predicted", arrow::utf8()),
                          arrow::field("count", arrow::int64())},
                         {Finish(truth_names), Finish(predicted_names), Finish(counts)}, error);
    out.per_class = Make({arrow::field("class", arrow::utf8()), arrow::field("rows", arrow::int64()),
                          arrow::field("precision", arrow::float64()), arrow::field("recall", arrow::float64()),
                          arrow::field("f1", arrow::float64())},
                         {Finish(class_names), Finish(class_rows), Finish(precision), Finish(recall), Finish(f1)}, error);
    if (!out.confusion || !out.per_class) return false;

    // Worst rows, in the panel's order.
    arrow::Int64Builder dataset_rows;
    if (m.regression_mode) {
        arrow::DoubleBuilder truth, predicted, errors;
        const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
        const size_t samples = width ? m.regression_truth.size() / width : 0;
        for (size_t r = 0; r < view.largest_errors.size() && r < worst_rows; ++r) {
            const size_t i = static_cast<size_t>(view.largest_errors[r]);
            const double y = m.regression_truth[i * width], p = m.regression_predictions[i * width];
            AppendDatasetRow(dataset_rows, m, i, samples);
            (void)truth.Append(y);
            (void)predicted.Append(p);
            (void)errors.Append(p - y);
        }
        out.worst_rows = Make({arrow::field("dataset_row", arrow::int64()), arrow::field("true", arrow::float64()),
                               arrow::field("predicted", arrow::float64()), arrow::field("error", arrow::float64())},
                              {Finish(dataset_rows), Finish(truth), Finish(predicted), Finish(errors)}, error);
    } else {
        arrow::StringBuilder truth, predicted;
        arrow::DoubleBuilder confidence;
        const size_t samples = std::min(m.predictions.size(), m.ground_truth.size());
        for (size_t r = 0; r < view.mistakes.size() && r < worst_rows; ++r) {
            const size_t i = static_cast<size_t>(view.mistakes[r]);
            AppendDatasetRow(dataset_rows, m, i, samples);
            (void)truth.Append(test_report::ClassName(m, m.ground_truth[i]));
            (void)predicted.Append(test_report::ClassName(m, m.predictions[i]));
            (void)confidence.Append(i < m.confidences.size() ? m.confidences[i] : std::nan(""));
        }
        out.worst_rows = Make({arrow::field("dataset_row", arrow::int64()), arrow::field("true", arrow::utf8()),
                               arrow::field("predicted", arrow::utf8()), arrow::field("confidence", arrow::float64())},
                              {Finish(dataset_rows), Finish(truth), Finish(predicted), Finish(confidence)}, error);
    }
    return out.worst_rows != nullptr;
}

}  // namespace cyxwiz
