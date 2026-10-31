#include "test_results_export.h"

#include "test_results_presentation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace cyxwiz {

namespace {

const char* Objective(const TestingMetrics& m) {
    return m.causal_lm_mode ? "causal_lm" : m.regression_mode ? "regression" : m.metric_learning_mode ? "metric_learning"
                                                                                                    : "classification";
}

// A CSV field: quoted when it holds a comma, quote or line break.
std::string Field(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) return text;
    std::string quoted = "\"";
    for (const char c : text) quoted += c == '"' ? std::string("\"\"") : std::string(1, c);
    return quoted + "\"";
}

bool Fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool WriteFile(const std::string& path, const std::string& content, std::string* error) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) return Fail(error, "cannot open " + path + " for writing");
    file << content;
    file.close();
    if (!file) return Fail(error, "could not write " + path + " (is the disk full?)");
    return true;
}

bool Classification(const TestingMetrics& m) {
    return !m.regression_mode && !m.causal_lm_mode;
}

}  // namespace

bool WriteTestResultsCsv(const TestingMetrics& m, const std::string& path, std::string* error) {
    std::ostringstream out;
    out << std::setprecision(6);
    out << "Test Results Summary\nMetric,Value\n";
    out << "Objective," << Objective(m) << "\n";
    out << "Quick test," << (m.partial ? "yes" : "no") << "\n";
    if (std::isfinite(m.test_loss)) out << "Loss," << m.test_loss << "\n";  // a table read as a test has none
    out << "Samples," << m.total_samples << "\n";
    if (!m.source_dataset.empty()) out << "Tested table," << Field(m.source_dataset) << "\n";
    if (m.causal_lm_mode) {
        out << "Valid target tokens," << m.total_target_values << "\n";
        out << "Token accuracy," << m.test_accuracy << "\n";
    } else if (m.regression_mode) {
        out << "Target values," << m.total_target_values << "\n";
        out << "MAE," << m.test_mae << "\n";
        out << "RMSE," << m.test_rmse << "\n";
    } else {
        out << "Accuracy," << m.test_accuracy << "\n";
        out << "Macro precision," << m.macro_precision << "\n";
        out << "Macro recall," << m.macro_recall << "\n";
        out << "Macro F1," << m.macro_f1 << "\n";
        out << "Weighted F1," << m.weighted_f1 << "\n";
        if (m.roc && m.roc->success) out << "AUC," << m.roc->auc << "\n";
        if (m.pr && m.pr->success) out << "Average precision," << m.pr->average_precision << "\n";
    }
    out << "Time (seconds)," << m.total_time_seconds << "\n";
    out << "Samples per second," << m.samples_per_second << "\n";

    if (Classification(m)) {
        out << "\nPer-class metrics\nClass,Precision,Recall,F1,Found (TP),False alarms (FP),Missed (FN),Support\n";
        for (size_t i = 0; i < m.per_class_metrics.size(); ++i) {
            const auto& c = m.per_class_metrics[i];
            out << Field(test_report::ClassName(m, static_cast<int>(i))) << "," << c.precision << "," << c.recall << ","
                << c.f1_score << "," << c.true_positives << "," << c.false_positives << "," << c.false_negatives << ","
                << c.support << "\n";
        }
        const auto& conf = m.confusion_matrix;
        out << "\nConfusion matrix (rows: actual, columns: predicted)\n";
        for (int j = 0; j < conf.num_classes; ++j) out << "," << Field(test_report::ClassName(m, j));
        out << "\n";
        for (int i = 0; i < conf.num_classes && static_cast<size_t>(i) < conf.matrix.size(); ++i) {
            out << Field(test_report::ClassName(m, i));
            for (const int value : conf.matrix[static_cast<size_t>(i)]) out << "," << value;
            out << "\n";
        }
        const size_t samples = std::min(m.predictions.size(), m.ground_truth.size());
        if (samples > 0) {
            const bool by_dataset = m.row_ids.size() == samples;
            out << "\nPredictions\nIndex," << (by_dataset ? "Dataset row," : "") << "Predicted,Actual,Correct,Confidence\n";
            for (size_t i = 0; i < samples; ++i) {
                out << i << ",";
                if (by_dataset) out << m.row_ids[i] + 1 << ",";
                out << Field(test_report::ClassName(m, m.predictions[i])) << ","
                    << Field(test_report::ClassName(m, m.ground_truth[i])) << ","
                    << (m.predictions[i] == m.ground_truth[i] ? "yes" : "no") << ","
                    << (i < m.confidences.size() ? m.confidences[i] : 0.0f) << "\n";
            }
        }
    }
    // A regression's kept rows: true and predicted per target.
    const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
    if (m.regression_mode && width > 0 && !m.regression_truth.empty() &&
        m.regression_truth.size() == m.regression_predictions.size()) {
        const size_t rows = m.regression_truth.size() / width;
        const bool by_dataset = m.row_ids.size() == rows;
        out << "\nRows\nIndex" << (by_dataset ? ",Dataset row" : "");
        for (size_t k = 0; k < width; ++k) {
            const std::string n = width == 1 ? "" : " " + std::to_string(k + 1);
            out << ",True" << n << ",Predicted" << n << ",Error" << n;
        }
        out << "\n";
        for (size_t i = 0; i < rows; ++i) {
            out << i;
            if (by_dataset) out << "," << m.row_ids[i] + 1;
            for (size_t k = 0; k < width; ++k) {
                const float y = m.regression_truth[i * width + k], p = m.regression_predictions[i * width + k];
                out << "," << y << "," << p << "," << (p - y);
            }
            out << "\n";
        }
    }
    return WriteFile(path, out.str(), error);
}

bool WriteTestResultsJson(const TestingMetrics& m, const std::string& path, std::string* error) {
    nlohmann::json j;
    j["objective"] = Objective(m);
    j["quick_test"] = m.partial;
    if (m.partial) {
        j["batch_limit"] = m.batch_limit;
        j["total_batches"] = m.total_batches;
    }
    if (std::isfinite(m.test_loss)) j["loss"] = m.test_loss;
    j["total_samples"] = m.total_samples;
    j["time_seconds"] = m.total_time_seconds;
    j["samples_per_second"] = m.samples_per_second;
    if (!m.source_dataset.empty()) j["tested_table"] = m.source_dataset;
    if (m.causal_lm_mode) {
        j["valid_target_tokens"] = m.total_target_values;
        j["token_accuracy"] = m.test_accuracy;
    } else if (m.regression_mode) {
        j["total_target_values"] = m.total_target_values;
        j["mae"] = m.test_mae;
        j["rmse"] = m.test_rmse;
        const size_t width = static_cast<size_t>(std::max(m.regression_width, 0));
        if (width > 0 && !m.regression_truth.empty() && m.regression_truth.size() == m.regression_predictions.size()) {
            const size_t rows = m.regression_truth.size() / width;
            nlohmann::json out_rows = nlohmann::json::array();
            for (size_t i = 0; i < rows; ++i) {
                nlohmann::json row = {
                    {"true", std::vector<float>(m.regression_truth.begin() + static_cast<std::ptrdiff_t>(i * width),
                                                m.regression_truth.begin() + static_cast<std::ptrdiff_t>((i + 1) * width))},
                    {"predicted", std::vector<float>(m.regression_predictions.begin() + static_cast<std::ptrdiff_t>(i * width),
                                                     m.regression_predictions.begin() + static_cast<std::ptrdiff_t>((i + 1) * width))}};
                if (m.row_ids.size() == rows) row["dataset_row"] = m.row_ids[i] + 1;
                out_rows.push_back(std::move(row));
            }
            j["rows"] = out_rows;
        }
    } else {
        j["accuracy"] = m.test_accuracy;
        j["macro_precision"] = m.macro_precision;
        j["macro_recall"] = m.macro_recall;
        j["macro_f1"] = m.macro_f1;
        j["weighted_f1"] = m.weighted_f1;
    }
    if (m.pair_metrics) {
        const auto& p = *m.pair_metrics;
        j["pair_metrics"] = {{"accuracy", p.accuracy},
                             {"threshold", p.threshold},
                             {"pair_count", p.pair_count},
                             {"positive_distance_mean", p.positive_distance_mean},
                             {"negative_distance_mean", p.negative_distance_mean}};
    }
    if (m.retrieval_metrics) {
        const auto& r = *m.retrieval_metrics;
        j["retrieval_metrics"] = {{"k", r.k}, {"recall_at_k", r.recall_at_k}, {"mean_reciprocal_rank", r.mean_reciprocal_rank}};
    }
    if (Classification(m)) {
        nlohmann::json classes = nlohmann::json::array();
        for (size_t i = 0; i < m.per_class_metrics.size(); ++i) {
            const auto& c = m.per_class_metrics[i];
            classes.push_back({{"class_id", c.class_id},
                               {"class_name", test_report::ClassName(m, static_cast<int>(i))},
                               {"precision", c.precision},
                               {"recall", c.recall},
                               {"f1_score", c.f1_score},
                               {"support", c.support},
                               {"true_positives", c.true_positives},
                               {"false_positives", c.false_positives},
                               {"false_negatives", c.false_negatives}});
        }
        j["per_class_metrics"] = classes;
        nlohmann::json names = nlohmann::json::array();
        for (int i = 0; i < m.confusion_matrix.num_classes; ++i) names.push_back(test_report::ClassName(m, i));
        j["confusion_matrix"] = {{"class_names", names}, {"matrix", m.confusion_matrix.matrix}};
        const size_t samples = std::min(m.predictions.size(), m.ground_truth.size());
        nlohmann::json predictions = nlohmann::json::array();
        for (size_t i = 0; i < samples; ++i)
        {
            nlohmann::json p = {{"predicted", m.predictions[i]},
                                {"actual", m.ground_truth[i]},
                                {"confidence", i < m.confidences.size() ? m.confidences[i] : 0.0f}};
            if (m.row_ids.size() == samples) p["dataset_row"] = m.row_ids[i] + 1;
            predictions.push_back(std::move(p));
        }
        j["predictions"] = predictions;
        if (m.roc && m.roc->success)
            j["roc"] = {{"auc", m.roc->auc},
                        {"fpr", m.roc->fpr},
                        {"tpr", m.roc->tpr},
                        {"class_auc", m.roc->class_auc},
                        {"class_fpr", m.roc->class_fpr},
                        {"class_tpr", m.roc->class_tpr}};
        if (m.pr && m.pr->success)
            j["precision_recall"] = {{"average_precision", m.pr->average_precision},
                                     {"recall", m.pr->recall},
                                     {"precision", m.pr->precision},
                                     {"class_ap", m.pr->class_ap},
                                     {"class_recall", m.pr->class_recall},
                                     {"class_precision", m.pr->class_precision}};
        if (!m.curves_note.empty()) j["curves_note"] = m.curves_note;
    }
    return WriteFile(path, j.dump(2) + "\n", error);
}

}  // namespace cyxwiz
