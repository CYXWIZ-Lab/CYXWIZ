// Prevent Windows min/max macros from interfering with standard functions.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"
#include "classification_metric_contract.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <spdlog/spdlog.h>

#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

ConfusionMatrixData ModelEvaluation::ComputeConfusionMatrix(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::string>& class_names) {
    ConfusionMatrixData result;
    if (y_true.empty()) {
        result.error_message = "Inputs must not be empty";
        return result;
    }
    if (y_true.size() != y_pred.size()) {
        result.error_message = "True and predicted labels must have equal length";
        return result;
    }
    if (y_true.size() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
        result.error_message = "Sample count exceeds the public result range";
        return result;
    }

    try {
        std::set<int> class_set(y_true.begin(), y_true.end());
        class_set.insert(y_pred.begin(), y_pred.end());
        const std::vector<int> classes(class_set.begin(), class_set.end());
        if (!class_names.empty() && class_names.size() != classes.size()) {
            result.error_message =
                "Class-name count must match the sorted union of labels";
            return result;
        }

        result.n_classes = static_cast<int>(classes.size());
        result.total_samples = static_cast<int>(y_true.size());
        std::map<int, int> class_to_index;
        for (size_t index = 0; index < classes.size(); ++index) {
            class_to_index.emplace(classes[index], static_cast<int>(index));
        }

        result.matrix.assign(
            result.n_classes, std::vector<int>(result.n_classes, 0));
        for (size_t sample = 0; sample < y_true.size(); ++sample) {
            const int true_index = class_to_index.at(y_true[sample]);
            const int predicted_index = class_to_index.at(y_pred[sample]);
            ++result.matrix[true_index][predicted_index];
        }

        if (!class_names.empty()) {
            result.class_names = class_names;
        } else {
            result.class_names.reserve(classes.size());
            for (int label : classes) {
                result.class_names.push_back("Class " + std::to_string(label));
            }
        }

        result.precision.resize(result.n_classes, 0.0);
        result.recall.resize(result.n_classes, 0.0);
        result.f1_scores.resize(result.n_classes, 0.0);
        result.support.resize(result.n_classes, 0);
        int correct = 0;
        for (int class_index = 0;
             class_index < result.n_classes;
             ++class_index) {
            const int true_positive =
                result.matrix[class_index][class_index];
            int false_positive = 0;
            int false_negative = 0;
            for (int other = 0; other < result.n_classes; ++other) {
                if (other != class_index) {
                    false_positive += result.matrix[other][class_index];
                    false_negative += result.matrix[class_index][other];
                }
                result.support[class_index] +=
                    result.matrix[class_index][other];
            }
            correct += true_positive;

            const double precision_denominator =
                static_cast<double>(true_positive) + false_positive;
            const double recall_denominator =
                static_cast<double>(true_positive) + false_negative;
            result.precision[class_index] = precision_denominator > 0.0
                ? static_cast<double>(true_positive) / precision_denominator
                : 0.0;
            result.recall[class_index] = recall_denominator > 0.0
                ? static_cast<double>(true_positive) / recall_denominator
                : 0.0;
            const double f1_denominator =
                result.precision[class_index] + result.recall[class_index];
            result.f1_scores[class_index] = f1_denominator > 0.0
                ? 2.0 * result.precision[class_index] *
                      result.recall[class_index] / f1_denominator
                : 0.0;
        }

        result.accuracy =
            static_cast<double>(correct) / result.total_samples;
        const double class_count = static_cast<double>(result.n_classes);
        result.macro_precision = std::accumulate(
            result.precision.begin(), result.precision.end(), 0.0) / class_count;
        result.macro_recall = std::accumulate(
            result.recall.begin(), result.recall.end(), 0.0) / class_count;
        result.macro_f1 = std::accumulate(
            result.f1_scores.begin(), result.f1_scores.end(), 0.0) / class_count;
        for (int class_index = 0;
             class_index < result.n_classes;
             ++class_index) {
            result.weighted_f1 += result.f1_scores[class_index] *
                                  result.support[class_index];
        }
        result.weighted_f1 /= result.total_samples;
        result.success = true;
        spdlog::debug(
            "Confusion matrix computed: {} classes, {} samples, accuracy={:.4f}",
            result.n_classes, result.total_samples, result.accuracy);
    } catch (const std::exception& error) {
        result.error_message = std::string("Exception: ") + error.what();
        spdlog::error("ComputeConfusionMatrix failed: {}", error.what());
    }
    return result;
}

BinaryMetrics ModelEvaluation::ComputeBinaryMetrics(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    double threshold) {
    BinaryMetrics result;
    result.threshold = threshold;
    if (!classification_metric_detail::ValidateBinaryScores(
            y_true, y_scores, result.error_message, threshold)) {
        return result;
    }

    for (size_t index = 0; index < y_true.size(); ++index) {
        const int prediction = y_scores[index] >= threshold ? 1 : 0;
        if (prediction == 1 && y_true[index] == 1) {
            ++result.tp;
        } else if (prediction == 1) {
            ++result.fp;
        } else if (y_true[index] == 0) {
            ++result.tn;
        } else {
            ++result.fn;
        }
    }

    const double true_positive = static_cast<double>(result.tp);
    const double false_positive = static_cast<double>(result.fp);
    const double true_negative = static_cast<double>(result.tn);
    const double false_negative = static_cast<double>(result.fn);
    const double precision_denominator = true_positive + false_positive;
    const double recall_denominator = true_positive + false_negative;
    const double specificity_denominator = true_negative + false_positive;
    result.precision = precision_denominator > 0.0
        ? true_positive / precision_denominator
        : 0.0;
    result.recall = recall_denominator > 0.0
        ? true_positive / recall_denominator
        : 0.0;
    result.specificity = specificity_denominator > 0.0
        ? true_negative / specificity_denominator
        : 0.0;
    const double f1_denominator = result.precision + result.recall;
    result.f1 = f1_denominator > 0.0
        ? 2.0 * result.precision * result.recall / f1_denominator
        : 0.0;

    if (recall_denominator > 0.0 && specificity_denominator > 0.0) {
        result.balanced_accuracy =
            (result.recall + result.specificity) / 2.0;
    } else if (recall_denominator > 0.0) {
        result.balanced_accuracy = result.recall;
    } else {
        result.balanced_accuracy = result.specificity;
    }

    const double mcc_numerator =
        true_positive * true_negative - false_positive * false_negative;
    const double mcc_denominator = std::sqrt(
        (true_positive + false_positive) *
        (true_positive + false_negative) *
        (true_negative + false_positive) *
        (true_negative + false_negative));
    result.mcc = mcc_denominator > 0.0
        ? mcc_numerator / mcc_denominator
        : 0.0;
    result.success = true;
    return result;
}

ClassificationReport ModelEvaluation::GenerateClassificationReport(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::string>& class_names) {
    ClassificationReport report;
    report.confusion_matrix =
        ComputeConfusionMatrix(y_true, y_pred, class_names);
    if (!report.confusion_matrix.success) {
        report.error_message = report.confusion_matrix.error_message;
        return report;
    }

    report.overall_metrics["accuracy"] = report.confusion_matrix.accuracy;
    report.overall_metrics["macro_precision"] =
        report.confusion_matrix.macro_precision;
    report.overall_metrics["macro_recall"] =
        report.confusion_matrix.macro_recall;
    report.overall_metrics["macro_f1"] = report.confusion_matrix.macro_f1;
    report.overall_metrics["weighted_f1"] =
        report.confusion_matrix.weighted_f1;
    for (int class_index = 0;
         class_index < report.confusion_matrix.n_classes;
         ++class_index) {
        auto& metrics = report.per_class_metrics[
            report.confusion_matrix.class_names[class_index]];
        metrics["precision"] = report.confusion_matrix.precision[class_index];
        metrics["recall"] = report.confusion_matrix.recall[class_index];
        metrics["f1"] = report.confusion_matrix.f1_scores[class_index];
        metrics["support"] =
            static_cast<double>(report.confusion_matrix.support[class_index]);
    }
    report.success = true;
    return report;
}

double ModelEvaluation::FindOptimalThreshold(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    const std::string& criterion) {
    if (criterion != "f1" && criterion != "youden" &&
        criterion != "balanced") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::string error_message;
    if (!classification_metric_detail::ValidateBinaryScores(
            y_true, y_scores, error_message)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::set<double> thresholds(y_scores.begin(), y_scores.end());
    double best_threshold = std::numeric_limits<double>::quiet_NaN();
    double best_score = -std::numeric_limits<double>::infinity();
    for (double threshold : thresholds) {
        const BinaryMetrics metrics =
            ComputeBinaryMetrics(y_true, y_scores, threshold);
        if (!metrics.success) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double score = metrics.f1;
        if (criterion == "youden") {
            score = metrics.recall + metrics.specificity - 1.0;
        } else if (criterion == "balanced") {
            score = metrics.balanced_accuracy;
        }
        if (score > best_score) {
            best_score = score;
            best_threshold = threshold;
        }
    }
    return best_threshold;
}

} // namespace cyxwiz
