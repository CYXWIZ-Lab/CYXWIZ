// Prevent Windows min/max macros from interfering with standard library functions.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"

#include <algorithm>
#include <cmath>
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

    if (y_true.size() != y_pred.size() || y_true.empty()) {
        result.error_message = "Invalid input: sizes don't match or empty";
        return result;
    }

    try {
        // Find unique classes
        std::set<int> classes_set(y_true.begin(), y_true.end());
        classes_set.insert(y_pred.begin(), y_pred.end());
        std::vector<int> classes(classes_set.begin(), classes_set.end());
        std::sort(classes.begin(), classes.end());

        result.n_classes = static_cast<int>(classes.size());
        result.total_samples = static_cast<int>(y_true.size());

        // Create class index mapping
        std::map<int, int> class_to_idx;
        for (size_t i = 0; i < classes.size(); ++i) {
            class_to_idx[classes[i]] = static_cast<int>(i);
        }

        // Initialize confusion matrix
        result.matrix.assign(result.n_classes, std::vector<int>(result.n_classes, 0));

        // Fill confusion matrix
        for (size_t i = 0; i < y_true.size(); ++i) {
            int true_idx = class_to_idx[y_true[i]];
            int pred_idx = class_to_idx[y_pred[i]];
            result.matrix[true_idx][pred_idx]++;
        }

        // Set class names
        if (!class_names.empty() && class_names.size() == static_cast<size_t>(result.n_classes)) {
            result.class_names = class_names;
        } else {
            result.class_names.resize(result.n_classes);
            for (int i = 0; i < result.n_classes; ++i) {
                result.class_names[i] = "Class " + std::to_string(classes[i]);
            }
        }

        // Compute per-class metrics
        result.precision.resize(result.n_classes);
        result.recall.resize(result.n_classes);
        result.f1_scores.resize(result.n_classes);
        result.support.resize(result.n_classes);

        int correct = 0;
        for (int i = 0; i < result.n_classes; ++i) {
            int tp = result.matrix[i][i];
            int fp = 0, fn = 0;

            for (int j = 0; j < result.n_classes; ++j) {
                if (j != i) {
                    fp += result.matrix[j][i];  // Column sum - diagonal
                    fn += result.matrix[i][j];  // Row sum - diagonal
                }
                if (i == j) correct += result.matrix[i][j];
            }

            result.support[i] = 0;
            for (int j = 0; j < result.n_classes; ++j) {
                result.support[i] += result.matrix[i][j];
            }

            // Precision = TP / (TP + FP)
            double denom_prec = tp + fp;
            result.precision[i] = (denom_prec > 0) ? static_cast<double>(tp) / denom_prec : 0.0;

            // Recall = TP / (TP + FN)
            double denom_rec = tp + fn;
            result.recall[i] = (denom_rec > 0) ? static_cast<double>(tp) / denom_rec : 0.0;

            // F1 = 2 * (precision * recall) / (precision + recall)
            double denom_f1 = result.precision[i] + result.recall[i];
            result.f1_scores[i] = (denom_f1 > 0) ? 2.0 * result.precision[i] * result.recall[i] / denom_f1 : 0.0;
        }

        // Overall accuracy
        result.accuracy = static_cast<double>(correct) / result.total_samples;

        // Macro averages (unweighted)
        result.macro_precision = std::accumulate(result.precision.begin(), result.precision.end(), 0.0) / result.n_classes;
        result.macro_recall = std::accumulate(result.recall.begin(), result.recall.end(), 0.0) / result.n_classes;
        result.macro_f1 = std::accumulate(result.f1_scores.begin(), result.f1_scores.end(), 0.0) / result.n_classes;

        // Weighted F1
        result.weighted_f1 = 0.0;
        for (int i = 0; i < result.n_classes; ++i) {
            result.weighted_f1 += result.f1_scores[i] * result.support[i];
        }
        result.weighted_f1 /= result.total_samples;

        result.success = true;
        spdlog::debug("Confusion matrix computed: {} classes, {} samples, accuracy={:.4f}",
                      result.n_classes, result.total_samples, result.accuracy);

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        spdlog::error("ComputeConfusionMatrix failed: {}", e.what());
    }

    return result;
}

BinaryMetrics ModelEvaluation::ComputeBinaryMetrics(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    double threshold) {

    BinaryMetrics result;
    result.threshold = threshold;

    if (y_true.size() != y_scores.size() || y_true.empty()) {
        return result;
    }

    // Count TP, FP, TN, FN
    for (size_t i = 0; i < y_true.size(); ++i) {
        int pred = (y_scores[i] >= threshold) ? 1 : 0;
        int actual = y_true[i];

        if (pred == 1 && actual == 1) result.tp++;
        else if (pred == 1 && actual == 0) result.fp++;
        else if (pred == 0 && actual == 0) result.tn++;
        else result.fn++;
    }

    // Compute metrics
    double denom;

    // Precision
    denom = result.tp + result.fp;
    result.precision = (denom > 0) ? static_cast<double>(result.tp) / denom : 0.0;

    // Recall (Sensitivity, TPR)
    denom = result.tp + result.fn;
    result.recall = (denom > 0) ? static_cast<double>(result.tp) / denom : 0.0;

    // Specificity (TNR)
    denom = result.tn + result.fp;
    result.specificity = (denom > 0) ? static_cast<double>(result.tn) / denom : 0.0;

    // F1
    denom = result.precision + result.recall;
    result.f1 = (denom > 0) ? 2.0 * result.precision * result.recall / denom : 0.0;

    // Balanced accuracy
    result.balanced_accuracy = (result.recall + result.specificity) / 2.0;

    // Matthews Correlation Coefficient
    double mcc_num = static_cast<double>(result.tp * result.tn - result.fp * result.fn);
    double mcc_denom = std::sqrt(
        static_cast<double>(result.tp + result.fp) *
        static_cast<double>(result.tp + result.fn) *
        static_cast<double>(result.tn + result.fp) *
        static_cast<double>(result.tn + result.fn));
    result.mcc = (mcc_denom > 0) ? mcc_num / mcc_denom : 0.0;

    return result;
}

ClassificationReport ModelEvaluation::GenerateClassificationReport(
    const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    const std::vector<std::string>& class_names) {

    ClassificationReport report;

    // Get confusion matrix
    report.confusion_matrix = ComputeConfusionMatrix(y_true, y_pred, class_names);

    if (!report.confusion_matrix.success) {
        report.error_message = report.confusion_matrix.error_message;
        return report;
    }

    // Overall metrics
    report.overall_metrics["accuracy"] = report.confusion_matrix.accuracy;
    report.overall_metrics["macro_precision"] = report.confusion_matrix.macro_precision;
    report.overall_metrics["macro_recall"] = report.confusion_matrix.macro_recall;
    report.overall_metrics["macro_f1"] = report.confusion_matrix.macro_f1;
    report.overall_metrics["weighted_f1"] = report.confusion_matrix.weighted_f1;

    // Per-class metrics
    for (int i = 0; i < report.confusion_matrix.n_classes; ++i) {
        std::string cls_name = report.confusion_matrix.class_names[i];
        report.per_class_metrics[cls_name]["precision"] = report.confusion_matrix.precision[i];
        report.per_class_metrics[cls_name]["recall"] = report.confusion_matrix.recall[i];
        report.per_class_metrics[cls_name]["f1"] = report.confusion_matrix.f1_scores[i];
        report.per_class_metrics[cls_name]["support"] = static_cast<double>(report.confusion_matrix.support[i]);
    }

    report.success = true;
    return report;
}

double ModelEvaluation::FindOptimalThreshold(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    const std::string& criterion) {

    if (y_true.empty() || y_scores.empty()) {
        return 0.5;
    }

    // Generate thresholds to test
    std::vector<double> thresholds;
    std::set<double> unique_scores(y_scores.begin(), y_scores.end());
    for (double s : unique_scores) {
        thresholds.push_back(s);
    }
    std::sort(thresholds.begin(), thresholds.end());

    double best_threshold = 0.5;
    double best_score = -1.0;

    for (double thresh : thresholds) {
        auto metrics = ComputeBinaryMetrics(y_true, y_scores, thresh);

        double score = 0.0;
        if (criterion == "f1") {
            score = metrics.f1;
        } else if (criterion == "youden") {
            // Youden's J = Sensitivity + Specificity - 1
            score = metrics.recall + metrics.specificity - 1.0;
        } else if (criterion == "balanced") {
            score = metrics.balanced_accuracy;
        }

        if (score > best_score) {
            best_score = score;
            best_threshold = thresh;
        }
    }

    return best_threshold;
}

} // namespace cyxwiz
