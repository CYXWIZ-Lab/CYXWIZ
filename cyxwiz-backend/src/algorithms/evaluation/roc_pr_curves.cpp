// Prevent Windows min/max macros from interfering with numeric_limits.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"
#include "classification_metric_contract.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <spdlog/spdlog.h>
#include <string>

#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

std::vector<size_t> ModelEvaluation::ArgSort(
    const std::vector<double>& values,
    bool descending) {
    std::vector<size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    if (descending) {
        std::stable_sort(
            indices.begin(), indices.end(),
            [&values](size_t lhs, size_t rhs) {
                return values[lhs] > values[rhs];
            });
    } else {
        std::stable_sort(
            indices.begin(), indices.end(),
            [&values](size_t lhs, size_t rhs) {
                return values[lhs] < values[rhs];
            });
    }
    return indices;
}

ROCCurveData ModelEvaluation::ComputeROC(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores) {
    ROCCurveData result;
    if (!classification_metric_detail::ValidateBinaryScores(
            y_true, y_scores, result.error_message)) {
        return result;
    }

    const size_t positive_count = static_cast<size_t>(
        std::count(y_true.begin(), y_true.end(), 1));
    const size_t negative_count = y_true.size() - positive_count;
    if (positive_count == 0 || negative_count == 0) {
        result.error_message =
            "ROC requires both positive and negative target samples";
        return result;
    }

    try {
        const auto sorted_indices = ArgSort(y_scores, true);
        result.fpr.reserve(y_true.size() + 1);
        result.tpr.reserve(y_true.size() + 1);
        result.thresholds.reserve(y_true.size() + 1);
        result.fpr.push_back(0.0);
        result.tpr.push_back(0.0);
        result.thresholds.push_back(
            std::numeric_limits<double>::infinity());

        size_t true_positive = 0;
        size_t false_positive = 0;
        size_t position = 0;
        while (position < sorted_indices.size()) {
            const double threshold = y_scores[sorted_indices[position]];
            do {
                if (y_true[sorted_indices[position]] == 1) {
                    ++true_positive;
                } else {
                    ++false_positive;
                }
                ++position;
            } while (
                position < sorted_indices.size() &&
                y_scores[sorted_indices[position]] == threshold);

            result.fpr.push_back(
                static_cast<double>(false_positive) / negative_count);
            result.tpr.push_back(
                static_cast<double>(true_positive) / positive_count);
            result.thresholds.push_back(threshold);
        }

        result.auc = ComputeAUC(result.fpr, result.tpr);
        if (!std::isfinite(result.auc)) {
            result.error_message = "ROC integration produced an invalid AUC";
            return result;
        }
        result.success = true;
        spdlog::debug(
            "ROC curve computed (native host): {} points, AUC={:.4f}",
            result.fpr.size(), result.auc);
    } catch (const std::exception& error) {
        result.error_message = std::string("Exception: ") + error.what();
        spdlog::error("ComputeROC failed: {}", error.what());
    }
    return result;
}

ROCCurveData ModelEvaluation::ComputeMulticlassROC(
    const std::vector<int>& y_true,
    const std::vector<std::vector<double>>& y_scores) {
    ROCCurveData result;
    size_t class_count = 0;
    if (!classification_metric_detail::ValidateMulticlassScores(
            y_true, y_scores, class_count, result.error_message)) {
        return result;
    }

    try {
        result.class_fpr.resize(class_count);
        result.class_tpr.resize(class_count);
        result.class_thresholds.resize(class_count);
        result.class_auc.resize(class_count);
        for (size_t class_index = 0;
             class_index < class_count;
             ++class_index) {
            std::vector<int> binary_labels(y_true.size());
            std::vector<double> class_scores(y_true.size());
            for (size_t sample = 0; sample < y_true.size(); ++sample) {
                binary_labels[sample] =
                    y_true[sample] == static_cast<int>(class_index) ? 1 : 0;
                class_scores[sample] = y_scores[sample][class_index];
            }

            const auto class_roc = ComputeROC(binary_labels, class_scores);
            if (!class_roc.success) {
                result.error_message =
                    "Class " + std::to_string(class_index) + ": " +
                    class_roc.error_message;
                return result;
            }
            result.class_fpr[class_index] = class_roc.fpr;
            result.class_tpr[class_index] = class_roc.tpr;
            result.class_thresholds[class_index] = class_roc.thresholds;
            result.class_auc[class_index] = class_roc.auc;
        }

        result.auc = std::accumulate(
            result.class_auc.begin(), result.class_auc.end(), 0.0) /
            static_cast<double>(class_count);
        result.success = true;
    } catch (const std::exception& error) {
        result.error_message = std::string("Exception: ") + error.what();
    }
    return result;
}

PRCurveData ModelEvaluation::ComputePRCurve(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores) {
    PRCurveData result;
    if (!classification_metric_detail::ValidateBinaryScores(
            y_true, y_scores, result.error_message)) {
        return result;
    }

    const size_t positive_count = static_cast<size_t>(
        std::count(y_true.begin(), y_true.end(), 1));
    if (positive_count == 0) {
        result.error_message =
            "Precision-recall requires at least one positive target sample";
        return result;
    }

    try {
        const auto sorted_indices = ArgSort(y_scores, true);
        std::vector<double> descending_precision;
        std::vector<double> descending_recall;
        std::vector<double> descending_thresholds;
        descending_precision.reserve(y_true.size());
        descending_recall.reserve(y_true.size());
        descending_thresholds.reserve(y_true.size());

        size_t true_positive = 0;
        size_t predicted_positive = 0;
        double previous_recall = 0.0;
        size_t position = 0;
        while (position < sorted_indices.size()) {
            const double threshold = y_scores[sorted_indices[position]];
            do {
                if (y_true[sorted_indices[position]] == 1) {
                    ++true_positive;
                }
                ++predicted_positive;
                ++position;
            } while (
                position < sorted_indices.size() &&
                y_scores[sorted_indices[position]] == threshold);

            const double precision =
                static_cast<double>(true_positive) / predicted_positive;
            const double recall =
                static_cast<double>(true_positive) / positive_count;
            descending_precision.push_back(precision);
            descending_recall.push_back(recall);
            descending_thresholds.push_back(threshold);
            result.average_precision +=
                precision * (recall - previous_recall);
            previous_recall = recall;
        }

        result.precision.assign(
            descending_precision.rbegin(), descending_precision.rend());
        result.recall.assign(
            descending_recall.rbegin(), descending_recall.rend());
        result.thresholds.assign(
            descending_thresholds.rbegin(), descending_thresholds.rend());
        result.precision.push_back(1.0);
        result.recall.push_back(0.0);
        result.success = true;
        spdlog::debug(
            "PR curve computed (native host): {} points, AP={:.4f}",
            result.precision.size(), result.average_precision);
    } catch (const std::exception& error) {
        result.error_message = std::string("Exception: ") + error.what();
    }
    return result;
}

PRCurveData ModelEvaluation::ComputeMulticlassPRCurve(
    const std::vector<int>& y_true,
    const std::vector<std::vector<double>>& y_scores) {
    PRCurveData result;
    size_t class_count = 0;
    if (!classification_metric_detail::ValidateMulticlassScores(
            y_true, y_scores, class_count, result.error_message)) {
        return result;
    }

    try {
        result.class_precision.resize(class_count);
        result.class_recall.resize(class_count);
        result.class_thresholds.resize(class_count);
        result.class_ap.resize(class_count);
        for (size_t class_index = 0;
             class_index < class_count;
             ++class_index) {
            std::vector<int> binary_labels(y_true.size());
            std::vector<double> class_scores(y_true.size());
            for (size_t sample = 0; sample < y_true.size(); ++sample) {
                binary_labels[sample] =
                    y_true[sample] == static_cast<int>(class_index) ? 1 : 0;
                class_scores[sample] = y_scores[sample][class_index];
            }

            const auto class_pr =
                ComputePRCurve(binary_labels, class_scores);
            if (!class_pr.success) {
                result.error_message =
                    "Class " + std::to_string(class_index) + ": " +
                    class_pr.error_message;
                return result;
            }
            result.class_precision[class_index] = class_pr.precision;
            result.class_recall[class_index] = class_pr.recall;
            result.class_thresholds[class_index] = class_pr.thresholds;
            result.class_ap[class_index] = class_pr.average_precision;
        }

        result.average_precision = std::accumulate(
            result.class_ap.begin(), result.class_ap.end(), 0.0) /
            static_cast<double>(class_count);
        result.success = true;
    } catch (const std::exception& error) {
        result.error_message = std::string("Exception: ") + error.what();
    }
    return result;
}

double ModelEvaluation::ComputeAUC(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    bool nondecreasing = true;
    bool nonincreasing = true;
    for (size_t index = 0; index < x.size(); ++index) {
        if (!std::isfinite(x[index]) || !std::isfinite(y[index])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (index == 0) {
            continue;
        }
        nondecreasing = nondecreasing && x[index] >= x[index - 1];
        nonincreasing = nonincreasing && x[index] <= x[index - 1];
    }
    if (!nondecreasing && !nonincreasing) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    long double area = 0.0L;
    for (size_t index = 1; index < x.size(); ++index) {
        area += static_cast<long double>(x[index] - x[index - 1]) *
                static_cast<long double>(y[index] + y[index - 1]) /
                2.0L;
    }
    if (nonincreasing && !nondecreasing) {
        area = -area;
    }
    return static_cast<double>(area);
}

} // namespace cyxwiz
