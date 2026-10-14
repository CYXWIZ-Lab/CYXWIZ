// Prevent Windows min/max macros from interfering with numeric_limits.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"
#include "classification_metric_contract.h"
#include "../arrayfire_host_values.h"
#include "../arrayfire_indexed.h"

#include <arrayfire.h>

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

namespace {

// Cumulative counts at each distinct score, scores descending (sklearn's
// _binary_clf_curve): the scores are sorted on the device, the labels summed
// along them, and the last position of every run of equal scores kept.
struct ThresholdCounts {
    af::array thresholds;  // distinct scores, descending
    af::array tps;         // positives scoring >= threshold
    af::array fps;         // negatives scoring >= threshold
};

ThresholdCounts CountsAtThresholds(const std::vector<int>& y_true, const std::vector<double>& y_scores) {
    const dim_t n = static_cast<dim_t>(y_scores.size());
    const af::array scores = HostValuesToDevice(y_scores);
    const af::array labels = af::array(n, y_true.data()).as(scores.type());
    af::array sorted;
    af::array order;
    SortWithIndex(sorted, order, scores, 0, false);
    const af::array tps = af::accum(labels(order));
    const af::array fps = (af::range(af::dim4(n), 0, scores.type()) + 1.0) - tps;
    af::array ends = af::constant(static_cast<unsigned>(n - 1), 1, u32);
    if (n > 1) {
        const af::array group_ends = af::where(af::diff1(sorted) != 0);
        if (group_ends.elements() > 0) ends = af::join(0, group_ends.as(u32), ends);
    }
    return {sorted(ends), tps(ends), fps(ends)};
}

}  // namespace

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
        const ThresholdCounts counts = CountsAtThresholds(y_true, y_scores);
        const af::array zero = af::constant(0.0, 1, counts.tps.type());
        const af::array fpr = af::join(0, zero, counts.fps / static_cast<double>(negative_count));
        const af::array tpr = af::join(0, zero, counts.tps / static_cast<double>(positive_count));
        // Trapezoids between successive points (fpr is non-decreasing).
        const dim_t points = fpr.elements();
        const af::array seq_hi = af::seq(1, static_cast<double>(points - 1));
        const af::array seq_lo = af::seq(0, static_cast<double>(points - 2));
        const af::array area = (fpr(seq_hi) - fpr(seq_lo)) * (tpr(seq_hi) + tpr(seq_lo)) * 0.5;
        result.auc = af::sum<double>(area);

        result.fpr = DeviceToHostValues(fpr, "ModelEvaluation::ComputeROC");
        result.tpr = DeviceToHostValues(tpr, "ModelEvaluation::ComputeROC");
        result.thresholds = DeviceToHostValues(counts.thresholds, "ModelEvaluation::ComputeROC");
        result.thresholds.insert(result.thresholds.begin(), std::numeric_limits<double>::infinity());
        if (!std::isfinite(result.auc)) {
            result.error_message = "ROC integration produced an invalid AUC";
            return result;
        }
        result.success = true;
        spdlog::debug("ROC curve computed on the device: {} points, AUC={:.4f}", result.fpr.size(), result.auc);
    } catch (const af::exception& error) {
        result.error_message = DeviceError("ModelEvaluation::ComputeROC", error);
        spdlog::error("{}", result.error_message);
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
        const ThresholdCounts counts = CountsAtThresholds(y_true, y_scores);
        const af::array precision = counts.tps / (counts.tps + counts.fps);
        const af::array recall = counts.tps / static_cast<double>(positive_count);
        // Non-interpolated average precision: sum of precision x recall step
        // (the first step starts from recall 0).
        result.average_precision = af::sum<double>(precision(0) * recall(0));
        if (recall.elements() > 1) {
            const af::array hi = af::seq(1, static_cast<double>(recall.elements() - 1));
            const af::array lo = af::seq(0, static_cast<double>(recall.elements() - 2));
            result.average_precision += af::sum<double>(precision(hi) * (recall(hi) - recall(lo)));
        }

        // Reported recall-descending (thresholds ascending), then the (1, 0) end.
        const std::vector<double> desc_precision = DeviceToHostValues(precision, "ModelEvaluation::ComputePRCurve");
        const std::vector<double> desc_recall = DeviceToHostValues(recall, "ModelEvaluation::ComputePRCurve");
        const std::vector<double> desc_thresholds =
            DeviceToHostValues(counts.thresholds, "ModelEvaluation::ComputePRCurve");
        result.precision.assign(desc_precision.rbegin(), desc_precision.rend());
        result.recall.assign(desc_recall.rbegin(), desc_recall.rend());
        result.thresholds.assign(desc_thresholds.rbegin(), desc_thresholds.rend());
        result.precision.push_back(1.0);
        result.recall.push_back(0.0);
        result.success = true;
        spdlog::debug("PR curve computed on the device: {} points, AP={:.4f}", result.precision.size(),
                      result.average_precision);
    } catch (const af::exception& error) {
        result.error_message = DeviceError("ModelEvaluation::ComputePRCurve", error);
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
