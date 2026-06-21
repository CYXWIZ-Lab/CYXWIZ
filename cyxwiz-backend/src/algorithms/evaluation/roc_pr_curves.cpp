// Prevent Windows min/max macros from interfering with std::numeric_limits and af::max/min
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "cyxwiz/model_evaluation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

// Ensure Windows min/max macros are undefined after all includes
#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#endif

namespace cyxwiz {

// GPU availability check (cached)
static bool s_use_gpu = false;
static bool s_gpu_checked = false;

static bool CheckGPUAvailable() {
    if (s_gpu_checked) return s_use_gpu;
    s_gpu_checked = true;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::Backend backend = af::getActiveBackend();
        s_use_gpu = (backend == AF_BACKEND_CUDA || backend == AF_BACKEND_OPENCL);
        if (s_use_gpu) {
            spdlog::info("[ModelEvaluation] GPU acceleration enabled");
        }
    } catch (const af::exception& e) {
        spdlog::warn("[ModelEvaluation] GPU check failed: {}", e.what());
        s_use_gpu = false;
    }
#endif
    return s_use_gpu;
}

std::vector<size_t> ModelEvaluation::ArgSort(const std::vector<double>& v, bool descending) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    if (descending) {
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
            return v[i1] > v[i2];
        });
    } else {
        std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
            return v[i1] < v[i2];
        });
    }
    return idx;
}

ROCCurveData ModelEvaluation::ComputeROC(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores) {

    ROCCurveData result;

    if (y_true.size() != y_scores.size() || y_true.empty()) {
        result.error_message = "Invalid input: sizes don't match or empty";
        return result;
    }

    try {
        int n = static_cast<int>(y_true.size());

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (CheckGPUAvailable() && n > 1000) {
            try {
                // Use ArrayFire for efficient computation
                af::array af_labels(n, y_true.data());
                af::array af_scores(n, y_scores.data());

                // Sort by scores descending
                af::array sorted_scores, sort_idx;
                af::sort(sorted_scores, sort_idx, af_scores, 0, false);  // descending

                // Reorder labels by sorted indices
                af::array sorted_labels = af_labels(sort_idx);

                // Count positives and negatives
                int n_pos = static_cast<int>(af::sum<int>(af_labels == 1));
                int n_neg = n - n_pos;

                if (n_pos == 0 || n_neg == 0) {
                    result.error_message = "Need both positive and negative samples";
                    return result;
                }

                // Compute cumulative sums for TPR and FPR
                af::array is_pos = (sorted_labels == 1).as(f64);
                af::array is_neg = (sorted_labels == 0).as(f64);

                af::array cum_tp = af::accum(is_pos);
                af::array cum_fp = af::accum(is_neg);

                // Convert to host
                std::vector<double> cum_tp_host(n), cum_fp_host(n), scores_host(n);
                cum_tp.host(cum_tp_host.data());
                cum_fp.host(cum_fp_host.data());
                sorted_scores.host(scores_host.data());

                // Build ROC curve with unique thresholds
                result.fpr.reserve(n + 2);
                result.tpr.reserve(n + 2);
                result.thresholds.reserve(n + 2);

                // Start point (0, 0)
                result.fpr.push_back(0.0);
                result.tpr.push_back(0.0);
                result.thresholds.push_back(scores_host[0] + 1.0);  // Above max score

                double prev_score = scores_host[0] + 1.0;
                for (int i = 0; i < n; ++i) {
                    double score = scores_host[i];
                    if (score != prev_score) {
                        result.fpr.push_back(cum_fp_host[i > 0 ? i - 1 : 0] / n_neg);
                        result.tpr.push_back(cum_tp_host[i > 0 ? i - 1 : 0] / n_pos);
                        result.thresholds.push_back(score);
                    }
                    prev_score = score;
                }

                // End point (1, 1)
                result.fpr.push_back(1.0);
                result.tpr.push_back(1.0);
                result.thresholds.push_back(scores_host.back() - 1.0);  // Below min score

                // Compute AUC using trapezoidal rule
                result.auc = ComputeAUC(result.fpr, result.tpr);

                result.success = true;
                spdlog::debug("ROC curve computed (GPU): {} points, AUC={:.4f}", result.fpr.size(), result.auc);
                return result;

            } catch (const af::exception& e) {
                spdlog::warn("ComputeROC ArrayFire error, fallback to CPU: {}", e.what());
            }
        }
#endif

        // CPU fallback - sort indices by scores
        auto sorted_idx = ArgSort(y_scores, true);  // descending

        // Count positives and negatives
        int n_pos = 0, n_neg = 0;
        for (int label : y_true) {
            if (label == 1) n_pos++;
            else n_neg++;
        }

        if (n_pos == 0 || n_neg == 0) {
            result.error_message = "Need both positive and negative samples";
            return result;
        }

        // Build ROC curve
        result.fpr.reserve(n + 2);
        result.tpr.reserve(n + 2);
        result.thresholds.reserve(n + 2);

        result.fpr.push_back(0.0);
        result.tpr.push_back(0.0);
        result.thresholds.push_back(y_scores[sorted_idx[0]] + 1.0);

        int cum_tp = 0, cum_fp = 0;
        double prev_score = y_scores[sorted_idx[0]] + 1.0;

        for (int i = 0; i < n; ++i) {
            size_t idx = sorted_idx[i];
            double score = y_scores[idx];

            if (score != prev_score) {
                result.fpr.push_back(static_cast<double>(cum_fp) / n_neg);
                result.tpr.push_back(static_cast<double>(cum_tp) / n_pos);
                result.thresholds.push_back(score);
            }

            if (y_true[idx] == 1) cum_tp++;
            else cum_fp++;

            prev_score = score;
        }

        result.fpr.push_back(1.0);
        result.tpr.push_back(1.0);
        result.thresholds.push_back(y_scores[sorted_idx.back()] - 1.0);

        result.auc = ComputeAUC(result.fpr, result.tpr);

        result.success = true;
        spdlog::debug("ROC curve computed (CPU): {} points, AUC={:.4f}", result.fpr.size(), result.auc);

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
        spdlog::error("ComputeROC failed: {}", e.what());
    }

    return result;
}

ROCCurveData ModelEvaluation::ComputeMulticlassROC(
    const std::vector<int>& y_true,
    const std::vector<std::vector<double>>& y_scores) {

    ROCCurveData result;

    if (y_true.empty() || y_scores.empty() || y_true.size() != y_scores.size()) {
        result.error_message = "Invalid input";
        return result;
    }

    try {
        // Find number of classes
        int n_classes = static_cast<int>(y_scores[0].size());

        result.class_fpr.resize(n_classes);
        result.class_tpr.resize(n_classes);
        result.class_auc.resize(n_classes);

        // Compute one-vs-rest ROC for each class
        for (int c = 0; c < n_classes; ++c) {
            // Binary labels for this class
            std::vector<int> binary_labels(y_true.size());
            std::vector<double> class_scores(y_true.size());

            for (size_t i = 0; i < y_true.size(); ++i) {
                binary_labels[i] = (y_true[i] == c) ? 1 : 0;
                class_scores[i] = y_scores[i][c];
            }

            auto class_roc = ComputeROC(binary_labels, class_scores);
            if (class_roc.success) {
                result.class_fpr[c] = class_roc.fpr;
                result.class_tpr[c] = class_roc.tpr;
                result.class_auc[c] = class_roc.auc;
            }
        }

        // Macro-average AUC
        result.auc = std::accumulate(result.class_auc.begin(), result.class_auc.end(), 0.0) / n_classes;

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
    }

    return result;
}

PRCurveData ModelEvaluation::ComputePRCurve(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores) {

    PRCurveData result;

    if (y_true.size() != y_scores.size() || y_true.empty()) {
        result.error_message = "Invalid input";
        return result;
    }

    try {
        int n = static_cast<int>(y_true.size());

#ifdef CYXWIZ_HAS_ARRAYFIRE
        if (CheckGPUAvailable() && n > 1000) {
            try {
                // Use ArrayFire for efficient sorting
                af::array af_labels(n, y_true.data());
                af::array af_scores(n, y_scores.data());

                // Sort by scores descending
                af::array sorted_scores, sort_idx;
                af::sort(sorted_scores, sort_idx, af_scores, 0, false);

                af::array sorted_labels = af_labels(sort_idx);

                // Count total positives
                int n_pos = static_cast<int>(af::sum<int>(af_labels == 1));

                if (n_pos == 0) {
                    result.error_message = "No positive samples";
                    return result;
                }

                // Compute cumulative TP and total predictions
                af::array is_pos = (sorted_labels == 1).as(f64);
                af::array cum_tp = af::accum(is_pos);
                af::array total_pred = af::range(af::dim4(n), 0, f64) + 1.0;

                // Precision = cum_tp / total_pred
                // Recall = cum_tp / n_pos
                af::array precision_arr = cum_tp / total_pred;
                af::array recall_arr = cum_tp / static_cast<double>(n_pos);

                // Convert to host
                std::vector<double> precision_host(n), recall_host(n), scores_host(n);
                precision_arr.host(precision_host.data());
                recall_arr.host(recall_host.data());
                sorted_scores.host(scores_host.data());

                // Build PR curve with unique thresholds
                result.precision.reserve(n + 1);
                result.recall.reserve(n + 1);
                result.thresholds.reserve(n + 1);

                // Start point (recall=0, precision=1)
                result.recall.push_back(0.0);
                result.precision.push_back(1.0);
                result.thresholds.push_back(scores_host[0] + 1.0);

                double prev_score = scores_host[0] + 1.0;
                for (int i = 0; i < n; ++i) {
                    double score = scores_host[i];
                    if (score != prev_score || i == n - 1) {
                        result.recall.push_back(recall_host[i]);
                        result.precision.push_back(precision_host[i]);
                        result.thresholds.push_back(score);
                    }
                    prev_score = score;
                }

                // Compute average precision (area under PR curve)
                result.average_precision = 0.0;
                for (size_t i = 1; i < result.recall.size(); ++i) {
                    double delta_recall = result.recall[i] - result.recall[i - 1];
                    result.average_precision += result.precision[i] * delta_recall;
                }

                result.success = true;
                spdlog::debug("PR curve computed (GPU): {} points, AP={:.4f}", result.precision.size(), result.average_precision);
                return result;

            } catch (const af::exception& e) {
                spdlog::warn("ComputePRCurve ArrayFire error, fallback to CPU: {}", e.what());
            }
        }
#endif

        // CPU fallback
        auto sorted_idx = ArgSort(y_scores, true);

        // Count total positives
        int n_pos = 0;
        for (int label : y_true) {
            if (label == 1) n_pos++;
        }

        if (n_pos == 0) {
            result.error_message = "No positive samples";
            return result;
        }

        result.precision.reserve(n + 1);
        result.recall.reserve(n + 1);
        result.thresholds.reserve(n + 1);

        result.recall.push_back(0.0);
        result.precision.push_back(1.0);
        result.thresholds.push_back(y_scores[sorted_idx[0]] + 1.0);

        int cum_tp = 0;
        double prev_score = y_scores[sorted_idx[0]] + 1.0;

        for (int i = 0; i < n; ++i) {
            size_t idx = sorted_idx[i];
            double score = y_scores[idx];

            if (y_true[idx] == 1) cum_tp++;

            if (score != prev_score || i == n - 1) {
                result.recall.push_back(static_cast<double>(cum_tp) / n_pos);
                result.precision.push_back(static_cast<double>(cum_tp) / (i + 1));
                result.thresholds.push_back(score);
            }

            prev_score = score;
        }

        // Compute average precision
        result.average_precision = 0.0;
        for (size_t i = 1; i < result.recall.size(); ++i) {
            double delta_recall = result.recall[i] - result.recall[i - 1];
            result.average_precision += result.precision[i] * delta_recall;
        }

        result.success = true;
        spdlog::debug("PR curve computed (CPU): {} points, AP={:.4f}", result.precision.size(), result.average_precision);

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
    }

    return result;
}

PRCurveData ModelEvaluation::ComputeMulticlassPRCurve(
    const std::vector<int>& y_true,
    const std::vector<std::vector<double>>& y_scores) {

    PRCurveData result;

    if (y_true.empty() || y_scores.empty()) {
        result.error_message = "Invalid input";
        return result;
    }

    try {
        int n_classes = static_cast<int>(y_scores[0].size());

        result.class_precision.resize(n_classes);
        result.class_recall.resize(n_classes);
        result.class_ap.resize(n_classes);

        for (int c = 0; c < n_classes; ++c) {
            std::vector<int> binary_labels(y_true.size());
            std::vector<double> class_scores(y_true.size());

            for (size_t i = 0; i < y_true.size(); ++i) {
                binary_labels[i] = (y_true[i] == c) ? 1 : 0;
                class_scores[i] = y_scores[i][c];
            }

            auto class_pr = ComputePRCurve(binary_labels, class_scores);
            if (class_pr.success) {
                result.class_precision[c] = class_pr.precision;
                result.class_recall[c] = class_pr.recall;
                result.class_ap[c] = class_pr.average_precision;
            }
        }

        // Mean AP
        result.average_precision = std::accumulate(result.class_ap.begin(), result.class_ap.end(), 0.0) / n_classes;

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Exception: ") + e.what();
    }

    return result;
}

double ModelEvaluation::ComputeAUC(
    const std::vector<double>& x,
    const std::vector<double>& y) {

    if (x.size() != y.size() || x.size() < 2) {
        return 0.0;
    }

    // Trapezoidal rule
    double auc = 0.0;
    for (size_t i = 1; i < x.size(); ++i) {
        auc += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0;
    }
    return std::abs(auc);  // Ensure positive
}

} // namespace cyxwiz
