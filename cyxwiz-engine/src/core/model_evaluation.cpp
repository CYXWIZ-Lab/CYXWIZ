#include "model_evaluation.h"
#include <arrayfire.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <set>
#include <spdlog/spdlog.h>

namespace cyxwiz {

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

ROCCurveData ModelEvaluation::ComputeROC(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores) {

    ROCCurveData result;

    if (y_true.size() != y_scores.size() || y_true.empty()) {
        result.error_message = "Invalid input: sizes don't match or empty";
        return result;
    }

    try {
        // Use ArrayFire for efficient computation
        int n = static_cast<int>(y_true.size());

        // Convert to ArrayFire arrays
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
        spdlog::debug("ROC curve computed: {} points, AUC={:.4f}", result.fpr.size(), result.auc);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
        spdlog::error("ComputeROC ArrayFire error: {}", e.what());
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
        // Using interpolated precision at recall levels
        result.average_precision = 0.0;
        for (size_t i = 1; i < result.recall.size(); ++i) {
            double delta_recall = result.recall[i] - result.recall[i - 1];
            result.average_precision += result.precision[i] * delta_recall;
        }

        result.success = true;
        spdlog::debug("PR curve computed: {} points, AP={:.4f}", result.precision.size(), result.average_precision);

    } catch (const af::exception& e) {
        result.error_message = std::string("ArrayFire error: ") + e.what();
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

std::vector<std::pair<std::vector<int>, std::vector<int>>> ModelEvaluation::KFoldSplit(
    int n_samples,
    int n_folds,
    bool shuffle,
    unsigned int seed) {

    std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;

    if (n_samples < n_folds || n_folds < 2) {
        return folds;
    }

    // Create index array
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 gen(seed);
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Compute fold sizes
    int fold_size = n_samples / n_folds;
    int remainder = n_samples % n_folds;

    int start = 0;
    for (int f = 0; f < n_folds; ++f) {
        int current_fold_size = fold_size + (f < remainder ? 1 : 0);
        int end = start + current_fold_size;

        std::vector<int> test_idx(indices.begin() + start, indices.begin() + end);
        std::vector<int> train_idx;
        train_idx.reserve(n_samples - current_fold_size);

        for (int i = 0; i < start; ++i) train_idx.push_back(indices[i]);
        for (int i = end; i < n_samples; ++i) train_idx.push_back(indices[i]);

        folds.emplace_back(train_idx, test_idx);
        start = end;
    }

    return folds;
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> ModelEvaluation::StratifiedKFoldSplit(
    const std::vector<int>& labels,
    int n_folds,
    bool shuffle,
    unsigned int seed) {

    std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;
    int n_samples = static_cast<int>(labels.size());

    if (n_samples < n_folds || n_folds < 2) {
        return folds;
    }

    // Group indices by class
    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < n_samples; ++i) {
        class_indices[labels[i]].push_back(i);
    }

    // Shuffle within each class if requested
    if (shuffle) {
        std::mt19937 gen(seed);
        for (auto& [cls, indices] : class_indices) {
            std::shuffle(indices.begin(), indices.end(), gen);
        }
    }

    // Initialize fold indices
    std::vector<std::vector<int>> fold_indices(n_folds);

    // Distribute samples from each class across folds
    for (auto& [cls, indices] : class_indices) {
        int n_cls = static_cast<int>(indices.size());
        int fold_size = n_cls / n_folds;
        int remainder = n_cls % n_folds;

        int start = 0;
        for (int f = 0; f < n_folds; ++f) {
            int current_size = fold_size + (f < remainder ? 1 : 0);
            for (int i = start; i < start + current_size; ++i) {
                fold_indices[f].push_back(indices[i]);
            }
            start += current_size;
        }
    }

    // Create train/test splits
    for (int f = 0; f < n_folds; ++f) {
        std::vector<int> test_idx = fold_indices[f];
        std::vector<int> train_idx;
        train_idx.reserve(n_samples - test_idx.size());

        for (int g = 0; g < n_folds; ++g) {
            if (g != f) {
                train_idx.insert(train_idx.end(), fold_indices[g].begin(), fold_indices[g].end());
            }
        }

        folds.emplace_back(train_idx, test_idx);
    }

    return folds;
}

std::vector<int> ModelEvaluation::GenerateTrainSizes(
    int n_samples,
    int n_points,
    double min_ratio,
    double max_ratio) {

    std::vector<int> sizes;

    if (n_points < 2 || n_samples < 10) {
        sizes.push_back(n_samples);
        return sizes;
    }

    int min_size = static_cast<int>(n_samples * min_ratio);
    int max_size = static_cast<int>(n_samples * max_ratio);

    min_size = (min_size < 10) ? 10 : min_size;
    max_size = (max_size > n_samples) ? n_samples : max_size;

    // Generate linearly spaced sizes
    for (int i = 0; i < n_points; ++i) {
        double ratio = static_cast<double>(i) / (n_points - 1);
        int size = min_size + static_cast<int>(ratio * (max_size - min_size));
        sizes.push_back(size);
    }

    return sizes;
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
