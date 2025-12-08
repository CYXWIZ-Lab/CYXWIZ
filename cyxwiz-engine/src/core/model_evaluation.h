#pragma once

#include <vector>
#include <string>
#include <map>

namespace cyxwiz {

// Cross-validation result
struct CrossValidationResult {
    std::vector<double> train_scores;
    std::vector<double> val_scores;
    double mean_train_score = 0.0;
    double mean_val_score = 0.0;
    double std_train_score = 0.0;
    double std_val_score = 0.0;
    int n_folds = 0;
    bool success = false;
    std::string error_message;
};

// Confusion matrix with derived metrics
struct ConfusionMatrixData {
    std::vector<std::vector<int>> matrix;
    std::vector<std::string> class_names;
    int n_classes = 0;
    int total_samples = 0;

    // Overall metrics
    double accuracy = 0.0;
    double macro_precision = 0.0;
    double macro_recall = 0.0;
    double macro_f1 = 0.0;
    double weighted_f1 = 0.0;

    // Per-class metrics
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1_scores;
    std::vector<int> support;  // Number of samples per class

    bool success = false;
    std::string error_message;
};

// ROC curve data
struct ROCCurveData {
    std::vector<double> fpr;        // False positive rate
    std::vector<double> tpr;        // True positive rate
    std::vector<double> thresholds;
    double auc = 0.0;               // Area under curve

    // For multi-class (one-vs-rest)
    std::vector<std::vector<double>> class_fpr;
    std::vector<std::vector<double>> class_tpr;
    std::vector<double> class_auc;

    bool success = false;
    std::string error_message;
};

// Precision-Recall curve data
struct PRCurveData {
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> thresholds;
    double average_precision = 0.0;  // Area under PR curve

    // For multi-class
    std::vector<std::vector<double>> class_precision;
    std::vector<std::vector<double>> class_recall;
    std::vector<double> class_ap;

    bool success = false;
    std::string error_message;
};

// Learning curve data
struct LearningCurveData {
    std::vector<int> train_sizes;
    std::vector<double> train_scores_mean;
    std::vector<double> train_scores_std;
    std::vector<double> val_scores_mean;
    std::vector<double> val_scores_std;

    // Individual fold scores (optional)
    std::vector<std::vector<double>> train_scores;
    std::vector<std::vector<double>> val_scores;

    std::string scoring_metric;  // "accuracy", "f1", "loss", etc.
    bool success = false;
    std::string error_message;
};

// Classification report (summary of all metrics)
struct ClassificationReport {
    ConfusionMatrixData confusion_matrix;
    std::map<std::string, double> overall_metrics;
    std::map<std::string, std::map<std::string, double>> per_class_metrics;

    bool success = false;
    std::string error_message;
};

// Binary classification metrics at a specific threshold
struct BinaryMetrics {
    double threshold = 0.5;
    int tp = 0, fp = 0, tn = 0, fn = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    double specificity = 0.0;
    double balanced_accuracy = 0.0;
    double mcc = 0.0;  // Matthews correlation coefficient
};

class ModelEvaluation {
public:
    // Confusion matrix computation
    static ConfusionMatrixData ComputeConfusionMatrix(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred,
        const std::vector<std::string>& class_names = {});

    // ROC curve (binary classification)
    static ROCCurveData ComputeROC(
        const std::vector<int>& y_true,
        const std::vector<double>& y_scores);

    // ROC curve (multi-class, one-vs-rest)
    static ROCCurveData ComputeMulticlassROC(
        const std::vector<int>& y_true,
        const std::vector<std::vector<double>>& y_scores);

    // Precision-Recall curve (binary)
    static PRCurveData ComputePRCurve(
        const std::vector<int>& y_true,
        const std::vector<double>& y_scores);

    // Precision-Recall curve (multi-class)
    static PRCurveData ComputeMulticlassPRCurve(
        const std::vector<int>& y_true,
        const std::vector<std::vector<double>>& y_scores);

    // Binary metrics at threshold
    static BinaryMetrics ComputeBinaryMetrics(
        const std::vector<int>& y_true,
        const std::vector<double>& y_scores,
        double threshold = 0.5);

    // AUC computation (trapezoidal rule)
    static double ComputeAUC(
        const std::vector<double>& x,
        const std::vector<double>& y);

    // Cross-validation fold generation
    static std::vector<std::pair<std::vector<int>, std::vector<int>>> KFoldSplit(
        int n_samples,
        int n_folds,
        bool shuffle = true,
        unsigned int seed = 42);

    // Stratified K-fold split (maintains class proportions)
    static std::vector<std::pair<std::vector<int>, std::vector<int>>> StratifiedKFoldSplit(
        const std::vector<int>& labels,
        int n_folds,
        bool shuffle = true,
        unsigned int seed = 42);

    // Learning curve computation helper
    static std::vector<int> GenerateTrainSizes(
        int n_samples,
        int n_points = 5,
        double min_ratio = 0.1,
        double max_ratio = 1.0);

    // Full classification report
    static ClassificationReport GenerateClassificationReport(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred,
        const std::vector<std::string>& class_names = {});

    // Optimal threshold finder (maximizes F1 or Youden's J)
    static double FindOptimalThreshold(
        const std::vector<int>& y_true,
        const std::vector<double>& y_scores,
        const std::string& criterion = "f1");  // "f1", "youden", "balanced"

private:
    // Helper: sort by scores and return indices
    static std::vector<size_t> ArgSort(const std::vector<double>& v, bool descending = true);
};

} // namespace cyxwiz
