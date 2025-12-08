#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <map>
#include <string>
#include <optional>
#include <cmath>

namespace cyxwiz {

// Result structure for all transforms
struct CYXWIZ_API TransformResult {
    std::vector<std::vector<double>> transformed_data;
    std::map<std::string, double> params;  // Store for inverse transform
    std::string method;
    bool success = false;
    std::string error_message;
};

// Statistics for a single column
struct CYXWIZ_API ColumnStats {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double std_dev = 0.0;
    double median = 0.0;
    double q1 = 0.0;  // 25th percentile
    double q3 = 0.0;  // 75th percentile
    double iqr = 0.0; // Interquartile range
    int count = 0;
    bool has_negatives = false;
    bool has_zeros = false;
};

// Box-Cox lambda finder result
struct CYXWIZ_API BoxCoxLambdaResult {
    double optimal_lambda = 0.0;
    std::vector<double> lambdas_tested;
    std::vector<double> log_likelihoods;
    double best_log_likelihood = 0.0;
    bool success = false;
};

// Normality test result
struct CYXWIZ_API NormalityTestResult {
    double statistic = 0.0;
    double p_value = 0.0;
    bool is_normal = false;  // Based on alpha = 0.05
    std::string test_name;
};

class CYXWIZ_API DataTransform {
public:
    // ========== Normalization (Min-Max Scaling) ==========
    // Transforms data to [range_min, range_max] (default [0, 1])
    // Formula: x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min
    static TransformResult Normalize(const std::vector<std::vector<double>>& data,
                                     double range_min = 0.0, double range_max = 1.0);

    // Normalize single column
    static TransformResult NormalizeColumn(const std::vector<double>& data,
                                           double range_min = 0.0, double range_max = 1.0);

    // ========== Standardization (Z-Score) ==========
    // Transforms data to have mean=0 and std=1
    // Formula: z = (x - mean) / std
    static TransformResult Standardize(const std::vector<std::vector<double>>& data);

    // Standardize single column
    static TransformResult StandardizeColumn(const std::vector<double>& data);

    // ========== Log Transforms ==========
    // Natural log, log10, log2
    // log1p: log(1 + x) - safe for values close to 0
    static TransformResult LogTransform(const std::vector<std::vector<double>>& data,
                                        const std::string& base = "natural",  // "natural", "log10", "log2"
                                        bool use_log1p = true);

    static TransformResult LogTransformColumn(const std::vector<double>& data,
                                              const std::string& base = "natural",
                                              bool use_log1p = true);

    // ========== Box-Cox Transform ==========
    // Power transform for positive data to achieve normality
    // y = (x^lambda - 1) / lambda   if lambda != 0
    // y = log(x)                    if lambda == 0
    static TransformResult BoxCox(const std::vector<std::vector<double>>& data,
                                  double lambda = 0.0,  // 0 = auto-find optimal
                                  bool auto_lambda = true);

    static TransformResult BoxCoxColumn(const std::vector<double>& data,
                                        double lambda = 0.0,
                                        bool auto_lambda = true);

    // Find optimal lambda for Box-Cox
    static BoxCoxLambdaResult FindOptimalLambda(const std::vector<double>& data,
                                                double lambda_min = -5.0,
                                                double lambda_max = 5.0,
                                                int n_steps = 100);

    // ========== Yeo-Johnson Transform ==========
    // Similar to Box-Cox but supports negative values
    static TransformResult YeoJohnson(const std::vector<std::vector<double>>& data,
                                      double lambda = 0.0,
                                      bool auto_lambda = true);

    static TransformResult YeoJohnsonColumn(const std::vector<double>& data,
                                            double lambda = 0.0,
                                            bool auto_lambda = true);

    // ========== Robust Scaling ==========
    // Uses median and IQR instead of mean and std
    // More robust to outliers
    // Formula: x_scaled = (x - median) / IQR
    static TransformResult RobustScale(const std::vector<std::vector<double>>& data);

    static TransformResult RobustScaleColumn(const std::vector<double>& data);

    // ========== Max Abs Scaling ==========
    // Scales by maximum absolute value to [-1, 1]
    // Formula: x_scaled = x / max(|x|)
    static TransformResult MaxAbsScale(const std::vector<std::vector<double>>& data);

    static TransformResult MaxAbsScaleColumn(const std::vector<double>& data);

    // ========== Quantile Transform ==========
    // Transform features to follow a uniform or normal distribution
    static TransformResult QuantileTransform(const std::vector<std::vector<double>>& data,
                                             const std::string& output_distribution = "uniform",  // "uniform" or "normal"
                                             int n_quantiles = 1000);

    // ========== Power Transform ==========
    // General power transform: y = x^power
    static TransformResult PowerTransform(const std::vector<std::vector<double>>& data,
                                          double power);

    // ========== Inverse Transforms ==========
    static std::vector<std::vector<double>> InverseTransform(
        const std::vector<std::vector<double>>& data,
        const TransformResult& original_result);

    static std::vector<double> InverseTransformColumn(
        const std::vector<double>& data,
        const TransformResult& original_result);

    // ========== Statistics ==========
    static ColumnStats ComputeColumnStats(const std::vector<double>& data);

    // Shapiro-Wilk normality test (for small samples < 5000)
    static NormalityTestResult ShapiroWilkTest(const std::vector<double>& data);

    // D'Agostino-Pearson normality test
    static NormalityTestResult DAgostinoPearsonTest(const std::vector<double>& data);

    // ========== Utility Functions ==========
    // Check if data can be transformed (e.g., for log, need positive values)
    static bool CanApplyLogTransform(const std::vector<double>& data, bool use_log1p);
    static bool CanApplyBoxCox(const std::vector<double>& data);

    // Detect outliers using IQR method
    static std::vector<int> DetectOutliers(const std::vector<double>& data, double iqr_multiplier = 1.5);

    // Get percentile value
    static double GetPercentile(std::vector<double> data, double percentile);
};

} // namespace cyxwiz
