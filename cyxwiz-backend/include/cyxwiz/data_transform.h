#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <map>
#include <string>

namespace cyxwiz {

// Column transforms, column statistics and the normality test behind the data
// transformation panels. Everything runs on ArrayFire on the active device (its
// CPU backend is the CPU option); a build without ArrayFire reports an error.
// Each input column is a std::vector<double>; columns may differ in length.

// Result structure for all transforms
struct CYXWIZ_API TransformResult {
    std::vector<std::vector<double>> transformed_data;
    std::map<std::string, double> params;  // fitted values per column (col<i>_<name>)
    std::string method;
    bool success = false;
    std::string error_message;
};

// Statistics for a single column. Standard deviation is the population one (divide
// by n), as Standardize uses; percentiles interpolate linearly (numpy "linear").
// count is 0 and error_message is set when the column could not be summarised.
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
    std::string error_message;
};

// Maximum-likelihood lambda search (Box-Cox or Yeo-Johnson). lambdas_tested and
// log_likelihoods are the coarse profile across [lambda_min, lambda_max];
// optimal_lambda is refined beyond the grid spacing.
struct CYXWIZ_API BoxCoxLambdaResult {
    double optimal_lambda = 0.0;
    std::vector<double> lambdas_tested;
    std::vector<double> log_likelihoods;
    double best_log_likelihood = 0.0;
    bool success = false;
    std::string error_message;
};

// Normality test result
struct CYXWIZ_API NormalityTestResult {
    double statistic = 0.0;
    double p_value = 0.0;
    bool is_normal = false;  // p_value > 0.05
    bool computed = false;   // false: see error_message (e.g. too few values)
    std::string test_name;
    std::string error_message;
};

// scikit-learn RobustScaler options: centre on the median, scale by the spread
// between two percentiles (a zero spread scales by 1).
struct CYXWIZ_API RobustScaleOptions {
    double quantile_min = 25.0;  // percent, 0 <= quantile_min < quantile_max <= 100
    double quantile_max = 75.0;
    bool with_centering = true;
    bool with_scaling = true;
};

class CYXWIZ_API DataTransform {
public:
    // ========== Normalization (Min-Max Scaling) ==========
    // x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min
    static TransformResult Normalize(const std::vector<std::vector<double>>& data,
                                     double range_min = 0.0, double range_max = 1.0);

    // ========== Standardization (Z-Score) ==========
    // z = (x - mean) / std, population std (scikit-learn StandardScaler)
    static TransformResult Standardize(const std::vector<std::vector<double>>& data);

    // ========== Log Transforms ==========
    // base "natural", "log10" or "log2"; log1p takes log(1 + x)
    static TransformResult LogTransform(const std::vector<std::vector<double>>& data,
                                        const std::string& base = "natural",
                                        bool use_log1p = true);

    // ========== Box-Cox Transform ==========
    // Strictly positive data. y = (x^lambda - 1) / lambda, or log(x) at lambda 0.
    // auto_lambda: maximum-likelihood lambda (scipy.stats.boxcox).
    static TransformResult BoxCoxColumn(const std::vector<double>& data,
                                        double lambda = 0.0,
                                        bool auto_lambda = true);

    static BoxCoxLambdaResult FindOptimalLambda(const std::vector<double>& data,
                                                double lambda_min = -5.0,
                                                double lambda_max = 5.0,
                                                int n_steps = 100);

    // ========== Yeo-Johnson Transform ==========
    // Any real data (scipy.stats.yeojohnson).
    static TransformResult YeoJohnsonColumn(const std::vector<double>& data,
                                            double lambda = 0.0,
                                            bool auto_lambda = true);

    static BoxCoxLambdaResult FindOptimalYeoJohnsonLambda(const std::vector<double>& data,
                                                          double lambda_min = -5.0,
                                                          double lambda_max = 5.0,
                                                          int n_steps = 100);

    // ========== Robust Scaling ==========
    // x_scaled = (x - median) / (q_max - q_min) (scikit-learn RobustScaler). Params per
    // column: col<i>_median (the centre used, 0 without centering) and col<i>_iqr (the
    // scale used, 1 without scaling or for a zero spread).
    static TransformResult RobustScale(const std::vector<std::vector<double>>& data,
                                       const RobustScaleOptions& options = {});

    // ========== Max Abs Scaling ==========
    // x_scaled = x / max(|x|)
    static TransformResult MaxAbsScale(const std::vector<std::vector<double>>& data);

    // ========== Quantile Transform ==========
    // Maps each column through min(n_quantiles, n) empirical quantiles onto a
    // uniform or standard normal distribution (scikit-learn QuantileTransformer,
    // fitted on the whole column: no subsampling).
    static TransformResult QuantileTransform(const std::vector<std::vector<double>>& data,
                                             const std::string& output_distribution = "uniform",  // or "normal"
                                             int n_quantiles = 1000);

    // ========== Statistics ==========
    static ColumnStats ComputeColumnStats(const std::vector<double>& data);

    // D'Agostino-Pearson K^2 test (scipy.stats.normaltest), at least 20 values.
    static NormalityTestResult DAgostinoPearsonTest(const std::vector<double>& data);

    // ========== Input checks ==========
    static bool CanApplyLogTransform(const std::vector<double>& data, bool use_log1p);
    static bool CanApplyBoxCox(const std::vector<double>& data);
};

} // namespace cyxwiz
