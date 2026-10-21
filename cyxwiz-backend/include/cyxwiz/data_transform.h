#pragma once

#include "cyxwiz/api_export.h"
#include <vector>
#include <map>
#include <string>

namespace cyxwiz {

// Column transforms, column statistics and the normality test behind the data
// transformation panels. Everything runs on ArrayFire on the active device (its
// CPU backend is the CPU option).
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
    double skewness = 0.0;  // Moment skewness m3 / m2^1.5 (scipy.stats.skew); 0 for constant data
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

// scikit-learn StandardScaler options.
struct CYXWIZ_API StandardizeOptions {
    bool with_mean = true;  // subtract the mean
    bool with_std = true;   // divide by the population std (a zero std divides by 1)
};

// Rows flagged as outliers in one column (FindOutliers).
struct CYXWIZ_API OutlierFlags {
    std::vector<unsigned char> is_outlier;  // one flag per value
    size_t count = 0;
    bool success = false;
    std::string error_message;
};

class CYXWIZ_API DataTransform {
public:
    // ========== Normalization (Min-Max Scaling) ==========
    // x_scaled = (x - min) / (max - min) * (range_max - range_min) + range_min
    // (scikit-learn MinMaxScaler: a constant column maps to range_min).
    static TransformResult Normalize(const std::vector<std::vector<double>>& data,
                                     double range_min = 0.0, double range_max = 1.0);

    // ========== Standardization (Z-Score) ==========
    // z = (x - mean) / std, population std (scikit-learn StandardScaler). Params per
    // column: col<i>_mean (the centre used) and col<i>_std (the scale used).
    static TransformResult Standardize(const std::vector<std::vector<double>>& data,
                                       const StandardizeOptions& options = {});

    // Standardization in batches (data bigger than memory, TOFIX154): each
    // batch's per-column moments come from the device (Moments), merge exactly
    // across batches (CombineMoments, Chan et al.), and give the center and
    // scale Standardize would use on all the data (StandardizeParameters).
    struct ColumnMoments {
        double count = 0.0;
        double mean = 0.0;
        double m2 = 0.0;  // sum of squared deviations from the mean
        double min = 0.0;
        double max = 0.0;
    };
    static bool Moments(const std::vector<std::vector<double>>& data, std::vector<ColumnMoments>& moments,
                        std::string& error);
    static void CombineMoments(ColumnMoments& total, const ColumnMoments& batch);
    // False (with no center/scale) for missing or non-finite values.
    static bool StandardizeParameters(const ColumnMoments& moments, const StandardizeOptions& options,
                                      double& center, double& scale);

    // Normalize with given per-column min and max (the streamed MinMaxScaler:
    // the extremes merged over every batch); the same formula as Normalize.
    static TransformResult NormalizeColumns(const std::vector<std::vector<double>>& data,
                                            const std::vector<double>& lows, const std::vector<double>& highs,
                                            double range_min = 0.0, double range_max = 1.0);

    // (x - center) / scale per column with given values (re-applying a fitted
    // scaler); one center and one scale per column, scales non-zero.
    static TransformResult ScaleColumns(const std::vector<std::vector<double>>& data,
                                        const std::vector<double>& centers,
                                        const std::vector<double>& scales);

    // ========== Outliers ==========
    // "iqr": outside [q1 - t * IQR, q3 + t * IQR] (linear quartiles); "zscore":
    // |x - mean| / std > t (population std, scipy.stats.zscore; none when the std
    // is 0).
    static OutlierFlags FindOutliers(const std::vector<double>& data, const std::string& method, double threshold);

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
    // QuantileTransform in two parts, so a fitted transform can be saved and
    // re-applied (the Quantile Transform node, TOFIX129 D2): the column's
    // min(n_quantiles, n) quantile references (non-decreasing), and the
    // mapping of other values through them. QuantileTransform = fit + apply.
    static bool QuantileFit(const std::vector<double>& data, int n_quantiles, std::vector<double>& quantiles,
                            std::string& error);
    static TransformResult QuantileApply(const std::vector<std::vector<double>>& data,
                                         const std::vector<std::vector<double>>& quantiles,
                                         const std::string& output_distribution = "uniform");

    // ========== Statistics ==========
    static ColumnStats ComputeColumnStats(const std::vector<double>& data);

    // Counts in `bins` equal-width bins over [low, high], the last bin holding
    // `high` (numpy.histogram); values outside are not counted. low == high:
    // every value equal to it in the middle bin. Empty when bins < 1, high <
    // low or the device fails.
    static std::vector<int64_t> Histogram(const std::vector<double>& data, int bins, double low, double high);

    // D'Agostino-Pearson K^2 test (scipy.stats.normaltest), at least 20 values.
    static NormalityTestResult DAgostinoPearsonTest(const std::vector<double>& data);

    // ========== Input checks ==========
    static bool CanApplyBoxCox(const std::vector<double>& data);
};

} // namespace cyxwiz
