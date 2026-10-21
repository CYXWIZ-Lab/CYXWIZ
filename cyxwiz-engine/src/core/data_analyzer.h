#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <variant>

namespace cyxwiz {

// Forward declarations
class DataTable;

/**
 * Data type classification for columns
 */
enum class ColumnDataType {
    Unknown,
    Numeric,        // Continuous floating-point values
    Integer,        // Discrete integer values
    Categorical,    // String/categorical values
    Boolean,        // True/false values
    DateTime,       // Date/time values
    Mixed           // Mixed types in column
};

// ========== Phase 4: Statistical Tools ==========

/**
 * Hypothesis test types
 */
enum class TestType {
    OneSampleTTest,
    TwoSampleTTest,
    PairedTTest,
    OneWayANOVA,
    ChiSquare,
    MannWhitneyU
};

/**
 * Hypothesis test result
 */
struct HypothesisTestResult {
    TestType test_type = TestType::OneSampleTTest;
    double test_statistic = 0.0;
    double p_value = 1.0;
    double df = 0.0;  // Degrees of freedom
    double df2 = 0.0; // Second df for F-test
    bool reject_null = false;  // At alpha = 0.05
    std::string interpretation;
    double effect_size = 0.0;  // Cohen's d or eta-squared
    double confidence_interval_low = 0.0;
    double confidence_interval_high = 0.0;
    double mean_diff = 0.0;  // Difference in means
    double se_diff = 0.0;    // Standard error of difference
};

/**
 * Distribution types for fitting
 */
enum class DistributionType {
    Normal,
    Uniform,
    Exponential,
    LogNormal,
    Gamma
};

/**
 * Distribution fit result
 */
struct DistributionFitResult {
    DistributionType type = DistributionType::Normal;
    std::string name;
    std::map<std::string, double> parameters;  // e.g., {"mu": 0.0, "sigma": 1.0}
    double ks_statistic = 0.0;  // Kolmogorov-Smirnov
    double ks_p_value = 1.0;
    double aic = 0.0;  // Akaike Information Criterion
    double bic = 0.0;  // Bayesian Information Criterion
    double log_likelihood = 0.0;
    bool good_fit = false;  // KS p-value > 0.05
};

/**
 * Regression types
 */
enum class RegressionType {
    Linear,
    Polynomial,
    Multiple
};

/**
 * Regression analysis result
 */
struct RegressionResult {
    RegressionType type = RegressionType::Linear;
    std::vector<std::string> predictor_names;
    std::string response_name;
    std::vector<double> coefficients;  // [intercept, b1, b2, ...]
    std::vector<double> std_errors;
    std::vector<double> t_values;
    std::vector<double> p_values;
    double r_squared = 0.0;
    double adjusted_r_squared = 0.0;
    double f_statistic = 0.0;
    double f_p_value = 1.0;
    double mse = 0.0;  // Mean prediction squared error: SSE / n
    double rmse = 0.0; // Root mean prediction squared error
    double mae = 0.0;  // Mean Absolute Error
    double residual_variance = 0.0; // Unbiased error variance: SSE / df_resid
    double residual_standard_error = 0.0; // sqrt(residual_variance)
    std::vector<double> residuals;
    std::vector<double> predicted;
    size_t n = 0;      // Sample size
    size_t df_model = 0;  // Degrees of freedom for model
    size_t df_resid = 0;  // Degrees of freedom for residuals
};

/**
 * Convert test type to string
 */
inline const char* TestTypeToString(TestType type) {
    switch (type) {
        case TestType::OneSampleTTest: return "One-Sample t-Test";
        case TestType::TwoSampleTTest: return "Two-Sample t-Test";
        case TestType::PairedTTest: return "Paired t-Test";
        case TestType::OneWayANOVA: return "One-Way ANOVA";
        case TestType::ChiSquare: return "Chi-Square Test";
        case TestType::MannWhitneyU: return "Mann-Whitney U Test";
        default: return "Unknown";
    }
}

/**
 * Convert distribution type to string
 */
inline const char* DistributionTypeToString(DistributionType type) {
    switch (type) {
        case DistributionType::Normal: return "Normal";
        case DistributionType::Uniform: return "Uniform";
        case DistributionType::Exponential: return "Exponential";
        case DistributionType::LogNormal: return "Log-Normal";
        case DistributionType::Gamma: return "Gamma";
        default: return "Unknown";
    }
}

/**
 * DataAnalyzer - statistics for the Tools panels (hypothesis tests,
 * regression, distribution fits). Exploring a dataset (missing values,
 * outliers, correlation, distribution) is Data Studio's Explore job, in SQL
 * (core/data_explore); it uses the distribution fits here.
 */
class DataAnalyzer {
public:
    DataAnalyzer() = default;
    ~DataAnalyzer() = default;

    // ========== Utility Functions ==========

    /**
     * Get numeric values from a column (filters out nulls)
     */
    static std::vector<double> GetNumericValues(const DataTable& table, size_t column_index);

    /**
     * Detect the data type of a column
     */
    static ColumnDataType DetectColumnType(const DataTable& table, size_t column_index);

    /**
     * Check if a cell value is null/missing
     */
    static bool IsNull(const std::variant<std::string, double, int64_t, std::monostate>& value);

    /**
     * Convert cell value to double (returns nullopt if not convertible)
     */
    static std::optional<double> ToDouble(const std::variant<std::string, double, int64_t, std::monostate>& value);

    // ========== Statistical Functions ==========

    static double Mean(const std::vector<double>& values);
    static double StdDev(const std::vector<double>& values, double mean);
    static double Variance(const std::vector<double>& values, double mean);
    static double Percentile(std::vector<double> values, double p);  // p in [0, 100]

    // ========== Hypothesis Testing ==========

    /**
     * One-sample t-test: test if sample mean equals hypothesized mean
     */
    static HypothesisTestResult OneSampleTTest(const std::vector<double>& sample, double mu0, double alpha = 0.05);

    /**
     * Two-sample t-test: compare means of two independent samples
     */
    static HypothesisTestResult TwoSampleTTest(const std::vector<double>& sample1,
                                                const std::vector<double>& sample2,
                                                bool equal_variance = true,
                                                double alpha = 0.05);

    /**
     * Paired t-test: compare paired observations
     */
    static HypothesisTestResult PairedTTest(const std::vector<double>& sample1,
                                             const std::vector<double>& sample2,
                                             double alpha = 0.05);

    /**
     * One-way ANOVA: compare means of multiple groups
     */
    static HypothesisTestResult OneWayANOVA(const std::vector<std::vector<double>>& groups,
                                             double alpha = 0.05);

    /**
     * Chi-square test for independence
     */
    static HypothesisTestResult ChiSquareTest(const std::vector<std::vector<double>>& contingency,
                                               double alpha = 0.05);

    /**
     * Student's t cumulative distribution function
     */
    static double StudentTCDF(double t, double df);

    /**
     * F cumulative distribution function
     */
    static double FCDF(double f, double df1, double df2);

    /**
     * Chi-square cumulative distribution function
     */
    static double ChiSquareCDF(double x, double df);

    /**
     * Critical value for t-distribution
     */
    static double TInv(double alpha, double df);

    // ========== Distribution Fitting ==========

    /**
     * Fit normal distribution to data
     */
    static DistributionFitResult FitNormal(const std::vector<double>& data);

    /**
     * Fit uniform distribution to data
     */
    static DistributionFitResult FitUniform(const std::vector<double>& data);

    /**
     * Fit exponential distribution to data
     */
    static DistributionFitResult FitExponential(const std::vector<double>& data);

    /**
     * Fit log-normal distribution to data
     */
    static DistributionFitResult FitLogNormal(const std::vector<double>& data);

    /**
     * Fit all supported distributions and rank by fit quality
     */
    static std::vector<DistributionFitResult> FitAllDistributions(const std::vector<double>& data);

    /**
     * Kolmogorov-Smirnov test for distribution fit
     */
    static std::pair<double, double> KolmogorovSmirnovTest(const std::vector<double>& data,
                                                            DistributionType dist_type,
                                                            const std::map<std::string, double>& params);

    /**
     * Normal CDF
     */
    static double NormalCDF(double x, double mu = 0.0, double sigma = 1.0);

    // ========== Regression Analysis ==========

    /**
     * Simple linear regression: y = b0 + b1*x
     */
    static RegressionResult LinearRegression(const std::vector<double>& x,
                                              const std::vector<double>& y);

    /**
     * Polynomial regression: y = b0 + b1*x + b2*x^2 + ... + bk*x^k
     */
    static RegressionResult PolynomialRegression(const std::vector<double>& x,
                                                  const std::vector<double>& y,
                                                  int degree);

    /**
     * Multiple linear regression: y = b0 + b1*x1 + b2*x2 + ...
     */
    static RegressionResult MultipleLinearRegression(const std::vector<std::vector<double>>& X,
                                                      const std::vector<double>& y,
                                                      const std::vector<std::string>& predictor_names = {});

};

} // namespace cyxwiz
