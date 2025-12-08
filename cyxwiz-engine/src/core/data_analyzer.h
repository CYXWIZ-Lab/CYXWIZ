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

/**
 * Histogram bin for numeric data visualization
 */
struct HistogramBin {
    double low = 0.0;
    double high = 0.0;
    size_t count = 0;
    float percentage = 0.0f;
};

/**
 * Top value entry for categorical data
 */
struct TopValue {
    std::string value;
    size_t count = 0;
    float percentage = 0.0f;
};

/**
 * Column profile containing comprehensive statistics
 */
struct ColumnProfile {
    std::string name;
    ColumnDataType dtype = ColumnDataType::Unknown;

    // Basic counts
    size_t total_count = 0;
    size_t non_null_count = 0;
    size_t null_count = 0;
    size_t unique_count = 0;
    float null_percentage = 0.0f;

    // Numeric statistics (only valid for Numeric/Integer types)
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double std_dev = 0.0;
    double variance = 0.0;
    double q1 = 0.0;          // 25th percentile
    double q3 = 0.0;          // 75th percentile
    double iqr = 0.0;         // Interquartile range
    double skewness = 0.0;
    double kurtosis = 0.0;
    double sum = 0.0;

    // Distribution
    std::vector<HistogramBin> histogram;

    // For categorical data
    std::vector<TopValue> top_values;

    // Memory estimate (bytes)
    size_t memory_estimate = 0;

    // Helper methods
    bool IsNumeric() const {
        return dtype == ColumnDataType::Numeric || dtype == ColumnDataType::Integer;
    }

    std::string GetDTypeString() const;
};

/**
 * Full dataset profile
 */
struct DataProfile {
    std::string source_name;
    size_t row_count = 0;
    size_t column_count = 0;
    size_t total_nulls = 0;
    float null_percentage = 0.0f;
    size_t memory_estimate = 0;

    std::vector<ColumnProfile> columns;

    // Get column by name
    const ColumnProfile* GetColumn(const std::string& name) const;
    ColumnProfile* GetColumn(const std::string& name);
};

/**
 * Correlation result between two columns
 */
struct CorrelationResult {
    std::string col1;
    std::string col2;
    double pearson = 0.0;
    double spearman = 0.0;     // Not implemented initially
    size_t sample_count = 0;   // Number of valid pairs used
};

/**
 * Full correlation matrix
 */
struct CorrelationMatrix {
    std::vector<std::string> column_names;
    std::vector<std::vector<double>> matrix;  // N x N matrix of Pearson correlations
    size_t row_count = 0;                     // Original data rows used

    // Get correlation between two columns
    double Get(size_t i, size_t j) const {
        if (i < matrix.size() && j < matrix[i].size()) {
            return matrix[i][j];
        }
        return 0.0;
    }

    double Get(const std::string& col1, const std::string& col2) const;
};

/**
 * Missing value analysis result
 */
struct MissingValueAnalysis {
    std::string source_name;
    size_t total_cells = 0;
    size_t total_missing = 0;
    float missing_percentage = 0.0f;

    // Per-column missing info
    struct ColumnMissing {
        std::string name;
        size_t missing_count = 0;
        float missing_percentage = 0.0f;
        std::vector<size_t> missing_indices;  // Row indices with missing values (limited to first 1000)
    };
    std::vector<ColumnMissing> columns;

    // Rows with any missing values
    size_t rows_with_missing = 0;
    float rows_with_missing_percentage = 0.0f;

    // Complete rows (no missing values)
    size_t complete_rows = 0;
};

/**
 * Outlier detection methods
 */
enum class OutlierMethod {
    IQR,            // Interquartile range (default factor: 1.5)
    ZScore,         // Z-score (default threshold: 3.0)
    ModifiedZScore  // Modified Z-score using MAD (default threshold: 3.5)
};

/**
 * Single outlier entry
 */
struct OutlierEntry {
    size_t row_index = 0;
    double value = 0.0;
    double score = 0.0;  // Z-score or IQR distance
    bool is_low = false; // Below lower bound (vs above upper bound)
};

/**
 * Outlier detection result for a column
 */
struct OutlierResult {
    std::string column_name;
    OutlierMethod method = OutlierMethod::IQR;
    double parameter = 1.5;  // IQR factor or Z-score threshold

    // Bounds used for detection
    double lower_bound = 0.0;
    double upper_bound = 0.0;

    // Statistics used
    double mean = 0.0;
    double std_dev = 0.0;
    double median = 0.0;
    double mad = 0.0;  // Median Absolute Deviation
    double q1 = 0.0;
    double q3 = 0.0;

    // Outliers found
    std::vector<OutlierEntry> outliers;
    size_t outlier_count = 0;
    float outlier_percentage = 0.0f;
    size_t total_valid = 0;  // Non-null values analyzed
};

// ========== Phase 4: Statistical Tools ==========

/**
 * Extended descriptive statistics result
 */
struct DescriptiveStats {
    double count = 0;
    double mean = 0.0;
    double median = 0.0;
    double mode = 0.0;
    double std_dev = 0.0;
    double variance = 0.0;
    double min = 0.0;
    double max = 0.0;
    double range = 0.0;
    double q1 = 0.0;
    double q3 = 0.0;
    double iqr = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double sum = 0.0;
    double sem = 0.0;  // Standard Error of Mean
    double cv = 0.0;   // Coefficient of Variation
    std::vector<double> percentiles;  // 5, 10, 25, 50, 75, 90, 95
};

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
    double mse = 0.0;  // Mean Squared Error
    double rmse = 0.0; // Root Mean Squared Error
    double mae = 0.0;  // Mean Absolute Error
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
 * DataAnalyzer - Core statistics engine for data analysis
 *
 * Provides comprehensive data profiling, correlation analysis,
 * missing value detection, and outlier identification.
 */
class DataAnalyzer {
public:
    DataAnalyzer() = default;
    ~DataAnalyzer() = default;

    // ========== Data Profiling ==========

    /**
     * Generate comprehensive profile for a DataTable
     * @param table The data table to analyze
     * @param histogram_bins Number of bins for histograms (default: 20)
     * @param top_n Number of top values to include for categorical (default: 10)
     * @return Complete data profile
     */
    DataProfile ProfileTable(const DataTable& table, int histogram_bins = 20, int top_n = 10);

    /**
     * Profile a single column
     */
    ColumnProfile ProfileColumn(const DataTable& table, const std::string& column_name,
                                 int histogram_bins = 20, int top_n = 10);
    ColumnProfile ProfileColumn(const DataTable& table, size_t column_index,
                                 int histogram_bins = 20, int top_n = 10);

    // ========== Correlation Analysis ==========

    /**
     * Compute correlation matrix for all numeric columns
     */
    CorrelationMatrix ComputeCorrelationMatrix(const DataTable& table);

    /**
     * Compute correlation between two specific columns
     */
    CorrelationResult ComputeCorrelation(const DataTable& table,
                                          const std::string& col1,
                                          const std::string& col2);

    // ========== Missing Value Analysis ==========

    /**
     * Analyze missing values in the table
     * @param max_indices_per_column Maximum missing indices to store per column
     */
    MissingValueAnalysis AnalyzeMissingValues(const DataTable& table,
                                               size_t max_indices_per_column = 1000);

    // ========== Outlier Detection ==========

    /**
     * Detect outliers in a column using specified method
     */
    OutlierResult DetectOutliers(const DataTable& table,
                                  const std::string& column_name,
                                  OutlierMethod method = OutlierMethod::IQR,
                                  double parameter = 0.0);  // 0 = use default

    OutlierResult DetectOutliers(const DataTable& table,
                                  size_t column_index,
                                  OutlierMethod method = OutlierMethod::IQR,
                                  double parameter = 0.0);

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
    static double Median(std::vector<double> values);  // Note: modifies input
    static double StdDev(const std::vector<double>& values, double mean);
    static double Variance(const std::vector<double>& values, double mean);
    static double Percentile(std::vector<double> values, double p);  // p in [0, 100]
    static double Skewness(const std::vector<double>& values, double mean, double std_dev);
    static double Kurtosis(const std::vector<double>& values, double mean, double std_dev);
    static double MAD(std::vector<double> values, double median);  // Median Absolute Deviation

    /**
     * Compute Pearson correlation coefficient
     */
    static double PearsonCorrelation(const std::vector<double>& x, const std::vector<double>& y);

    // ========== Phase 4: Statistical Methods ==========

    /**
     * Compute extended descriptive statistics
     */
    static DescriptiveStats ComputeDescriptiveStats(const std::vector<double>& data);

    /**
     * Calculate mode (most frequent value)
     */
    static double Mode(const std::vector<double>& data);

    /**
     * Calculate Standard Error of Mean
     */
    static double SEM(const std::vector<double>& data);

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

    /**
     * Generate theoretical quantiles for QQ-plot
     */
    static std::vector<double> TheoreticalQuantiles(size_t n, DistributionType type,
                                                     const std::map<std::string, double>& params);

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

private:
    // Helper to build histogram
    std::vector<HistogramBin> BuildHistogram(const std::vector<double>& values,
                                              double min_val, double max_val,
                                              int num_bins);

    // Helper to get top N values for categorical data
    std::vector<TopValue> GetTopValues(const std::vector<std::string>& values, int top_n);
};

/**
 * Get string representation of column data type
 */
inline std::string ColumnProfile::GetDTypeString() const {
    switch (dtype) {
        case ColumnDataType::Numeric: return "Numeric";
        case ColumnDataType::Integer: return "Integer";
        case ColumnDataType::Categorical: return "Categorical";
        case ColumnDataType::Boolean: return "Boolean";
        case ColumnDataType::DateTime: return "DateTime";
        case ColumnDataType::Mixed: return "Mixed";
        default: return "Unknown";
    }
}

/**
 * Get string representation of outlier detection method
 */
inline const char* OutlierMethodToString(OutlierMethod method) {
    switch (method) {
        case OutlierMethod::IQR: return "IQR";
        case OutlierMethod::ZScore: return "Z-Score";
        case OutlierMethod::ModifiedZScore: return "Modified Z-Score";
        default: return "Unknown";
    }
}

} // namespace cyxwiz
