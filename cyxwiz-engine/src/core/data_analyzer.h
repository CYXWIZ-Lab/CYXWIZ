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
 * DataAnalyzer - column helpers and the distribution fits behind Data
 * Studio's Distribution tab. Exploring a dataset is Data Studio's Explore job,
 * in SQL (core/data_explore); hypothesis tests and regression are
 * cyxwiz::StatisticalTests (cyxwiz-backend, on the device; TOFIX129 D3).
 */
class DataAnalyzer {
public:
    DataAnalyzer() = default;
    ~DataAnalyzer() = default;

    // ========== Utility Functions ==========

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
    static double Variance(const std::vector<double>& values, double mean);

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

};

} // namespace cyxwiz
