#include "data_analyzer.h"
#include "../data/data_table.h"
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cyxwiz {

// ========== Statistical Functions ==========

double DataAnalyzer::Mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double DataAnalyzer::Variance(const std::vector<double>& values, double mean) {
    if (values.size() < 2) return 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(values.size() - 1);  // Sample variance
}

// ========== Utility Functions ==========

bool DataAnalyzer::IsNull(const std::variant<std::string, double, int64_t, std::monostate>& value) {
    if (std::holds_alternative<std::monostate>(value)) return true;
    if (std::holds_alternative<std::string>(value)) {
        const auto& str = std::get<std::string>(value);
        return str.empty() || str == "NA" || str == "N/A" || str == "null" ||
               str == "NULL" || str == "nan" || str == "NaN" || str == "None";
    }
    if (std::holds_alternative<double>(value)) {
        return std::isnan(std::get<double>(value));
    }
    return false;
}

std::optional<double> DataAnalyzer::ToDouble(const std::variant<std::string, double, int64_t, std::monostate>& value) {
    if (std::holds_alternative<double>(value)) {
        double d = std::get<double>(value);
        if (!std::isnan(d) && !std::isinf(d)) return d;
        return std::nullopt;
    }
    if (std::holds_alternative<int64_t>(value)) {
        return static_cast<double>(std::get<int64_t>(value));
    }
    if (std::holds_alternative<std::string>(value)) {
        const auto& str = std::get<std::string>(value);
        if (str.empty()) return std::nullopt;
        try {
            size_t pos = 0;
            double d = std::stod(str, &pos);
            if (pos == str.size() && !std::isnan(d) && !std::isinf(d)) {
                return d;
            }
        } catch (...) {}
    }
    return std::nullopt;
}

ColumnDataType DataAnalyzer::DetectColumnType(const DataTable& table, size_t column_index) {
    size_t numeric_count = 0;
    size_t integer_count = 0;
    size_t string_count = 0;
    size_t bool_count = 0;
    size_t valid_count = 0;

    for (size_t i = 0; i < std::min(table.GetRowCount(), size_t(1000)); i++) {
        auto cell = table.GetCell(i, column_index);

        if (IsNull(cell)) continue;
        valid_count++;

        if (std::holds_alternative<double>(cell)) {
            double d = std::get<double>(cell);
            if (d == std::floor(d) && std::abs(d) < 1e15) {
                integer_count++;
            }
            numeric_count++;
        } else if (std::holds_alternative<int64_t>(cell)) {
            integer_count++;
            numeric_count++;
        } else if (std::holds_alternative<std::string>(cell)) {
            const auto& str = std::get<std::string>(cell);
            // Check if it's a boolean
            if (str == "true" || str == "false" || str == "True" || str == "False" ||
                str == "TRUE" || str == "FALSE" || str == "0" || str == "1") {
                bool_count++;
            }
            // Check if it's numeric
            try {
                size_t pos = 0;
                double d = std::stod(str, &pos);
                if (pos == str.size()) {
                    if (d == std::floor(d) && std::abs(d) < 1e15) {
                        integer_count++;
                    }
                    numeric_count++;
                } else {
                    string_count++;
                }
            } catch (...) {
                string_count++;
            }
        }
    }

    if (valid_count == 0) return ColumnDataType::Unknown;

    double numeric_ratio = static_cast<double>(numeric_count) / valid_count;
    double integer_ratio = static_cast<double>(integer_count) / valid_count;
    double string_ratio = static_cast<double>(string_count) / valid_count;
    double bool_ratio = static_cast<double>(bool_count) / valid_count;

    if (bool_ratio > 0.9) return ColumnDataType::Boolean;
    if (numeric_ratio > 0.9) {
        if (integer_ratio > 0.9) return ColumnDataType::Integer;
        return ColumnDataType::Numeric;
    }
    if (string_ratio > 0.5) return ColumnDataType::Categorical;
    if (numeric_ratio > 0.5 || string_ratio > 0.5) return ColumnDataType::Mixed;

    return ColumnDataType::Unknown;
}

// ========== Normal distribution (the fits' KS test) ==========

double DataAnalyzer::NormalCDF(double x, double mu, double sigma) {
    if (sigma <= 0) return 0.0;
    double z = (x - mu) / sigma;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// ========== Distribution Fitting ==========

// Maximum-likelihood standard deviation: the root mean squared deviation
// (divides by n, not n - 1).
static double MaximumLikelihoodSigma(const std::vector<double>& values, double mean) {
    double sum_sq = 0.0;
    for (double v : values) sum_sq += (v - mean) * (v - mean);
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

DistributionFitResult DataAnalyzer::FitNormal(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Normal;
    result.name = "Normal";

    if (data.size() < 3) return result;

    // Maximum-likelihood estimates (scipy.stats.norm.fit): sigma divides by n.
    double mu = Mean(data);
    double sigma = MaximumLikelihoodSigma(data, mu);

    result.parameters["mu"] = mu;
    result.parameters["sigma"] = sigma;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double ll = 0.0;
    for (double x : data) {
        double z = (x - mu) / sigma;
        ll += -0.5 * z * z - std::log(sigma) - 0.5 * std::log(2.0 * M_PI);
    }
    result.log_likelihood = ll;

    // AIC and BIC
    result.aic = -2.0 * ll + 2.0 * 2.0;  // 2 parameters
    result.bic = -2.0 * ll + 2.0 * std::log(n);

    // KS test
    auto ks = KolmogorovSmirnovTest(data, DistributionType::Normal, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitUniform(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Uniform;
    result.name = "Uniform";

    if (data.empty()) return result;

    double a = *std::min_element(data.begin(), data.end());
    double b = *std::max_element(data.begin(), data.end());

    result.parameters["a"] = a;
    result.parameters["b"] = b;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double range = b - a;
    if (range > 0) {
        result.log_likelihood = -n * std::log(range);
    }

    result.aic = -2.0 * result.log_likelihood + 2.0 * 2.0;
    result.bic = -2.0 * result.log_likelihood + 2.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::Uniform, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitExponential(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::Exponential;
    result.name = "Exponential";

    if (data.empty()) return result;

    // Check for positive values
    for (double x : data) {
        if (x < 0) {
            result.good_fit = false;
            return result;
        }
    }

    double lambda = 1.0 / Mean(data);
    result.parameters["lambda"] = lambda;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    result.log_likelihood = n * std::log(lambda) - lambda * sum;

    result.aic = -2.0 * result.log_likelihood + 2.0 * 1.0;
    result.bic = -2.0 * result.log_likelihood + 1.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::Exponential, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

DistributionFitResult DataAnalyzer::FitLogNormal(const std::vector<double>& data) {
    DistributionFitResult result;
    result.type = DistributionType::LogNormal;
    result.name = "Log-Normal";

    if (data.empty()) return result;

    // Check for positive values
    std::vector<double> log_data;
    for (double x : data) {
        if (x <= 0) {
            result.good_fit = false;
            return result;
        }
        log_data.push_back(std::log(x));
    }

    // Maximum-likelihood estimates (scipy.stats.lognorm.fit, floc=0).
    double mu = Mean(log_data);
    double sigma = MaximumLikelihoodSigma(log_data, mu);

    result.parameters["mu"] = mu;
    result.parameters["sigma"] = sigma;

    // Log-likelihood
    double n = static_cast<double>(data.size());
    double ll = 0.0;
    for (size_t i = 0; i < data.size(); i++) {
        double z = (log_data[i] - mu) / sigma;
        ll += -0.5 * z * z - std::log(sigma * data[i]) - 0.5 * std::log(2.0 * M_PI);
    }
    result.log_likelihood = ll;

    result.aic = -2.0 * ll + 2.0 * 2.0;
    result.bic = -2.0 * ll + 2.0 * std::log(n);

    auto ks = KolmogorovSmirnovTest(data, DistributionType::LogNormal, result.parameters);
    result.ks_statistic = ks.first;
    result.ks_p_value = ks.second;
    result.good_fit = result.ks_p_value > 0.05;

    return result;
}

std::vector<DistributionFitResult> DataAnalyzer::FitAllDistributions(const std::vector<double>& data) {
    std::vector<DistributionFitResult> results;

    results.push_back(FitNormal(data));
    results.push_back(FitUniform(data));

    // Only fit positive-value distributions if data is positive
    bool all_positive = std::all_of(data.begin(), data.end(), [](double x) { return x > 0; });
    if (all_positive) {
        results.push_back(FitExponential(data));
        results.push_back(FitLogNormal(data));
    }

    // Sort by AIC (lower is better)
    std::sort(results.begin(), results.end(), [](const DistributionFitResult& a, const DistributionFitResult& b) {
        return a.aic < b.aic;
    });

    return results;
}

std::pair<double, double> DataAnalyzer::KolmogorovSmirnovTest(const std::vector<double>& data,
                                                               DistributionType dist_type,
                                                               const std::map<std::string, double>& params) {
    if (data.empty()) return {0.0, 1.0};

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    double n = static_cast<double>(data.size());
    double D = 0.0;

    for (size_t i = 0; i < sorted_data.size(); i++) {
        double F_empirical = static_cast<double>(i + 1) / n;
        double F_theoretical = 0.0;

        switch (dist_type) {
            case DistributionType::Normal: {
                double mu = params.count("mu") ? params.at("mu") : 0.0;
                double sigma = params.count("sigma") ? params.at("sigma") : 1.0;
                F_theoretical = NormalCDF(sorted_data[i], mu, sigma);
                break;
            }
            case DistributionType::Uniform: {
                double a = params.count("a") ? params.at("a") : 0.0;
                double b = params.count("b") ? params.at("b") : 1.0;
                if (b > a) {
                    F_theoretical = (sorted_data[i] - a) / (b - a);
                    F_theoretical = std::max(0.0, std::min(1.0, F_theoretical));
                }
                break;
            }
            case DistributionType::Exponential: {
                double lambda = params.count("lambda") ? params.at("lambda") : 1.0;
                F_theoretical = 1.0 - std::exp(-lambda * sorted_data[i]);
                break;
            }
            case DistributionType::LogNormal: {
                double mu = params.count("mu") ? params.at("mu") : 0.0;
                double sigma = params.count("sigma") ? params.at("sigma") : 1.0;
                if (sorted_data[i] > 0) {
                    F_theoretical = NormalCDF(std::log(sorted_data[i]), mu, sigma);
                }
                break;
            }
            default:
                F_theoretical = 0.5;
        }

        double d1 = std::abs(F_empirical - F_theoretical);
        double d2 = std::abs((static_cast<double>(i) / n) - F_theoretical);
        D = std::max(D, std::max(d1, d2));
    }

    // p-value from the asymptotic Kolmogorov distribution with Stephens'
    // small-sample correction: Q(l) = 2 sum_k (-1)^(k-1) exp(-2 k^2 l^2)
    // (scipy.stats.kstwobign.sf). Summing the whole series, not only its
    // first term, keeps large p-values right; near l = 0 the series stops
    // converging and Q is 1 to double precision.
    double sqrt_n = std::sqrt(n);
    double lambda_ks = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D;
    double p_value = 1.0;
    if (lambda_ks > 0.2) {
        double sum = 0.0;
        for (int k = 1; k <= 100; ++k) {
            const double term = std::exp(-2.0 * k * k * lambda_ks * lambda_ks);
            sum += (k % 2 == 1) ? term : -term;
            if (term < 1e-17) break;
        }
        p_value = std::max(0.0, std::min(1.0, 2.0 * sum));
    }

    return {D, p_value};
}

} // namespace cyxwiz
