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

double DataAnalyzer::StdDev(const std::vector<double>& values, double mean) {
    return std::sqrt(Variance(values, mean));
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

double DataAnalyzer::Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(values.begin(), values.end());
    if (p >= 100.0) return *std::max_element(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    double index = (p / 100.0) * static_cast<double>(values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper || upper >= values.size()) {
        return values[lower];
    }

    double fraction = index - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
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

std::vector<double> DataAnalyzer::GetNumericValues(const DataTable& table, size_t column_index) {
    std::vector<double> values;
    values.reserve(table.GetRowCount());

    for (size_t i = 0; i < table.GetRowCount(); i++) {
        auto opt = ToDouble(table.GetCell(i, column_index));
        if (opt.has_value()) {
            values.push_back(opt.value());
        }
    }

    return values;
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

// ========== Hypothesis Testing Implementations ==========

double DataAnalyzer::NormalCDF(double x, double mu, double sigma) {
    if (sigma <= 0) return 0.0;
    double z = (x - mu) / sigma;
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

// Continued fraction of the regularized incomplete beta function, evaluated
// with the modified Lentz method (converges fast for x < (a + 1) / (a + b + 2)).
static double BetaContinuedFraction(double a, double b, double x) {
    constexpr double kTiny = 1e-300;
    constexpr double kEps = 1e-15;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kTiny) d = kTiny;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= 300; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kTiny) d = kTiny;
        c = 1.0 + aa / c;
        if (std::abs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::abs(step - 1.0) < kEps) break;
    }
    return h;
}

// Regularized incomplete beta function I_x(a, b).
static double betaInc(double a, double b, double x) {
    if (x <= 0) return 0.0;
    if (x >= 1) return 1.0;
    const double front = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                                  a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * BetaContinuedFraction(a, b, x) / a;
    }
    return 1.0 - front * BetaContinuedFraction(b, a, 1.0 - x) / b;
}

double DataAnalyzer::StudentTCDF(double t, double df) {
    if (df <= 0) return 0.5;

    double x = df / (df + t * t);
    double prob = 0.5 * betaInc(df / 2.0, 0.5, x);

    return t < 0 ? prob : 1.0 - prob;
}

double DataAnalyzer::FCDF(double f, double df1, double df2) {
    if (f <= 0 || df1 <= 0 || df2 <= 0) return 0.0;

    double x = df1 * f / (df1 * f + df2);
    return betaInc(df1 / 2.0, df2 / 2.0, x);
}

double DataAnalyzer::ChiSquareCDF(double x, double df) {
    if (x <= 0 || df <= 0) return 0.0;

    // Regularized lower incomplete gamma P(df / 2, x / 2).
    const double a = df / 2.0;
    const double z = x / 2.0;
    const double front = std::exp(-z + a * std::log(z) - std::lgamma(a));

    if (z < a + 1.0) {
        // Series: converges quickly below a + 1.
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n <= 1000; ++n) {
            term *= z / (a + n);
            sum += term;
            if (std::abs(term) < std::abs(sum) * 1e-16) break;
        }
        return std::min(1.0, sum * front);
    }

    // Continued fraction for the upper tail Q (modified Lentz).
    constexpr double kTiny = 1e-300;
    double b = z + 1.0 - a;
    double c = 1.0 / kTiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < kTiny) d = kTiny;
        c = b + an / c;
        if (std::abs(c) < kTiny) c = kTiny;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::abs(step - 1.0) < 1e-16) break;
    }
    return std::max(0.0, 1.0 - front * h);
}

double DataAnalyzer::TInv(double alpha, double df) {
    // Two-tailed critical value: t with P(T < t) = 1 - alpha / 2. The CDF is
    // monotone, so bracket the root and bisect (Newton from 1.96 could step
    // past the flat tail and never come back).
    const double target = 1.0 - alpha / 2.0;
    double lo = 0.0;
    double hi = 2.0;
    while (StudentTCDF(hi, df) < target && hi < 1e8) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-13 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (StudentTCDF(mid, df) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

HypothesisTestResult DataAnalyzer::OneSampleTTest(const std::vector<double>& sample, double mu0, double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::OneSampleTTest;

    if (sample.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2)";
        return result;
    }

    double n = static_cast<double>(sample.size());
    double mean = Mean(sample);
    double sd = StdDev(sample, mean);
    double se = sd / std::sqrt(n);

    if (se == 0) {
        result.interpretation = "Standard error is zero (no variance in sample)";
        return result;
    }

    result.test_statistic = (mean - mu0) / se;
    result.df = n - 1;
    result.mean_diff = mean - mu0;
    result.se_diff = se;

    // Two-tailed p-value
    result.p_value = 2.0 * (1.0 - StudentTCDF(std::abs(result.test_statistic), result.df));

    // Confidence interval
    double t_crit = TInv(alpha, result.df);
    result.confidence_interval_low = mean - t_crit * se;
    result.confidence_interval_high = mean + t_crit * se;

    // Effect size (Cohen's d)
    result.effect_size = (mean - mu0) / sd;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Sample mean significantly differs from " +
                               std::to_string(mu0) + " (p = " +
                               std::to_string(result.p_value) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant difference from " +
                               std::to_string(mu0) + " (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::TwoSampleTTest(const std::vector<double>& sample1,
                                                   const std::vector<double>& sample2,
                                                   bool equal_variance,
                                                   double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::TwoSampleTTest;

    if (sample1.size() < 2 || sample2.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2 in one or both groups)";
        return result;
    }

    double n1 = static_cast<double>(sample1.size());
    double n2 = static_cast<double>(sample2.size());
    double mean1 = Mean(sample1);
    double mean2 = Mean(sample2);
    double var1 = Variance(sample1, mean1);
    double var2 = Variance(sample2, mean2);

    double se, df;

    if (equal_variance) {
        // Pooled variance
        double sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
        se = std::sqrt(sp2 * (1.0 / n1 + 1.0 / n2));
        df = n1 + n2 - 2;
    } else {
        // Welch's approximation
        se = std::sqrt(var1 / n1 + var2 / n2);
        double num = std::pow(var1 / n1 + var2 / n2, 2);
        double den = std::pow(var1 / n1, 2) / (n1 - 1) + std::pow(var2 / n2, 2) / (n2 - 1);
        df = num / den;
    }

    if (se == 0) {
        result.interpretation = "Standard error is zero";
        return result;
    }

    result.test_statistic = (mean1 - mean2) / se;
    result.df = df;
    result.mean_diff = mean1 - mean2;
    result.se_diff = se;

    result.p_value = 2.0 * (1.0 - StudentTCDF(std::abs(result.test_statistic), result.df));

    double t_crit = TInv(alpha, result.df);
    result.confidence_interval_low = (mean1 - mean2) - t_crit * se;
    result.confidence_interval_high = (mean1 - mean2) + t_crit * se;

    // Cohen's d
    double pooled_sd = std::sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
    result.effect_size = pooled_sd > 0 ? (mean1 - mean2) / pooled_sd : 0;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Significant difference between groups (p = " +
                               std::to_string(result.p_value) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant difference (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::PairedTTest(const std::vector<double>& sample1,
                                                const std::vector<double>& sample2,
                                                double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::PairedTTest;

    if (sample1.size() != sample2.size()) {
        result.interpretation = "Sample sizes must be equal for paired test";
        return result;
    }

    if (sample1.size() < 2) {
        result.interpretation = "Insufficient sample size (n < 2)";
        return result;
    }

    // Calculate differences
    std::vector<double> diff(sample1.size());
    for (size_t i = 0; i < sample1.size(); i++) {
        diff[i] = sample1[i] - sample2[i];
    }

    // One-sample t-test on differences
    result = OneSampleTTest(diff, 0.0, alpha);
    result.test_type = TestType::PairedTTest;

    return result;
}

HypothesisTestResult DataAnalyzer::OneWayANOVA(const std::vector<std::vector<double>>& groups,
                                                double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::OneWayANOVA;

    if (groups.size() < 2) {
        result.interpretation = "Need at least 2 groups for ANOVA";
        return result;
    }

    size_t k = groups.size();  // Number of groups
    size_t N = 0;              // Total sample size
    for (const auto& g : groups) {
        if (g.empty()) {
            result.interpretation = "Empty group detected";
            return result;
        }
        N += g.size();
    }

    // Calculate grand mean
    double grand_sum = 0.0;
    for (const auto& g : groups) {
        grand_sum += std::accumulate(g.begin(), g.end(), 0.0);
    }
    double grand_mean = grand_sum / static_cast<double>(N);

    // Calculate SSB (between groups) and SSW (within groups)
    double SSB = 0.0;
    double SSW = 0.0;

    for (const auto& g : groups) {
        double group_mean = Mean(g);
        double n = static_cast<double>(g.size());

        SSB += n * std::pow(group_mean - grand_mean, 2);

        for (double v : g) {
            SSW += std::pow(v - group_mean, 2);
        }
    }

    double df_between = static_cast<double>(k - 1);
    double df_within = static_cast<double>(N - k);

    if (df_within <= 0) {
        result.interpretation = "Insufficient degrees of freedom";
        return result;
    }

    double MSB = SSB / df_between;
    double MSW = SSW / df_within;

    if (MSW == 0) {
        result.interpretation = "Within-group variance is zero";
        return result;
    }

    result.test_statistic = MSB / MSW;  // F-statistic
    result.df = df_between;
    result.df2 = df_within;

    result.p_value = 1.0 - FCDF(result.test_statistic, df_between, df_within);

    // Effect size: eta-squared
    double SST = SSB + SSW;
    result.effect_size = SST > 0 ? SSB / SST : 0;

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Significant differences among groups (p = " +
                               std::to_string(result.p_value) + ", eta^2 = " +
                               std::to_string(result.effect_size) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant differences (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
}

HypothesisTestResult DataAnalyzer::ChiSquareTest(const std::vector<std::vector<double>>& contingency,
                                                  double alpha) {
    HypothesisTestResult result;
    result.test_type = TestType::ChiSquare;

    if (contingency.empty() || contingency[0].empty()) {
        result.interpretation = "Empty contingency table";
        return result;
    }

    size_t rows = contingency.size();
    size_t cols = contingency[0].size();

    // Calculate row totals, column totals, and grand total
    std::vector<double> row_totals(rows, 0.0);
    std::vector<double> col_totals(cols, 0.0);
    double grand_total = 0.0;

    for (size_t i = 0; i < rows; i++) {
        if (contingency[i].size() != cols) {
            result.interpretation = "Inconsistent column counts";
            return result;
        }
        for (size_t j = 0; j < cols; j++) {
            row_totals[i] += contingency[i][j];
            col_totals[j] += contingency[i][j];
            grand_total += contingency[i][j];
        }
    }

    if (grand_total == 0) {
        result.interpretation = "Total count is zero";
        return result;
    }

    // Calculate chi-square statistic
    double chi2 = 0.0;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double expected = (row_totals[i] * col_totals[j]) / grand_total;
            if (expected > 0) {
                chi2 += std::pow(contingency[i][j] - expected, 2) / expected;
            }
        }
    }

    result.test_statistic = chi2;
    result.df = static_cast<double>((rows - 1) * (cols - 1));

    result.p_value = 1.0 - ChiSquareCDF(chi2, result.df);

    // Effect size: Cramér's V
    const size_t min_dimension = std::min(rows - 1, cols - 1);
    const double min_dim = static_cast<double>(min_dimension);
    if (min_dim > 0 && grand_total > 0) {
        result.effect_size = std::sqrt(chi2 / (grand_total * min_dim));
    }

    result.reject_null = result.p_value < alpha;

    if (result.reject_null) {
        result.interpretation = "Reject H0: Variables are associated (p = " +
                               std::to_string(result.p_value) + ", Cramer's V = " +
                               std::to_string(result.effect_size) + ")";
    } else {
        result.interpretation = "Fail to reject H0: No significant association (p = " +
                               std::to_string(result.p_value) + ")";
    }

    return result;
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

// ========== Regression Analysis ==========

RegressionResult DataAnalyzer::LinearRegression(const std::vector<double>& x,
                                                 const std::vector<double>& y) {
    RegressionResult result;
    result.type = RegressionType::Linear;

    if (x.size() != y.size() || x.size() < 3) {
        return result;
    }

    size_t n = x.size();
    result.n = n;
    result.df_model = 1;
    result.df_resid = n - 2;

    double mean_x = Mean(x);
    double mean_y = Mean(y);

    // Calculate slope and intercept
    double ss_xy = 0.0, ss_xx = 0.0;
    for (size_t i = 0; i < n; i++) {
        ss_xy += (x[i] - mean_x) * (y[i] - mean_y);
        ss_xx += (x[i] - mean_x) * (x[i] - mean_x);
    }

    if (ss_xx == 0) return result;

    double b1 = ss_xy / ss_xx;  // Slope
    double b0 = mean_y - b1 * mean_x;  // Intercept

    result.coefficients = {b0, b1};
    result.predictor_names = {"intercept", "x"};

    // Calculate predictions and residuals
    result.predicted.resize(n);
    result.residuals.resize(n);
    double ss_res = 0.0, ss_tot = 0.0;
    double mae = 0.0;

    for (size_t i = 0; i < n; i++) {
        result.predicted[i] = b0 + b1 * x[i];
        result.residuals[i] = y[i] - result.predicted[i];
        ss_res += result.residuals[i] * result.residuals[i];
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        mae += std::abs(result.residuals[i]);
    }

    result.r_squared = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
    result.adjusted_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - 2);
    result.mse = ss_res / n;
    result.rmse = std::sqrt(result.mse);
    result.mae = mae / n;
    result.residual_variance = ss_res / result.df_resid;
    result.residual_standard_error = std::sqrt(result.residual_variance);

    // Standard errors
    double se_b1 = std::sqrt(result.residual_variance / ss_xx);
    double se_b0 = std::sqrt(
        result.residual_variance *
        (1.0 / n + mean_x * mean_x / ss_xx));

    result.std_errors = {se_b0, se_b1};

    // t-values and p-values
    result.t_values = {b0 / se_b0, b1 / se_b1};
    const double residual_df = static_cast<double>(result.df_resid);
    result.p_values = {
        2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[0]), residual_df)),
        2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[1]), residual_df))
    };

    // F-statistic
    double ss_reg = ss_tot - ss_res;
    result.f_statistic = result.residual_variance > 0
        ? (ss_reg / 1) / result.residual_variance
        : 0;
    result.f_p_value = 1.0 - FCDF(result.f_statistic, 1.0, residual_df);

    return result;
}

RegressionResult DataAnalyzer::PolynomialRegression(const std::vector<double>& x,
                                                     const std::vector<double>& y,
                                                     int degree) {
    RegressionResult result;
    result.type = RegressionType::Polynomial;

    if (x.size() != y.size() || x.size() < static_cast<size_t>(degree + 2)) {
        return result;
    }

    // Build design matrix X = [1, x, x^2, ..., x^degree]
    size_t n = x.size();
    size_t p = degree + 1;

    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    for (size_t i = 0; i < n; i++) {
        X[i][0] = 1.0;
        for (int j = 1; j <= degree; j++) {
            X[i][j] = std::pow(x[i], j);
        }
    }

    // Build predictor names
    std::vector<std::string> names = {"intercept"};
    for (int j = 1; j <= degree; j++) {
        names.push_back("x^" + std::to_string(j));
    }

    return MultipleLinearRegression(X, y, names);
}

RegressionResult DataAnalyzer::MultipleLinearRegression(const std::vector<std::vector<double>>& X,
                                                         const std::vector<double>& y,
                                                         const std::vector<std::string>& predictor_names) {
    RegressionResult result;
    result.type = RegressionType::Multiple;

    size_t n = X.size();
    if (n == 0 || X[0].empty() || y.size() != n) {
        return result;
    }

    size_t p = X[0].size();  // Number of predictors including intercept
    result.n = n;
    result.df_model = p - 1;
    result.df_resid = n - p;

    if (n <= p) return result;

    // Normal equations: (X'X)β = X'y
    // Using Gauss-Jordan elimination

    // Compute X'X
    std::vector<std::vector<double>> XtX(p, std::vector<double>(p, 0.0));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            for (size_t k = 0; k < n; k++) {
                XtX[i][j] += X[k][i] * X[k][j];
            }
        }
    }

    // Compute X'y
    std::vector<double> Xty(p, 0.0);
    for (size_t i = 0; i < p; i++) {
        for (size_t k = 0; k < n; k++) {
            Xty[i] += X[k][i] * y[k];
        }
    }

    // Solve using Gauss-Jordan with pivoting
    std::vector<std::vector<double>> augmented(p, std::vector<double>(p + 1));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            augmented[i][j] = XtX[i][j];
        }
        augmented[i][p] = Xty[i];
    }

    // Forward elimination with partial pivoting
    for (size_t col = 0; col < p; col++) {
        // Find pivot
        size_t max_row = col;
        for (size_t row = col + 1; row < p; row++) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[max_row][col])) {
                max_row = row;
            }
        }
        std::swap(augmented[col], augmented[max_row]);

        if (std::abs(augmented[col][col]) < 1e-10) {
            // Matrix is singular
            return result;
        }

        // Eliminate
        for (size_t row = 0; row < p; row++) {
            if (row != col) {
                double factor = augmented[row][col] / augmented[col][col];
                for (size_t j = col; j <= p; j++) {
                    augmented[row][j] -= factor * augmented[col][j];
                }
            }
        }
    }

    // Extract solution
    result.coefficients.resize(p);
    for (size_t i = 0; i < p; i++) {
        result.coefficients[i] = augmented[i][p] / augmented[i][i];
    }

    // Set predictor names
    if (predictor_names.size() == p) {
        result.predictor_names = predictor_names;
    } else {
        result.predictor_names.push_back("intercept");
        for (size_t i = 1; i < p; i++) {
            result.predictor_names.push_back("x" + std::to_string(i));
        }
    }

    // Calculate predictions and residuals
    double mean_y = Mean(y);
    result.predicted.resize(n);
    result.residuals.resize(n);
    double ss_res = 0.0, ss_tot = 0.0;
    double mae = 0.0;

    for (size_t i = 0; i < n; i++) {
        result.predicted[i] = 0.0;
        for (size_t j = 0; j < p; j++) {
            result.predicted[i] += result.coefficients[j] * X[i][j];
        }
        result.residuals[i] = y[i] - result.predicted[i];
        ss_res += result.residuals[i] * result.residuals[i];
        ss_tot += (y[i] - mean_y) * (y[i] - mean_y);
        mae += std::abs(result.residuals[i]);
    }

    result.r_squared = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0;
    result.adjusted_r_squared = 1.0 - (1.0 - result.r_squared) * (n - 1) / (n - p);
    result.mse = ss_res / n;
    result.rmse = std::sqrt(result.mse);
    result.mae = mae / n;
    result.residual_variance = ss_res / result.df_resid;
    result.residual_standard_error = std::sqrt(result.residual_variance);

    // Compute (X'X)^-1 for standard errors
    // We already have the result of Gauss-Jordan, extract diagonal
    std::vector<std::vector<double>> XtX_inv(p, std::vector<double>(p, 0.0));

    // Re-solve with identity matrix to get inverse
    std::vector<std::vector<double>> aug2(p, std::vector<double>(2 * p));
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            aug2[i][j] = XtX[i][j];
            aug2[i][p + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t col = 0; col < p; col++) {
        size_t max_row = col;
        for (size_t row = col + 1; row < p; row++) {
            if (std::abs(aug2[row][col]) > std::abs(aug2[max_row][col])) {
                max_row = row;
            }
        }
        std::swap(aug2[col], aug2[max_row]);

        if (std::abs(aug2[col][col]) > 1e-10) {
            double pivot = aug2[col][col];
            for (size_t j = 0; j < 2 * p; j++) {
                aug2[col][j] /= pivot;
            }
            for (size_t row = 0; row < p; row++) {
                if (row != col) {
                    double factor = aug2[row][col];
                    for (size_t j = 0; j < 2 * p; j++) {
                        aug2[row][j] -= factor * aug2[col][j];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < p; j++) {
            XtX_inv[i][j] = aug2[i][p + j];
        }
    }

    // Standard errors
    result.std_errors.resize(p);
    result.t_values.resize(p);
    result.p_values.resize(p);

    const double residual_df = static_cast<double>(result.df_resid);

    for (size_t i = 0; i < p; i++) {
        result.std_errors[i] =
            std::sqrt(result.residual_variance * XtX_inv[i][i]);
        if (result.std_errors[i] > 0) {
            result.t_values[i] = result.coefficients[i] / result.std_errors[i];
            result.p_values[i] =
                2.0 * (1.0 - StudentTCDF(std::abs(result.t_values[i]), residual_df));
        }
    }

    // F-statistic
    double ss_reg = ss_tot - ss_res;
    const double model_df = static_cast<double>(result.df_model);
    double ms_reg = ss_reg / model_df;
    result.f_statistic = result.residual_variance > 0
        ? ms_reg / result.residual_variance
        : 0;
    result.f_p_value = 1.0 - FCDF(result.f_statistic, model_df, residual_df);

    return result;
}

} // namespace cyxwiz
