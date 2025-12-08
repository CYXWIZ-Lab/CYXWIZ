// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// ============================================================================
// Utility Functions
// ============================================================================

double TimeSeries::Mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double TimeSeries::Variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = Mean(data);
    double sum = 0.0;
    for (double x : data) {
        sum += (x - m) * (x - m);
    }
    return sum / (data.size() - 1);
}

double TimeSeries::StdDev(const std::vector<double>& data) {
    return std::sqrt(Variance(data));
}

std::vector<double> TimeSeries::RollingMean(const std::vector<double>& data, int window) {
    if (window <= 0 || data.size() < static_cast<size_t>(window)) {
        return {};
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    double sum = 0.0;
    for (int i = 0; i < window; i++) {
        sum += data[i];
    }
    result.push_back(sum / window);

    for (size_t i = window; i < data.size(); i++) {
        sum += data[i] - data[i - window];
        result.push_back(sum / window);
    }

    return result;
}

std::vector<double> TimeSeries::RollingStd(const std::vector<double>& data, int window) {
    if (window <= 1 || data.size() < static_cast<size_t>(window)) {
        return {};
    }

    std::vector<double> result;
    result.reserve(data.size() - window + 1);

    for (size_t i = 0; i <= data.size() - window; i++) {
        double sum = 0.0, sum_sq = 0.0;
        for (int j = 0; j < window; j++) {
            sum += data[i + j];
            sum_sq += data[i + j] * data[i + j];
        }
        double mean = sum / window;
        double var = (sum_sq / window) - (mean * mean);
        result.push_back(std::sqrt((std::max)(0.0, var)));
    }

    return result;
}

std::vector<double> TimeSeries::CenteredMovingAverage(const std::vector<double>& data, int window) {
    if (window <= 0 || data.size() < static_cast<size_t>(window)) {
        return data;
    }

    std::vector<double> result(data.size(), 0.0);
    int half = window / 2;

    // For even window, we need an extra average
    bool even_window = (window % 2 == 0);

    for (size_t i = 0; i < data.size(); i++) {
        int start = static_cast<int>(i) - half;
        int end = static_cast<int>(i) + half;

        if (even_window) {
            // For even window, take average of two moving averages
            if (start < 0 || end >= static_cast<int>(data.size())) {
                result[i] = data[i];  // Edge: keep original
            } else {
                double sum1 = 0.0, sum2 = 0.0;
                for (int j = start; j < end; j++) {
                    sum1 += data[j];
                }
                for (int j = start + 1; j <= end; j++) {
                    sum2 += data[j];
                }
                result[i] = (sum1 / window + sum2 / window) / 2.0;
            }
        } else {
            if (start < 0 || end >= static_cast<int>(data.size())) {
                result[i] = data[i];
            } else {
                double sum = 0.0;
                for (int j = start; j <= end; j++) {
                    sum += data[j];
                }
                result[i] = sum / window;
            }
        }
    }

    return result;
}

// ============================================================================
// Decomposition
// ============================================================================

DecompositionResult TimeSeries::Decompose(
    const std::vector<double>& data,
    int period,
    const std::string& method
) {
    DecompositionResult result;
    result.original = data;
    result.period = period;
    result.method = method;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    if (period < 2) {
        result.error_message = "Period must be >= 2";
        return result;
    }

    if (data.size() < static_cast<size_t>(2 * period)) {
        result.error_message = "Data length must be at least 2 * period";
        return result;
    }

    try {
        int n = static_cast<int>(data.size());

        // Step 1: Compute trend using centered moving average
        result.trend = CenteredMovingAverage(data, period);

        // Step 2: Detrend the series
        std::vector<double> detrended(n);
        if (method == "multiplicative") {
            for (int i = 0; i < n; i++) {
                if (std::abs(result.trend[i]) > 1e-10) {
                    detrended[i] = data[i] / result.trend[i];
                } else {
                    detrended[i] = 1.0;
                }
            }
        } else {
            // Additive
            for (int i = 0; i < n; i++) {
                detrended[i] = data[i] - result.trend[i];
            }
        }

        // Step 3: Compute seasonal component (average by period position)
        std::vector<double> seasonal_avg(period, 0.0);
        std::vector<int> counts(period, 0);

        for (int i = 0; i < n; i++) {
            int pos = i % period;
            seasonal_avg[pos] += detrended[i];
            counts[pos]++;
        }

        for (int i = 0; i < period; i++) {
            if (counts[i] > 0) {
                seasonal_avg[i] /= counts[i];
            }
        }

        // Normalize seasonal component
        double seasonal_mean = std::accumulate(seasonal_avg.begin(), seasonal_avg.end(), 0.0) / period;
        if (method == "multiplicative") {
            for (int i = 0; i < period; i++) {
                seasonal_avg[i] /= seasonal_mean;
            }
        } else {
            for (int i = 0; i < period; i++) {
                seasonal_avg[i] -= seasonal_mean;
            }
        }

        // Expand seasonal component to full length
        result.seasonal.resize(n);
        for (int i = 0; i < n; i++) {
            result.seasonal[i] = seasonal_avg[i % period];
        }

        // Step 4: Compute residual
        result.residual.resize(n);
        if (method == "multiplicative") {
            for (int i = 0; i < n; i++) {
                if (std::abs(result.trend[i] * result.seasonal[i]) > 1e-10) {
                    result.residual[i] = data[i] / (result.trend[i] * result.seasonal[i]);
                } else {
                    result.residual[i] = 1.0;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                result.residual[i] = data[i] - result.trend[i] - result.seasonal[i];
            }
        }

        // Compute strength metrics
        double var_residual = Variance(result.residual);
        double var_detrended = Variance(detrended);
        double var_deseasoned(0.0);

        std::vector<double> deseasoned(n);
        if (method == "multiplicative") {
            for (int i = 0; i < n; i++) {
                if (std::abs(result.seasonal[i]) > 1e-10) {
                    deseasoned[i] = data[i] / result.seasonal[i];
                } else {
                    deseasoned[i] = data[i];
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                deseasoned[i] = data[i] - result.seasonal[i];
            }
        }
        var_deseasoned = Variance(deseasoned);

        // Trend strength: 1 - Var(residual)/Var(deseasoned)
        if (var_deseasoned > 1e-10) {
            result.trend_strength = (std::max)(0.0, 1.0 - var_residual / var_deseasoned);
        }

        // Seasonal strength: 1 - Var(residual)/Var(detrended)
        if (var_detrended > 1e-10) {
            result.seasonal_strength = (std::max)(0.0, 1.0 - var_residual / var_detrended);
        }

        result.residual_variance = var_residual;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Decomposition failed: ") + e.what();
    }

    return result;
}

DecompositionResult TimeSeries::STLDecompose(
    const std::vector<double>& data,
    int period,
    int seasonal_window,
    int trend_window
) {
    // Simplified STL - uses LOESS-like smoothing
    DecompositionResult result;
    result.original = data;
    result.period = period;
    result.method = "stl";

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    if (period < 2) {
        result.error_message = "Period must be >= 2";
        return result;
    }

    // Auto trend window
    if (trend_window < 0) {
        trend_window = static_cast<int>(std::ceil(1.5 * period / (1.0 - 1.5 / seasonal_window)));
        if (trend_window % 2 == 0) trend_window++;
    }

    try {
        int n = static_cast<int>(data.size());

        // Initialize components
        result.trend.resize(n, 0.0);
        result.seasonal.resize(n, 0.0);
        result.residual.resize(n, 0.0);

        // Initial trend estimate
        result.trend = CenteredMovingAverage(data, period);

        // Iterate STL
        for (int iter = 0; iter < 2; iter++) {
            // Detrend
            std::vector<double> detrended(n);
            for (int i = 0; i < n; i++) {
                detrended[i] = data[i] - result.trend[i];
            }

            // Compute seasonal by averaging across periods
            std::vector<double> seasonal_avg(period, 0.0);
            std::vector<int> counts(period, 0);

            for (int i = 0; i < n; i++) {
                seasonal_avg[i % period] += detrended[i];
                counts[i % period]++;
            }

            for (int i = 0; i < period; i++) {
                if (counts[i] > 0) {
                    seasonal_avg[i] /= counts[i];
                }
            }

            // Center seasonal
            double smean = std::accumulate(seasonal_avg.begin(), seasonal_avg.end(), 0.0) / period;
            for (int i = 0; i < period; i++) {
                seasonal_avg[i] -= smean;
            }

            // Apply seasonal smoothing (simple moving average within each subseries)
            for (int i = 0; i < n; i++) {
                result.seasonal[i] = seasonal_avg[i % period];
            }

            // Deseasonalize
            std::vector<double> deseasoned(n);
            for (int i = 0; i < n; i++) {
                deseasoned[i] = data[i] - result.seasonal[i];
            }

            // Update trend with smoother
            result.trend = CenteredMovingAverage(deseasoned, trend_window);
        }

        // Final residual
        for (int i = 0; i < n; i++) {
            result.residual[i] = data[i] - result.trend[i] - result.seasonal[i];
        }

        // Compute strength metrics
        std::vector<double> detrended(n), deseasoned(n);
        for (int i = 0; i < n; i++) {
            detrended[i] = data[i] - result.trend[i];
            deseasoned[i] = data[i] - result.seasonal[i];
        }

        double var_residual = Variance(result.residual);
        double var_detrended = Variance(detrended);
        double var_deseasoned = Variance(deseasoned);

        if (var_deseasoned > 1e-10) {
            result.trend_strength = (std::max)(0.0, 1.0 - var_residual / var_deseasoned);
        }
        if (var_detrended > 1e-10) {
            result.seasonal_strength = (std::max)(0.0, 1.0 - var_residual / var_detrended);
        }

        result.residual_variance = var_residual;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("STL decomposition failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// Autocorrelation
// ============================================================================

AutocorrelationResult TimeSeries::ComputeACF(const std::vector<double>& data, int max_lag) {
    AutocorrelationResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());

    // Default max_lag
    if (max_lag < 0) {
        max_lag = (std::min)(n / 2, 40);
    }
    max_lag = (std::min)(max_lag, n - 1);
    result.max_lag = max_lag;

    try {
        double mean = Mean(data);
        double var = 0.0;
        for (double x : data) {
            var += (x - mean) * (x - mean);
        }

        if (var < 1e-10) {
            result.error_message = "Zero variance in data";
            return result;
        }

        result.acf.resize(max_lag + 1);
        result.lags.resize(max_lag + 1);

        for (int lag = 0; lag <= max_lag; lag++) {
            double sum = 0.0;
            for (int i = 0; i < n - lag; i++) {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            result.acf[lag] = sum / var;
            result.lags[lag] = static_cast<double>(lag);
        }

        // Confidence bounds (95% CI for white noise)
        double ci = 1.96 / std::sqrt(static_cast<double>(n));
        result.confidence_upper.resize(max_lag + 1, ci);
        result.confidence_lower.resize(max_lag + 1, -ci);

        // Find significant lags
        for (int lag = 1; lag <= max_lag; lag++) {
            if (std::abs(result.acf[lag]) > ci) {
                result.significant_acf_lags.push_back(lag);
            }
        }

        // Suggest MA order based on ACF cutoff
        for (int q = max_lag; q >= 0; q--) {
            if (std::abs(result.acf[q]) > ci) {
                result.suggested_ma_order = q;
                break;
            }
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("ACF computation failed: ") + e.what();
    }

    return result;
}

AutocorrelationResult TimeSeries::ComputePACF(const std::vector<double>& data, int max_lag) {
    AutocorrelationResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());

    if (max_lag < 0) {
        max_lag = (std::min)(n / 2, 40);
    }
    max_lag = (std::min)(max_lag, n - 1);
    result.max_lag = max_lag;

    try {
        // First compute ACF
        auto acf_result = ComputeACF(data, max_lag);
        if (!acf_result.success) {
            result.error_message = acf_result.error_message;
            return result;
        }

        result.acf = acf_result.acf;
        result.lags = acf_result.lags;

        // Durbin-Levinson algorithm for PACF
        result.pacf.resize(max_lag + 1);
        result.pacf[0] = 1.0;

        if (max_lag > 0) {
            result.pacf[1] = result.acf[1];
        }

        std::vector<double> phi_prev(max_lag + 1), phi_curr(max_lag + 1);
        phi_prev[1] = result.acf[1];

        for (int k = 2; k <= max_lag; k++) {
            double num = result.acf[k];
            double den = 1.0;

            for (int j = 1; j < k; j++) {
                num -= phi_prev[j] * result.acf[k - j];
                den -= phi_prev[j] * result.acf[j];
            }

            if (std::abs(den) < 1e-10) {
                phi_curr[k] = 0.0;
            } else {
                phi_curr[k] = num / den;
            }

            result.pacf[k] = phi_curr[k];

            // Update phi
            for (int j = 1; j < k; j++) {
                phi_curr[j] = phi_prev[j] - phi_curr[k] * phi_prev[k - j];
            }
            phi_prev = phi_curr;
        }

        // Confidence bounds
        double ci = 1.96 / std::sqrt(static_cast<double>(n));
        result.confidence_upper.resize(max_lag + 1, ci);
        result.confidence_lower.resize(max_lag + 1, -ci);

        // Find significant lags
        for (int lag = 1; lag <= max_lag; lag++) {
            if (std::abs(result.pacf[lag]) > ci) {
                result.significant_pacf_lags.push_back(lag);
            }
        }

        // Suggest AR order based on PACF cutoff
        for (int p = max_lag; p >= 0; p--) {
            if (std::abs(result.pacf[p]) > ci) {
                result.suggested_ar_order = p;
                break;
            }
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("PACF computation failed: ") + e.what();
    }

    return result;
}

AutocorrelationResult TimeSeries::ComputeACFPACF(const std::vector<double>& data, int max_lag) {
    auto result = ComputePACF(data, max_lag);

    // Ljung-Box test
    if (result.success && !result.acf.empty()) {
        result.ljung_box_pvalue = LjungBoxTest(data, (std::min)(10, result.max_lag));
    }

    return result;
}

double TimeSeries::LjungBoxTest(const std::vector<double>& data, int lags) {
    int n = static_cast<int>(data.size());
    if (n < lags + 1) return 1.0;

    auto acf = ComputeACF(data, lags);
    if (!acf.success) return 1.0;

    // Q = n(n+2) * sum(r_k^2 / (n-k))
    double q = 0.0;
    for (int k = 1; k <= lags; k++) {
        q += (acf.acf[k] * acf.acf[k]) / (n - k);
    }
    q *= n * (n + 2);

    // p-value from chi-squared distribution (approximate using normal for simplicity)
    // For proper implementation, would use chi-squared CDF
    // Using simple approximation: chi-squared(k) ~ N(k, 2k) for large k
    double mean = static_cast<double>(lags);
    double std = std::sqrt(2.0 * lags);
    double z = (q - mean) / std;

    // Approximate p-value (one-tailed, upper)
    double pvalue = 0.5 * std::erfc(z / std::sqrt(2.0));
    return pvalue;
}

// ============================================================================
// Stationarity Tests
// ============================================================================

StationarityResult TimeSeries::TestStationarity(const std::vector<double>& data, int max_lags) {
    StationarityResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    try {
        // Run ADF test
        auto adf = ADFTest(data, max_lags);
        result.adf_statistic = adf.adf_statistic;
        result.adf_pvalue = adf.adf_pvalue;
        result.adf_critical = adf.adf_critical;
        result.adf_stationary = adf.adf_stationary;

        // Run KPSS test
        auto kpss = KPSSTest(data, "c");
        result.kpss_statistic = kpss.kpss_statistic;
        result.kpss_pvalue = kpss.kpss_pvalue;
        result.kpss_critical = kpss.kpss_critical;
        result.kpss_stationary = kpss.kpss_stationary;

        // Combined result
        // ADF: reject H0 (unit root) => stationary
        // KPSS: fail to reject H0 (stationary) => stationary
        result.is_stationary = result.adf_stationary && result.kpss_stationary;

        // Suggest differencing
        if (!result.is_stationary) {
            // Try differencing and test again
            auto diff1 = Difference(data, 1);
            auto adf1 = ADFTest(diff1, max_lags);
            if (adf1.adf_stationary) {
                result.suggested_differencing = 1;
            } else {
                auto diff2 = Difference(data, 2);
                auto adf2 = ADFTest(diff2, max_lags);
                result.suggested_differencing = adf2.adf_stationary ? 2 : 1;
            }
        }

        // Rolling statistics
        int window = (std::max)(10, static_cast<int>(data.size()) / 10);
        result.rolling_mean = RollingMean(data, window);
        result.rolling_std = RollingStd(data, window);
        result.rolling_window = window;

        // Analysis text
        std::string analysis;
        if (result.is_stationary) {
            analysis = "Series appears stationary (ADF rejects unit root, KPSS accepts stationarity)";
        } else if (result.adf_stationary && !result.kpss_stationary) {
            analysis = "Series may be difference-stationary (ADF rejects, KPSS rejects)";
        } else if (!result.adf_stationary && result.kpss_stationary) {
            analysis = "Conflicting results - series may be trend-stationary";
        } else {
            analysis = "Series appears non-stationary. Suggested differencing: d=" +
                       std::to_string(result.suggested_differencing);
        }
        result.analysis = analysis;

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Stationarity test failed: ") + e.what();
    }

    return result;
}

StationarityResult TimeSeries::ADFTest(const std::vector<double>& data, int max_lags) {
    StationarityResult result;

    int n = static_cast<int>(data.size());
    if (n < 10) {
        result.error_message = "Data too short for ADF test";
        return result;
    }

    // Auto lag selection
    if (max_lags < 0) {
        max_lags = static_cast<int>(std::pow(n - 1, 1.0 / 3.0));
    }
    max_lags = (std::min)(max_lags, n / 2 - 1);

    try {
        // Simplified ADF: test regression of Δy_t on y_{t-1} and lags of Δy
        // ADF statistic is t-statistic on y_{t-1} coefficient

        // Compute first difference
        std::vector<double> dy(n - 1);
        for (int i = 0; i < n - 1; i++) {
            dy[i] = data[i + 1] - data[i];
        }

        // For simplicity, use basic OLS estimate
        // y = a + b*x + error
        // Here: Δy_t = a + rho*y_{t-1} + error (simplified no lag terms)

        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        int count = n - 1 - max_lags;

        for (int i = max_lags; i < n - 1; i++) {
            double x = data[i];  // y_{t-1}
            double y = dy[i];     // Δy_t
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        double mean_x = sum_x / count;
        double mean_y = sum_y / count;
        double sxx = sum_xx - count * mean_x * mean_x;
        double sxy = sum_xy - count * mean_x * mean_y;

        if (std::abs(sxx) < 1e-10) {
            result.error_message = "Singular matrix in ADF regression";
            return result;
        }

        double rho = sxy / sxx;
        double intercept = mean_y - rho * mean_x;

        // Residual variance
        double sse = 0;
        for (int i = max_lags; i < n - 1; i++) {
            double predicted = intercept + rho * data[i];
            double resid = dy[i] - predicted;
            sse += resid * resid;
        }
        double mse = sse / (count - 2);
        double se_rho = std::sqrt(mse / sxx);

        result.adf_statistic = rho / se_rho;

        // Critical values (approximate for n >= 100)
        result.adf_critical["1%"] = -3.43;
        result.adf_critical["5%"] = -2.86;
        result.adf_critical["10%"] = -2.57;

        // p-value approximation (MacKinnon 1994 approximation)
        // Very rough approximation
        double stat = result.adf_statistic;
        if (stat < -3.43) {
            result.adf_pvalue = 0.01;
        } else if (stat < -2.86) {
            result.adf_pvalue = 0.05;
        } else if (stat < -2.57) {
            result.adf_pvalue = 0.10;
        } else {
            result.adf_pvalue = 0.5;  // Rough estimate
        }

        result.adf_stationary = (result.adf_statistic < result.adf_critical["5%"]);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("ADF test failed: ") + e.what();
    }

    return result;
}

StationarityResult TimeSeries::KPSSTest(const std::vector<double>& data, const std::string& regression) {
    StationarityResult result;

    int n = static_cast<int>(data.size());
    if (n < 10) {
        result.error_message = "Data too short for KPSS test";
        return result;
    }

    try {
        // KPSS test statistic
        // H0: series is stationary around a deterministic trend

        double mean = Mean(data);

        // Residuals from regression on constant (or constant + trend)
        std::vector<double> residuals(n);
        if (regression == "ct") {
            // Linear trend: y = a + b*t + e
            double sum_t = 0, sum_y = 0, sum_tt = 0, sum_ty = 0;
            for (int i = 0; i < n; i++) {
                double t = static_cast<double>(i);
                sum_t += t;
                sum_y += data[i];
                sum_tt += t * t;
                sum_ty += t * data[i];
            }
            double mean_t = sum_t / n;
            double mean_y = sum_y / n;
            double stt = sum_tt - n * mean_t * mean_t;
            double sty = sum_ty - n * mean_t * mean_y;
            double b = sty / stt;
            double a = mean_y - b * mean_t;

            for (int i = 0; i < n; i++) {
                residuals[i] = data[i] - a - b * i;
            }
        } else {
            // Constant only
            for (int i = 0; i < n; i++) {
                residuals[i] = data[i] - mean;
            }
        }

        // Cumulative sum of residuals
        std::vector<double> S(n);
        S[0] = residuals[0];
        for (int i = 1; i < n; i++) {
            S[i] = S[i - 1] + residuals[i];
        }

        // Variance estimator (Newey-West with automatic bandwidth)
        int bandwidth = static_cast<int>(4.0 * std::pow(n / 100.0, 0.25));

        double s2 = 0;
        for (double r : residuals) {
            s2 += r * r;
        }
        s2 /= n;

        // Add autocovariance terms
        for (int lag = 1; lag <= bandwidth; lag++) {
            double gamma = 0;
            for (int i = lag; i < n; i++) {
                gamma += residuals[i] * residuals[i - lag];
            }
            gamma /= n;
            double weight = 1.0 - static_cast<double>(lag) / (bandwidth + 1);
            s2 += 2 * weight * gamma;
        }

        // KPSS statistic
        double sum_S2 = 0;
        for (double s : S) {
            sum_S2 += s * s;
        }
        result.kpss_statistic = sum_S2 / (n * n * s2);

        // Critical values
        if (regression == "ct") {
            result.kpss_critical["1%"] = 0.216;
            result.kpss_critical["5%"] = 0.146;
            result.kpss_critical["10%"] = 0.119;
        } else {
            result.kpss_critical["1%"] = 0.739;
            result.kpss_critical["5%"] = 0.463;
            result.kpss_critical["10%"] = 0.347;
        }

        // p-value approximation
        double crit5 = result.kpss_critical["5%"];
        if (result.kpss_statistic > result.kpss_critical["1%"]) {
            result.kpss_pvalue = 0.01;
        } else if (result.kpss_statistic > crit5) {
            result.kpss_pvalue = 0.05;
        } else if (result.kpss_statistic > result.kpss_critical["10%"]) {
            result.kpss_pvalue = 0.10;
        } else {
            result.kpss_pvalue = 0.5;
        }

        // For KPSS, we REJECT H0 (stationarity) if statistic > critical value
        result.kpss_stationary = (result.kpss_statistic < crit5);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("KPSS test failed: ") + e.what();
    }

    return result;
}

std::vector<double> TimeSeries::Difference(const std::vector<double>& data, int order) {
    if (order <= 0 || data.size() <= static_cast<size_t>(order)) {
        return data;
    }

    std::vector<double> result = data;
    for (int d = 0; d < order; d++) {
        std::vector<double> diff(result.size() - 1);
        for (size_t i = 0; i < result.size() - 1; i++) {
            diff[i] = result[i + 1] - result[i];
        }
        result = diff;
    }
    return result;
}

std::vector<double> TimeSeries::SeasonalDifference(const std::vector<double>& data, int period, int order) {
    if (order <= 0 || period <= 0 || data.size() <= static_cast<size_t>(period * order)) {
        return data;
    }

    std::vector<double> result = data;
    for (int d = 0; d < order; d++) {
        std::vector<double> diff(result.size() - period);
        for (size_t i = 0; i < result.size() - period; i++) {
            diff[i] = result[i + period] - result[i];
        }
        result = diff;
    }
    return result;
}

// ============================================================================
// Seasonality Detection
// ============================================================================

SeasonalityResult TimeSeries::DetectSeasonality(const std::vector<double>& data, int min_period, int max_period) {
    SeasonalityResult result;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    int n = static_cast<int>(data.size());
    if (max_period < 0) {
        max_period = n / 2;
    }
    max_period = (std::min)(max_period, n / 2);

    if (min_period < 2) min_period = 2;
    if (max_period <= min_period) {
        result.error_message = "Invalid period range";
        return result;
    }

    try {
        // Compute periodogram
        auto [periodogram, frequencies] = Periodogram(data);
        result.periodogram = periodogram;
        result.frequencies = frequencies;

        // Convert to periods
        result.periods.resize(frequencies.size());
        for (size_t i = 0; i < frequencies.size(); i++) {
            if (frequencies[i] > 1e-10) {
                result.periods[i] = 1.0 / frequencies[i];
            } else {
                result.periods[i] = 0;
            }
        }

        // Find peaks in periodogram within valid period range
        double max_power = 0;
        double total_power = 0;

        for (size_t i = 1; i < periodogram.size(); i++) {
            total_power += periodogram[i];
        }

        for (size_t i = 1; i < periodogram.size() - 1; i++) {
            double period = result.periods[i];
            if (period >= min_period && period <= max_period) {
                // Local maximum check
                if (periodogram[i] > periodogram[i - 1] && periodogram[i] > periodogram[i + 1]) {
                    int period_int = static_cast<int>(std::round(period));
                    double strength = periodogram[i] / total_power;

                    result.candidate_periods.push_back(period_int);
                    result.candidate_strengths.push_back(strength);

                    if (periodogram[i] > max_power) {
                        max_power = periodogram[i];
                        result.detected_period = period_int;
                        result.strength = strength;
                    }
                }
            }
        }

        // Also check ACF for confirmation
        auto acf = ComputeACF(data, max_period);
        if (acf.success) {
            for (int lag = min_period; lag <= max_period && lag < static_cast<int>(acf.acf.size()) - 1; lag++) {
                if (acf.acf[lag] > acf.acf[lag - 1] && acf.acf[lag] > acf.acf[lag + 1] &&
                    acf.acf[lag] > 0.1) {
                    result.acf_peaks.push_back(lag);
                }
            }
        }

        // Determine if seasonality is significant
        result.has_seasonality = (result.strength > 0.05 && result.detected_period >= min_period);

        // Analysis text
        if (result.has_seasonality) {
            result.analysis = "Seasonality detected with period " + std::to_string(result.detected_period) +
                              " (strength: " + std::to_string(result.strength) + ")";
        } else {
            result.analysis = "No significant seasonality detected";
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Seasonality detection failed: ") + e.what();
    }

    return result;
}

std::pair<std::vector<double>, std::vector<double>> TimeSeries::Periodogram(const std::vector<double>& data) {
    int n = static_cast<int>(data.size());
    if (n == 0) return {{}, {}};

    double mean = Mean(data);

    // Compute raw periodogram using DFT
    std::vector<double> periodogram;
    std::vector<double> frequencies;

    int n_freq = n / 2 + 1;
    periodogram.resize(n_freq);
    frequencies.resize(n_freq);

    for (int k = 0; k < n_freq; k++) {
        double cos_sum = 0, sin_sum = 0;
        for (int t = 0; t < n; t++) {
            double angle = TWO_PI * k * t / n;
            cos_sum += (data[t] - mean) * std::cos(angle);
            sin_sum += (data[t] - mean) * std::sin(angle);
        }
        periodogram[k] = (cos_sum * cos_sum + sin_sum * sin_sum) / n;
        frequencies[k] = static_cast<double>(k) / n;
    }

    return {periodogram, frequencies};
}

// ============================================================================
// Forecasting
// ============================================================================

double TimeSeries::OptimizeESAlpha(const std::vector<double>& data) {
    // Grid search for optimal alpha
    double best_alpha = 0.3;
    double best_mse = std::numeric_limits<double>::max();

    for (double alpha = 0.1; alpha <= 0.9; alpha += 0.1) {
        double level = data[0];
        double sse = 0;

        for (size_t i = 1; i < data.size(); i++) {
            double error = data[i] - level;
            sse += error * error;
            level = alpha * data[i] + (1 - alpha) * level;
        }

        double mse = sse / (data.size() - 1);
        if (mse < best_mse) {
            best_mse = mse;
            best_alpha = alpha;
        }
    }

    return best_alpha;
}

ForecastResult TimeSeries::SimpleES(const std::vector<double>& data, int horizon, double alpha) {
    ForecastResult result;
    result.method = "Simple Exponential Smoothing";
    result.horizon = horizon;

    if (data.empty()) {
        result.error_message = "Empty data";
        return result;
    }

    if (horizon <= 0) {
        result.error_message = "Horizon must be positive";
        return result;
    }

    try {
        // Optimize alpha if needed
        if (alpha < 0 || alpha > 1) {
            alpha = OptimizeESAlpha(data);
        }

        result.parameters["alpha"] = alpha;

        // Fit model
        int n = static_cast<int>(data.size());
        double level = data[0];

        result.fitted_values.resize(n);
        result.fitted_values[0] = level;

        double sse = 0;
        double sae = 0;
        double sape = 0;
        int ape_count = 0;

        for (int i = 1; i < n; i++) {
            double forecast = level;
            result.fitted_values[i] = forecast;

            double error = data[i] - forecast;
            sse += error * error;
            sae += std::abs(error);
            if (std::abs(data[i]) > 1e-10) {
                sape += std::abs(error / data[i]);
                ape_count++;
            }

            level = alpha * data[i] + (1 - alpha) * level;
        }

        // Metrics
        result.mse = sse / (n - 1);
        result.rmse = std::sqrt(result.mse);
        result.mae = sae / (n - 1);
        result.mape = (ape_count > 0) ? (sape / ape_count) * 100 : 0;

        // Generate forecasts
        result.forecast.resize(horizon, level);

        // Prediction intervals (approximate)
        double sigma = std::sqrt(result.mse);
        result.lower_bound.resize(horizon);
        result.upper_bound.resize(horizon);

        for (int h = 0; h < horizon; h++) {
            // Variance increases with horizon for SES
            double var_h = result.mse * (1 + h * alpha * alpha);
            double se = std::sqrt(var_h);
            result.lower_bound[h] = level - 1.96 * se;
            result.upper_bound[h] = level + 1.96 * se;
        }

        result.model_summary = "Simple ES: alpha=" + std::to_string(alpha) +
                               ", RMSE=" + std::to_string(result.rmse);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("SimpleES failed: ") + e.what();
    }

    return result;
}

std::pair<double, double> TimeSeries::OptimizeHoltParams(const std::vector<double>& data) {
    double best_alpha = 0.3, best_beta = 0.1;
    double best_mse = std::numeric_limits<double>::max();

    for (double alpha = 0.1; alpha <= 0.9; alpha += 0.2) {
        for (double beta = 0.05; beta <= 0.5; beta += 0.1) {
            double level = data[0];
            double trend = data[1] - data[0];
            double sse = 0;

            for (size_t i = 1; i < data.size(); i++) {
                double forecast = level + trend;
                double error = data[i] - forecast;
                sse += error * error;

                double new_level = alpha * data[i] + (1 - alpha) * (level + trend);
                trend = beta * (new_level - level) + (1 - beta) * trend;
                level = new_level;
            }

            double mse = sse / (data.size() - 1);
            if (mse < best_mse) {
                best_mse = mse;
                best_alpha = alpha;
                best_beta = beta;
            }
        }
    }

    return {best_alpha, best_beta};
}

ForecastResult TimeSeries::HoltLinear(const std::vector<double>& data, int horizon,
                                       double alpha, double beta, bool damped) {
    ForecastResult result;
    result.method = damped ? "Damped Holt's Method" : "Holt's Linear Method";
    result.horizon = horizon;

    if (data.size() < 2) {
        result.error_message = "Need at least 2 data points";
        return result;
    }

    try {
        // Optimize if needed
        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) {
            auto [opt_alpha, opt_beta] = OptimizeHoltParams(data);
            alpha = opt_alpha;
            beta = opt_beta;
        }

        double phi = damped ? 0.9 : 1.0;

        result.parameters["alpha"] = alpha;
        result.parameters["beta"] = beta;
        if (damped) result.parameters["phi"] = phi;

        int n = static_cast<int>(data.size());
        double level = data[0];
        double trend = data[1] - data[0];

        result.fitted_values.resize(n);
        result.fitted_values[0] = level;

        double sse = 0, sae = 0, sape = 0;
        int ape_count = 0;

        for (int i = 1; i < n; i++) {
            double forecast = level + phi * trend;
            result.fitted_values[i] = forecast;

            double error = data[i] - forecast;
            sse += error * error;
            sae += std::abs(error);
            if (std::abs(data[i]) > 1e-10) {
                sape += std::abs(error / data[i]);
                ape_count++;
            }

            double new_level = alpha * data[i] + (1 - alpha) * (level + phi * trend);
            trend = beta * (new_level - level) + (1 - beta) * phi * trend;
            level = new_level;
        }

        result.mse = sse / (n - 1);
        result.rmse = std::sqrt(result.mse);
        result.mae = sae / (n - 1);
        result.mape = (ape_count > 0) ? (sape / ape_count) * 100 : 0;

        // Forecasts
        result.forecast.resize(horizon);
        result.lower_bound.resize(horizon);
        result.upper_bound.resize(horizon);

        double sigma = std::sqrt(result.mse);
        double phi_sum = 0;

        for (int h = 0; h < horizon; h++) {
            phi_sum += std::pow(phi, h + 1);
            result.forecast[h] = level + phi_sum * trend;

            double var_h = result.mse * (1 + (h + 1) * alpha * alpha);
            double se = std::sqrt(var_h);
            result.lower_bound[h] = result.forecast[h] - 1.96 * se;
            result.upper_bound[h] = result.forecast[h] + 1.96 * se;
        }

        result.model_summary = result.method + ": alpha=" + std::to_string(alpha) +
                               ", beta=" + std::to_string(beta);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("HoltLinear failed: ") + e.what();
    }

    return result;
}

std::tuple<double, double, double> TimeSeries::OptimizeHWParams(
    const std::vector<double>& data, int period, const std::string& seasonal_type) {

    double best_alpha = 0.3, best_beta = 0.1, best_gamma = 0.1;
    double best_mse = std::numeric_limits<double>::max();

    int n = static_cast<int>(data.size());

    for (double alpha = 0.2; alpha <= 0.8; alpha += 0.3) {
        for (double beta = 0.05; beta <= 0.3; beta += 0.1) {
            for (double gamma = 0.1; gamma <= 0.5; gamma += 0.2) {
                // Initialize
                double level = 0;
                for (int i = 0; i < period; i++) level += data[i];
                level /= period;

                double trend = 0;
                for (int i = 0; i < period; i++) {
                    trend += (data[i + period] - data[i]) / period;
                }
                trend /= period;

                std::vector<double> seasonal(period);
                for (int i = 0; i < period; i++) {
                    if (seasonal_type == "multiplicative") {
                        seasonal[i] = data[i] / level;
                    } else {
                        seasonal[i] = data[i] - level;
                    }
                }

                double sse = 0;
                for (int t = period; t < n; t++) {
                    int s = t % period;
                    double forecast;
                    if (seasonal_type == "multiplicative") {
                        forecast = (level + trend) * seasonal[s];
                    } else {
                        forecast = level + trend + seasonal[s];
                    }

                    double error = data[t] - forecast;
                    sse += error * error;

                    double new_level, new_seasonal;
                    if (seasonal_type == "multiplicative") {
                        new_level = alpha * data[t] / seasonal[s] + (1 - alpha) * (level + trend);
                        new_seasonal = gamma * data[t] / new_level + (1 - gamma) * seasonal[s];
                    } else {
                        new_level = alpha * (data[t] - seasonal[s]) + (1 - alpha) * (level + trend);
                        new_seasonal = gamma * (data[t] - new_level) + (1 - gamma) * seasonal[s];
                    }

                    trend = beta * (new_level - level) + (1 - beta) * trend;
                    level = new_level;
                    seasonal[s] = new_seasonal;
                }

                double mse = sse / (n - period);
                if (mse < best_mse) {
                    best_mse = mse;
                    best_alpha = alpha;
                    best_beta = beta;
                    best_gamma = gamma;
                }
            }
        }
    }

    return {best_alpha, best_beta, best_gamma};
}

ForecastResult TimeSeries::HoltWinters(const std::vector<double>& data, int horizon,
                                        int period, const std::string& seasonal_type,
                                        double alpha, double beta, double gamma) {
    ForecastResult result;
    result.method = "Holt-Winters (" + seasonal_type + ")";
    result.horizon = horizon;

    int n = static_cast<int>(data.size());

    // Auto-detect period if needed
    if (period < 2) {
        auto seasonality = DetectSeasonality(data);
        if (seasonality.has_seasonality) {
            period = seasonality.detected_period;
        } else {
            period = 12;  // Default
        }
    }

    if (n < 2 * period) {
        result.error_message = "Need at least 2 complete periods";
        return result;
    }

    try {
        // Optimize parameters if needed
        if (alpha < 0 || beta < 0 || gamma < 0) {
            auto [opt_a, opt_b, opt_g] = OptimizeHWParams(data, period, seasonal_type);
            alpha = opt_a;
            beta = opt_b;
            gamma = opt_g;
        }

        result.parameters["alpha"] = alpha;
        result.parameters["beta"] = beta;
        result.parameters["gamma"] = gamma;
        result.parameters["period"] = static_cast<double>(period);

        // Initialize components
        double level = 0;
        for (int i = 0; i < period; i++) level += data[i];
        level /= period;

        double trend = 0;
        for (int i = 0; i < period; i++) {
            trend += (data[i + period] - data[i]) / period;
        }
        trend /= period;

        std::vector<double> seasonal(period);
        for (int i = 0; i < period; i++) {
            if (seasonal_type == "multiplicative") {
                seasonal[i] = data[i] / level;
            } else {
                seasonal[i] = data[i] - level;
            }
        }

        // Fit model
        result.fitted_values.resize(n);
        for (int i = 0; i < period; i++) {
            result.fitted_values[i] = data[i];  // No prediction for initialization
        }

        double sse = 0, sae = 0, sape = 0;
        int ape_count = 0;

        for (int t = period; t < n; t++) {
            int s = t % period;
            double forecast;
            if (seasonal_type == "multiplicative") {
                forecast = (level + trend) * seasonal[s];
            } else {
                forecast = level + trend + seasonal[s];
            }
            result.fitted_values[t] = forecast;

            double error = data[t] - forecast;
            sse += error * error;
            sae += std::abs(error);
            if (std::abs(data[t]) > 1e-10) {
                sape += std::abs(error / data[t]);
                ape_count++;
            }

            // Update
            double new_level, new_seasonal;
            if (seasonal_type == "multiplicative") {
                new_level = alpha * data[t] / seasonal[s] + (1 - alpha) * (level + trend);
                new_seasonal = gamma * data[t] / new_level + (1 - gamma) * seasonal[s];
            } else {
                new_level = alpha * (data[t] - seasonal[s]) + (1 - alpha) * (level + trend);
                new_seasonal = gamma * (data[t] - new_level) + (1 - gamma) * seasonal[s];
            }

            trend = beta * (new_level - level) + (1 - beta) * trend;
            level = new_level;
            seasonal[s] = new_seasonal;
        }

        int fit_count = n - period;
        result.mse = sse / fit_count;
        result.rmse = std::sqrt(result.mse);
        result.mae = sae / fit_count;
        result.mape = (ape_count > 0) ? (sape / ape_count) * 100 : 0;

        // Forecasts
        result.forecast.resize(horizon);
        result.lower_bound.resize(horizon);
        result.upper_bound.resize(horizon);

        double sigma = std::sqrt(result.mse);

        for (int h = 0; h < horizon; h++) {
            int s = (n + h) % period;
            if (seasonal_type == "multiplicative") {
                result.forecast[h] = (level + (h + 1) * trend) * seasonal[s];
            } else {
                result.forecast[h] = level + (h + 1) * trend + seasonal[s];
            }

            // Approximate prediction interval
            double se = sigma * std::sqrt(1.0 + 0.1 * (h + 1));
            result.lower_bound[h] = result.forecast[h] - 1.96 * se;
            result.upper_bound[h] = result.forecast[h] + 1.96 * se;
        }

        result.model_summary = "Holt-Winters: alpha=" + std::to_string(alpha) +
                               ", beta=" + std::to_string(beta) +
                               ", gamma=" + std::to_string(gamma) +
                               ", period=" + std::to_string(period);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("HoltWinters failed: ") + e.what();
    }

    return result;
}

ForecastResult TimeSeries::MovingAverageForecast(const std::vector<double>& data, int window, int horizon) {
    ForecastResult result;
    result.method = "Moving Average (window=" + std::to_string(window) + ")";
    result.horizon = horizon;

    if (data.size() < static_cast<size_t>(window)) {
        result.error_message = "Not enough data for window size";
        return result;
    }

    try {
        int n = static_cast<int>(data.size());

        // Compute rolling mean
        result.fitted_values = RollingMean(data, window);

        // Pad beginning with NaN-like values (use first valid value)
        std::vector<double> fitted(n);
        int padding = window - 1;
        for (int i = 0; i < padding; i++) {
            fitted[i] = result.fitted_values.empty() ? data[i] : result.fitted_values[0];
        }
        for (size_t i = 0; i < result.fitted_values.size(); i++) {
            fitted[padding + i] = result.fitted_values[i];
        }
        result.fitted_values = fitted;

        // Last MA value for forecasting
        double last_ma = 0;
        for (int i = n - window; i < n; i++) {
            last_ma += data[i];
        }
        last_ma /= window;

        // All forecasts are the same (flat)
        result.forecast.resize(horizon, last_ma);

        // Compute MSE
        double sse = 0;
        for (int i = window; i < n; i++) {
            double error = data[i] - result.fitted_values[i];
            sse += error * error;
        }
        result.mse = sse / (n - window);
        result.rmse = std::sqrt(result.mse);

        // Prediction intervals
        double sigma = result.rmse;
        result.lower_bound.resize(horizon);
        result.upper_bound.resize(horizon);
        for (int h = 0; h < horizon; h++) {
            result.lower_bound[h] = last_ma - 1.96 * sigma;
            result.upper_bound[h] = last_ma + 1.96 * sigma;
        }

        result.model_summary = "MA(" + std::to_string(window) + ") RMSE=" + std::to_string(result.rmse);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("MovingAverage failed: ") + e.what();
    }

    return result;
}

ForecastResult TimeSeries::ARIMA(const std::vector<double>& data, int horizon, int p, int d, int q) {
    ForecastResult result;
    result.horizon = horizon;

    if (data.size() < 10) {
        result.error_message = "Not enough data for ARIMA";
        return result;
    }

    try {
        // Auto order selection if needed
        if (p < 0 || d < 0 || q < 0) {
            // Simple heuristic
            auto stat = TestStationarity(data);
            d = stat.suggested_differencing;

            auto diff_data = Difference(data, d);
            auto acf_pacf = ComputeACFPACF(diff_data);

            p = (std::min)(acf_pacf.suggested_ar_order, 3);
            q = (std::min)(acf_pacf.suggested_ma_order, 3);
        }

        result.method = "ARIMA(" + std::to_string(p) + "," + std::to_string(d) + "," + std::to_string(q) + ")";
        result.parameters["p"] = static_cast<double>(p);
        result.parameters["d"] = static_cast<double>(d);
        result.parameters["q"] = static_cast<double>(q);

        // Apply differencing
        auto diff_data = Difference(data, d);
        int n = static_cast<int>(diff_data.size());

        // For simplicity, use AR model only (full ARIMA would require more sophisticated estimation)
        // Fit AR(p) using Yule-Walker
        std::vector<double> ar_coeffs(p, 0.0);

        if (p > 0) {
            auto acf = ComputeACF(diff_data, p);
            if (acf.success && p <= static_cast<int>(acf.acf.size())) {
                // Solve Yule-Walker equations (simplified)
                // For AR(1): phi_1 = rho_1
                // For AR(2): solve 2x2 system, etc.
                if (p == 1) {
                    ar_coeffs[0] = acf.acf[1];
                } else {
                    // Use simple approximation for higher orders
                    auto pacf = ComputePACF(diff_data, p);
                    for (int i = 0; i < p && i < static_cast<int>(pacf.pacf.size()); i++) {
                        ar_coeffs[i] = pacf.pacf[i + 1];
                    }
                }
            }
        }

        // Fit and compute residuals
        result.fitted_values.resize(data.size());
        double mean_diff = Mean(diff_data);

        std::vector<double> residuals;
        for (int t = p; t < n; t++) {
            double pred = mean_diff;
            for (int i = 0; i < p; i++) {
                pred += ar_coeffs[i] * (diff_data[t - 1 - i] - mean_diff);
            }
            double resid = diff_data[t] - pred;
            residuals.push_back(resid);
        }

        result.mse = Variance(residuals);
        result.rmse = std::sqrt(result.mse);

        // Generate forecasts on differenced scale
        std::vector<double> diff_forecast(horizon);
        std::vector<double> extended = diff_data;

        for (int h = 0; h < horizon; h++) {
            double pred = mean_diff;
            int t = static_cast<int>(extended.size());
            for (int i = 0; i < p && (t - 1 - i) >= 0; i++) {
                pred += ar_coeffs[i] * (extended[t - 1 - i] - mean_diff);
            }
            diff_forecast[h] = pred;
            extended.push_back(pred);
        }

        // Integrate back (reverse differencing)
        result.forecast.resize(horizon);
        double last_value = data.back();

        if (d == 0) {
            result.forecast = diff_forecast;
        } else if (d == 1) {
            for (int h = 0; h < horizon; h++) {
                last_value += diff_forecast[h];
                result.forecast[h] = last_value;
            }
        } else {
            // For d > 1, need to track more values
            std::vector<double> values(d, data.back());
            for (int h = 0; h < horizon; h++) {
                double new_val = diff_forecast[h];
                for (int i = d - 1; i >= 0; i--) {
                    new_val += values[i];
                }
                result.forecast[h] = new_val;
                // Shift values
                for (int i = d - 1; i > 0; i--) {
                    values[i] = values[i - 1] + diff_forecast[h];
                }
                if (d > 0) values[0] = result.forecast[h];
            }
        }

        // Prediction intervals
        double sigma = result.rmse;
        result.lower_bound.resize(horizon);
        result.upper_bound.resize(horizon);
        for (int h = 0; h < horizon; h++) {
            double se = sigma * std::sqrt(1.0 + 0.1 * h);
            result.lower_bound[h] = result.forecast[h] - 1.96 * se;
            result.upper_bound[h] = result.forecast[h] + 1.96 * se;
        }

        result.model_summary = result.method + " RMSE=" + std::to_string(result.rmse);
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("ARIMA failed: ") + e.what();
    }

    return result;
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

std::vector<double> TimeSeries::GenerateWhiteNoise(int n, double mean, double std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, std);

    for (int i = 0; i < n; i++) {
        result[i] = dist(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateRandomWalk(int n, double start, double std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, std);

    result[0] = start;
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] + dist(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateTrendSeasonal(int n, double trend_slope,
                                                       double seasonal_amplitude,
                                                       int period, double noise_std) {
    std::vector<double> result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < n; i++) {
        double trend = trend_slope * i;
        double seasonal = seasonal_amplitude * std::sin(TWO_PI * i / period);
        result[i] = trend + seasonal + noise(gen);
    }
    return result;
}

std::vector<double> TimeSeries::GenerateAR(int n, const std::vector<double>& coeffs, double noise_std) {
    int p = static_cast<int>(coeffs.size());
    std::vector<double> result(n, 0.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    // Initialize with noise
    for (int i = 0; i < p; i++) {
        result[i] = noise(gen);
    }

    for (int i = p; i < n; i++) {
        double val = noise(gen);
        for (int j = 0; j < p; j++) {
            val += coeffs[j] * result[i - 1 - j];
        }
        result[i] = val;
    }
    return result;
}

std::vector<double> TimeSeries::GenerateMA(int n, const std::vector<double>& coeffs, double noise_std) {
    int q = static_cast<int>(coeffs.size());
    std::vector<double> result(n);
    std::vector<double> errors(n + q);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < n + q; i++) {
        errors[i] = noise(gen);
    }

    for (int i = 0; i < n; i++) {
        double val = errors[i + q];
        for (int j = 0; j < q; j++) {
            val += coeffs[j] * errors[i + q - 1 - j];
        }
        result[i] = val;
    }
    return result;
}

std::vector<double> TimeSeries::GenerateARIMA(int n, const std::vector<double>& ar_coeffs,
                                               const std::vector<double>& ma_coeffs,
                                               int d, double noise_std) {
    // Generate ARMA first
    int p = static_cast<int>(ar_coeffs.size());
    int q = static_cast<int>(ma_coeffs.size());

    std::vector<double> errors(n + q);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_std);

    for (size_t i = 0; i < errors.size(); i++) {
        errors[i] = noise(gen);
    }

    std::vector<double> arma(n, 0.0);
    int start = (p > 1) ? p : 1;
    for (int i = 0; i < start; i++) {
        arma[i] = errors[i + q];
    }

    for (int i = start; i < n; i++) {
        double val = errors[i + q];
        for (int j = 0; j < p; j++) {
            val += ar_coeffs[j] * arma[i - 1 - j];
        }
        for (int j = 0; j < q; j++) {
            val += ma_coeffs[j] * errors[i + q - 1 - j];
        }
        arma[i] = val;
    }

    // Integrate d times
    std::vector<double> result = arma;
    for (int i = 0; i < d; i++) {
        for (int j = 1; j < static_cast<int>(result.size()); j++) {
            result[j] += result[j - 1];
        }
    }

    return result;
}

} // namespace cyxwiz
