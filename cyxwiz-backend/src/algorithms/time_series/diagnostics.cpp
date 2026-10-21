// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cyxwiz {
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

} // namespace cyxwiz
