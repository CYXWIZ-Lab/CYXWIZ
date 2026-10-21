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
        // Simplified ADF: test regression of Î”y_t on y_{t-1} and lags of Î”y
        // ADF statistic is t-statistic on y_{t-1} coefficient

        // Compute first difference
        std::vector<double> dy(n - 1);
        for (int i = 0; i < n - 1; i++) {
            dy[i] = data[i + 1] - data[i];
        }

        // For simplicity, use basic OLS estimate
        // y = a + b*x + error
        // Here: Î”y_t = a + rho*y_{t-1} + error (simplified no lag terms)

        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        int count = n - 1 - max_lags;

        for (int i = max_lags; i < n - 1; i++) {
            double x = data[i];  // y_{t-1}
            double y = dy[i];     // Î”y_t
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

} // namespace cyxwiz
