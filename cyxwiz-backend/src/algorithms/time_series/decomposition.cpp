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

} // namespace cyxwiz
