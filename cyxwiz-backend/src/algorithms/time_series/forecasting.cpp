// Prevent Windows min/max macros from interfering with std::min/std::max
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cyxwiz/time_series.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyxwiz {

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

    if (data.size() < 2) {
        result.error_message = "Need at least 2 data points";
        return result;
    }

    if (horizon < 0) {
        result.error_message = "Horizon must be non-negative";
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

} // namespace cyxwiz
