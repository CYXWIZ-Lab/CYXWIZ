#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <random>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

struct CYXWIZ_API DecompositionResult {
    std::vector<double> trend;                    // Trend component
    std::vector<double> seasonal;                 // Seasonal component
    std::vector<double> residual;                 // Residual component
    std::vector<double> original;                 // Original series (for reference)
    int period = 0;                               // Seasonal period used
    std::string method;                           // "additive" or "multiplicative"
    double trend_strength = 0.0;                  // Strength of trend (0-1)
    double seasonal_strength = 0.0;               // Strength of seasonality (0-1)
    double residual_variance = 0.0;               // Variance of residuals
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API AutocorrelationResult {
    std::vector<double> acf;                      // Autocorrelation function
    std::vector<double> pacf;                     // Partial autocorrelation function
    std::vector<double> lags;                     // Lag values (0, 1, 2, ...)
    std::vector<double> confidence_upper;         // 95% CI upper bound
    std::vector<double> confidence_lower;         // 95% CI lower bound
    int max_lag = 0;                              // Maximum lag computed
    double ljung_box_statistic = 0.0;             // Ljung-Box Q statistic
    double ljung_box_pvalue = 0.0;                // p-value for white noise test
    std::vector<int> significant_acf_lags;        // Lags where ACF is significant
    std::vector<int> significant_pacf_lags;       // Lags where PACF is significant
    int suggested_ar_order = 0;                   // Suggested AR(p) order
    int suggested_ma_order = 0;                   // Suggested MA(q) order
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API StationarityResult {
    // Augmented Dickey-Fuller Test
    double adf_statistic = 0.0;                   // ADF test statistic
    double adf_pvalue = 0.0;                      // p-value
    std::map<std::string, double> adf_critical;   // Critical values (1%, 5%, 10%)
    bool adf_stationary = false;                  // Result: is stationary?

    // KPSS Test
    double kpss_statistic = 0.0;                  // KPSS test statistic
    double kpss_pvalue = 0.0;                     // p-value
    std::map<std::string, double> kpss_critical;  // Critical values
    bool kpss_stationary = false;                 // Result: is stationary?

    // Combined result
    bool is_stationary = false;                   // Overall conclusion
    int suggested_differencing = 0;               // Suggested d for ARIMA
    std::string analysis;                         // Text interpretation

    // Rolling statistics (for visual check)
    std::vector<double> rolling_mean;
    std::vector<double> rolling_std;
    int rolling_window = 0;

    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API SeasonalityResult {
    bool has_seasonality = false;                 // Is seasonality detected?
    int detected_period = 0;                      // Primary detected period
    double strength = 0.0;                        // Seasonality strength (0-1)

    // Spectral analysis
    std::vector<double> periodogram;              // Spectral density
    std::vector<double> frequencies;              // Frequency axis
    std::vector<double> periods;                  // Period axis (1/freq)

    // Multiple seasonality support
    std::vector<int> candidate_periods;           // All detected periods
    std::vector<double> candidate_strengths;      // Corresponding strengths

    // ACF-based detection
    std::vector<int> acf_peaks;                   // Peaks in ACF

    std::string analysis;                         // Text interpretation
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API ForecastResult {
    std::vector<double> forecast;                 // Point forecasts
    std::vector<double> lower_bound;              // Lower prediction interval (95%)
    std::vector<double> upper_bound;              // Upper prediction interval (95%)
    std::vector<double> fitted_values;            // In-sample fitted values

    // Accuracy metrics
    double mse = 0.0;                             // Mean Squared Error
    double rmse = 0.0;                            // Root Mean Squared Error
    double mae = 0.0;                             // Mean Absolute Error
    double mape = 0.0;                            // Mean Absolute Percentage Error
    double aic = 0.0;                             // Akaike Information Criterion
    double bic = 0.0;                             // Bayesian Information Criterion

    // Model parameters
    std::string method;                           // Model name
    std::map<std::string, double> parameters;     // Fitted parameters
    std::string model_summary;                    // Text summary

    int horizon = 0;                              // Forecast horizon
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Time Series Analysis Class
// ============================================================================

class CYXWIZ_API TimeSeries {
public:
    // ==================== Decomposition ====================

    /**
     * Classical time series decomposition (moving average based)
     * @param data Input time series
     * @param period Seasonal period (e.g., 12 for monthly data with yearly seasonality)
     * @param method "additive" or "multiplicative"
     * @return DecompositionResult with trend, seasonal, residual components
     */
    static DecompositionResult Decompose(
        const std::vector<double>& data,
        int period,
        const std::string& method = "additive"
    );

    /**
     * STL Decomposition (Seasonal-Trend decomposition using Loess)
     * More robust than classical decomposition
     * @param data Input time series
     * @param period Seasonal period
     * @param seasonal_window Window for seasonal smoothing (odd number, >= 7)
     * @param trend_window Window for trend smoothing (-1 for auto)
     * @return DecompositionResult
     */
    static DecompositionResult STLDecompose(
        const std::vector<double>& data,
        int period,
        int seasonal_window = 7,
        int trend_window = -1
    );

    // ==================== Autocorrelation ====================

    /**
     * Compute Autocorrelation Function (ACF)
     * @param data Input time series
     * @param max_lag Maximum lag to compute (default: min(n/2, 40))
     * @return AutocorrelationResult with ACF values and confidence bounds
     */
    static AutocorrelationResult ComputeACF(
        const std::vector<double>& data,
        int max_lag = -1
    );

    /**
     * Compute Partial Autocorrelation Function (PACF)
     * Uses Durbin-Levinson recursion
     * @param data Input time series
     * @param max_lag Maximum lag to compute
     * @return AutocorrelationResult with PACF values
     */
    static AutocorrelationResult ComputePACF(
        const std::vector<double>& data,
        int max_lag = -1
    );

    /**
     * Compute both ACF and PACF together
     * @param data Input time series
     * @param max_lag Maximum lag
     * @return AutocorrelationResult with both ACF and PACF
     */
    static AutocorrelationResult ComputeACFPACF(
        const std::vector<double>& data,
        int max_lag = -1
    );

    /**
     * Ljung-Box test for white noise
     * @param data Input time series (often residuals)
     * @param lags Number of lags to test
     * @return p-value (low = not white noise)
     */
    static double LjungBoxTest(
        const std::vector<double>& data,
        int lags = 10
    );

    // ==================== Stationarity Tests ====================

    /**
     * Test stationarity using ADF and KPSS tests
     * @param data Input time series
     * @param max_lags Maximum lags for ADF test (-1 for auto)
     * @return StationarityResult with test statistics and interpretation
     */
    static StationarityResult TestStationarity(
        const std::vector<double>& data,
        int max_lags = -1
    );

    /**
     * Augmented Dickey-Fuller test
     * H0: series has a unit root (non-stationary)
     * @param data Input time series
     * @param max_lags Maximum lags
     * @return StationarityResult (only ADF fields populated)
     */
    static StationarityResult ADFTest(
        const std::vector<double>& data,
        int max_lags = -1
    );

    /**
     * KPSS test (Kwiatkowski-Phillips-Schmidt-Shin)
     * H0: series is stationary
     * @param data Input time series
     * @param regression "c" for constant, "ct" for constant + trend
     * @return StationarityResult (only KPSS fields populated)
     */
    static StationarityResult KPSSTest(
        const std::vector<double>& data,
        const std::string& regression = "c"
    );

    /**
     * Difference a time series
     * @param data Input time series
     * @param order Differencing order (1 = first difference, 2 = second, etc.)
     * @return Differenced series
     */
    static std::vector<double> Difference(
        const std::vector<double>& data,
        int order = 1
    );

    /**
     * Seasonal difference
     * @param data Input time series
     * @param period Seasonal period
     * @param order Number of seasonal differences
     * @return Seasonally differenced series
     */
    static std::vector<double> SeasonalDifference(
        const std::vector<double>& data,
        int period,
        int order = 1
    );

    // ==================== Seasonality Detection ====================

    /**
     * Detect seasonality in time series
     * Uses spectral analysis and ACF peaks
     * @param data Input time series
     * @param min_period Minimum period to consider
     * @param max_period Maximum period (-1 for n/2)
     * @return SeasonalityResult with detected periods and strengths
     */
    static SeasonalityResult DetectSeasonality(
        const std::vector<double>& data,
        int min_period = 2,
        int max_period = -1
    );

    /**
     * Compute periodogram (spectral density)
     * @param data Input time series
     * @return Periodogram values and corresponding frequencies
     */
    static std::pair<std::vector<double>, std::vector<double>> Periodogram(
        const std::vector<double>& data
    );

    // ==================== Forecasting ====================

    /**
     * Simple Exponential Smoothing (SES)
     * For series without trend or seasonality
     * @param data Input time series
     * @param horizon Number of periods to forecast
     * @param alpha Smoothing parameter (0-1, -1 for auto)
     * @return ForecastResult
     */
    static ForecastResult SimpleES(
        const std::vector<double>& data,
        int horizon,
        double alpha = -1
    );

    /**
     * Holt's Linear Trend Method
     * For series with trend but no seasonality
     * @param data Input time series
     * @param horizon Forecast horizon
     * @param alpha Level smoothing (0-1, -1 for auto)
     * @param beta Trend smoothing (0-1, -1 for auto)
     * @param damped Use damped trend
     * @return ForecastResult
     */
    static ForecastResult HoltLinear(
        const std::vector<double>& data,
        int horizon,
        double alpha = -1,
        double beta = -1,
        bool damped = false
    );

    /**
     * Holt-Winters Exponential Smoothing
     * For series with trend and seasonality
     * @param data Input time series
     * @param horizon Forecast horizon
     * @param period Seasonal period (-1 for auto-detect)
     * @param seasonal_type "additive" or "multiplicative"
     * @param alpha Level smoothing (-1 for auto)
     * @param beta Trend smoothing (-1 for auto)
     * @param gamma Seasonal smoothing (-1 for auto)
     * @return ForecastResult
     */
    static ForecastResult HoltWinters(
        const std::vector<double>& data,
        int horizon,
        int period = -1,
        const std::string& seasonal_type = "additive",
        double alpha = -1,
        double beta = -1,
        double gamma = -1
    );

    /**
     * Simple Moving Average forecast
     * @param data Input time series
     * @param window Moving average window
     * @param horizon Forecast horizon
     * @return ForecastResult
     */
    static ForecastResult MovingAverageForecast(
        const std::vector<double>& data,
        int window,
        int horizon
    );

    /**
     * ARIMA forecasting (simplified implementation)
     * @param data Input time series
     * @param horizon Forecast horizon
     * @param p AR order (-1 for auto)
     * @param d Differencing order (-1 for auto)
     * @param q MA order (-1 for auto)
     * @return ForecastResult
     */
    static ForecastResult ARIMA(
        const std::vector<double>& data,
        int horizon,
        int p = -1,
        int d = -1,
        int q = -1
    );

    // ==================== Utility Functions ====================

    /**
     * Compute rolling/moving mean
     * @param data Input data
     * @param window Window size
     * @return Rolling mean (shorter by window-1)
     */
    static std::vector<double> RollingMean(
        const std::vector<double>& data,
        int window
    );

    /**
     * Compute rolling/moving standard deviation
     * @param data Input data
     * @param window Window size
     * @return Rolling std (shorter by window-1)
     */
    static std::vector<double> RollingStd(
        const std::vector<double>& data,
        int window
    );

    /**
     * Compute mean
     */
    static double Mean(const std::vector<double>& data);

    /**
     * Compute variance
     */
    static double Variance(const std::vector<double>& data);

    /**
     * Compute standard deviation
     */
    static double StdDev(const std::vector<double>& data);

    // ==================== Synthetic Data Generation ====================

    /**
     * Generate white noise
     */
    static std::vector<double> GenerateWhiteNoise(
        int n,
        double mean = 0.0,
        double std = 1.0
    );

    /**
     * Generate random walk
     */
    static std::vector<double> GenerateRandomWalk(
        int n,
        double start = 0.0,
        double std = 1.0
    );

    /**
     * Generate trend + seasonal + noise
     */
    static std::vector<double> GenerateTrendSeasonal(
        int n,
        double trend_slope,
        double seasonal_amplitude,
        int period,
        double noise_std
    );

    /**
     * Generate AR(p) process
     */
    static std::vector<double> GenerateAR(
        int n,
        const std::vector<double>& coeffs,
        double noise_std = 1.0
    );

    /**
     * Generate MA(q) process
     */
    static std::vector<double> GenerateMA(
        int n,
        const std::vector<double>& coeffs,
        double noise_std = 1.0
    );

    /**
     * Generate ARIMA(p,d,q) process
     */
    static std::vector<double> GenerateARIMA(
        int n,
        const std::vector<double>& ar_coeffs,
        const std::vector<double>& ma_coeffs,
        int d,
        double noise_std = 1.0
    );

private:
    // Helper functions
    static std::vector<double> CenteredMovingAverage(
        const std::vector<double>& data,
        int window
    );

    static double OptimizeESAlpha(
        const std::vector<double>& data
    );

    static std::pair<double, double> OptimizeHoltParams(
        const std::vector<double>& data
    );

    static std::tuple<double, double, double> OptimizeHWParams(
        const std::vector<double>& data,
        int period,
        const std::string& seasonal_type
    );
};

} // namespace cyxwiz
