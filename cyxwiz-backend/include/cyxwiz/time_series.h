// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
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
    // STL settings used (0 when classical)
    int seasonal_window = 0;
    int trend_window = 0;
    int low_pass_window = 0;
    bool robust = false;
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API AutocorrelationResult {
    std::vector<double> acf;                      // Autocorrelation function
    std::vector<double> pacf;                     // Partial autocorrelation function
    std::vector<double> lags;                     // Lag values (0, 1, 2, ...)
    std::vector<double> confidence_upper;         // ACF band (Bartlett: grows with the lag)
    std::vector<double> confidence_lower;
    std::vector<double> pacf_confidence_upper;    // PACF band (z / sqrt(n))
    std::vector<double> pacf_confidence_lower;
    double confidence_level = 0.95;
    int max_lag = 0;                              // Maximum lag computed
    double ljung_box_statistic = 0.0;             // Ljung-Box Q at ljung_box_lags
    double ljung_box_pvalue = 0.0;                // its chi-squared p-value
    int ljung_box_lags = 0;                       // min(10, max_lag)
    std::vector<double> ljung_box_q;              // Q and p at every lag 1 .. max_lag (index 0 unused)
    std::vector<double> ljung_box_p;
    std::vector<int> significant_acf_lags;        // Lags where ACF is significant
    std::vector<int> significant_pacf_lags;       // Lags where PACF is significant
    int suggested_ar_order = 0;                   // Suggested AR(p) order
    int suggested_ma_order = 0;                   // Suggested MA(q) order
    bool success = false;
    std::string error_message;
};

struct CYXWIZ_API StationarityResult {
    std::string regression = "c";                 // "c" (level) or "ct" (level and trend)

    // Augmented Dickey-Fuller Test (statsmodels adfuller: lags by AIC, MacKinnon p and critical values)
    double adf_statistic = 0.0;                   // ADF test statistic
    double adf_pvalue = 0.0;                      // p-value
    std::map<std::string, double> adf_critical;   // Critical values (1%, 5%, 10%)
    bool adf_stationary = false;                  // p < 0.05: a unit root is rejected
    int adf_lags = 0;                             // lags used
    int adf_nobs = 0;                             // observations in the regression
    double adf_icbest = 0.0;                      // the AIC of the chosen lag

    // KPSS Test (statsmodels kpss: Newey-West lags by Hobijn, p interpolated in the 1992 table)
    double kpss_statistic = 0.0;                  // KPSS test statistic
    double kpss_pvalue = 0.0;                     // p-value, clipped to 0.01 .. 0.10
    std::map<std::string, double> kpss_critical;  // Critical values
    bool kpss_stationary = false;                 // p >= 0.05: stationarity is not rejected
    int kpss_lags = 0;

    // The same tests on the differenced series (d = 1, 2), for the suggested d
    struct Differenced {
        int d = 0;
        double adf_statistic = 0.0, adf_pvalue = 0.0;
        double kpss_statistic = 0.0, kpss_pvalue = 0.0;
        bool stationary = false;                  // stationary enough: KPSS p >= 0.05 and ADF p < 0.10
    };
    std::vector<Differenced> differenced;

    // Combined result
    bool is_stationary = false;                   // Overall conclusion
    int suggested_differencing = 0;               // The smallest d that is stationary enough (-1: not even at 2)
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

    // Multiple seasonality support (ranked by their share of the detrended power)
    std::vector<int> candidate_periods;           // All detected periods
    std::vector<double> candidate_strengths;      // Each one's share of the power (0-1)
    std::vector<int> harmonic_of;                 // The stronger period this one divides (0: none)
    double detected_share = 0.0;                  // The primary period's share of the power

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
    double sse = 0.0;                             // Sum of squared one-step errors
    double log_likelihood = 0.0;                  // ARIMA: the Gaussian log-likelihood
    int n_params = 0;                             // Parameters counted by AIC

    // Model parameters
    std::string method;                           // Model name
    std::map<std::string, double> parameters;     // Fitted parameters (alpha, beta, gamma, phi, initial_*, ar.i, ma.i, const, sigma2)
    std::string model_summary;                    // Text summary
    std::vector<double> residuals;                // y - fitted

    double level = 0.95;                          // The prediction interval's level
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
        int trend_window = -1,
        int low_pass_window = -1,
        bool robust = false
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
        int max_lag = -1,
        double confidence = 0.95
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
        int max_lags = -1,
        const std::string& regression = "c"
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
        int max_lags = -1,
        const std::string& regression = "c",
        bool autolag = true
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
        const std::string& regression = "c",
        int nlags = -1
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
        double alpha = -1,
        double level = 0.95
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
        bool damped = false,
        double level = 0.95
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
        double gamma = -1,
        double level = 0.95
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
        int q = -1,
        double level = 0.95
    );

    /**
     * Seasonal naive forecast: each future value repeats the value one period
     * (or several) before it; the interval grows with the number of cycles
     * ahead (Hyndman's formula).
     */
    static ForecastResult SeasonalNaive(
        const std::vector<double>& data,
        int horizon,
        int period,
        double level = 0.95
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

    // ==================== Windowing for ML ====================

    /**
     * Configuration for sliding window dataset creation
     */
    struct WindowConfig {
        int window_size = 10;          // Lookback window (number of past timesteps)
        int forecast_horizon = 1;      // How many steps ahead to predict
        int stride = 1;               // Step between consecutive windows
        std::vector<int> lag_values;   // Additional lag features (e.g., {1, 7, 30})
        std::vector<int> rolling_windows; // Rolling stat windows (e.g., {7, 30})
        bool add_diff_features = false;   // Add differenced features
        bool normalize = false;        // Normalize each window
    };

    /**
     * Result of windowing operation
     */
    struct WindowResult {
        std::vector<std::vector<double>> X;  // Input windows [num_windows, window_size * features]
        std::vector<std::vector<double>> y;  // Target values [num_windows, forecast_horizon]
        size_t num_windows = 0;
        size_t input_features = 0;     // Features per timestep
        size_t target_features = 0;
        bool success = false;
        std::string error_message;
    };

    /**
     * Create sliding window dataset from time series
     * @param data Input time series (single feature)
     * @param config Window configuration
     * @return WindowResult with X (inputs) and y (targets)
     */
    static WindowResult CreateWindows(
        const std::vector<double>& data,
        const WindowConfig& config
    );

    /**
     * Create sliding window dataset from multivariate time series
     * @param data Input data [num_timesteps][num_features]
     * @param target_col Index of the target column
     * @param config Window configuration
     * @return WindowResult
     */
    static WindowResult CreateMultivariateWindows(
        const std::vector<std::vector<double>>& data,
        int target_col,
        const WindowConfig& config
    );

    /**
     * Add engineered time-series features to data
     * @param data Input time series
     * @param lag_values Lag features to add (e.g., {1, 7, 30})
     * @param rolling_windows Rolling mean/std windows (e.g., {7, 30})
     * @param add_diff Add first-difference feature
     * @return Multivariate data [num_timesteps][num_features]
     *         Features: [original, lag_1, ..., lag_n, roll_mean_w1, roll_std_w1, ..., diff]
     */
    static std::vector<std::vector<double>> AddFeatures(
        const std::vector<double>& data,
        const std::vector<int>& lag_values = {},
        const std::vector<int>& rolling_windows = {},
        bool add_diff = false
    );

    /**
     * Chronological train/val/test split (preserves temporal order)
     * @param num_samples Total number of samples
     * @param train_ratio Training ratio (default: 0.7)
     * @param val_ratio Validation ratio (default: 0.15)
     * @return Tuple of (train_end_idx, val_end_idx) - test is the rest
     */
    static std::pair<size_t, size_t> ChronologicalSplit(
        size_t num_samples,
        double train_ratio = 0.7,
        double val_ratio = 0.15
    );

};

} // namespace cyxwiz
