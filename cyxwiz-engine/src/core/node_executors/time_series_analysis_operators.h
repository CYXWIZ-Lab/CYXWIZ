#pragma once

// The time series analysis nodes (TOFIX129 D6 step 3). Each reads one numeric
// signal column in table order (rows with a missing value are refused: fill
// or drop them first) and runs cyxwiz::TimeSeries, whose figures match
// statsmodels (test_computation_truth_time_series). Decomposition appends
// its parts; ACF, PACF, Stationarity Test and Seasonality Detector emit a
// table of their own; ARIMA and Exp. Smoothing append fitted and residual
// columns and, with a horizon, future rows holding forecast / low / high
// (the time column continues its step there, every other column is empty).

#include "pipeline_operator.h"
#include <string>
#include <utility>

namespace cyxwiz {

/**
 * Decomposition: trend, seasonal and residual columns appended to the input.
 *
 * Params:
 *   signal_col (required)            numeric column
 *   period (default -1)              seasonal period; -1 finds it from the detrended periodogram
 *   method (default "additive")      "additive" / "multiplicative" (classical)
 *   algorithm (default "stl")        "stl" / "classical" (classical leaves half a period empty at each end)
 *   seasonal_window (default 7)      STL: the seasonal loess window (odd, >= 3)
 *   robust (default false)           STL: robustness iterations
 */
class TimeSeriesDecompositionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TimeSeriesDecomposition"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int period_ = -1;
    std::string method_ = "additive";
    std::string algorithm_ = "stl";
    int seasonal_window_ = 7;
    bool robust_ = false;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * ARIMA: an ARIMA(p, d, q) fit by exact maximum likelihood; fitted and
 * residual columns on the input rows, and `horizon` future rows with
 * forecast / low / high at the chosen level.
 *
 * Params:
 *   signal_col (required)   numeric column
 *   p, d, q (default -1)    orders; -1 chooses them (d by the stationarity tests, p and q by AIC)
 *   horizon (default 12)    future rows (0: the in-sample fit only)
 *   level (default 0.95)    the prediction interval's level
 *   time_col (default "")   a date, timestamp or numeric column continued over the future rows
 */
class ARIMAOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ARIMA"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int p_ = -1, d_ = -1, q_ = -1;
    int horizon_ = 12;
    double level_ = 0.95;
    std::string time_col_;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * Exp. Smoothing: simple, Holt (optionally damped) or Holt-Winters; fitted
 * and residual columns, and `horizon` future rows with forecast / low / high.
 *
 * Params:
 *   signal_col (required)          numeric column
 *   method (default "holt_winters") "simple" / "holt" / "holt_winters"
 *   seasonal (default "additive")  Holt-Winters season: "additive" / "multiplicative"
 *   alpha, beta, gamma (default -1) smoothing weights; -1 estimates them
 *   period (default -1)            Holt-Winters period; -1 finds it
 *   damped (default false)         damped trend (Holt)
 *   horizon (default 12), level (default 0.95), time_col (default ""): as ARIMA
 */
class ExponentialSmoothingOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ExponentialSmoothing"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    std::string method_ = "holt_winters";
    std::string seasonal_ = "additive";
    double alpha_ = -1.0, beta_ = -1.0, gamma_ = -1.0;
    int period_ = -1;
    bool damped_ = false;
    int horizon_ = 12;
    double level_ = 0.95;
    std::string time_col_;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * ACF: one row per lag 0 .. max_lag: lag, acf, confidence_lower,
 * confidence_upper (Bartlett band at `confidence`), significant,
 * ljung_box_q, ljung_box_p (the white-noise test up to that lag).
 *
 * Params: signal_col (required), max_lag (default -1: min(n/2, 40)), confidence (default 0.95).
 */
class ACFOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ACF"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int max_lag_ = -1;
    double confidence_ = 0.95;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * PACF: one row per lag: lag, pacf, confidence_lower, confidence_upper
 * (z / sqrt(n) at `confidence`), significant. Params as ACF.
 */
class PACFOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "PACF"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int max_lag_ = -1;
    double confidence_ = 0.95;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * Stationarity Test: ADF and KPSS on the series and on its first and second
 * differences, one row each: differencing, adf_statistic, adf_pvalue,
 * adf_lags, kpss_statistic, kpss_pvalue, kpss_lags, stationary (KPSS p >= 0.05
 * and ADF p < 0.10), suggested_differencing (the smallest d that is enough,
 * -1 when none), analysis.
 *
 * Params: signal_col (required), max_lags (default -1: 12 (n/100)^0.25, chosen by AIC),
 *         regression (default "c": around a level; "ct": around a trend).
 */
class StationarityTestOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "StationarityTest"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int max_lags_ = -1;
    std::string regression_ = "c";
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * Seasonality Detector: the candidate periods of the detrended periodogram,
 * ranked by their share of the power: period, share, harmonic_of (the
 * stronger period this one divides, 0 for none), is_primary, has_seasonality,
 * strength (the primary's: how much of the detrended series a season of that
 * length explains), analysis.
 *
 * Params: signal_col (required), min_period (default 2), max_period (default -1: a quarter of the rows, four cycles).
 */
class SeasonalityDetectorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SeasonalityDetector"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }

private:
    std::string signal_col_;
    int min_period_ = 2;
    int max_period_ = -1;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * SeasonalNaiveOperator - deterministic rolling forecast baseline.
 *
 * Consumes the canonical wide table emitted by TimeSeriesWindow:
 * `x_0..x_N`, `y`, `y_1..y_H`. For every input window it repeats the
 * most recent complete seasonal cycle and emits one long-form row per
 * forecast horizon:
 *
 *   window_index, horizon, actual, prediction, error
 *
 * TimeSeriesSplit metadata is carried forward when present, so FilterRows
 * can select `__partition__ = 2` before RegressionMetrics evaluates the
 * held-out test baseline. Target indices are also preserved as
 * `__target_index` when TimeSeriesWindow metadata is available.
 *
 * Params:
 *   seasonal_period (required, int >= 1) - observations per cycle.
 */
class SeasonalNaiveOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SeasonalNaive"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;
    arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    int seasonal_period_ = 0;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
