#pragma once

#include "pipeline_operator.h"
#include <string>

namespace cyxwiz {

/**
 * TimeSeriesDecompositionOperator — Cat-1 Band 1 pipeline operator.
 *
 * Classical additive/multiplicative decomposition of a time-series
 * column into trend + seasonal + residual via
 * `TimeSeries::Decompose`. Appends three new columns to the input
 * table (same row count, no alignment break).
 *
 * Closes the TimeSeriesDecomposition dead NodeType from the
 * "Phase 5 time-series" block in tofix.md.
 *
 * Params:
 *   signal_col (required)               — numeric column to decompose.
 *   period     (required, int)          — seasonal period (e.g. 12 for
 *                                          monthly-with-yearly-seasonality).
 *   method     (default "additive")     — "additive" / "multiplicative".
 *   algorithm  (default "classical")    — "classical" / "stl" (STL uses
 *                                          Loess smoothing, more robust).
 */
class TimeSeriesDecompositionOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "TimeSeriesDecomposition"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int period_ = 0;
    std::string method_ = "additive";
    std::string algorithm_ = "classical";
};

/**
 * ARIMAOperator — Cat-1 Band 1 pipeline operator (in-sample fit only).
 *
 * Fits an ARIMA(p, d, q) model to a signal column via
 * `TimeSeries::ARIMA` and appends `fitted` + `residual` columns.
 * Row count is preserved — the forecast horizon goes to zero so
 * only in-sample fitted values are written back.
 *
 * Out-of-sample forecasting (horizon > 0) would add future rows to
 * the table; that schema change is deferred to a separate Cat-1
 * "forecast-rows" operator or a Cat-2 visualization panel. See
 * tofix.md for reasoning.
 *
 * Params:
 *   signal_col (required)         — numeric column to model.
 *   p          (default -1, auto) — AR order.
 *   d          (default -1, auto) — differencing order.
 *   q          (default -1, auto) — MA order.
 */
class ARIMAOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ARIMA"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int p_ = -1;
    int d_ = -1;
    int q_ = -1;
};

/**
 * ExponentialSmoothingOperator — Cat-1 Band 1 pipeline operator.
 *
 * Fits a Simple ES / Holt / Holt-Winters model to a signal column
 * (selected via `method` param) and appends `fitted` + `residual`
 * columns. Horizon=0 — in-sample fit only, same rationale as
 * ARIMAOperator.
 *
 * Params:
 *   signal_col (required)             — numeric column.
 *   method     (default "simple")     — "simple" / "holt" / "holt_winters".
 *   alpha      (default -1, auto)     — level smoothing.
 *   beta       (default -1, auto)     — trend smoothing (holt/hw).
 *   gamma      (default -1, auto)     — seasonal smoothing (hw).
 *   period     (default -1, auto)     — seasonal period (hw).
 *   damped     (default false)        — damped trend (holt).
 */
class ExponentialSmoothingOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ExponentialSmoothing"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    std::string method_ = "simple";
    double alpha_ = -1.0;
    double beta_ = -1.0;
    double gamma_ = -1.0;
    int period_ = -1;
    bool damped_ = false;
};

/**
 * ACFOperator - Cat-1 Band 1 analysis operator.
 *
 * Emits a lag-indexed table: lag, acf, confidence_lower,
 * confidence_upper, significant.
 */
class ACFOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "ACF"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int max_lag_ = -1;
};

/**
 * PACFOperator - Cat-1 Band 1 analysis operator.
 *
 * Emits a lag-indexed table: lag, pacf, confidence_lower,
 * confidence_upper, significant.
 */
class PACFOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "PACF"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int max_lag_ = -1;
};

/**
 * StationarityTestOperator - Cat-1 Band 1 analysis operator.
 *
 * Emits a one-row summary table with ADF/KPSS statistics and the
 * suggested ARIMA differencing order.
 */
class StationarityTestOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "StationarityTest"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int max_lags_ = -1;
};

/**
 * SeasonalityDetectorOperator - Cat-1 Band 1 analysis operator.
 *
 * Emits candidate periods with strength scores. If no candidate exists,
 * emits one row containing the primary detection fields.
 */
class SeasonalityDetectorOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "SeasonalityDetector"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params,
                   std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

private:
    std::string signal_col_;
    int min_period_ = 2;
    int max_period_ = -1;
};

} // namespace cyxwiz
