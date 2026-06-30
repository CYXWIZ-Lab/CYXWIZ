#pragma once

#include "pipeline_operator.h"
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * FFTOperator — Cat-1 Band 1 pipeline operator.
 *
 * Treats one numeric column of the input table as a single 1D
 * time-domain signal and computes its FFT via
 * `SignalProcessing::FFT`. Output is a NEW table with one row per
 * FFT bin and three columns: `frequency (f64)`, `magnitude (f64)`,
 * `phase (f64)`.
 *
 * Row count CHANGES — the output has as many rows as the FFT
 * produces bins (not the input sample count). This breaks "carry
 * all input columns forward" pipelines; the output is
 * frequency-domain data that doesn't align with the time-domain
 * input rows. Downstream nodes must understand this.
 *
 * For per-window / per-sample FFT (the usual ML feature-extraction
 * pattern), chain `TimeSeriesWindow` → FFT with a Cat-1 operator
 * that runs FFT per row of the windowed table. That's out of
 * scope for v1 — deferred in tofix.
 *
 * Params:
 *   signal_col    (required)    — numeric column to FFT.
 *   sample_rate   (default 1.0) — Hz; controls frequency axis scaling.
 */
class FFTOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "FFT"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string signal_col_;
    double sample_rate_ = 1.0;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * Convolve1DOperator — Cat-1 Band 1 pipeline operator.
 *
 * Convolves one signal column with a user-specified FIR kernel
 * (passed as a comma-separated parameter, NOT through the Kernel
 * input pin — the materializer doesn't wire Tensor-typed pins).
 * The Kernel input pin remains visually present but inert.
 *
 * Mode is forced to "same" so the output column has the same row
 * count as the input — keeps all other input columns aligned and
 * preserves downstream compatibility.
 *
 * Params:
 *   signal_col  (required)        — numeric column to convolve.
 *   kernel      (required, csv)   — comma-sep kernel taps, e.g.
 *                                    "0.25,0.5,0.25" for a 3-tap
 *                                    moving-average smoother.
 */
class Convolve1DOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "Convolve1D"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string signal_col_;
    std::vector<double> kernel_;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * FilterDesignerOperator — Cat-1 Band 1 pipeline operator.
 *
 * Designs a Butterworth-style filter from params and applies it
 * to one signal column in place. The NodeType name
 * `FilterDesigner` in `node_editor.h` is a misnomer carried over
 * from the panel era when design and apply were separate —
 * this operator does both so the pipeline sees a filtered signal,
 * not a filter-coefficients output.
 *
 * Params:
 *   signal_col    (required)            — numeric column to filter.
 *   filter_type   (default "lowpass")   — "lowpass"/"highpass"/
 *                                          "bandpass"/"bandstop".
 *   cutoff        (default 0.5)         — primary cutoff (Hz, or
 *                                          normalized 0..0.5 of fs).
 *   cutoff_high   (optional)            — upper cutoff for band*
 *                                          filters; must be > cutoff.
 *   sample_rate   (default 1.0)         — Hz.
 *   order         (default 4)           — filter order.
 */
class FilterDesignerOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "FilterDesigner"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) override;

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override;

    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

private:
    std::string signal_col_;
    std::string filter_type_ = "lowpass";
    double cutoff_ = 0.5;
    double cutoff_high_ = 0.0;
    double sample_rate_ = 1.0;
    int order_ = 4;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
