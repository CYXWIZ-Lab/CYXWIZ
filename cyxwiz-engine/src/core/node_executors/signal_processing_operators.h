#pragma once

// The signal nodes' operators (TOFIX129 D7): FFT (a one-sided spectrum
// table, a branch), IFFT (that table back to a signal), Filter Designer and
// Convolution 1D (a filtered column beside the others), Wavelet Transform
// (the parts and the denoised signal beside the others). Every figure comes
// from cyxwiz::SignalProcessing on the device, checked against numpy / scipy /
// PyWavelets in test_computation_truth_signal; the column contracts are
// checked in test_pipeline_executor_operator_routing.

#include "pipeline_operator.h"
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

/**
 * FFTOperator — the one-sided spectrum of one numeric column as a NEW table
 * (n / 2 + 1 rows): `frequency` (per second at sample_rate), `period`
 * (1 / frequency; null at 0), `amplitude` (2 |X| / sum of the window; the
 * amplitude of a tone), `phase` (radians), `power_density` (scipy periodogram,
 * density), `real` and `imag` (numpy.fft.rfft of the windowed, mean-removed
 * signal, so IFFT can take the table back). The row count changes, so this
 * node is a branch for looking at a signal, not a step on the way to training.
 *
 * Params:
 *   signal_col    (required)       — numeric column.
 *   sample_rate   (default 1.0)    — samples per second (per unit, per row).
 *   window        (default "hann") — "hann", "hamming", "blackman", "boxcar".
 *   remove_mean   (default true)   — the mean taken out first.
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
    std::string window_ = "hann";
    bool remove_mean_ = true;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * IFFTOperator — the FFT node's table back to a signal: a NEW table with one
 * column `signal` of `samples` rows (0: 2 (bins - 1), the even length) from
 * the `real` and `imag` columns (numpy.fft.irfft). What comes back is the
 * signal the FFT node transformed: windowed and mean-removed as it was.
 *
 * Params:
 *   samples   (default 0) — the length of the signal the spectrum came from.
 */
class IFFTOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "IFFT"; }
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
    int samples_ = 0;
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * Convolve1DOperator — one signal column convolved with a kernel given as a
 * comma-separated list, zero-phase (the kernel centred on each row, the
 * rows beyond the ends repeating the edge value), added as a new float64
 * column `<signal_col>_filtered` beside every input column.
 *
 * Params:
 *   signal_col  (required)        — numeric column.
 *   kernel      (required, csv)   — the taps, e.g. "0.25,0.5,0.25".
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
 * FilterDesignerOperator — a windowed-sinc FIR filter (scipy.signal.firwin)
 * designed from the params and applied zero-phase to one signal column,
 * added as a new float64 column `<signal_col>_filtered` beside every input
 * column.
 *
 * Params:
 *   signal_col    (required)            — numeric column.
 *   filter_type   (default "lowpass")   — "lowpass" / "highpass" / "bandpass" / "bandstop".
 *   cutoff        (default 0.1)         — the cutoff (per second at sample_rate; below Nyquist).
 *   cutoff_high   (band filters)        — the upper cutoff; above cutoff, below Nyquist.
 *   sample_rate   (default 1.0)         — samples per second (per unit, per row).
 *   taps          (default 31)          — coefficients; odd for highpass / bandstop.
 *   window        (default "hamming")   — "hamming", "hann", "blackman", "boxcar".
 *   order                               — the old parameter: 2 order + 1 taps, read when `taps` is not set.
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
    double cutoff_ = 0.1;
    double cutoff_high_ = 0.0;
    double sample_rate_ = 1.0;
    int order_ = 15;  // the old parameter: 2 * order + 1 taps (kept until `taps` is set)
    int taps_ = 0;    // the number of coefficients (scipy.signal.firwin); 0: from order_
    std::string window_ = "hamming";
    PipelineOperatorProgressCallback progress_callback_;
};

/**
 * WaveletTransformOperator — pywt.wavedec of one signal column (symmetric
 * mode), each part brought back to the rows and added as float64 columns
 * `<signal_col>_A<level>`, `<signal_col>_D<level>` .. `<signal_col>_D1`
 * beside every input column, plus `<signal_col>_denoised` when denoising is
 * on (the details thresholded at the universal threshold, or the one given).
 *
 * Params:
 *   signal_col  (required)        — numeric column.
 *   wavelet     (default "db4")   — haar, db1..db8, sym2..sym8, coif1..coif5.
 *   level       (default 3)       — levels (at most pywt.dwt_max_level).
 *   denoise     (default "off")   — "off", "soft", "hard".
 *   threshold   (default 0)       — 0: the universal threshold sigma sqrt(2 ln n).
 */
class WaveletTransformOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "WaveletTransform"; }
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
    std::string wavelet_ = "db4";
    int level_ = 3;
    std::string denoise_ = "off";
    double threshold_ = 0.0;
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
