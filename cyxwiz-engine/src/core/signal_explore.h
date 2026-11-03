#pragma once

// The answers behind Data Studio's Explore tab Signal (TOFIX129 D7, approved
// mockup d7_signal_mockup.html): one signal column over the time column D6's
// reader gives (dates, numbers, or the row order), its sample rate from the
// step (1 / step: Hz for dates, per unit for numbers, per row for the order)
// or typed, in four looks: Spectrum (amplitude or Welch power, peaks in the
// spectrum and in the rows), Spectrogram (STFT, the hop raised for long
// series), Filter (a firwin FIR or a kernel applied zero-phase) and Wavelet
// (wavedec parts brought back to the rows, energy by scale, denoising). The
// arithmetic is cyxwiz::SignalProcessing on the device (checked against
// numpy / scipy / PyWavelets in test_computation_truth_signal); what is new
// here (the unit, the peaks' choices, the kernels, the parts, the node
// parameters) is checked in test_signal_explore.

#include "data_explore.h"
#include "series_explore.h"

#include <cyxwiz/signal_processing.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::signal {

struct SignalRequest {
    std::string time;          // "" = the row order
    std::string value;
    double sample_rate = 0.0;  // 0: 1 / the step; else typed (per second for dates, per unit, per row)
};

// Frequencies are shown as cycles per `seconds` ("per hour"), chosen so the
// Nyquist frequency reads as at least 10 (UnitFor).
struct FrequencyUnit {
    std::string name = "Hz";   // "Hz", "per minute", "per hour", "per day", "per week", "per month", "per year", "per unit", "per row"
    double seconds = 1.0;      // numbers: 1 unit; order: 1 row
};

struct SignalData {
    series::Series series;     // t, values, labels, step, rows / used / unordered, kind
    double sample_rate = 0.0;  // samples per second (dates), per unit (numbers), per row (order)
    bool rate_from_step = true;
    FrequencyUnit unit;
    double step_seconds = 1.0;  // 1 / sample_rate: the time one row spans
};

struct SignalAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    SignalData data;
};

// Reads the column over its time (series::ReadSeries) and sets the rate and
// the unit. The rows stay in time order; uneven spacing is in series.step.even.
SignalAnswer ReadSignal(const std::string& table, const SignalRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// Pure: the unit for a step (the smallest of second, minute, hour, day, week,
// month, year with Nyquist >= 10 per unit; "per unit" / "per row" otherwise).
FrequencyUnit UnitFor(series::TimeKind kind, double sample_rate);

// ---- Spectrum

struct SpectrumRequest {
    SignalRequest base;
    std::string window = "hann";     // "hann", "hamming", "blackman", "boxcar"
    bool remove_mean = true;
    bool remove_trend = false;       // a least-squares line taken out first (a trending series otherwise puts its whole length on top)
    int welch_segment = 0;           // 0: the whole series; else Welch over segments of this many rows (the view turns it on above 100,000 rows)
    double peak_prominence_sd = 1.0; // rows: a peak stands out by this many SDs
    int peak_gap_rows = 0;           // rows: peaks at least this far apart (0: 10 rows)
};

struct SpectrumPeak {
    int bin = 0;
    double frequency = 0.0;    // per second (dates) / unit / row
    double amplitude = 0.0;    // the amplitude (whole series) or the power density (Welch) at the peak
    double prominence = 0.0;
};

struct RowPeak {
    int index = 0;             // row of the series (sorted)
    double t = 0.0;
    double value = 0.0;
    double prominence = 0.0;
};

struct SpectrumAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    SpectrumRequest request;
    SignalData data;
    SpectrumResult spectrum;              // frequencies per second / unit / row
    std::vector<SpectrumPeak> peaks;      // the strongest first, at most 10
    std::vector<RowPeak> row_peaks;       // in time order
    double row_sd = 0.0;                  // the SD the prominence is in
    int row_gap = 0;                      // the gap used (rows)
    double median_gap_seconds = 0.0;      // between row peaks (0: fewer than 2)
};

SpectrumAnswer RunSpectrum(const std::string& table, const SpectrumRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// ---- Spectrogram

struct SpectrogramRequest {
    SignalRequest base;
    int window_rows = 256;
    int hop_rows = 64;
    std::string window = "hann";
    double range_db = 60.0;    // the floor below the loudest bin
    int max_frames = 4000;     // the hop is raised to keep within it (D6's long-series rule)
};

struct SpectrogramAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    SpectrogramRequest request;
    SignalData data;
    SpectrogramResult spectrogram;
    int hop_used = 0;
    bool hop_raised = false;
    double loudest_db = 0.0;   // 10 log10 of the largest density
    double loudest_frequency = 0.0;
    double floor_db = 0.0;     // loudest - range
};

SpectrogramAnswer RunSpectrogram(const std::string& table, const SpectrogramRequest& request, const QueryRunner& run,
                                 const explore::MissingTexts& texts);

// ---- Filter

enum class FilterMode { Design, Kernel };
enum class FilterKind { Lowpass, Highpass, Bandpass, Bandstop };
enum class KernelPreset { MovingAverage, Gaussian, Difference, SecondDifference, Typed };

struct FilterRequest {
    SignalRequest base;
    FilterMode mode = FilterMode::Design;
    FilterKind kind = FilterKind::Lowpass;
    double cut = 0.0;          // per second / unit / row (0: a tenth of Nyquist)
    double cut_high = 0.0;     // bandpass / bandstop
    int taps = 101;
    std::string window = "hamming";
    KernelPreset preset = KernelPreset::MovingAverage;
    int kernel_size = 5;
    std::vector<double> kernel_typed;
};

struct FilterAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    FilterRequest request;
    SignalData data;
    FilterCoefficients filter;        // Design: b and the response; Kernel: b = the kernel, response too
    std::vector<double> kernel;       // the taps applied
    std::vector<double> filtered;     // one value per row, centred (zero-phase)
    double rms_original = 0.0, rms_kept = 0.0, rms_removed = 0.0;
    double half_power = NAN;          // Design low / high-pass: where |H| crosses 1 / sqrt 2 (per second / unit / row)
    int edge_rows = 0;                // rows at each end that lean on the edge ((taps - 1) / 2)
};

FilterAnswer RunFilter(const std::string& table, const FilterRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// Pure: a kernel of `size` taps (moving average 1/n; Gaussian with sigma =
// size / 6, normalised; difference [-1, 0, 1] / 2 and second difference
// [1, -2, 1] whatever the size), or the typed list.
std::vector<double> Kernel(KernelPreset preset, int size, const std::vector<double>& typed);

// ---- Wavelet

enum class DenoiseMode { Off, Soft, Hard };

struct WaveletRequest {
    SignalRequest base;
    std::string wavelet = "db4";
    int levels = 0;            // 0: min(4, the maximum)
    DenoiseMode denoise = DenoiseMode::Off;
    double threshold = 0.0;    // 0: the universal threshold
};

struct WaveletPart {
    std::string name;          // "A4", "D4" .. "D1"
    int level = 0;
    bool approximation = false;
    double scale_low = 0.0;    // seconds (or units / rows): the shortest cycle in the part
    double scale_high = 0.0;   // 0 for the approximation ("and longer")
    size_t coefficients = 0;
    double energy_share = 0.0;
    std::vector<double> rows;  // the part brought back to the rows (one value per row)
};

struct WaveletAnswer {
    std::string error;
    bool ok() const { return error.empty(); }
    WaveletRequest request;
    SignalData data;
    WaveletResult coefficients;
    std::vector<WaveletPart> parts;   // the approximation, then the coarsest detail to the finest
    double reconstruction_error = 0.0;  // max |rebuilt - column|
    DenoiseResult denoised;             // when denoising is on
};

WaveletAnswer RunWavelet(const std::string& table, const WaveletRequest& request, const QueryRunner& run, const explore::MissingTexts& texts);

// ---- Into the graph

// The FFT node (a branch: a one-sided spectrum table) with the look's settings.
std::map<std::string, std::string> NodeParameters(const SpectrumRequest& request);
// "FilterDesigner" (Design) or "Convolution1D" (Kernel), and its parameters; both add <col>_filtered beside the columns.
const char* FilterNodeType(const FilterRequest& request);
std::map<std::string, std::string> NodeParameters(const FilterRequest& request);
// The Wavelet Transform node: the parts as columns, and the denoised column when on.
std::map<std::string, std::string> NodeParameters(const WaveletRequest& request);

// ---- Pure helpers (test_signal_explore)

const char* KindName(FilterKind kind);        // "lowpass" ..
const char* PresetName(KernelPreset preset);  // "moving_average" ..
double Rms(const std::vector<double>& values);

}  // namespace cyxwiz::signal
