#include "signal_explore.h"

#include <cyxwiz/statistical_tests.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace cyxwiz::signal {

namespace {

using series::TimeKind;

std::string Number(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

double Mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double StandardDeviation(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = Mean(v);
    double s = 0.0;
    for (const double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// The response's first crossing below 1 / sqrt 2 from the pass side.
double HalfPower(const FilterCoefficients& f, FilterKind kind) {
    const auto& mag = f.freq_response_mag;
    const auto& axis = f.freq_axis;
    if (mag.size() < 2 || mag.size() != axis.size()) return NAN;
    const double target = 1.0 / std::sqrt(2.0);
    const auto cross = [&](size_t a, size_t b) {
        const double m0 = mag[a], m1 = mag[b];
        if (m1 == m0) return axis[b];
        return axis[a] + (target - m0) / (m1 - m0) * (axis[b] - axis[a]);
    };
    if (kind == FilterKind::Lowpass) {
        for (size_t i = 1; i < mag.size(); ++i)
            if (mag[i] < target && mag[i - 1] >= target) return cross(i - 1, i);
    } else if (kind == FilterKind::Highpass) {
        for (size_t i = mag.size() - 1; i > 0; --i)
            if (mag[i - 1] < target && mag[i] >= target) return cross(i, i - 1);
    }
    return NAN;
}

}  // namespace

// ---- the signal and its unit

FrequencyUnit UnitFor(TimeKind kind, double sample_rate) {
    if (kind == TimeKind::Numbers) return {"per unit", 1.0};
    if (kind == TimeKind::Order) return {"per row", 1.0};
    static const std::pair<const char*, double> kUnits[] = {
        {"Hz", 1.0}, {"per minute", 60.0}, {"per hour", 3600.0}, {"per day", series::kDaySeconds},
        {"per week", 7.0 * series::kDaySeconds}, {"per month", 30.436875 * series::kDaySeconds}, {"per year", 365.2425 * series::kDaySeconds}};
    const double nyquist = sample_rate / 2.0;
    for (const auto& [name, seconds] : kUnits)
        if (nyquist * seconds >= 10.0) return {name, seconds};
    return {"per year", 365.2425 * series::kDaySeconds};
}

SignalAnswer ReadSignal(const std::string& table, const SignalRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    SignalAnswer a;
    series::SeriesRequest base;
    base.time = request.time;
    base.value = request.value;
    base.period = 1;  // the period is Series' business; 1 skips its periodogram
    series::SeriesAnswer read = series::ReadSeries(table, base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.data.series = std::move(read.series);
    const series::Series& s = a.data.series;
    if (request.sample_rate > 0.0) {
        a.data.sample_rate = request.sample_rate;
        a.data.rate_from_step = false;
    } else {
        const double step = s.kind == TimeKind::Order ? 1.0 : s.step.seconds;
        if (!(step > 0.0)) return a.error = "The rows are all at the same time; the sample rate cannot be read from them. Type it.", a;
        a.data.sample_rate = 1.0 / step;
        a.data.rate_from_step = true;
    }
    a.data.step_seconds = 1.0 / a.data.sample_rate;
    a.data.unit = UnitFor(s.kind, a.data.sample_rate);
    return a;
}

// ---- Spectrum

SpectrumAnswer RunSpectrum(const std::string& table, const SpectrumRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    SpectrumAnswer a;
    a.request = request;
    SignalAnswer read = ReadSignal(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.data = std::move(read.data);
    a.request.base.sample_rate = a.data.sample_rate;  // resolved (the node needs it)
    const std::vector<double>& x = a.data.series.values;
    const double fs = a.data.sample_rate;
    // The trend out first when asked: a least-squares line on the row index
    // (its residuals have no mean either).
    std::vector<double> detrended;
    if (request.remove_trend) {
        std::vector<double> index(x.size());
        std::iota(index.begin(), index.end(), 0.0);
        const LinearFit fit = StatisticalTests::FitLinear({index}, x, {"row"});
        if (!fit.ok) return a.error = "The trend could not be fitted: " + fit.error, a;
        detrended = fit.residuals;
    }
    const std::vector<double>& y = request.remove_trend ? detrended : x;
    if (request.welch_segment > 0) {
        a.spectrum = SignalProcessing::Welch(y, fs, request.welch_segment, request.window);
    } else {
        a.spectrum = SignalProcessing::Spectrum(y, fs, request.window, request.remove_mean);
    }
    if (!a.spectrum.success) return a.error = a.spectrum.error_message, a;
    // Peaks in the spectrum: by prominence (a twentieth of the tallest bin), the strongest first.
    const std::vector<double>& height = request.welch_segment > 0 ? a.spectrum.psd : a.spectrum.amplitude;
    double tallest = 0.0;
    for (size_t k = 1; k < height.size(); ++k) tallest = std::max(tallest, height[k]);
    if (tallest > 0.0) {
        for (const auto& p : SignalProcessing::FindPeaksByProminence(height, tallest / 20.0, 1)) {
            if (p.index == 0) continue;  // the mean is not a cycle
            a.peaks.push_back({p.index, a.spectrum.frequencies[static_cast<size_t>(p.index)], p.value, p.prominence});
        }
        std::stable_sort(a.peaks.begin(), a.peaks.end(), [](const SpectrumPeak& l, const SpectrumPeak& r) { return l.amplitude > r.amplitude; });
        if (a.peaks.size() > 10) a.peaks.resize(10);
    }
    // Peaks in the rows: prominence in SDs, a gap in rows.
    a.row_sd = StandardDeviation(x);
    a.row_gap = request.peak_gap_rows > 0 ? request.peak_gap_rows : 10;
    if (a.row_sd > 0.0) {
        for (const auto& p : SignalProcessing::FindPeaksByProminence(x, request.peak_prominence_sd * a.row_sd, a.row_gap))
            a.row_peaks.push_back({p.index, a.data.series.t[static_cast<size_t>(p.index)], p.value, p.prominence});
    }
    if (a.row_peaks.size() >= 2) {
        std::vector<double> gaps;
        for (size_t i = 1; i < a.row_peaks.size(); ++i)
            gaps.push_back(static_cast<double>(a.row_peaks[i].index - a.row_peaks[i - 1].index) * a.data.step_seconds);
        a.median_gap_seconds = Median(gaps);
    }
    return a;
}

// ---- Spectrogram

SpectrogramAnswer RunSpectrogram(const std::string& table, const SpectrogramRequest& request, const QueryRunner& run,
                                 const explore::MissingTexts& texts) {
    SpectrogramAnswer a;
    a.request = request;
    SignalAnswer read = ReadSignal(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.data = std::move(read.data);
    a.request.base.sample_rate = a.data.sample_rate;  // resolved (the node needs it)
    const std::vector<double>& x = a.data.series.values;
    const int n = static_cast<int>(x.size());
    if (request.window_rows < 2 || request.window_rows > n)
        return a.error = "The window must be 2 .. " + std::to_string(n) + " rows (" + std::to_string(request.window_rows) + " typed).", a;
    if (request.hop_rows < 1) return a.error = "The hop must be at least 1 row.", a;
    // At most max_frames frames: a longer hop when the series is long (and said).
    a.hop_used = request.hop_rows;
    if (request.max_frames > 0) {
        const int frames = (n - request.window_rows) / a.hop_used + 1;
        if (frames > request.max_frames) {
            a.hop_used = (n - request.window_rows + request.max_frames - 1) / request.max_frames;
            a.hop_raised = true;
        }
    }
    a.spectrogram = SignalProcessing::ComputeSpectrogram(x, request.window_rows, a.hop_used, a.data.sample_rate, request.window);
    if (!a.spectrogram.success) return a.error = a.spectrogram.error_message, a;
    double loudest = 0.0;
    int loudest_bin = 0;
    for (const auto& frame : a.spectrogram.spectrogram)
        for (size_t b = 1; b < frame.size(); ++b)
            if (frame[b] > loudest) {
                loudest = frame[b];
                loudest_bin = static_cast<int>(b);
            }
    a.loudest_db = loudest > 0.0 ? 10.0 * std::log10(loudest) : -300.0;
    a.loudest_frequency = a.spectrogram.frequencies.empty() ? 0.0 : a.spectrogram.frequencies[static_cast<size_t>(loudest_bin)];
    a.floor_db = a.loudest_db - request.range_db;
    return a;
}

// ---- Filter

std::vector<double> Kernel(KernelPreset preset, int size, const std::vector<double>& typed) {
    switch (preset) {
        case KernelPreset::MovingAverage: {
            if (size < 1) return {};
            return std::vector<double>(static_cast<size_t>(size), 1.0 / size);
        }
        case KernelPreset::Gaussian: {
            if (size < 1) return {};
            std::vector<double> k(static_cast<size_t>(size));
            const double sigma = std::max(1e-9, size / 6.0), centre = 0.5 * (size - 1);
            double sum = 0.0;
            for (int i = 0; i < size; ++i) {
                k[static_cast<size_t>(i)] = std::exp(-0.5 * (i - centre) * (i - centre) / (sigma * sigma));
                sum += k[static_cast<size_t>(i)];
            }
            for (double& v : k) v /= sum;
            return k;
        }
        case KernelPreset::Difference: return {-0.5, 0.0, 0.5};
        case KernelPreset::SecondDifference: return {1.0, -2.0, 1.0};
        case KernelPreset::Typed: return typed;
    }
    return {};
}

double Rms(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double s = 0.0;
    for (const double v : values) s += v * v;
    return std::sqrt(s / static_cast<double>(values.size()));
}

FilterAnswer RunFilter(const std::string& table, const FilterRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    FilterAnswer a;
    a.request = request;
    SignalAnswer read = ReadSignal(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.data = std::move(read.data);
    a.request.base.sample_rate = a.data.sample_rate;  // resolved (the node needs it)
    const std::vector<double>& x = a.data.series.values;
    const double fs = a.data.sample_rate, nyquist = fs / 2.0;
    if (request.mode == FilterMode::Design) {
        const double cut = request.cut > 0.0 ? request.cut : nyquist / 10.0;
        a.request.cut = cut;
        const bool band = request.kind == FilterKind::Bandpass || request.kind == FilterKind::Bandstop;
        a.filter = SignalProcessing::DesignFIR(KindName(request.kind), cut, band ? request.cut_high : 0.0, fs, request.taps, request.window);
        if (!a.filter.success) return a.error = a.filter.error_message, a;
        a.kernel = a.filter.b;
        a.half_power = HalfPower(a.filter, request.kind);
    } else {
        a.kernel = Kernel(request.preset, request.kernel_size, request.kernel_typed);
        if (a.kernel.empty()) return a.error = "The kernel is empty: pick a size, or type its values.", a;
        for (const double v : a.kernel)
            if (!std::isfinite(v)) return a.error = "The kernel has a value that is not a number.", a;
        if (a.kernel.size() > x.size()) return a.error = "The kernel is longer than the series.", a;
        a.filter.b = a.kernel;
        a.filter.a = {1.0};
        a.filter.sample_rate = fs;
        a.filter.order = static_cast<int>(a.kernel.size());
        a.filter.filter_type = PresetName(request.preset);
        SignalProcessing::ComputeFrequencyResponse(a.filter);
        a.filter.success = a.filter.error_message.empty();
        if (!a.filter.success) return a.error = a.filter.error_message, a;
    }
    if (a.kernel.size() > x.size()) return a.error = "The filter has more taps than the series has rows.", a;
    a.filtered = SignalProcessing::ApplyFilter(x, a.filter);
    if (a.filtered.size() != x.size()) return a.error = "The filter could not be applied.", a;
    std::vector<double> removed(x.size());
    for (size_t i = 0; i < x.size(); ++i) removed[i] = x[i] - a.filtered[i];
    a.rms_original = Rms(x);
    a.rms_kept = Rms(a.filtered);
    a.rms_removed = Rms(removed);
    a.edge_rows = static_cast<int>((a.kernel.size() - 1) / 2);
    return a;
}

// ---- Wavelet

WaveletAnswer RunWavelet(const std::string& table, const WaveletRequest& request, const QueryRunner& run, const explore::MissingTexts& texts) {
    WaveletAnswer a;
    a.request = request;
    SignalAnswer read = ReadSignal(table, request.base, run, texts);
    if (!read.ok()) return a.error = read.error, a;
    a.data = std::move(read.data);
    a.request.base.sample_rate = a.data.sample_rate;  // resolved (the node needs it)
    const std::vector<double>& x = a.data.series.values;
    const int max_levels = SignalProcessing::WaveletMaxLevel(static_cast<int>(x.size()), request.wavelet);
    if (max_levels < 1) return a.error = "Too few rows for " + request.wavelet + " (" + std::to_string(x.size()) + ").", a;
    const int levels = request.levels > 0 ? request.levels : std::min(4, max_levels);
    a.request.levels = levels;
    a.coefficients = SignalProcessing::DWT(x, request.wavelet, levels);
    if (!a.coefficients.success) return a.error = a.coefficients.error_message, a;
    // Energy shares, and each part alone brought back to the rows.
    double total = 0.0;
    std::vector<double> energy(static_cast<size_t>(levels) + 1, 0.0);
    for (const double c : a.coefficients.approximation) energy[0] += c * c;
    for (int level = 0; level < levels; ++level)
        for (const double c : a.coefficients.details[static_cast<size_t>(level)]) energy[static_cast<size_t>(level) + 1] += c * c;
    for (const double e : energy) total += e;
    const double step = a.data.step_seconds;
    const auto alone = [&](int keep_level) {  // -1: the approximation
        WaveletResult only = a.coefficients;
        if (keep_level >= 0) std::fill(only.approximation.begin(), only.approximation.end(), 0.0);
        for (int level = 0; level < levels; ++level)
            if (level != keep_level) std::fill(only.details[static_cast<size_t>(level)].begin(), only.details[static_cast<size_t>(level)].end(), 0.0);
        return SignalProcessing::IDWT(only);
    };
    WaveletPart approx;
    approx.name = "A" + std::to_string(levels);
    approx.level = levels;
    approx.approximation = true;
    approx.scale_low = step * std::pow(2.0, levels);
    approx.scale_high = 0.0;
    approx.coefficients = a.coefficients.approximation.size();
    approx.energy_share = total > 0.0 ? energy[0] / total : 0.0;
    approx.rows = alone(-1);
    a.parts.push_back(std::move(approx));
    for (int level = levels; level >= 1; --level) {
        WaveletPart part;
        part.name = "D" + std::to_string(level);
        part.level = level;
        part.scale_low = step * std::pow(2.0, level - 1);
        part.scale_high = step * std::pow(2.0, level);
        part.coefficients = a.coefficients.details[static_cast<size_t>(level - 1)].size();
        part.energy_share = total > 0.0 ? energy[static_cast<size_t>(level)] / total : 0.0;
        part.rows = alone(level - 1);
        a.parts.push_back(std::move(part));
    }
    for (const auto& part : a.parts)
        if (part.rows.size() != x.size()) return a.error = "A part could not be brought back to the rows.", a;
    const std::vector<double> rebuilt = SignalProcessing::IDWT(a.coefficients);
    if (rebuilt.size() == x.size())
        for (size_t i = 0; i < x.size(); ++i) a.reconstruction_error = std::max(a.reconstruction_error, std::fabs(rebuilt[i] - x[i]));
    if (request.denoise != DenoiseMode::Off) {
        a.denoised = SignalProcessing::WaveletDenoise(x, request.wavelet, levels, request.threshold, request.denoise == DenoiseMode::Soft);
        if (!a.denoised.success) return a.error = a.denoised.error_message, a;
    }
    return a;
}

// ---- Into the graph

const char* KindName(FilterKind kind) {
    switch (kind) {
        case FilterKind::Lowpass: return "lowpass";
        case FilterKind::Highpass: return "highpass";
        case FilterKind::Bandpass: return "bandpass";
        case FilterKind::Bandstop: return "bandstop";
    }
    return "lowpass";
}

const char* PresetName(KernelPreset preset) {
    switch (preset) {
        case KernelPreset::MovingAverage: return "moving_average";
        case KernelPreset::Gaussian: return "gaussian";
        case KernelPreset::Difference: return "difference";
        case KernelPreset::SecondDifference: return "second_difference";
        case KernelPreset::Typed: return "typed";
    }
    return "typed";
}

std::map<std::string, std::string> NodeParameters(const SpectrumRequest& r) {
    return {{"signal_col", r.base.value}, {"sample_rate", Number(r.base.sample_rate)}, {"window", r.window},
            {"remove_mean", r.remove_mean ? "true" : "false"}};
}

const char* FilterNodeType(const FilterRequest& r) { return r.mode == FilterMode::Design ? "FilterDesigner" : "Convolution1D"; }

std::map<std::string, std::string> NodeParameters(const FilterRequest& r) {
    if (r.mode == FilterMode::Design) {
        return {{"signal_col", r.base.value}, {"filter_type", KindName(r.kind)}, {"cutoff", Number(r.cut)}, {"cutoff_high", Number(r.cut_high)},
                {"sample_rate", Number(r.base.sample_rate)}, {"taps", std::to_string(r.taps)}, {"window", r.window}};
    }
    std::string kernel;
    for (const double v : Kernel(r.preset, r.kernel_size, r.kernel_typed)) kernel += (kernel.empty() ? "" : ",") + Number(v);
    return {{"signal_col", r.base.value}, {"kernel", kernel}};
}

std::map<std::string, std::string> NodeParameters(const WaveletRequest& r) {
    return {{"signal_col", r.base.value}, {"wavelet", r.wavelet}, {"level", std::to_string(r.levels)},
            {"denoise", r.denoise == DenoiseMode::Off ? "off" : r.denoise == DenoiseMode::Soft ? "soft" : "hard"},
            {"threshold", Number(r.threshold)}};
}

}  // namespace cyxwiz::signal
