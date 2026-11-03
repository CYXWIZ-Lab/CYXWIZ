#ifdef _WIN32
#define NOMINMAX
#endif

#include <cyxwiz/signal_processing.h>

#include "signal_device.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

using namespace signal_detail;

// ============================================================================
// One-sided spectra (numpy.fft.rfft and scipy.signal.periodogram / welch on
// the ArrayFire device, TOFIX129 D7)
// ============================================================================

namespace {

// |X|^2 of the first half + 1 rows of a batched FFT, scaled to a density
// (scipy: 1 / (fs sum w^2)), the interior bins doubled for the mirror half.
af::array OneSidedDensity(const af::array& spectrum, int nperseg, double sample_rate, double window_energy) {
    const dim_t bins = nperseg / 2 + 1;
    af::array power = af::real(spectrum * af::conjg(spectrum))(af::seq(0, static_cast<double>(bins - 1)), af::span);
    power = power / (sample_rate * window_energy);
    if (bins > 2) {
        const dim_t last = nperseg % 2 == 0 ? bins - 2 : bins - 1;  // Nyquist stays single when n is even
        power(af::seq(1, static_cast<double>(last)), af::span) *= 2.0;
    }
    return power;
}

}  // namespace

SpectrumResult SignalProcessing::Spectrum(const std::vector<double>& signal, double sample_rate, const std::string& window,
                                          bool remove_mean) {
    SpectrumResult result;
    result.sample_rate = sample_rate;
    result.window = window;
    result.error_message = Refusal(signal, 2, "Spectrum");
    if (!result.error_message.empty()) return result;
    if (!(sample_rate > 0.0)) {
        result.error_message = "Spectrum: the sample rate must be positive";
        return result;
    }
    const int n = static_cast<int>(signal.size());
    const std::vector<double> w = Window(window, n, true);
    if (w.empty()) {
        result.error_message = "Spectrum: unknown window '" + window + "' (hann, hamming, blackman, boxcar)";
        return result;
    }
    result.n = n;
    try {
        af::array x = ToDevice(signal);
        if (remove_mean) x = x - af::mean<double>(x);
        const af::array spectrum = af::fft(x * ToDevice(w));
        const dim_t bins = n / 2 + 1;
        af::array amplitude = af::abs(spectrum)(af::seq(0, static_cast<double>(bins - 1))) * (2.0 / SumOf(w));
        amplitude(0) /= 2.0;
        if (n % 2 == 0) amplitude(static_cast<int>(bins - 1)) /= 2.0;
        result.amplitude = ToHost(amplitude, "SignalProcessing::Spectrum");
        result.complex_output = ComplexToHost(spectrum(af::seq(0, static_cast<double>(bins - 1))), "SignalProcessing::Spectrum");
        result.psd = ToHost(OneSidedDensity(spectrum, n, sample_rate, SumOfSquares(w)), "SignalProcessing::Spectrum");
        result.frequencies.resize(static_cast<size_t>(bins));
        for (dim_t k = 0; k < bins; ++k) result.frequencies[static_cast<size_t>(k)] = static_cast<double>(k) * sample_rate / n;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::Spectrum", e);
    }
    return result;
}

SpectrumResult SignalProcessing::Welch(const std::vector<double>& signal, double sample_rate, int nperseg, const std::string& window) {
    SpectrumResult result;
    result.sample_rate = sample_rate;
    result.window = window;
    result.error_message = Refusal(signal, 2, "Welch");
    if (!result.error_message.empty()) return result;
    if (!(sample_rate > 0.0)) {
        result.error_message = "Welch: the sample rate must be positive";
        return result;
    }
    const int n = static_cast<int>(signal.size());
    if (nperseg < 2) {
        result.error_message = "Welch: a segment needs at least 2 samples";
        return result;
    }
    nperseg = std::min(nperseg, n);  // scipy: a segment longer than the signal becomes the signal
    const std::vector<double> w = Window(window, nperseg, true);
    if (w.empty()) {
        result.error_message = "Welch: unknown window '" + window + "' (hann, hamming, blackman, boxcar)";
        return result;
    }
    const int hop = nperseg - nperseg / 2;
    result.n = nperseg;
    try {
        // Every segment as a column, its mean removed, windowed, one batched FFT.
        af::array frames = af::unwrap(ToDevice(signal), nperseg, 1, hop, 1, 0, 0, true);
        const dim_t segments = frames.dims(1);
        frames = frames - af::tile(af::mean(frames, 0), nperseg, 1);
        frames = frames * af::tile(ToDevice(w), 1, static_cast<unsigned>(segments));
        const af::array spectrum = af::fft(frames);
        const af::array density = OneSidedDensity(spectrum, nperseg, sample_rate, SumOfSquares(w));
        result.psd = ToHost(af::mean(density, 1), "SignalProcessing::Welch");
        result.segments = static_cast<int>(segments);
        const dim_t bins = nperseg / 2 + 1;
        result.frequencies.resize(static_cast<size_t>(bins));
        for (dim_t k = 0; k < bins; ++k) result.frequencies[static_cast<size_t>(k)] = static_cast<double>(k) * sample_rate / nperseg;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::Welch", e);
    }
    return result;
}

// ============================================================================
// Spectrogram: scipy.signal.spectrogram (psd, density), one batched FFT
// ============================================================================

SpectrogramResult SignalProcessing::ComputeSpectrogram(
    const std::vector<double>& signal,
    int window_size,
    int hop_size,
    double sample_rate,
    const std::string& window_type) {
    SpectrogramResult result;
    result.error_message = Refusal(signal, 2, "Spectrogram");
    if (!result.error_message.empty()) return result;
    const int n = static_cast<int>(signal.size());
    if (!(sample_rate > 0.0)) {
        result.error_message = "Spectrogram: the sample rate must be positive";
        return result;
    }
    if (window_size < 2 || window_size > n) {
        result.error_message = "Spectrogram: the window must be 2 .. " + std::to_string(n) + " samples (" +
                               std::to_string(window_size) + " asked)";
        return result;
    }
    if (hop_size < 1) {
        result.error_message = "Spectrogram: the hop must be at least 1 sample";
        return result;
    }
    const std::vector<double> w = Window(window_type, window_size, true);
    if (w.empty()) {
        result.error_message = "Spectrogram: unknown window '" + window_type + "' (hann, hamming, blackman, boxcar)";
        return result;
    }
    try {
        af::array frames = af::unwrap(ToDevice(signal), window_size, 1, hop_size, 1, 0, 0, true);
        const dim_t num_frames = frames.dims(1);
        frames = frames * af::tile(ToDevice(w), 1, static_cast<unsigned>(num_frames));
        const af::array density = OneSidedDensity(af::fft(frames), window_size, sample_rate, SumOfSquares(w));
        const dim_t bins = window_size / 2 + 1;
        const std::vector<double> values = ToHost(density, "SignalProcessing::Spectrogram");  // column-major: bin fastest
        result.spectrogram.assign(static_cast<size_t>(num_frames), std::vector<double>(static_cast<size_t>(bins)));
        for (dim_t f = 0; f < num_frames; ++f)
            for (dim_t b = 0; b < bins; ++b)
                result.spectrogram[static_cast<size_t>(f)][static_cast<size_t>(b)] = values[static_cast<size_t>(f * bins + b)];
        result.frequencies.resize(static_cast<size_t>(bins));
        for (dim_t k = 0; k < bins; ++k) result.frequencies[static_cast<size_t>(k)] = static_cast<double>(k) * sample_rate / window_size;
        // Frame centres, as scipy: (nperseg / 2 + i hop) / fs.
        result.times.resize(static_cast<size_t>(num_frames));
        for (dim_t f = 0; f < num_frames; ++f)
            result.times[static_cast<size_t>(f)] = (window_size / 2.0 + static_cast<double>(f) * hop_size) / sample_rate;
        result.num_frames = static_cast<int>(num_frames);
        result.num_bins = static_cast<int>(bins);
        result.duration = n / sample_rate;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::Spectrogram", e);
    }
    return result;
}

} // namespace cyxwiz
