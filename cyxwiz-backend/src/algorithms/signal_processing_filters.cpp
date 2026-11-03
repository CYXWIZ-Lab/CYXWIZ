#ifdef _WIN32
#define NOMINMAX
#endif

#include <cyxwiz/signal_processing.h>

#include "signal_device.h"

#include <cmath>
#include <string>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

using namespace signal_detail;

namespace {
constexpr double PI = 3.14159265358979323846;

double Sinc(double x) { return x == 0.0 ? 1.0 : std::sin(PI * x) / (PI * x); }
}  // namespace

// ============================================================================
// Filter Design: scipy.signal.firwin (TOFIX129 D7)
// ============================================================================

FilterCoefficients SignalProcessing::DesignFIR(const std::string& filter_type, double cutoff_low, double cutoff_high,
                                               double sample_rate, int taps, const std::string& window) {
    FilterCoefficients result;
    result.filter_type = filter_type;
    result.cutoff_low = cutoff_low;
    result.cutoff_high = cutoff_high;
    result.sample_rate = sample_rate;
    result.order = taps;
    result.window = window;
    const bool band = filter_type == "bandpass" || filter_type == "bandstop";
    if (filter_type != "lowpass" && filter_type != "highpass" && !band) {
        result.error_message = "Filter type must be lowpass, highpass, bandpass or bandstop (got '" + filter_type + "')";
        return result;
    }
    if (!(sample_rate > 0.0)) {
        result.error_message = "The sample rate must be positive";
        return result;
    }
    const double nyquist = sample_rate / 2.0;
    if (!(cutoff_low > 0.0) || !(cutoff_low < nyquist)) {
        result.error_message = "The cutoff must lie between 0 and the Nyquist frequency " + std::to_string(nyquist);
        return result;
    }
    if (band && (!(cutoff_high > cutoff_low) || !(cutoff_high < nyquist))) {
        result.error_message = "The upper cutoff must lie between the lower cutoff and the Nyquist frequency " + std::to_string(nyquist);
        return result;
    }
    if (taps < 3) {
        result.error_message = "A filter needs at least 3 taps";
        return result;
    }
    const bool pass_zero = filter_type == "lowpass" || filter_type == "bandstop";
    // scipy: pass_nyquist = (number of cutoffs is odd) xor pass_zero; it needs an odd length.
    const bool pass_nyquist = (band ? false : true) != pass_zero;
    if (pass_nyquist && taps % 2 == 0) {
        result.error_message = "A " + filter_type + " filter needs an odd number of taps (it passes the Nyquist frequency)";
        return result;
    }
    const std::vector<double> win = Window(window, taps, false);
    if (win.empty()) {
        result.error_message = "Unknown window '" + window + "' (hamming, hann, blackman, boxcar)";
        return result;
    }
    // The pass bands in Nyquist units: [0, c] lowpass, [c, 1] highpass, [c1, c2] bandpass, [0, c1] + [c2, 1] bandstop.
    std::vector<double> edges;
    if (pass_zero) edges.push_back(0.0);
    edges.push_back(cutoff_low / nyquist);
    if (band) edges.push_back(cutoff_high / nyquist);
    if (pass_nyquist) edges.push_back(1.0);
    // The windowed sinc of each band (the coefficient table is made on the host).
    const double alpha = 0.5 * (taps - 1);
    std::vector<double> h(static_cast<size_t>(taps), 0.0);
    for (int i = 0; i < taps; ++i) {
        const double m = i - alpha;
        for (size_t b = 0; b + 1 < edges.size(); b += 2) {
            const double left = edges[b], right = edges[b + 1];
            h[static_cast<size_t>(i)] += right * Sinc(right * m) - left * Sinc(left * m);
        }
        h[static_cast<size_t>(i)] *= win[static_cast<size_t>(i)];
    }
    // scipy scale=True: unit gain at the centre of the first pass band (0 for a
    // lowpass, Nyquist for a highpass, the band centre otherwise).
    const double left = edges[0], right = edges[1];
    const double scale_frequency = left == 0.0 ? 0.0 : (right == 1.0 ? 1.0 : 0.5 * (left + right));
    double s = 0.0;
    for (int i = 0; i < taps; ++i) s += h[static_cast<size_t>(i)] * std::cos(PI * (i - alpha) * scale_frequency);
    if (s != 0.0)
        for (double& v : h) v /= s;
    result.b = h;
    result.a = {1.0};
    ComputeFrequencyResponse(result);
    result.success = result.error_message.empty();
    return result;
}

FilterCoefficients SignalProcessing::DesignLowpass(double cutoff_freq, double sample_rate, int taps) {
    return DesignFIR("lowpass", cutoff_freq, 0.0, sample_rate, taps);
}

FilterCoefficients SignalProcessing::DesignHighpass(double cutoff_freq, double sample_rate, int taps) {
    return DesignFIR("highpass", cutoff_freq, 0.0, sample_rate, taps);
}

FilterCoefficients SignalProcessing::DesignBandpass(double low_freq, double high_freq, double sample_rate, int taps) {
    return DesignFIR("bandpass", low_freq, high_freq, sample_rate, taps);
}

FilterCoefficients SignalProcessing::DesignBandstop(double low_freq, double high_freq, double sample_rate, int taps) {
    return DesignFIR("bandstop", low_freq, high_freq, sample_rate, taps);
}

std::vector<double> SignalProcessing::ApplyFilter(const std::vector<double>& signal, const FilterCoefficients& filter) {
    if (signal.empty() || filter.b.empty()) return signal;
    // Zero-phase FIR: the kernel centred on each sample (numpy.convolve "same").
    const ConvolutionResult out = Convolve1D(signal, filter.b, "same");
    return out.success ? out.output : signal;
}

void SignalProcessing::ComputeFrequencyResponse(FilterCoefficients& filter, int num_points) {
    filter.freq_axis.clear();
    filter.freq_response_mag.clear();
    filter.freq_response_phase.clear();
    if (filter.b.empty() || num_points < 2 || !(filter.sample_rate > 0.0)) return;
    const int taps = static_cast<int>(filter.b.size());
    // H(w) = sum_k b_k e^{-i w k} on w = 0 .. pi inclusive (scipy.signal.freqz on
    // that grid): the cosine and sine tables are one outer product each, the
    // sums one matrix product each, on the device.
    std::vector<double> omega(static_cast<size_t>(num_points)), index(static_cast<size_t>(taps));
    for (int i = 0; i < num_points; ++i) omega[static_cast<size_t>(i)] = PI * i / (num_points - 1);
    for (int k = 0; k < taps; ++k) index[static_cast<size_t>(k)] = k;
    try {
        const af::array phase = af::matmul(ToDevice(omega), af::transpose(ToDevice(index)));  // [points, taps]
        const af::array b = ToDevice(filter.b);
        const af::array re = af::matmul(af::cos(phase), b);
        const af::array im = af::matmul(-af::sin(phase), b);
        filter.freq_response_mag = ToHost(af::sqrt(re * re + im * im), "SignalProcessing::ComputeFrequencyResponse");
        filter.freq_response_phase = ToHost(af::atan2(im, re), "SignalProcessing::ComputeFrequencyResponse");
    } catch (const af::exception& e) {
        filter.error_message = DeviceError("SignalProcessing::ComputeFrequencyResponse", e);
        return;
    }
    filter.freq_axis.resize(static_cast<size_t>(num_points));
    for (int i = 0; i < num_points; ++i) filter.freq_axis[static_cast<size_t>(i)] = omega[static_cast<size_t>(i)] * filter.sample_rate / (2.0 * PI);
}

} // namespace cyxwiz
