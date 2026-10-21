#include <cyxwiz/signal_processing.h>

#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cyxwiz {
namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;
} // namespace

// ============================================================================
// Filter Design
// ============================================================================

FilterCoefficients SignalProcessing::DesignLowpass(double cutoff_freq, double sample_rate, int order) {
    FilterCoefficients result;
    result.filter_type = "lowpass";
    result.cutoff_low = cutoff_freq;
    result.sample_rate = sample_rate;
    result.order = order;

    if (cutoff_freq <= 0 || cutoff_freq >= sample_rate / 2) {
        result.error_message = "Cutoff frequency must be between 0 and Nyquist frequency";
        return result;
    }

    try {
        // Design FIR lowpass using windowed sinc
        int N = order * 2 + 1;  // Filter length
        double fc = cutoff_freq / sample_rate;  // Normalized cutoff

        result.b.resize(N);
        auto window = HammingWindow(N);

        for (int i = 0; i < N; i++) {
            int n = i - order;
            if (n == 0) {
                result.b[i] = 2.0 * fc;
            } else {
                result.b[i] = std::sin(TWO_PI * fc * n) / (PI * n);
            }
            result.b[i] *= window[i];
        }

        // Normalize
        double sum = std::accumulate(result.b.begin(), result.b.end(), 0.0);
        for (auto& coef : result.b) {
            coef /= sum;
        }

        result.a = {1.0};  // FIR filter

        // Compute frequency response
        ComputeFrequencyResponse(result);

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Filter design failed: ") + e.what();
    }

    return result;
}

FilterCoefficients SignalProcessing::DesignHighpass(double cutoff_freq, double sample_rate, int order) {
    FilterCoefficients result;
    result.filter_type = "highpass";
    result.cutoff_low = cutoff_freq;
    result.sample_rate = sample_rate;
    result.order = order;

    if (cutoff_freq <= 0 || cutoff_freq >= sample_rate / 2) {
        result.error_message = "Cutoff frequency must be between 0 and Nyquist frequency";
        return result;
    }

    try {
        // Design highpass using spectral inversion of lowpass
        auto lowpass = DesignLowpass(cutoff_freq, sample_rate, order);
        if (!lowpass.success) {
            result.error_message = lowpass.error_message;
            return result;
        }

        result.b = lowpass.b;
        int N = static_cast<int>(result.b.size());

        // Spectral inversion
        for (int i = 0; i < N; i++) {
            result.b[i] = -result.b[i];
        }
        result.b[N / 2] += 1.0;

        result.a = {1.0};

        ComputeFrequencyResponse(result);

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Filter design failed: ") + e.what();
    }

    return result;
}

FilterCoefficients SignalProcessing::DesignBandpass(
    double low_freq, double high_freq, double sample_rate, int order) {

    FilterCoefficients result;
    result.filter_type = "bandpass";
    result.cutoff_low = low_freq;
    result.cutoff_high = high_freq;
    result.sample_rate = sample_rate;
    result.order = order;

    if (low_freq >= high_freq) {
        result.error_message = "Low cutoff must be less than high cutoff";
        return result;
    }

    if (low_freq <= 0 || high_freq >= sample_rate / 2) {
        result.error_message = "Cutoff frequencies must be between 0 and Nyquist frequency";
        return result;
    }

    try {
        // Design bandpass using windowed sinc
        int N = order * 2 + 1;
        double fc_low = low_freq / sample_rate;
        double fc_high = high_freq / sample_rate;

        result.b.resize(N);
        auto window = HammingWindow(N);

        for (int i = 0; i < N; i++) {
            int n = i - order;
            if (n == 0) {
                result.b[i] = 2.0 * (fc_high - fc_low);
            } else {
                double sinc_high = std::sin(TWO_PI * fc_high * n) / (PI * n);
                double sinc_low = std::sin(TWO_PI * fc_low * n) / (PI * n);
                result.b[i] = sinc_high - sinc_low;
            }
            result.b[i] *= window[i];
        }

        result.a = {1.0};

        ComputeFrequencyResponse(result);

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Filter design failed: ") + e.what();
    }

    return result;
}

FilterCoefficients SignalProcessing::DesignBandstop(
    double low_freq, double high_freq, double sample_rate, int order) {

    FilterCoefficients result;
    result.filter_type = "bandstop";
    result.cutoff_low = low_freq;
    result.cutoff_high = high_freq;
    result.sample_rate = sample_rate;
    result.order = order;

    if (low_freq >= high_freq) {
        result.error_message = "Low cutoff must be less than high cutoff";
        return result;
    }

    try {
        // Design bandstop using spectral inversion of bandpass
        auto bandpass = DesignBandpass(low_freq, high_freq, sample_rate, order);
        if (!bandpass.success) {
            result.error_message = bandpass.error_message;
            return result;
        }

        result.b = bandpass.b;
        int N = static_cast<int>(result.b.size());

        // Spectral inversion
        for (int i = 0; i < N; i++) {
            result.b[i] = -result.b[i];
        }
        result.b[N / 2] += 1.0;

        result.a = {1.0};

        ComputeFrequencyResponse(result);

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("Filter design failed: ") + e.what();
    }

    return result;
}

std::vector<double> SignalProcessing::ApplyFilter(
    const std::vector<double>& signal,
    const FilterCoefficients& filter) {

    if (signal.empty() || filter.b.empty()) {
        return signal;
    }

    // FIR filtering using convolution
    auto result = Convolve1D(signal, filter.b, "same");
    return result.output;
}

void SignalProcessing::ComputeFrequencyResponse(FilterCoefficients& filter, int num_points) {
    filter.freq_axis.resize(num_points);
    filter.freq_response_mag.resize(num_points);
    filter.freq_response_phase.resize(num_points);

    double nyquist = filter.sample_rate / 2.0;

    for (int i = 0; i < num_points; i++) {
        double freq = i * nyquist / (num_points - 1);
        filter.freq_axis[i] = freq;

        // Evaluate filter at this frequency
        double omega = TWO_PI * freq / filter.sample_rate;
        std::complex<double> H(0.0, 0.0);

        for (size_t k = 0; k < filter.b.size(); k++) {
            H += filter.b[k] * std::exp(std::complex<double>(0.0, -omega * k));
        }

        filter.freq_response_mag[i] = std::abs(H);
        filter.freq_response_phase[i] = std::arg(H);
    }
}

} // namespace cyxwiz