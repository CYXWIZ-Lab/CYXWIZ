#include <cyxwiz/signal_processing.h>

#include <complex>

namespace cyxwiz {

// ============================================================================
// Signal Analysis
// ============================================================================

std::vector<SignalProcessing::Peak> SignalProcessing::FindPeaks(
    const std::vector<double>& signal,
    double min_height,
    int min_distance) {

    std::vector<Peak> peaks;

    if (signal.size() < 3) return peaks;

    for (size_t i = 1; i < signal.size() - 1; i++) {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
            if (signal[i] >= min_height) {
                // Check minimum distance from previous peaks
                bool far_enough = true;
                for (const auto& p : peaks) {
                    if (std::abs(static_cast<int>(i) - p.index) < min_distance) {
                        far_enough = false;
                        break;
                    }
                }

                if (far_enough) {
                    Peak p;
                    p.index = static_cast<int>(i);
                    p.value = signal[i];
                    p.frequency = 0.0;  // Caller should set this if applicable
                    peaks.push_back(p);
                }
            }
        }
    }

    return peaks;
}

FFTResult SignalProcessing::PowerSpectralDensity(const std::vector<double>& signal, double sample_rate) {
    auto fft_result = FFT(signal, sample_rate);

    if (fft_result.success) {
        // Convert to power spectral density
        for (size_t i = 0; i < fft_result.magnitude.size(); i++) {
            fft_result.magnitude[i] = fft_result.magnitude[i] * fft_result.magnitude[i] / signal.size();
        }
    }

    return fft_result;
}

} // namespace cyxwiz