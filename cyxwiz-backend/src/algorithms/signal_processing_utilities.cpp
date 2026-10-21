#include <cyxwiz/signal_processing.h>

#include <algorithm>
#include <numeric>

namespace cyxwiz {

// ============================================================================
// Utility
// ============================================================================

std::vector<double> SignalProcessing::ZeroPadToPowerOf2(const std::vector<double>& signal) {
    int n = static_cast<int>(signal.size());
    int padded_size = NextPowerOf2(n);

    std::vector<double> padded(padded_size, 0.0);
    std::copy(signal.begin(), signal.end(), padded.begin());

    return padded;
}

std::vector<double> SignalProcessing::Resample(const std::vector<double>& signal, int target_size) {
    if (signal.empty() || target_size <= 0) return {};

    int n = static_cast<int>(signal.size());
    std::vector<double> resampled(target_size);

    for (int i = 0; i < target_size; i++) {
        double pos = static_cast<double>(i) * (n - 1) / (target_size - 1);
        int idx = static_cast<int>(pos);
        double frac = pos - idx;

        if (idx >= n - 1) {
            resampled[i] = signal[n - 1];
        } else {
            resampled[i] = signal[idx] * (1.0 - frac) + signal[idx + 1] * frac;
        }
    }

    return resampled;
}

std::vector<double> SignalProcessing::Normalize(const std::vector<double>& signal) {
    if (signal.empty()) return signal;

    double max_val = *std::max_element(signal.begin(), signal.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });

    max_val = std::abs(max_val);
    if (max_val < 1e-10) return signal;

    std::vector<double> normalized(signal.size());
    for (size_t i = 0; i < signal.size(); i++) {
        normalized[i] = signal[i] / max_val;
    }

    return normalized;
}

std::vector<double> SignalProcessing::RemoveDC(const std::vector<double>& signal) {
    if (signal.empty()) return signal;

    double mean = std::accumulate(signal.begin(), signal.end(), 0.0) / signal.size();

    std::vector<double> result(signal.size());
    for (size_t i = 0; i < signal.size(); i++) {
        result[i] = signal[i] - mean;
    }

    return result;
}

} // namespace cyxwiz