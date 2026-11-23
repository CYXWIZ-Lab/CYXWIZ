#include <cyxwiz/signal_processing.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

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

// scipy.signal.find_peaks(x, prominence=p, distance=d), step by step as scipy
// does it (TOFIX129 D7): a sequential host walk over the samples, since each
// peak's bases depend on its neighbours.
std::vector<SignalProcessing::ProminentPeak> SignalProcessing::FindPeaksByProminence(
    const std::vector<double>& signal, double prominence, int distance) {
    std::vector<ProminentPeak> out;
    const int n = static_cast<int>(signal.size());
    if (n < 3) return out;
    const auto& x = signal;
    // 1. Local maxima (_local_maxima_1d): a plateau counts once, at its middle.
    std::vector<int> peaks;
    for (int i = 1; i < n - 1;) {
        if (x[static_cast<size_t>(i - 1)] < x[static_cast<size_t>(i)]) {
            int ahead = i + 1;
            while (ahead < n - 1 && x[static_cast<size_t>(ahead)] == x[static_cast<size_t>(i)]) ++ahead;
            if (x[static_cast<size_t>(ahead)] < x[static_cast<size_t>(i)]) {
                peaks.push_back((i + ahead - 1) / 2);
                i = ahead;
                continue;
            }
            i = ahead;
            continue;
        }
        ++i;
    }
    // 2. Distance (_select_by_peak_distance): the highest peak first keeps
    // its place and drops every peak closer than `distance`; equal heights
    // go by position, the later one first (numpy's argsort reversed).
    if (distance > 1 && peaks.size() > 1) {
        std::vector<size_t> order(peaks.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return x[static_cast<size_t>(peaks[a])] < x[static_cast<size_t>(peaks[b])];
        });
        std::vector<bool> keep(peaks.size(), true);
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const size_t j = *it;
            if (!keep[j]) continue;
            for (size_t k = j; k-- > 0 && peaks[j] - peaks[k] < distance;) keep[k] = false;
            for (size_t k = j + 1; k < peaks.size() && peaks[k] - peaks[j] < distance; ++k) keep[k] = false;
        }
        std::vector<int> kept;
        for (size_t k = 0; k < peaks.size(); ++k)
            if (keep[k]) kept.push_back(peaks[k]);
        peaks.swap(kept);
    }
    // 3. Prominence (_peak_prominences, no wlen): the lowest point on each
    // side before the signal rises above the peak; the drop to the higher base.
    for (const int peak : peaks) {
        const double height = x[static_cast<size_t>(peak)];
        ProminentPeak p;
        p.index = peak;
        p.value = height;
        double left_min = height, right_min = height;
        p.left_base = peak;
        for (int i = peak; i >= 0 && x[static_cast<size_t>(i)] <= height; --i) {
            if (x[static_cast<size_t>(i)] < left_min) {
                left_min = x[static_cast<size_t>(i)];
                p.left_base = i;
            }
        }
        p.right_base = peak;
        for (int i = peak; i < n && x[static_cast<size_t>(i)] <= height; ++i) {
            if (x[static_cast<size_t>(i)] < right_min) {
                right_min = x[static_cast<size_t>(i)];
                p.right_base = i;
            }
        }
        p.prominence = height - std::max(left_min, right_min);
        if (p.prominence >= prominence) out.push_back(p);
    }
    return out;
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
