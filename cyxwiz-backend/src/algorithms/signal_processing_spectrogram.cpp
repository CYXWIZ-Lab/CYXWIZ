#include <cyxwiz/signal_processing.h>

#include <cmath>
#include <complex>
#include <vector>

namespace cyxwiz {

// ============================================================================
// Spectrogram
// ============================================================================

SpectrogramResult SignalProcessing::ComputeSpectrogram(
    const std::vector<double>& signal,
    int window_size,
    int hop_size,
    double sample_rate,
    const std::string& window_type) {

    SpectrogramResult result;

    if (signal.empty()) {
        result.error_message = "Empty signal";
        return result;
    }

    int n = static_cast<int>(signal.size());

    if (window_size > n) {
        result.error_message = "Window size larger than signal length";
        return result;
    }

    try {
        // Get window function
        std::vector<double> window;
        if (window_type == "hamming") {
            window = HammingWindow(window_size);
        } else if (window_type == "hann") {
            window = HannWindow(window_size);
        } else if (window_type == "blackman") {
            window = BlackmanWindow(window_size);
        } else {
            window = RectangularWindow(window_size);
        }

        // Calculate number of frames
        int num_frames = (n - window_size) / hop_size + 1;
        int num_bins = window_size / 2 + 1;

        result.spectrogram.resize(num_frames, std::vector<double>(num_bins));
        result.times.resize(num_frames);
        result.frequencies.resize(num_bins);

        // Frequency axis
        double freq_resolution = sample_rate / window_size;
        for (int i = 0; i < num_bins; i++) {
            result.frequencies[i] = i * freq_resolution;
        }

        // Compute STFT
        for (int frame = 0; frame < num_frames; frame++) {
            int start = frame * hop_size;
            result.times[frame] = start / sample_rate;

            // Extract and window the frame
            std::vector<double> windowed_frame(window_size);
            for (int i = 0; i < window_size; i++) {
                windowed_frame[i] = signal[start + i] * window[i];
            }

            // Compute FFT
            auto fft_result = FFT(windowed_frame, sample_rate);
            if (!fft_result.success) {
                result.error_message = "FFT failed at frame " + std::to_string(frame);
                return result;
            }

            // Store power spectrum (only positive frequencies)
            for (int i = 0; i < num_bins; i++) {
                double mag = fft_result.magnitude[i];
                result.spectrogram[frame][i] = mag * mag;  // Power
            }
        }

        result.num_frames = num_frames;
        result.num_bins = num_bins;
        result.duration = n / sample_rate;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("Spectrogram computation failed: ") + e.what();
    }

    return result;
}

} // namespace cyxwiz