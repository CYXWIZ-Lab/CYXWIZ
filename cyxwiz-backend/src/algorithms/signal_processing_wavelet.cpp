#include <cyxwiz/signal_processing.h>

#include <vector>

namespace cyxwiz {

// ============================================================================
// Wavelet Transform
// ============================================================================

WaveletResult SignalProcessing::DWT(
    const std::vector<double>& signal,
    const std::string& wavelet,
    int levels) {

    WaveletResult result;
    result.wavelet_name = wavelet;

    if (signal.empty()) {
        result.error_message = "Empty signal";
        return result;
    }

    int n = static_cast<int>(signal.size());
    result.original_size = n;

    if (levels <= 0) {
        result.error_message = "Levels must be positive";
        return result;
    }

    // Check if signal is long enough
    int min_length = static_cast<int>(std::pow(2, levels));
    if (n < min_length) {
        result.error_message = "Signal too short for " + std::to_string(levels) + " levels";
        return result;
    }

    try {
        std::vector<double> low_pass, high_pass;
        GetWaveletFilters(wavelet, low_pass, high_pass);

        result.details.resize(levels);

        std::vector<double> approx = signal;

        // Multi-level decomposition
        for (int level = 0; level < levels; level++) {
            std::vector<double> new_approx, detail;
            DWTDecompose(approx, low_pass, high_pass, new_approx, detail);

            result.details[level] = detail;
            approx = new_approx;
        }

        result.approximation = approx;
        result.levels = levels;
        result.success = true;

    } catch (const std::exception& e) {
        result.error_message = std::string("DWT failed: ") + e.what();
    }

    return result;
}

std::vector<double> SignalProcessing::IDWT(const WaveletResult& coeffs) {
    if (!coeffs.success || coeffs.approximation.empty()) {
        return {};
    }

    try {
        std::vector<double> low_pass, high_pass;
        GetWaveletFilters(coeffs.wavelet_name, low_pass, high_pass);

        std::vector<double> approx = coeffs.approximation;

        // Reconstruct from deepest level
        for (int level = coeffs.levels - 1; level >= 0; level--) {
            int target_size = static_cast<int>(coeffs.details[level].size()) * 2;
            approx = DWTReconstruct(approx, coeffs.details[level], low_pass, high_pass, target_size);
        }

        // Trim to original size
        if (static_cast<int>(approx.size()) > coeffs.original_size) {
            approx.resize(coeffs.original_size);
        }

        return approx;

    } catch (...) {
        return {};
    }
}

} // namespace cyxwiz