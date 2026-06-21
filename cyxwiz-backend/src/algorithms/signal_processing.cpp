#include <cyxwiz/signal_processing.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

// Constants
constexpr double PI = 3.14159265358979323846;
constexpr double TWO_PI = 2.0 * PI;

// ============================================================================
// Private Helpers
// ============================================================================

bool SignalProcessing::IsPowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int SignalProcessing::NextPowerOf2(int n) {
    if (n <= 0) return 1;
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

void SignalProcessing::GetWaveletFilters(
    const std::string& wavelet,
    std::vector<double>& low_pass,
    std::vector<double>& high_pass) {

    // Wavelet filter coefficients (scaling filter)
    if (wavelet == "haar" || wavelet == "db1") {
        double c = 1.0 / std::sqrt(2.0);
        low_pass = {c, c};
    } else if (wavelet == "db2") {
        low_pass = {
            0.48296291314469025,
            0.836516303737469,
            0.22414386804185735,
            -0.12940952255092145
        };
    } else if (wavelet == "db3") {
        low_pass = {
            0.3326705529509569,
            0.8068915093133388,
            0.4598775021193313,
            -0.13501102001039084,
            -0.08544127388224149,
            0.035226291882100656
        };
    } else if (wavelet == "db4") {
        low_pass = {
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            -0.02798376941698385,
            -0.18703481171888114,
            0.030841381835986965,
            0.032883011666982945,
            -0.010597401784997278
        };
    } else {
        // Default to Haar
        double c = 1.0 / std::sqrt(2.0);
        low_pass = {c, c};
    }

    // Generate high-pass filter from low-pass (QMF)
    int n = static_cast<int>(low_pass.size());
    high_pass.resize(n);
    for (int i = 0; i < n; i++) {
        high_pass[i] = ((i % 2 == 0) ? 1.0 : -1.0) * low_pass[n - 1 - i];
    }
}

void SignalProcessing::DWTDecompose(
    const std::vector<double>& signal,
    const std::vector<double>& low_pass,
    const std::vector<double>& high_pass,
    std::vector<double>& approx,
    std::vector<double>& detail) {

    int n = static_cast<int>(signal.size());
    int filter_len = static_cast<int>(low_pass.size());
    int out_len = (n + filter_len - 1) / 2;

    approx.resize(out_len, 0.0);
    detail.resize(out_len, 0.0);

    // Convolve and downsample by 2
    for (int i = 0; i < out_len; i++) {
        int idx = i * 2;
        for (int j = 0; j < filter_len; j++) {
            int sig_idx = idx - j;
            if (sig_idx >= 0 && sig_idx < n) {
                approx[i] += low_pass[j] * signal[sig_idx];
                detail[i] += high_pass[j] * signal[sig_idx];
            }
        }
    }
}

std::vector<double> SignalProcessing::DWTReconstruct(
    const std::vector<double>& approx,
    const std::vector<double>& detail,
    const std::vector<double>& low_pass,
    const std::vector<double>& high_pass,
    int original_size) {

    int n = static_cast<int>(approx.size());
    int filter_len = static_cast<int>(low_pass.size());

    // Upsample by 2
    std::vector<double> up_approx(n * 2, 0.0);
    std::vector<double> up_detail(n * 2, 0.0);

    for (int i = 0; i < n; i++) {
        up_approx[i * 2] = approx[i];
        up_detail[i * 2] = detail[i];
    }

    // Synthesis filters (time-reversed)
    std::vector<double> low_synth(low_pass.rbegin(), low_pass.rend());
    std::vector<double> high_synth(high_pass.rbegin(), high_pass.rend());

    // Convolve
    auto conv_approx = Convolve1D(up_approx, low_synth, "same");
    auto conv_detail = Convolve1D(up_detail, high_synth, "same");

    // Sum
    std::vector<double> result(conv_approx.output.size());
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = conv_approx.output[i] + conv_detail.output[i];
    }

    // Trim to original size
    if (static_cast<int>(result.size()) > original_size) {
        result.resize(original_size);
    }

    return result;
}

} // namespace cyxwiz
