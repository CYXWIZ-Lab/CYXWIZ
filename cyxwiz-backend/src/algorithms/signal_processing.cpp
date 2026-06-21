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
// FFT Operations
// ============================================================================

FFTResult SignalProcessing::FFT(const std::vector<double>& signal, double sample_rate) {
    FFTResult result;
    result.sample_rate = sample_rate;

    if (signal.empty()) {
        result.error_message = "Empty signal";
        return result;
    }

    int n = static_cast<int>(signal.size());
    result.n = n;

    try {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        // GPU-accelerated FFT using ArrayFire
        af::array sig(n, signal.data());
        af::array fft_result = af::fft(sig);

        // Get complex output
        std::vector<af::cfloat> complex_host(n);
        fft_result.host(complex_host.data());

        result.complex_output.resize(n);
        result.magnitude.resize(n);
        result.phase.resize(n);
        result.frequencies.resize(n);

        double freq_resolution = sample_rate / n;

        for (int i = 0; i < n; i++) {
            result.complex_output[i] = std::complex<double>(
                complex_host[i].real, complex_host[i].imag
            );
            result.magnitude[i] = std::abs(result.complex_output[i]);
            result.phase[i] = std::arg(result.complex_output[i]);

            // Frequency bins
            if (i <= n / 2) {
                result.frequencies[i] = i * freq_resolution;
            } else {
                result.frequencies[i] = (i - n) * freq_resolution;
            }
        }

        result.success = true;
#else
        // CPU fallback using Cooley-Tukey FFT
        // Pad to next power of 2
        int padded_size = NextPowerOf2(n);
        std::vector<std::complex<double>> data(padded_size, {0.0, 0.0});

        for (int i = 0; i < n; i++) {
            data[i] = std::complex<double>(signal[i], 0.0);
        }

        // Bit-reversal permutation
        int bits = static_cast<int>(std::log2(padded_size));
        for (int i = 0; i < padded_size; i++) {
            int j = 0;
            for (int k = 0; k < bits; k++) {
                if (i & (1 << k)) {
                    j |= (1 << (bits - 1 - k));
                }
            }
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }

        // Cooley-Tukey iterative FFT
        for (int len = 2; len <= padded_size; len *= 2) {
            double angle = -TWO_PI / len;
            std::complex<double> wlen(std::cos(angle), std::sin(angle));

            for (int i = 0; i < padded_size; i += len) {
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < len / 2; j++) {
                    std::complex<double> u = data[i + j];
                    std::complex<double> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        // Store results (only original n samples)
        result.complex_output.resize(n);
        result.magnitude.resize(n);
        result.phase.resize(n);
        result.frequencies.resize(n);

        double freq_resolution = sample_rate / n;

        for (int i = 0; i < n; i++) {
            result.complex_output[i] = data[i];
            result.magnitude[i] = std::abs(data[i]);
            result.phase[i] = std::arg(data[i]);

            if (i <= n / 2) {
                result.frequencies[i] = i * freq_resolution;
            } else {
                result.frequencies[i] = (i - n) * freq_resolution;
            }
        }

        result.success = true;
#endif
    } catch (const std::exception& e) {
        result.error_message = std::string("FFT failed: ") + e.what();
    }

    return result;
}

FFT2DResult SignalProcessing::FFT2D(const std::vector<std::vector<double>>& image) {
    FFT2DResult result;

    if (image.empty() || image[0].empty()) {
        result.error_message = "Empty image";
        return result;
    }

    int rows = static_cast<int>(image.size());
    int cols = static_cast<int>(image[0].size());
    result.rows = rows;
    result.cols = cols;

    try {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        // Flatten to 1D and create ArrayFire array
        std::vector<double> flat(rows * cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat[i * cols + j] = image[i][j];
            }
        }

        af::array img(cols, rows, flat.data());
        img = af::transpose(img);
        af::array fft_result = af::fft2(img);

        // Get complex output
        std::vector<af::cfloat> complex_host(rows * cols);
        af::transpose(fft_result).host(complex_host.data());

        result.complex_output.resize(rows, std::vector<std::complex<double>>(cols));
        result.magnitude.resize(rows, std::vector<double>(cols));
        result.phase.resize(rows, std::vector<double>(cols));

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                result.complex_output[i][j] = std::complex<double>(
                    complex_host[idx].real, complex_host[idx].imag
                );
                result.magnitude[i][j] = std::abs(result.complex_output[i][j]);
                result.phase[i][j] = std::arg(result.complex_output[i][j]);
            }
        }

        result.success = true;
#else
        // CPU fallback: Apply 1D FFT to each row, then each column
        result.complex_output.resize(rows, std::vector<std::complex<double>>(cols));
        result.magnitude.resize(rows, std::vector<double>(cols));
        result.phase.resize(rows, std::vector<double>(cols));

        // FFT on rows
        for (int i = 0; i < rows; i++) {
            auto row_fft = FFT(image[i], 1.0);
            if (!row_fft.success) {
                result.error_message = "2D FFT failed on row " + std::to_string(i);
                return result;
            }
            for (int j = 0; j < cols; j++) {
                result.complex_output[i][j] = row_fft.complex_output[j];
            }
        }

        // FFT on columns
        for (int j = 0; j < cols; j++) {
            std::vector<double> col(rows);
            for (int i = 0; i < rows; i++) {
                col[i] = result.complex_output[i][j].real();
            }
            auto col_fft = FFT(col, 1.0);
            if (!col_fft.success) {
                result.error_message = "2D FFT failed on column " + std::to_string(j);
                return result;
            }
            for (int i = 0; i < rows; i++) {
                result.complex_output[i][j] = col_fft.complex_output[i];
                result.magnitude[i][j] = std::abs(result.complex_output[i][j]);
                result.phase[i][j] = std::arg(result.complex_output[i][j]);
            }
        }

        result.success = true;
#endif
    } catch (const std::exception& e) {
        result.error_message = std::string("2D FFT failed: ") + e.what();
    }

    return result;
}

std::vector<double> SignalProcessing::IFFT(const std::vector<std::complex<double>>& spectrum) {
    if (spectrum.empty()) {
        return {};
    }

    int n = static_cast<int>(spectrum.size());

    try {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        // Convert to ArrayFire complex array
        std::vector<af::cfloat> complex_input(n);
        for (int i = 0; i < n; i++) {
            complex_input[i].real = static_cast<float>(spectrum[i].real());
            complex_input[i].imag = static_cast<float>(spectrum[i].imag());
        }

        af::array spec(n, complex_input.data());
        af::array ifft_result = af::ifft(spec);

        std::vector<float> real_output(n);
        af::real(ifft_result).host(real_output.data());

        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = static_cast<double>(real_output[i]);
        }

        return result;
#else
        // CPU fallback: Conjugate, FFT, conjugate, scale
        std::vector<std::complex<double>> conj_spectrum(n);
        for (int i = 0; i < n; i++) {
            conj_spectrum[i] = std::conj(spectrum[i]);
        }

        // Reuse FFT code structure
        int padded_size = NextPowerOf2(n);
        std::vector<std::complex<double>> data(padded_size, {0.0, 0.0});

        for (int i = 0; i < n; i++) {
            data[i] = conj_spectrum[i];
        }

        // Bit-reversal permutation
        int bits = static_cast<int>(std::log2(padded_size));
        for (int i = 0; i < padded_size; i++) {
            int j = 0;
            for (int k = 0; k < bits; k++) {
                if (i & (1 << k)) {
                    j |= (1 << (bits - 1 - k));
                }
            }
            if (i < j) {
                std::swap(data[i], data[j]);
            }
        }

        // Cooley-Tukey iterative FFT
        for (int len = 2; len <= padded_size; len *= 2) {
            double angle = -TWO_PI / len;
            std::complex<double> wlen(std::cos(angle), std::sin(angle));

            for (int i = 0; i < padded_size; i += len) {
                std::complex<double> w(1.0, 0.0);
                for (int j = 0; j < len / 2; j++) {
                    std::complex<double> u = data[i + j];
                    std::complex<double> v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        // Conjugate and scale
        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = std::conj(data[i]).real() / n;
        }

        return result;
#endif
    } catch (...) {
        return {};
    }
}

std::vector<std::vector<double>> SignalProcessing::IFFT2D(
    const std::vector<std::vector<std::complex<double>>>& spectrum) {

    if (spectrum.empty() || spectrum[0].empty()) {
        return {};
    }

    int rows = static_cast<int>(spectrum.size());
    int cols = static_cast<int>(spectrum[0].size());

    // Apply IFFT to columns first, then rows
    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));

    // IFFT on columns
    for (int j = 0; j < cols; j++) {
        std::vector<std::complex<double>> col(rows);
        for (int i = 0; i < rows; i++) {
            col[i] = spectrum[i][j];
        }
        auto col_ifft = IFFT(col);
        for (int i = 0; i < rows; i++) {
            temp[i][j] = std::complex<double>(col_ifft[i], 0.0);
        }
    }

    // IFFT on rows
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        std::vector<std::complex<double>> row(cols);
        for (int j = 0; j < cols; j++) {
            row[j] = temp[i][j];
        }
        auto row_ifft = IFFT(row);
        for (int j = 0; j < cols; j++) {
            result[i][j] = row_ifft[j];
        }
    }

    return result;
}

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
