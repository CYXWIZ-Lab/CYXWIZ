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
// Convolution
// ============================================================================

ConvolutionResult SignalProcessing::Convolve1D(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    const std::string& mode) {

    ConvolutionResult result;

    if (signal.empty() || kernel.empty()) {
        result.error_message = "Empty signal or kernel";
        return result;
    }

    int n = static_cast<int>(signal.size());
    int k = static_cast<int>(kernel.size());

    try {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        af::array sig(n, signal.data());
        af::array kern(k, kernel.data());

        af::convMode af_mode = AF_CONV_DEFAULT;  // "same"
        if (mode == "full") {
            af_mode = AF_CONV_EXPAND;
        }

        af::array conv_result = af::convolve1(sig, kern, af_mode);

        int output_size = static_cast<int>(conv_result.dims(0));
        result.output.resize(output_size);
        conv_result.host(result.output.data());

        // Handle "valid" mode manually
        if (mode == "valid") {
            int valid_size = n - k + 1;
            if (valid_size > 0) {
                int offset = (output_size - valid_size) / 2;
                std::vector<double> valid_output(result.output.begin() + offset,
                                                  result.output.begin() + offset + valid_size);
                result.output = valid_output;
            }
        }

        result.output_size = static_cast<int>(result.output.size());
        result.success = true;
#else
        // CPU fallback
        int full_size = n + k - 1;
        std::vector<double> full_output(full_size, 0.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                full_output[i + j] += signal[i] * kernel[j];
            }
        }

        if (mode == "full") {
            result.output = full_output;
        } else if (mode == "same") {
            int offset = k / 2;
            result.output.resize(n);
            for (int i = 0; i < n; i++) {
                result.output[i] = full_output[i + offset];
            }
        } else if (mode == "valid") {
            int valid_size = n - k + 1;
            if (valid_size > 0) {
                result.output.resize(valid_size);
                for (int i = 0; i < valid_size; i++) {
                    result.output[i] = full_output[i + k - 1];
                }
            }
        }

        result.output_size = static_cast<int>(result.output.size());
        result.success = true;
#endif
    } catch (const std::exception& e) {
        result.error_message = std::string("Convolution failed: ") + e.what();
    }

    return result;
}

Convolution2DResult SignalProcessing::Convolve2D(
    const std::vector<std::vector<double>>& image,
    const std::vector<std::vector<double>>& kernel,
    const std::string& mode) {

    Convolution2DResult result;

    if (image.empty() || image[0].empty() || kernel.empty() || kernel[0].empty()) {
        result.error_message = "Empty image or kernel";
        return result;
    }

    int img_rows = static_cast<int>(image.size());
    int img_cols = static_cast<int>(image[0].size());
    int kern_rows = static_cast<int>(kernel.size());
    int kern_cols = static_cast<int>(kernel[0].size());

    try {
#ifdef CYXWIZ_HAS_ARRAYFIRE
        // Flatten image and kernel
        std::vector<double> img_flat(img_rows * img_cols);
        for (int i = 0; i < img_rows; i++) {
            for (int j = 0; j < img_cols; j++) {
                img_flat[i * img_cols + j] = image[i][j];
            }
        }

        std::vector<double> kern_flat(kern_rows * kern_cols);
        for (int i = 0; i < kern_rows; i++) {
            for (int j = 0; j < kern_cols; j++) {
                kern_flat[i * kern_cols + j] = kernel[i][j];
            }
        }

        af::array img(img_cols, img_rows, img_flat.data());
        img = af::transpose(img);
        af::array kern(kern_cols, kern_rows, kern_flat.data());
        kern = af::transpose(kern);

        af::convMode af_mode = AF_CONV_DEFAULT;
        if (mode == "full") {
            af_mode = AF_CONV_EXPAND;
        }

        af::array conv_result = af::convolve2(img, kern, af_mode);

        int out_rows = static_cast<int>(conv_result.dims(0));
        int out_cols = static_cast<int>(conv_result.dims(1));

        std::vector<double> out_flat(out_rows * out_cols);
        af::transpose(conv_result).host(out_flat.data());

        result.output.resize(out_rows, std::vector<double>(out_cols));
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                result.output[i][j] = out_flat[i * out_cols + j];
            }
        }

        result.rows = out_rows;
        result.cols = out_cols;
        result.success = true;
#else
        // CPU fallback
        int out_rows, out_cols;
        int pad_top = 0, pad_left = 0;

        if (mode == "full") {
            out_rows = img_rows + kern_rows - 1;
            out_cols = img_cols + kern_cols - 1;
            pad_top = kern_rows - 1;
            pad_left = kern_cols - 1;
        } else if (mode == "same") {
            out_rows = img_rows;
            out_cols = img_cols;
            pad_top = kern_rows / 2;
            pad_left = kern_cols / 2;
        } else {  // valid
            out_rows = img_rows - kern_rows + 1;
            out_cols = img_cols - kern_cols + 1;
        }

        if (out_rows <= 0 || out_cols <= 0) {
            result.error_message = "Kernel larger than image for valid mode";
            return result;
        }

        result.output.resize(out_rows, std::vector<double>(out_cols, 0.0));

        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < kern_rows; ki++) {
                    for (int kj = 0; kj < kern_cols; kj++) {
                        int ii = i + ki - pad_top;
                        int jj = j + kj - pad_left;
                        if (ii >= 0 && ii < img_rows && jj >= 0 && jj < img_cols) {
                            sum += image[ii][jj] * kernel[kern_rows - 1 - ki][kern_cols - 1 - kj];
                        }
                    }
                }
                result.output[i][j] = sum;
            }
        }

        result.rows = out_rows;
        result.cols = out_cols;
        result.success = true;
#endif
    } catch (const std::exception& e) {
        result.error_message = std::string("2D Convolution failed: ") + e.what();
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

// ============================================================================
// Window Functions
// ============================================================================

std::vector<double> SignalProcessing::HammingWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.54 - 0.46 * std::cos(TWO_PI * i / (size - 1));
    }
    return window;
}

std::vector<double> SignalProcessing::HannWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.5 * (1.0 - std::cos(TWO_PI * i / (size - 1)));
    }
    return window;
}

std::vector<double> SignalProcessing::BlackmanWindow(int size) {
    std::vector<double> window(size);
    for (int i = 0; i < size; i++) {
        window[i] = 0.42 - 0.5 * std::cos(TWO_PI * i / (size - 1))
                    + 0.08 * std::cos(4.0 * PI * i / (size - 1));
    }
    return window;
}

std::vector<double> SignalProcessing::RectangularWindow(int size) {
    return std::vector<double>(size, 1.0);
}

// ============================================================================
// Signal Generation
// ============================================================================

std::vector<double> SignalProcessing::GenerateSineWave(
    double frequency, double sample_rate, int num_samples, double amplitude, double phase) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        signal[i] = amplitude * std::sin(TWO_PI * frequency * t + phase);
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateSquareWave(
    double frequency, double sample_rate, int num_samples, double amplitude) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;
    double period = 1.0 / frequency;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        double phase = std::fmod(t, period) / period;
        signal[i] = (phase < 0.5) ? amplitude : -amplitude;
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateSawtoothWave(
    double frequency, double sample_rate, int num_samples, double amplitude) {

    std::vector<double> signal(num_samples);
    double dt = 1.0 / sample_rate;
    double period = 1.0 / frequency;

    for (int i = 0; i < num_samples; i++) {
        double t = i * dt;
        double phase = std::fmod(t, period) / period;
        signal[i] = amplitude * (2.0 * phase - 1.0);
    }

    return signal;
}

std::vector<double> SignalProcessing::GenerateWhiteNoise(int num_samples, double amplitude) {
    std::vector<double> signal(num_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-amplitude, amplitude);

    for (int i = 0; i < num_samples; i++) {
        signal[i] = dis(gen);
    }

    return signal;
}

std::vector<double> SignalProcessing::AddNoise(const std::vector<double>& signal, double snr_db) {
    if (signal.empty()) return signal;

    // Calculate signal power
    double signal_power = 0.0;
    for (double s : signal) {
        signal_power += s * s;
    }
    signal_power /= signal.size();

    // Calculate noise power from SNR
    double noise_power = signal_power / std::pow(10.0, snr_db / 10.0);
    double noise_amplitude = std::sqrt(noise_power);

    // Generate noise and add to signal
    auto noise = GenerateWhiteNoise(static_cast<int>(signal.size()), noise_amplitude);
    std::vector<double> noisy_signal(signal.size());

    for (size_t i = 0; i < signal.size(); i++) {
        noisy_signal[i] = signal[i] + noise[i];
    }

    return noisy_signal;
}

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
