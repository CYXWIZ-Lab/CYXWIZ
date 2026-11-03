#ifdef _WIN32
#define NOMINMAX
#endif

#include <cyxwiz/signal_processing.h>

#include "signal_device.h"

#include <cmath>
#include <complex>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

using namespace signal_detail;

// ============================================================================
// FFT Operations (numpy.fft on the ArrayFire device, TOFIX129 D7)
// ============================================================================

FFTResult SignalProcessing::FFT(const std::vector<double>& signal, double sample_rate) {
    FFTResult result;
    result.sample_rate = sample_rate;
    result.error_message = Refusal(signal, 1, "FFT");
    if (!result.error_message.empty()) return result;
    if (!(sample_rate > 0.0)) {
        result.error_message = "FFT: the sample rate must be positive";
        return result;
    }
    const int n = static_cast<int>(signal.size());
    result.n = n;
    try {
        const af::array spectrum = af::fft(ToDevice(signal));
        result.complex_output = ComplexToHost(spectrum, "SignalProcessing::FFT");
        result.magnitude = ToHost(af::abs(spectrum), "SignalProcessing::FFT");
        result.phase = ToHost(af::arg(spectrum), "SignalProcessing::FFT");
        // numpy.fft.fftfreq(n, 1 / fs): 0 .. (n - 1) / 2 positive, then negative.
        result.frequencies.resize(n);
        for (int i = 0; i < n; ++i)
            result.frequencies[i] = (i <= (n - 1) / 2 ? i : i - n) * sample_rate / n;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::FFT", e);
    }
    return result;
}

FFT2DResult SignalProcessing::FFT2D(const std::vector<std::vector<double>>& image) {
    FFT2DResult result;
    if (image.empty() || image[0].empty()) {
        result.error_message = "FFT2D: the image is empty";
        return result;
    }
    const int rows = static_cast<int>(image.size());
    const int cols = static_cast<int>(image[0].size());
    result.rows = rows;
    result.cols = cols;
    try {
        std::vector<double> flat(static_cast<size_t>(rows) * cols);
        for (int i = 0; i < rows; ++i) {
            if (static_cast<int>(image[i].size()) != cols) {
                result.error_message = "FFT2D: the rows have different lengths";
                return result;
            }
            for (int j = 0; j < cols; ++j) flat[static_cast<size_t>(i) * cols + j] = image[i][j];
        }
        // Row-major on the host is column-major [cols, rows] on the device: transpose.
        const af::array img = af::transpose(af::moddims(ToDevice(flat), cols, rows));
        const af::array spectrum = af::fft2(img);
        const std::vector<std::complex<double>> values = ComplexToHost(af::transpose(spectrum), "SignalProcessing::FFT2D");
        result.complex_output.assign(rows, std::vector<std::complex<double>>(cols));
        result.magnitude.assign(rows, std::vector<double>(cols));
        result.phase.assign(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const std::complex<double> v = values[static_cast<size_t>(i) * cols + j];
                result.complex_output[i][j] = v;
                result.magnitude[i][j] = std::abs(v);
                result.phase[i][j] = std::arg(v);
            }
        }
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::FFT2D", e);
    }
    return result;
}

std::vector<double> SignalProcessing::IFFT(const std::vector<std::complex<double>>& spectrum) {
    if (spectrum.empty()) return {};
    const dim_t n = static_cast<dim_t>(spectrum.size());
    try {
        const af::array back = af::ifftNorm(ComplexToDevice(spectrum), 1.0 / static_cast<double>(n));
        return ToHost(af::real(back), "SignalProcessing::IFFT");
    } catch (const af::exception& e) {
        static_cast<void>(DeviceError("SignalProcessing::IFFT", e));  // the vector API has no place for the message
        return {};
    }
}

std::vector<std::vector<double>> SignalProcessing::IFFT2D(
    const std::vector<std::vector<std::complex<double>>>& spectrum) {
    if (spectrum.empty() || spectrum[0].empty()) return {};
    const int rows = static_cast<int>(spectrum.size());
    const int cols = static_cast<int>(spectrum[0].size());
    try {
        std::vector<std::complex<double>> flat(static_cast<size_t>(rows) * cols);
        for (int i = 0; i < rows; ++i) {
            if (static_cast<int>(spectrum[i].size()) != cols) return {};
            for (int j = 0; j < cols; ++j) flat[static_cast<size_t>(i) * cols + j] = spectrum[i][j];
        }
        const af::array spec = af::transpose(af::moddims(ComplexToDevice(flat), cols, rows));
        const af::array back = af::ifft2Norm(spec, 1.0 / static_cast<double>(static_cast<dim_t>(rows) * cols));
        const std::vector<double> values = ToHost(af::real(af::transpose(back)), "SignalProcessing::IFFT2D");
        std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) out[i][j] = values[static_cast<size_t>(i) * cols + j];
        return out;
    } catch (const af::exception& e) {
        static_cast<void>(DeviceError("SignalProcessing::IFFT2D", e));  // the vector API has no place for the message
        return {};
    }
}

} // namespace cyxwiz
