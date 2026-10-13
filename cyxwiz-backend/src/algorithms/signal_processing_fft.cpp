#include <cyxwiz/signal_processing.h>

#include "arrayfire_host_materialization.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {
namespace {
} // namespace

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
        // GPU-accelerated FFT using ArrayFire
        af::array sig(n, signal.data());
        af::array fft_result = af::fft(sig);

        // Get complex output
        std::vector<af::cfloat> complex_host(n);
        MaterializeArrayFireToHost(
            fft_result,
            complex_host.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "SignalProcessing::FFT",
            "arrayfire_native");

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
        MaterializeArrayFireToHost(
            af::transpose(fft_result),
            complex_host.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "SignalProcessing::FFT2D",
            "row_major_2d");

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
        // Convert to ArrayFire complex array
        std::vector<af::cfloat> complex_input(n);
        for (int i = 0; i < n; i++) {
            complex_input[i].real = static_cast<float>(spectrum[i].real());
            complex_input[i].imag = static_cast<float>(spectrum[i].imag());
        }

        af::array spec(n, complex_input.data());
        af::array ifft_result = af::ifft(spec);

        std::vector<float> real_output(n);
        MaterializeArrayFireToHost(
            af::real(ifft_result),
            real_output.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "SignalProcessing::IFFT",
            "arrayfire_native");

        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = static_cast<double>(real_output[i]);
        }

        return result;
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

} // namespace cyxwiz
