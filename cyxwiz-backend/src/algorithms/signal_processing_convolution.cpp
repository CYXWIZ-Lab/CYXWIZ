#ifdef _WIN32
#define NOMINMAX
#endif

#include <cyxwiz/signal_processing.h>

#include "signal_device.h"

#include <algorithm>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

using namespace signal_detail;

// ============================================================================
// Convolution (numpy.convolve on the ArrayFire device, TOFIX129 D7)
// ============================================================================

ConvolutionResult SignalProcessing::Convolve1D(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    const std::string& mode) {
    ConvolutionResult result;
    if (signal.empty() || kernel.empty()) {
        result.error_message = "Convolve1D: the signal and the kernel need at least one value each";
        return result;
    }
    if (mode != "full" && mode != "same" && mode != "valid") {
        result.error_message = "Convolve1D: mode must be full, same or valid (got '" + mode + "')";
        return result;
    }
    const dim_t n = static_cast<dim_t>(signal.size());
    const dim_t k = static_cast<dim_t>(kernel.size());
    try {
        // The full convolution (n + k - 1), then numpy's slice for the mode.
        const af::array full = af::convolve1(ToDevice(signal), ToDevice(kernel), AF_CONV_EXPAND);
        const dim_t longer = std::max(n, k), shorter = std::min(n, k);
        dim_t start = 0, length = n + k - 1;
        if (mode == "same") {
            length = longer;
            start = (shorter - 1) / 2;
        } else if (mode == "valid") {
            length = longer - shorter + 1;
            start = shorter - 1;
        }
        result.output = ToHost(full(af::seq(static_cast<double>(start), static_cast<double>(start + length - 1))),
                               "SignalProcessing::Convolve1D");
        result.output_size = static_cast<int>(result.output.size());
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::Convolve1D", e);
    }
    return result;
}

Convolution2DResult SignalProcessing::Convolve2D(
    const std::vector<std::vector<double>>& image,
    const std::vector<std::vector<double>>& kernel,
    const std::string& mode) {
    Convolution2DResult result;
    if (image.empty() || image[0].empty() || kernel.empty() || kernel[0].empty()) {
        result.error_message = "Convolve2D: the image and the kernel need at least one value each";
        return result;
    }
    if (mode != "full" && mode != "same" && mode != "valid") {
        result.error_message = "Convolve2D: mode must be full, same or valid (got '" + mode + "')";
        return result;
    }
    const int img_rows = static_cast<int>(image.size());
    const int img_cols = static_cast<int>(image[0].size());
    const int kern_rows = static_cast<int>(kernel.size());
    const int kern_cols = static_cast<int>(kernel[0].size());
    try {
        std::vector<double> img_flat(static_cast<size_t>(img_rows) * img_cols);
        for (int i = 0; i < img_rows; ++i) {
            if (static_cast<int>(image[i].size()) != img_cols) {
                result.error_message = "Convolve2D: the image rows have different lengths";
                return result;
            }
            for (int j = 0; j < img_cols; ++j) img_flat[static_cast<size_t>(i) * img_cols + j] = image[i][j];
        }
        std::vector<double> kern_flat(static_cast<size_t>(kern_rows) * kern_cols);
        for (int i = 0; i < kern_rows; ++i) {
            if (static_cast<int>(kernel[i].size()) != kern_cols) {
                result.error_message = "Convolve2D: the kernel rows have different lengths";
                return result;
            }
            for (int j = 0; j < kern_cols; ++j) kern_flat[static_cast<size_t>(i) * kern_cols + j] = kernel[i][j];
        }
        // Row-major host arrays are column-major [cols, rows] on the device.
        const af::array img = af::transpose(af::moddims(ToDevice(img_flat), img_cols, img_rows));
        const af::array kern = af::transpose(af::moddims(ToDevice(kern_flat), kern_cols, kern_rows));
        af::array out = af::convolve2(img, kern, mode == "same" ? AF_CONV_DEFAULT : AF_CONV_EXPAND);
        if (mode == "valid") {
            const int rows = img_rows - kern_rows + 1, cols = img_cols - kern_cols + 1;
            if (rows <= 0 || cols <= 0) {
                result.error_message = "Convolve2D: the kernel is larger than the image, so no valid part exists";
                return result;
            }
            out = out(af::seq(kern_rows - 1, kern_rows - 2 + rows), af::seq(kern_cols - 1, kern_cols - 2 + cols));
        }
        const int out_rows = static_cast<int>(out.dims(0)), out_cols = static_cast<int>(out.dims(1));
        const std::vector<double> values = ToHost(af::transpose(out), "SignalProcessing::Convolve2D");
        result.output.assign(out_rows, std::vector<double>(out_cols));
        for (int i = 0; i < out_rows; ++i)
            for (int j = 0; j < out_cols; ++j) result.output[i][j] = values[static_cast<size_t>(i) * out_cols + j];
        result.rows = out_rows;
        result.cols = out_cols;
        result.success = true;
    } catch (const af::exception& e) {
        result.error_message = DeviceError("SignalProcessing::Convolve2D", e);
    }
    return result;
}

} // namespace cyxwiz
