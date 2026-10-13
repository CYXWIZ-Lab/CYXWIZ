#include <cyxwiz/signal_processing.h>

#include "arrayfire_host_materialization.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <arrayfire.h>

namespace cyxwiz {

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
        af::array sig(n, signal.data());
        af::array kern(k, kernel.data());

        af::convMode af_mode = AF_CONV_DEFAULT;  // "same"
        if (mode == "full") {
            af_mode = AF_CONV_EXPAND;
        }

        af::array conv_result = af::convolve1(sig, kern, af_mode);

        int output_size = static_cast<int>(conv_result.dims(0));
        result.output.resize(output_size);
        MaterializeArrayFireToHost(
            conv_result,
            result.output.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "SignalProcessing::Convolve1D",
            "arrayfire_native");

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
        MaterializeArrayFireToHost(
            af::transpose(conv_result),
            out_flat.data(),
            ArrayFireHostSyncCategory::OutputMaterialization,
            "SignalProcessing::Convolve2D",
            "row_major_2d");

        result.output.resize(out_rows, std::vector<double>(out_cols));
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                result.output[i][j] = out_flat[i * out_cols + j];
            }
        }

        result.rows = out_rows;
        result.cols = out_cols;
        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = std::string("2D Convolution failed: ") + e.what();
    }

    return result;
}

} // namespace cyxwiz
