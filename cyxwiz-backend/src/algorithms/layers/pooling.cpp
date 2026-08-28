#include "cyxwiz/layers/pooling.h"
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

void LogPoolingFallbackOnce(
    const char* operation_name,
    const char* error_message,
    const Tensor& tensor,
    const char* tensor_name) {
    const BackendFallbackReason reason =
        ClassifyArrayFireBackendFallbackReason(error_message);
    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext(tensor_name, tensor.Shape()));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        operation_name,
        reason,
        error_message,
        context);
    const bool log_fallback =
        ShouldLogArrayFireBackendFallbackOnce(
            operation_name, reason, context);
    if (log_fallback) {
        spdlog::warn("{}",
                     BuildArrayFireBackendFallbackMessage(
                         operation_name, reason,
                         reason != BackendFallbackReason::CudaJitParamOverflow,
                         error_message, context));
    }
}

} // namespace
#endif

MaxPool2DLayer::MaxPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride == -1 ? pool_size : stride), padding_(padding) {
    if (pool_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument(
            "MaxPool2D requires positive pool_size/stride and non-negative padding");
    }
}

Tensor MaxPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Apply padding if needed
        if (padding_ > 0) {
            // Pad with -infinity for max pooling
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
            x.eval();
            // Note: For max pooling with zero padding, zeros will participate
            // in max computation but won't affect results if inputs are positive
        }

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        // Use af::unwrap to extract patches, then max
        // unwrap extracts patches into columns
        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));
        af::array indices = af::constant(0, af::dim4(out_h, out_w, channels, batch_size), af::dtype::s32);

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                af::array channel = x(af::span, af::span, c, b);

                // Extract patches using unwrap
                af::array patches = af::unwrap(channel, pool_size_, pool_size_,
                                                stride_, stride_);
                patches.eval();

                // patches shape: [pool_size*pool_size, num_patches]
                // Take max along first dimension
                af::array max_vals, max_idx;
                af::max(max_vals, max_idx, patches, 0);
                max_vals.eval();
                max_idx.eval();

                // Reshape to output spatial dimensions
                max_vals = af::moddims(max_vals, af::dim4(out_h, out_w));
                max_vals.eval();
                af::array reshaped_idx = af::moddims(max_idx, af::dim4(out_h, out_w));
                reshaped_idx.eval();

                output(af::span, af::span, c, b) = max_vals;
                indices(af::span, af::span, c, b) = reshaped_idx;
            }
        }
        output.eval();
        indices.eval();

        const std::vector<size_t> output_shape = {
            static_cast<size_t>(out_h), static_cast<size_t>(out_w),
            static_cast<size_t>(channels), static_cast<size_t>(batch_size)};
        max_indices_ = AfToTensor(indices).Reshape(output_shape);
        return AfToTensor(output).Reshape(output_shape);
    } catch (const af::exception& e) {
        LogPoolingFallbackOnce(
            "MaxPool2DLayer::Forward", e.what(), input, "input");
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "MaxPool2DLayer::Forward");
    ValidatePoolInput(input, "MaxPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(pool_size_) || padded_w < static_cast<size_t>(pool_size_)) {
        throw std::runtime_error("MaxPool2D pool window is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;

    Tensor output({out_h, out_w, channels, batch_size}, DataType::Float32);
    max_indices_ = Tensor({out_h, out_w, channels, batch_size}, DataType::Int32);
    const float* input_data = input.ReadData<float>();
    float* output_data = output.MutableData<float>();
    int32_t* index_data = max_indices_.MutableData<int32_t>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float max_value = -std::numeric_limits<float>::infinity();
                    int32_t max_index = 0;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            const float value = ih >= 0 && iw >= 0 &&
                                                        ih < static_cast<int>(in_h) &&
                                                        iw < static_cast<int>(in_w)
                                                    ? input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                                             static_cast<size_t>(iw),
                                                                             c, b, in_w, channels, batch_size)]
                                                    : 0.0f;
                            if (value > max_value) {
                                max_value = value;
                                max_index = static_cast<int32_t>(ph * pool_size_ + pw);
                            }
                        }
                    }
                    const size_t out_index = Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
                    output_data[out_index] = max_value;
                    index_data[out_index] = max_index;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);
        af::array indices = TensorToAf(max_indices_);

        int in_h = static_cast<int>(x.dims(0));
        int in_w = static_cast<int>(x.dims(1));
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        int out_h = static_cast<int>(grad_out.dims(0));
        int out_w = static_cast<int>(grad_out.dims(1));

        // Suppress unused variable warnings
        (void)in_h;
        (void)in_w;

        // Initialize gradient w.r.t. input
        af::array dx = af::constant(0.0f, x.dims());

        // Scatter gradients back to max positions
        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        // Get the max index within the pool window
                        int idx = indices(oh, ow, c, b).scalar<int>();
                        int pool_h = idx / pool_size_;
                        int pool_w = idx % pool_size_;

                        // Calculate input position
                        int ih = oh * stride_ + pool_h;
                        int iw = ow * stride_ + pool_w;

                        // Add gradient
                        dx(ih, iw, c, b) += grad_out(oh, ow, c, b);
                    }
                }
            }
        }

        dx.eval();
        return AfToTensor(dx).Reshape(cached_input_.Shape());
    } catch (const af::exception& e) {
        LogPoolingFallbackOnce(
            "MaxPool2DLayer::Backward", e.what(), grad_output, "grad_output");
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "MaxPool2DLayer::Backward");
    ValidatePoolInput(cached_input_, "MaxPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("MaxPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 4) {
        throw std::runtime_error("MaxPool2D backward expects [out_h, out_w, C, N] grad_output");
    }
    if (grad_shape[2] != input_shape[2] || grad_shape[3] != input_shape[3] ||
        max_indices_.Shape() != grad_shape) {
        throw std::runtime_error("MaxPool2D backward gradient/cache shape mismatch");
    }

    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t out_h = grad_shape[0];
    const size_t out_w = grad_shape[1];
    Tensor grad_input = Tensor::Zeros(input_shape, DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    const int32_t* index_data = max_indices_.ReadData<int32_t>();
    float* grad_input_data = grad_input.MutableData<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const size_t grad_index = Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
                    const int32_t local_index = index_data[grad_index];
                    const int ph = local_index / pool_size_;
                    const int pw = local_index % pool_size_;
                    const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                    const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                    if (ih >= 0 && iw >= 0 && ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                        grad_input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                    static_cast<size_t>(iw),
                                                    c, b, in_w, channels, batch_size)] += grad_data[grad_index];
                    }
                }
            }
        }
    }

    return grad_input;
}

// ============================================================================
// AvgPool2D Layer Implementation
// ============================================================================

AvgPool2DLayer::AvgPool2DLayer(int pool_size, int stride, int padding)
    : pool_size_(pool_size), stride_(stride == -1 ? pool_size : stride), padding_(padding) {
    if (pool_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument(
            "AvgPool2D requires positive pool_size/stride and non-negative padding");
    }
}

Tensor AvgPool2DLayer::Forward(const Tensor& input) {
    cached_input_ = input;

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array x = TensorToAf(input);

        // Apply padding if needed
        if (padding_ > 0) {
            x = af::pad(x, af::dim4(padding_, padding_, 0, 0),
                        af::dim4(padding_, padding_, 0, 0), AF_PAD_ZERO);
            x.eval();
        }

        dim_t in_h = x.dims(0);
        dim_t in_w = x.dims(1);
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        // Calculate output dimensions
        dim_t out_h = (in_h - pool_size_) / stride_ + 1;
        dim_t out_w = (in_w - pool_size_) / stride_ + 1;

        af::array output = af::constant(0.0f, af::dim4(out_h, out_w, channels, batch_size));

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                af::array channel = x(af::span, af::span, c, b);

                // Extract patches using unwrap
                af::array patches = af::unwrap(channel, pool_size_, pool_size_,
                                                stride_, stride_);
                patches.eval();

                // Take mean along first dimension
                af::array mean_vals = af::mean(patches, 0);
                mean_vals.eval();

                // Reshape to output spatial dimensions
                mean_vals = af::moddims(mean_vals, af::dim4(out_h, out_w));
                mean_vals.eval();

                output(af::span, af::span, c, b) = mean_vals;
            }
        }
        output.eval();

        return AfToTensor(output).Reshape({
            static_cast<size_t>(out_h), static_cast<size_t>(out_w),
            static_cast<size_t>(channels), static_cast<size_t>(batch_size)});
    } catch (const af::exception& e) {
        LogPoolingFallbackOnce(
            "AvgPool2DLayer::Forward", e.what(), input, "input");
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "AvgPool2DLayer::Forward");
    ValidatePoolInput(input, "AvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];
    const size_t padded_h = in_h + static_cast<size_t>(2 * padding_);
    const size_t padded_w = in_w + static_cast<size_t>(2 * padding_);
    if (padded_h < static_cast<size_t>(pool_size_) || padded_w < static_cast<size_t>(pool_size_)) {
        throw std::runtime_error("AvgPool2D pool window is larger than padded input");
    }
    const size_t out_h = (padded_h - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;
    const size_t out_w = (padded_w - static_cast<size_t>(pool_size_)) / static_cast<size_t>(stride_) + 1;

    Tensor output({out_h, out_w, channels, batch_size}, DataType::Float32);
    const float* input_data = input.ReadData<float>();
    float* output_data = output.MutableData<float>();
    const float scale = 1.0f / static_cast<float>(pool_size_ * pool_size_);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            if (ih >= 0 && iw >= 0 &&
                                ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                                sum += input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                              static_cast<size_t>(iw),
                                                              c, b, in_w, channels, batch_size)];
                            }
                        }
                    }
                    output_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)] = sum * scale;
                }
            }
        }
    }

    return output;
}

Tensor AvgPool2DLayer::Backward(const Tensor& grad_output) {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        af::array grad_out = TensorToAf(grad_output);
        af::array x = TensorToAf(cached_input_);

        int in_h = static_cast<int>(x.dims(0));
        int in_w = static_cast<int>(x.dims(1));
        int channels = static_cast<int>(x.dims(2));
        int batch_size = static_cast<int>((x.numdims() > 3) ? x.dims(3) : 1);

        int out_h = static_cast<int>(grad_out.dims(0));
        int out_w = static_cast<int>(grad_out.dims(1));

        // Suppress unused variable warnings
        (void)in_h;
        (void)in_w;

        // For average pooling, gradient is distributed equally
        float scale = 1.0f / (pool_size_ * pool_size_);

        af::array dx = af::constant(0.0f, x.dims());

        for (int c = 0; c < channels; c++) {
            for (int b = 0; b < batch_size; b++) {
                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float grad_val = grad_out(oh, ow, c, b).scalar<float>() * scale;

                        // Distribute gradient to all positions in the pool window
                        for (int ph = 0; ph < pool_size_; ph++) {
                            for (int pw = 0; pw < pool_size_; pw++) {
                                int ih = oh * stride_ + ph;
                                int iw = ow * stride_ + pw;
                                dx(ih, iw, c, b) += grad_val;
                            }
                        }
                    }
                }
            }
        }

        dx.eval();
        return AfToTensor(dx).Reshape(cached_input_.Shape());
    } catch (const af::exception& e) {
        LogPoolingFallbackOnce(
            "AvgPool2DLayer::Backward", e.what(), grad_output, "grad_output");
    }
#endif

    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "AvgPool2DLayer::Backward");
    ValidatePoolInput(cached_input_, "AvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("AvgPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const std::vector<size_t>& grad_shape = grad_output.Shape();
    if (grad_shape.size() != 4 || grad_shape[2] != input_shape[2] || grad_shape[3] != input_shape[3]) {
        throw std::runtime_error("AvgPool2D backward gradient shape mismatch");
    }

    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    const size_t out_h = grad_shape[0];
    const size_t out_w = grad_shape[1];
    Tensor grad_input = Tensor::Zeros(input_shape, DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    float* grad_input_data = grad_input.MutableData<float>();
    const float scale = 1.0f / static_cast<float>(pool_size_ * pool_size_);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    const float grad_value =
                        grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)] * scale;
                    for (int ph = 0; ph < pool_size_; ++ph) {
                        for (int pw = 0; pw < pool_size_; ++pw) {
                            const int ih = static_cast<int>(oh * static_cast<size_t>(stride_)) + ph - padding_;
                            const int iw = static_cast<int>(ow * static_cast<size_t>(stride_)) + pw - padding_;
                            if (ih >= 0 && iw >= 0 &&
                                ih < static_cast<int>(in_h) && iw < static_cast<int>(in_w)) {
                                grad_input_data[Pool4DIndex(static_cast<size_t>(ih),
                                                            static_cast<size_t>(iw),
                                                            c, b, in_w, channels, batch_size)] += grad_value;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// ============================================================================
// GlobalAvgPool2D Layer Implementation
// ============================================================================

Tensor GlobalAvgPool2DLayer::Forward(const Tensor& input) {
    ValidatePoolInput(input, "GlobalAvgPool2D");
    const std::vector<size_t>& shape = input.Shape();
    const size_t in_h = shape[0];
    const size_t in_w = shape[1];
    const size_t channels = shape[2];
    const size_t batch_size = shape[3];

    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("input", shape));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "GlobalAvgPool2DLayer::Forward",
        BackendFallbackReason::UnsupportedOperation,
        "ArrayFire implementation unavailable",
        context);
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "GlobalAvgPool2DLayer::Forward");
    cached_input_ = input;

    Tensor output({channels, batch_size}, DataType::Float32);
    const float* input_data = input.ReadData<float>();
    float* output_data = output.MutableData<float>();
    const float scale = 1.0f / static_cast<float>(in_h * in_w);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (size_t h = 0; h < in_h; ++h) {
                for (size_t w = 0; w < in_w; ++w) {
                    sum += input_data[Pool4DIndex(h, w, c, b, in_w, channels, batch_size)];
                }
            }
            output_data[c * batch_size + b] = sum * scale;
        }
    }

    return output;
}

Tensor GlobalAvgPool2DLayer::Backward(const Tensor& grad_output) {
    ValidatePoolInput(cached_input_, "GlobalAvgPool2D");
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error("GlobalAvgPool2D backward CPU fallback requires Float32 grad_output");
    }
    const std::vector<size_t>& input_shape = cached_input_.Shape();
    const size_t in_h = input_shape[0];
    const size_t in_w = input_shape[1];
    const size_t channels = input_shape[2];
    const size_t batch_size = input_shape[3];
    if (grad_output.Shape() != std::vector<size_t>{channels, batch_size}) {
        throw std::runtime_error("GlobalAvgPool2D backward gradient shape mismatch");
    }

    const std::string context = BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("grad_output", grad_output.Shape()));
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "GlobalAvgPool2DLayer::Backward",
        BackendFallbackReason::UnsupportedOperation,
        "ArrayFire implementation unavailable",
        context);
    const ScopedArrayFireHostSyncAttribution attribution(
        ArrayFireHostSyncCategory::LayerCpuPath,
        "GlobalAvgPool2DLayer::Backward");

    Tensor grad_input(input_shape, DataType::Float32);
    const float* grad_data = grad_output.ReadData<float>();
    float* grad_input_data = grad_input.MutableData<float>();
    const float scale = 1.0f / static_cast<float>(in_h * in_w);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            const float grad_value = grad_data[c * batch_size + b] * scale;
            for (size_t h = 0; h < in_h; ++h) {
                for (size_t w = 0; w < in_w; ++w) {
                    grad_input_data[Pool4DIndex(h, w, c, b, in_w, channels, batch_size)] = grad_value;
                }
            }
        }
    }

    return grad_input;
}

} // namespace cyxwiz
