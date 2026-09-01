#include "cyxwiz/layers/convolution.h"
#include "../arrayfire_backend_utils.h"
#include "conv2d_native.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace cyxwiz {

namespace {

struct Conv2DGeometry {
    size_t in_h;
    size_t in_w;
    size_t in_channels;
    size_t batch_size;
    size_t out_h;
    size_t out_w;
    size_t kernel_elements;
    size_t patch_count;
    size_t matrix_columns;
};

Conv2DGeometry ValidateConv2DForwardInput(const Tensor& input,
                                          int in_channels,
                                          int out_channels,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          const Tensor& weights,
                                          const Tensor& bias,
                                          bool use_bias) {
    constexpr const char* layer_name = "Conv2D";
    ValidateSpatial4DInput(input, layer_name);
    if (weights.GetDataType() != DataType::Float32 ||
        (use_bias && bias.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Conv2D requires Float32 parameters");
    }

    const std::vector<size_t> expected_weight_shape{
        static_cast<size_t>(kernel_size),
        static_cast<size_t>(kernel_size),
        static_cast<size_t>(in_channels),
        static_cast<size_t>(out_channels),
    };
    if (weights.Shape() != expected_weight_shape) {
        throw std::runtime_error("Conv2D forward weight shape mismatch");
    }
    if (use_bias &&
        bias.Shape() !=
            std::vector<size_t>{static_cast<size_t>(out_channels)}) {
        throw std::runtime_error("Conv2D forward bias shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    if (shape[2] != static_cast<size_t>(in_channels)) {
        throw std::runtime_error("Conv2D forward input channel mismatch");
    }

    const size_t padded_h = CheckedSpatialPaddedExtent(
        shape[0], padding, layer_name);
    const size_t padded_w = CheckedSpatialPaddedExtent(
        shape[1], padding, layer_name);
    const size_t kernel = static_cast<size_t>(kernel_size);
    if (padded_h < kernel || padded_w < kernel) {
        throw std::runtime_error(
            "Conv2D kernel is larger than padded input");
    }

    const size_t out_h =
        (padded_h - kernel) / static_cast<size_t>(stride) + 1;
    const size_t out_w =
        (padded_w - kernel) / static_cast<size_t>(stride) + 1;
    const size_t kernel_area = CheckedLayerProduct(
        kernel, kernel, layer_name, "kernel area");
    const size_t kernel_elements = CheckedLayerProduct(
        kernel_area, shape[2], layer_name, "kernel element count");
    const size_t patch_count = CheckedLayerProduct(
        out_h, out_w, layer_name, "patch count");
    const size_t matrix_columns = CheckedLayerProduct(
        patch_count, shape[3], layer_name, "matrix column count");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    (void)CheckedIntDim(shape[0], "Conv2D input height");
    (void)CheckedIntDim(shape[1], "Conv2D input width");
    (void)CheckedIntDim(shape[2], "Conv2D input channels");
    (void)CheckedIntDim(shape[3], "Conv2D batch size");
    (void)CheckedIntDim(out_h, "Conv2D output height");
    (void)CheckedIntDim(out_w, "Conv2D output width");
    (void)CheckedIntDim(kernel_elements, "Conv2D kernel element count");
    (void)CheckedIntDim(patch_count, "Conv2D patch count");
    (void)CheckedIntDim(matrix_columns, "Conv2D matrix column count");
#endif

    return {
        shape[0], shape[1], shape[2], shape[3],
        out_h, out_w, kernel_elements, patch_count, matrix_columns,
    };
}

std::vector<size_t> Conv2DOutputShape(const Conv2DGeometry& geometry,
                                      int out_channels) {
    return {
        geometry.out_h,
        geometry.out_w,
        static_cast<size_t>(out_channels),
        geometry.batch_size,
    };
}

Conv2DNativeGeometry BuildNativeGeometry(
    const Conv2DGeometry& geometry) {
    return {
        geometry.in_h,
        geometry.in_w,
        geometry.in_channels,
        geometry.batch_size,
        geometry.out_h,
        geometry.out_w,
    };
}

Conv2DNativeConfig BuildNativeConfig(int out_channels,
                                     int kernel_size,
                                     int stride,
                                     int padding,
                                     bool use_bias) {
    return {
        static_cast<size_t>(out_channels),
        kernel_size,
        stride,
        padding,
        use_bias,
    };
}

Conv2DGeometry ValidateConv2DBackwardInput(const Tensor& cached_input,
                                           const Tensor& grad_output,
                                           int in_channels,
                                           int out_channels,
                                           int kernel_size,
                                           int stride,
                                           int padding,
                                           const Tensor& weights,
                                           const Tensor& bias,
                                           bool use_bias) {
    const Conv2DGeometry geometry = ValidateConv2DForwardInput(
        cached_input, in_channels, out_channels, kernel_size, stride,
        padding, weights, bias, use_bias);
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "Conv2D backward requires Float32 grad_output");
    }
    if (grad_output.Shape() != Conv2DOutputShape(geometry, out_channels)) {
        throw std::runtime_error(
            "Conv2D backward gradient shape does not match Forward output");
    }
    return geometry;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

af::array Conv2DColumns(const af::array& input,
                        const Conv2DGeometry& geometry,
                        int kernel_size,
                        int stride,
                        int padding) {
    af::array patches = af::unwrap(
        input,
        kernel_size,
        kernel_size,
        stride,
        stride,
        padding,
        padding);
    patches = af::reorder(patches, 0, 2, 1, 3);
    return af::moddims(
        patches,
        static_cast<dim_t>(geometry.kernel_elements),
        static_cast<dim_t>(geometry.matrix_columns));
}

af::array Conv2DGradOutputColumns(const af::array& grad_output,
                                  const Conv2DGeometry& geometry,
                                  int out_channels) {
    af::array reordered = af::reorder(grad_output, 2, 0, 1, 3);
    return af::moddims(
        reordered,
        static_cast<dim_t>(out_channels),
        static_cast<dim_t>(geometry.matrix_columns));
}

#endif

} // namespace

Conv2DLayer::Conv2DLayer(int in_channels,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding,
                         bool use_bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      use_bias_(use_bias) {
    if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 ||
        stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument(
            "Conv2D requires positive channels/kernel/stride and "
            "non-negative padding");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const size_t kernel_area = CheckedLayerProduct(
        static_cast<size_t>(kernel_size_),
        static_cast<size_t>(kernel_size_),
        "Conv2D",
        "kernel area");
    const size_t fan_in_size = CheckedLayerProduct(
        kernel_area,
        static_cast<size_t>(in_channels_),
        "Conv2D",
        "fan-in");
    if (fan_in_size > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::overflow_error("Conv2D fan-in exceeds ArrayFire limit");
    }
    const int fan_in = static_cast<int>(fan_in_size);
    const af::dim4 weight_dims(
        kernel_size_, kernel_size_, in_channels_, out_channels_);
    weights_ = AfToTensor(KaimingUniform(fan_in, weight_dims));
    if (use_bias_) {
        bias_ = AfToTensor(
            af::constant(0.0f, af::dim4(out_channels_)));
    }
#else
    weights_ = Tensor::Random(
        {
            static_cast<size_t>(kernel_size_),
            static_cast<size_t>(kernel_size_),
            static_cast<size_t>(in_channels_),
            static_cast<size_t>(out_channels_),
        });
    if (use_bias_) {
        bias_ = Tensor::Zeros(
            {static_cast<size_t>(out_channels_)});
    }
#endif

    grad_weights_ = Tensor::Zeros(
        {
            static_cast<size_t>(kernel_size_),
            static_cast<size_t>(kernel_size_),
            static_cast<size_t>(in_channels_),
            static_cast<size_t>(out_channels_),
        });
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros(
            {static_cast<size_t>(out_channels_)});
    }
}

Tensor Conv2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    const Conv2DGeometry geometry = ValidateConv2DForwardInput(
        input,
        in_channels_,
        out_channels_,
        kernel_size_,
        stride_,
        padding_,
        weights_,
        bias_,
        use_bias_);
    const std::vector<size_t> output_shape =
        Conv2DOutputShape(geometry, out_channels_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback(
            "Conv2DLayer::Forward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire unwrap requires padding smaller than kernel size",
            input,
            "input");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   "Conv2DLayer::Forward")) {
        RecordLayerArrayFireFallback(
            "Conv2DLayer::Forward",
            "forced ArrayFire backend fallback test hook",
            input,
            "input");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const af::array columns = Conv2DColumns(
                TensorToAf(input),
                geometry,
                kernel_size_,
                stride_,
                padding_);
            const af::array filters = af::moddims(
                TensorToAf(weights_),
                static_cast<dim_t>(geometry.kernel_elements),
                static_cast<dim_t>(out_channels_));
            const af::array output_columns = af::matmulTN(filters, columns);
            af::array output = af::reorder(
                af::moddims(
                    output_columns,
                    static_cast<dim_t>(out_channels_),
                    static_cast<dim_t>(geometry.out_h),
                    static_cast<dim_t>(geometry.out_w),
                    static_cast<dim_t>(geometry.batch_size)),
                1,
                2,
                0,
                3);
            if (use_bias_) {
                const af::array bias = af::moddims(
                    TensorToAf(bias_),
                    1,
                    1,
                    static_cast<dim_t>(out_channels_),
                    1);
                output += af::tile(
                    bias,
                    af::dim4(
                        static_cast<dim_t>(geometry.out_h),
                        static_cast<dim_t>(geometry.out_w),
                        1,
                        static_cast<dim_t>(geometry.batch_size)));
            }
            output.eval();

            Tensor result = Tensor::FromSemanticArray(output, output_shape);
            cached_input_ = input;
            has_forward_ = true;
            return result;
        } catch (const af::exception& e) {
            RecordLayerArrayFireFallback(
                "Conv2DLayer::Forward", e.what(), input, "input");
        }
    }
#else
    RecordLayerArrayFireFallback(
        "Conv2DLayer::Forward",
        BackendFallbackReason::BackendUnavailable,
        "ArrayFire support is not compiled",
        input,
        "input");
#endif

    Tensor output = Conv2DForwardNative(
        input,
        weights_,
        bias_,
        BuildNativeGeometry(geometry),
        BuildNativeConfig(
            out_channels_, kernel_size_, stride_, padding_, use_bias_));
    cached_input_ = input;
    has_forward_ = true;
    return output;
}

Tensor Conv2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "Conv2DLayer::Backward requires a successful Forward call");
    }
    const Conv2DGeometry geometry = ValidateConv2DBackwardInput(
        cached_input_,
        grad_output,
        in_channels_,
        out_channels_,
        kernel_size_,
        stride_,
        padding_,
        weights_,
        bias_,
        use_bias_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback(
            "Conv2DLayer::Backward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire wrap requires padding smaller than kernel size",
            grad_output,
            "grad_output");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   "Conv2DLayer::Backward")) {
        RecordLayerArrayFireFallback(
            "Conv2DLayer::Backward",
            "forced ArrayFire backend fallback test hook",
            grad_output,
            "grad_output");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const af::array columns = Conv2DColumns(
                TensorToAf(cached_input_),
                geometry,
                kernel_size_,
                stride_,
                padding_);
            const af::array grad_columns = Conv2DGradOutputColumns(
                TensorToAf(grad_output), geometry, out_channels_);
            const af::array filters = af::moddims(
                TensorToAf(weights_),
                static_cast<dim_t>(geometry.kernel_elements),
                static_cast<dim_t>(out_channels_));

            af::array grad_weight = af::moddims(
                af::matmulNT(columns, grad_columns),
                kernel_size_,
                kernel_size_,
                in_channels_,
                out_channels_);
            grad_weight.eval();

            if (use_bias_) {
                af::array grad_bias = af::sum(
                    af::sum(af::sum(TensorToAf(grad_output), 0), 1), 3);
                grad_bias = af::moddims(grad_bias, out_channels_);
                grad_bias.eval();
                grad_bias_ = Tensor::FromSemanticArray(
                    grad_bias,
                    {static_cast<size_t>(out_channels_)});
            }

            af::array grad_patches = af::reorder(
                af::moddims(
                    af::matmul(filters, grad_columns),
                    static_cast<dim_t>(
                        geometry.kernel_elements / geometry.in_channels),
                    in_channels_,
                    static_cast<dim_t>(geometry.patch_count),
                    static_cast<dim_t>(geometry.batch_size)),
                0,
                2,
                1,
                3);
            af::array grad_input = af::wrap(
                grad_patches,
                static_cast<dim_t>(geometry.in_h),
                static_cast<dim_t>(geometry.in_w),
                kernel_size_,
                kernel_size_,
                stride_,
                stride_,
                padding_,
                padding_);
            grad_input.eval();

            grad_weights_ = Tensor::FromSemanticArray(
                grad_weight,
                {
                    static_cast<size_t>(kernel_size_),
                    static_cast<size_t>(kernel_size_),
                    static_cast<size_t>(in_channels_),
                    static_cast<size_t>(out_channels_),
                });
            return Tensor::FromSemanticArray(
                grad_input, cached_input_.Shape());
        } catch (const af::exception& e) {
            RecordLayerArrayFireFallback(
                "Conv2DLayer::Backward",
                e.what(),
                grad_output,
                "grad_output");
        }
    }
#else
    RecordLayerArrayFireFallback(
        "Conv2DLayer::Backward",
        BackendFallbackReason::BackendUnavailable,
        "ArrayFire support is not compiled",
        grad_output,
        "grad_output");
#endif

    return Conv2DBackwardNative(
        cached_input_,
        grad_output,
        weights_,
        grad_weights_,
        grad_bias_,
        BuildNativeGeometry(geometry),
        BuildNativeConfig(
            out_channels_, kernel_size_, stride_, padding_, use_bias_));
}

std::map<std::string, Tensor> Conv2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv2DLayer::SetParameters(
    const std::map<std::string, Tensor>& params) {
    bool changed = false;
    if (params.count("weights")) {
        weights_ = params.at("weights");
        changed = true;
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
        changed = true;
    }
    if (changed) {
        has_forward_ = false;
    }
}

} // namespace cyxwiz
