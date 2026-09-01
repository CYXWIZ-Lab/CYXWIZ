#include "cyxwiz/layers/convolution.h"

#include "../arrayfire_backend_utils.h"
#include "conv1d_native.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>


namespace cyxwiz {

namespace {

struct Conv1DGeometry {
    size_t input_length;
    size_t in_channels;
    size_t batch_size;
    size_t output_length;
    size_t kernel_elements;
    size_t matrix_columns;
};

std::vector<size_t> Conv1DWeightShape(int out_channels,
                                      int in_channels,
                                      int kernel_size) {
    return {
        static_cast<size_t>(out_channels),
        static_cast<size_t>(in_channels),
        static_cast<size_t>(kernel_size),
    };
}

std::vector<size_t> Conv1DOutputShape(const Conv1DGeometry& geometry,
                                      int out_channels) {
    return {
        geometry.output_length,
        static_cast<size_t>(out_channels),
        geometry.batch_size,
    };
}

Conv1DGeometry ValidateConv1DForwardInput(const Tensor& input,
                                          int in_channels,
                                          int out_channels,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          int dilation,
                                          const Tensor& weights,
                                          const Tensor& bias,
                                          bool use_bias) {
    if (input.GetDataType() != DataType::Float32) {
        throw std::runtime_error("Conv1D requires Float32 input");
    }
    if (input.Shape().size() != 3) {
        throw std::runtime_error("Conv1D expects [L, C, N] input");
    }
    if (input.Shape()[0] == 0 || input.Shape()[1] == 0 ||
        input.Shape()[2] == 0) {
        throw std::runtime_error(
            "Conv1D does not support empty dimensions");
    }
    if (weights.GetDataType() != DataType::Float32 ||
        (use_bias && bias.GetDataType() != DataType::Float32)) {
        throw std::runtime_error("Conv1D requires Float32 parameters");
    }
    if (weights.Shape() !=
        Conv1DWeightShape(out_channels, in_channels, kernel_size)) {
        throw std::runtime_error("Conv1D forward weight shape mismatch");
    }
    if (use_bias &&
        bias.Shape() !=
            std::vector<size_t>{static_cast<size_t>(out_channels)}) {
        throw std::runtime_error("Conv1D forward bias shape mismatch");
    }

    const std::vector<size_t>& shape = input.Shape();
    if (shape[1] != static_cast<size_t>(in_channels)) {
        throw std::runtime_error("Conv1D forward input channel mismatch");
    }

    const size_t dilated_span = CheckedLayerProduct(
        static_cast<size_t>(dilation),
        static_cast<size_t>(kernel_size - 1),
        "Conv1D",
        "effective kernel span");
    if (dilated_span == (std::numeric_limits<size_t>::max)()) {
        throw std::overflow_error("Conv1D effective kernel overflow");
    }
    const size_t effective_kernel = dilated_span + 1;
    const size_t padded_length = CheckedSpatialPaddedExtent(
        shape[0], padding, "Conv1D");
    if (padded_length < effective_kernel) {
        throw std::runtime_error(
            "Conv1D kernel is larger than padded input");
    }

    const size_t output_length =
        (padded_length - effective_kernel) /
            static_cast<size_t>(stride) +
        1;
    const size_t kernel_elements = CheckedLayerProduct(
        static_cast<size_t>(kernel_size),
        shape[1],
        "Conv1D",
        "kernel element count");
    const size_t matrix_columns = CheckedLayerProduct(
        output_length,
        shape[2],
        "Conv1D",
        "matrix column count");

#ifdef CYXWIZ_HAS_ARRAYFIRE
    (void)CheckedIntDim(shape[0], "Conv1D input length");
    (void)CheckedIntDim(shape[1], "Conv1D input channels");
    (void)CheckedIntDim(shape[2], "Conv1D batch size");
    (void)CheckedIntDim(output_length, "Conv1D output length");
    (void)CheckedIntDim(kernel_elements, "Conv1D kernel element count");
    (void)CheckedIntDim(matrix_columns, "Conv1D matrix column count");
#endif

    return {
        shape[0],
        shape[1],
        shape[2],
        output_length,
        kernel_elements,
        matrix_columns,
    };
}

Conv1DGeometry ValidateConv1DBackwardInput(const Tensor& cached_input,
                                           const Tensor& grad_output,
                                           int in_channels,
                                           int out_channels,
                                           int kernel_size,
                                           int stride,
                                           int padding,
                                           int dilation,
                                           const Tensor& weights,
                                           const Tensor& bias,
                                           bool use_bias) {
    const Conv1DGeometry geometry = ValidateConv1DForwardInput(
        cached_input,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        weights,
        bias,
        use_bias);
    if (grad_output.GetDataType() != DataType::Float32) {
        throw std::runtime_error(
            "Conv1D backward requires Float32 grad_output");
    }
    if (grad_output.Shape() != Conv1DOutputShape(geometry, out_channels)) {
        throw std::runtime_error(
            "Conv1D backward gradient shape does not match Forward output");
    }
    return geometry;
}

Conv1DNativeGeometry BuildNativeGeometry(const Conv1DGeometry& geometry) {
    return {
        geometry.input_length,
        geometry.in_channels,
        geometry.batch_size,
        geometry.output_length,
    };
}

Conv1DNativeConfig BuildNativeConfig(int out_channels,
                                     int kernel_size,
                                     int stride,
                                     int padding,
                                     int dilation,
                                     bool use_bias) {
    return {
        static_cast<size_t>(out_channels),
        kernel_size,
        stride,
        padding,
        dilation,
        use_bias,
    };
}

#ifdef CYXWIZ_HAS_ARRAYFIRE

af::array Conv1DColumns(const af::array& input,
                        const Conv1DGeometry& geometry,
                        int kernel_size,
                        int stride,
                        int padding) {
    const af::array input_4d = af::moddims(
        input,
        static_cast<dim_t>(geometry.input_length),
        1,
        static_cast<dim_t>(geometry.in_channels),
        static_cast<dim_t>(geometry.batch_size));
    af::array patches = af::unwrap(
        input_4d,
        kernel_size,
        1,
        stride,
        1,
        padding,
        0);
    patches = af::reorder(patches, 0, 2, 1, 3);
    return af::moddims(
        patches,
        static_cast<dim_t>(geometry.kernel_elements),
        static_cast<dim_t>(geometry.matrix_columns));
}

af::array Conv1DFilters(const Tensor& weights,
                        const Conv1DGeometry& geometry,
                        int out_channels) {
    const af::array reordered = af::reorder(
        TensorToAf(weights), 2, 1, 0, 3);
    return af::moddims(
        reordered,
        static_cast<dim_t>(geometry.kernel_elements),
        static_cast<dim_t>(out_channels));
}

af::array Conv1DGradOutputColumns(const af::array& grad_output,
                                  const Conv1DGeometry& geometry,
                                  int out_channels) {
    const af::array reordered = af::reorder(
        grad_output, 1, 0, 2, 3);
    return af::moddims(
        reordered,
        static_cast<dim_t>(out_channels),
        static_cast<dim_t>(geometry.matrix_columns));
}

#endif

} // namespace

Conv1DLayer::Conv1DLayer(int in_channels,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding,
                         int dilation,
                         bool use_bias)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      use_bias_(use_bias) {
    if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 ||
        stride_ <= 0 || padding_ < 0 || dilation_ <= 0) {
        throw std::invalid_argument(
            "Conv1D requires positive channels/kernel/stride/dilation and "
            "non-negative padding");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    const size_t fan_in_size = CheckedLayerProduct(
        static_cast<size_t>(in_channels_),
        static_cast<size_t>(kernel_size_),
        "Conv1D",
        "fan-in");
    if (fan_in_size > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::overflow_error("Conv1D fan-in exceeds ArrayFire limit");
    }
    weights_ = AfToTensor(XavierUniform(
        static_cast<int>(fan_in_size),
        out_channels_,
        af::dim4(out_channels_, in_channels_, kernel_size_)));
    if (use_bias_) {
        bias_ = AfToTensor(
            af::constant(0.0f, af::dim4(out_channels_)));
    }
#else
    weights_ = Tensor::Random(
        Conv1DWeightShape(out_channels_, in_channels_, kernel_size_));
    if (use_bias_) {
        bias_ = Tensor::Zeros({static_cast<size_t>(out_channels_)});
    }
#endif

    grad_weights_ = Tensor::Zeros(
        Conv1DWeightShape(out_channels_, in_channels_, kernel_size_));
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels_)});
    }
}

Tensor Conv1DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    const Conv1DGeometry geometry = ValidateConv1DForwardInput(
        input,
        in_channels_,
        out_channels_,
        kernel_size_,
        stride_,
        padding_,
        dilation_,
        weights_,
        bias_,
        use_bias_);
    const std::vector<size_t> output_shape =
        Conv1DOutputShape(geometry, out_channels_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (dilation_ != 1) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Forward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire unwrap does not support dilated Conv1D windows",
            input,
            "input");
        use_native_cpu = true;
    } else if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Forward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire unwrap requires padding smaller than kernel size",
            input,
            "input");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   "Conv1DLayer::Forward")) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Forward",
            "forced ArrayFire backend fallback test hook",
            input,
            "input");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const af::array columns = Conv1DColumns(
                TensorToAf(input),
                geometry,
                kernel_size_,
                stride_,
                padding_);
            const af::array filters = Conv1DFilters(
                weights_, geometry, out_channels_);
            const af::array output_columns = af::matmulTN(filters, columns);
            af::array output = af::moddims(
                af::reorder(
                    af::moddims(
                        output_columns,
                        out_channels_,
                        static_cast<dim_t>(geometry.output_length),
                        static_cast<dim_t>(geometry.batch_size)),
                    1,
                    0,
                    2,
                    3),
                static_cast<dim_t>(geometry.output_length),
                out_channels_,
                static_cast<dim_t>(geometry.batch_size));
            if (use_bias_) {
                const af::array bias = af::moddims(
                    TensorToAf(bias_), 1, out_channels_, 1);
                output += af::tile(
                    bias,
                    af::dim4(
                        static_cast<dim_t>(geometry.output_length),
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
                "Conv1DLayer::Forward", e.what(), input, "input");
        }
    }
#else
    RecordLayerArrayFireFallback(
        "Conv1DLayer::Forward",
        BackendFallbackReason::BackendUnavailable,
        "ArrayFire support is not compiled",
        input,
        "input");
#endif

    Tensor output = Conv1DForwardNative(
        input,
        weights_,
        bias_,
        BuildNativeGeometry(geometry),
        BuildNativeConfig(
            out_channels_,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            use_bias_));
    cached_input_ = input;
    has_forward_ = true;
    return output;
}

Tensor Conv1DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error(
            "Conv1DLayer::Backward requires a successful Forward call");
    }
    const Conv1DGeometry geometry = ValidateConv1DBackwardInput(
        cached_input_,
        grad_output,
        in_channels_,
        out_channels_,
        kernel_size_,
        stride_,
        padding_,
        dilation_,
        weights_,
        bias_,
        use_bias_);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    bool use_native_cpu = false;
    if (dilation_ != 1) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Backward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire wrap does not support dilated Conv1D windows",
            grad_output,
            "grad_output");
        use_native_cpu = true;
    } else if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Backward",
            BackendFallbackReason::UnsupportedShape,
            "ArrayFire wrap requires padding smaller than kernel size",
            grad_output,
            "grad_output");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting(
                   "Conv1DLayer::Backward")) {
        RecordLayerArrayFireFallback(
            "Conv1DLayer::Backward",
            "forced ArrayFire backend fallback test hook",
            grad_output,
            "grad_output");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const af::array columns = Conv1DColumns(
                TensorToAf(cached_input_),
                geometry,
                kernel_size_,
                stride_,
                padding_);
            const af::array grad_columns = Conv1DGradOutputColumns(
                TensorToAf(grad_output), geometry, out_channels_);
            const af::array filters = Conv1DFilters(
                weights_, geometry, out_channels_);

            af::array grad_weight = af::reorder(
                af::moddims(
                    af::matmulNT(columns, grad_columns),
                    kernel_size_,
                    in_channels_,
                    out_channels_),
                2,
                1,
                0,
                3);
            grad_weight.eval();

            if (use_bias_) {
                af::array grad_bias = af::sum(
                    af::sum(TensorToAf(grad_output), 0), 2);
                grad_bias = af::moddims(grad_bias, out_channels_);
                grad_bias.eval();
                grad_bias_ = Tensor::FromSemanticArray(
                    grad_bias,
                    {static_cast<size_t>(out_channels_)});
            }

            af::array grad_patches = af::reorder(
                af::moddims(
                    af::matmul(filters, grad_columns),
                    kernel_size_,
                    in_channels_,
                    static_cast<dim_t>(geometry.output_length),
                    static_cast<dim_t>(geometry.batch_size)),
                0,
                2,
                1,
                3);
            af::array grad_input = af::wrap(
                grad_patches,
                static_cast<dim_t>(geometry.input_length),
                1,
                kernel_size_,
                1,
                stride_,
                1,
                padding_,
                0);
            grad_input = af::moddims(
                grad_input,
                static_cast<dim_t>(geometry.input_length),
                static_cast<dim_t>(geometry.in_channels),
                static_cast<dim_t>(geometry.batch_size));
            grad_input.eval();

            grad_weights_ = Tensor::FromSemanticArray(
                grad_weight,
                Conv1DWeightShape(
                    out_channels_, in_channels_, kernel_size_));
            return Tensor::FromSemanticArray(
                grad_input, cached_input_.Shape());
        } catch (const af::exception& e) {
            RecordLayerArrayFireFallback(
                "Conv1DLayer::Backward",
                e.what(),
                grad_output,
                "grad_output");
        }
    }
#else
    RecordLayerArrayFireFallback(
        "Conv1DLayer::Backward",
        BackendFallbackReason::BackendUnavailable,
        "ArrayFire support is not compiled",
        grad_output,
        "grad_output");
#endif

    return Conv1DBackwardNative(
        cached_input_,
        grad_output,
        weights_,
        grad_weights_,
        grad_bias_,
        BuildNativeGeometry(geometry),
        BuildNativeConfig(
            out_channels_,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            use_bias_));
}

std::map<std::string, Tensor> Conv1DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void Conv1DLayer::SetParameters(
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
