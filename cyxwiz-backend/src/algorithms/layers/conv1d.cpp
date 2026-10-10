// Conv1D on [L, C, N], on the ArrayFire device only. A sparse gather built
// once per input length takes each output position's dilated window (zero
// padding reads nothing), and each sample's convolution is one batched matmul
// straight into the [L_out, F, N] layout; backward scatters the window
// gradients back with the transposed gather. No unwrap and no reorder:
// ArrayFire 3.10's CUDA backend fails to compile its unwrap and strided-copy
// kernels for these shapes. The CPU is ArrayFire's CPU backend; a build
// without ArrayFire refuses the layer.
#include "cyxwiz/layers/convolution.h"

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

#ifdef CYXWIZ_HAS_ARRAYFIRE

[[noreturn]] void ThrowConv1DDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}

#endif

} // namespace

// For one input length: G [C*k*L_out, L*C] gathers window o, tap j, channel
// c (row (c + C*j) + C*k*o) from input element (o*stride + j*dilation -
// padding, c) (column l + L*c); padding reads nothing. scatter is G^T.
struct Conv1DLayer::DeviceGather {
    size_t length = 0;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array gather;
    af::array scatter;
#endif
};

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
      use_bias_(use_bias),
      gather_(std::make_shared<DeviceGather>()) {
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
    throw std::runtime_error("Conv1D runs on ArrayFire, and this build has no ArrayFire");
#endif

    grad_weights_ = Tensor::Zeros(
        Conv1DWeightShape(out_channels_, in_channels_, kernel_size_));
    if (use_bias_) {
        grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels_)});
    }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

// The gather for this geometry, rebuilt only when the input length changes.
void EnsureConv1DGather(size_t& cached_length, af::array& gather, af::array& scatter,
                        const Conv1DGeometry& geometry, int kernel_size, int stride, int padding, int dilation) {
    if (cached_length == geometry.input_length && !gather.isempty()) return;
    const size_t length = geometry.input_length;
    const size_t channels = geometry.in_channels;
    const size_t taps = static_cast<size_t>(kernel_size);
    const size_t rows = channels * taps * geometry.output_length;
    const size_t columns = length * channels;
    std::vector<int> gather_offsets(rows + 1, 0), gather_columns;
    std::vector<int> scatter_offsets(columns + 1, 0);
    gather_columns.reserve(rows);
    size_t row = 0;
    for (size_t o = 0; o < geometry.output_length; ++o) {
        for (size_t j = 0; j < taps; ++j) {
            const long long position = static_cast<long long>(o) * stride + static_cast<long long>(j) * dilation -
                                       padding;
            for (size_t c = 0; c < channels; ++c, ++row) {
                gather_offsets[row + 1] = gather_offsets[row];
                if (position < 0 || position >= static_cast<long long>(length)) continue;
                const int column = static_cast<int>(static_cast<size_t>(position) + length * c);
                gather_columns.push_back(column);
                ++gather_offsets[row + 1];
                ++scatter_offsets[static_cast<size_t>(column) + 1];
            }
        }
    }
    for (size_t i = 0; i < columns; ++i) scatter_offsets[i + 1] += scatter_offsets[i];
    // The transpose: for each input element, the rows that read it.
    std::vector<int> scatter_rows(gather_columns.size());
    std::vector<int> fill(scatter_offsets.begin(), scatter_offsets.end() - 1);
    for (size_t r = 0; r < rows; ++r) {
        for (int e = gather_offsets[r]; e < gather_offsets[r + 1]; ++e) {
            scatter_rows[static_cast<size_t>(fill[static_cast<size_t>(gather_columns[static_cast<size_t>(e)])]++)] =
                static_cast<int>(r);
        }
    }
    const std::vector<float> ones(gather_columns.size(), 1.0f);
    const dim_t nnz = static_cast<dim_t>(ones.size());
    gather = af::sparse(static_cast<dim_t>(rows), static_cast<dim_t>(columns), nnz, ones.data(),
                              gather_offsets.data(), gather_columns.data(), f32, AF_STORAGE_CSR, afHost);
    scatter = af::sparse(static_cast<dim_t>(columns), static_cast<dim_t>(rows), nnz, ones.data(),
                               scatter_offsets.data(), scatter_rows.data(), f32, AF_STORAGE_CSR, afHost);
    cached_length = length;
}

// [C*k, L_out, N]: column o of sample n is window o, row c + C*j.
af::array Conv1DColumns(const af::array& gather, const Tensor& input, const Conv1DGeometry& geometry) {
    const af::array x = af::moddims(TensorToAf(input), static_cast<dim_t>(geometry.input_length * geometry.in_channels),
                                    static_cast<dim_t>(geometry.batch_size));
    return af::moddims(af::matmul(gather, x), static_cast<dim_t>(geometry.kernel_elements),
                       static_cast<dim_t>(geometry.output_length), static_cast<dim_t>(geometry.batch_size));
}

// Weights [F, C, k] -> [C*k, F] (row c + C*j), batched over N for the matmuls.
af::array Conv1DFilters(const Tensor& weights, const Conv1DGeometry& geometry, int out_channels) {
    const af::array filters = af::transpose(af::moddims(TensorToAf(weights), static_cast<dim_t>(out_channels),
                                                        static_cast<dim_t>(geometry.kernel_elements)));
    return af::tile(filters, 1, 1, static_cast<unsigned>(geometry.batch_size));
}

}  // namespace
#endif

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
    try {
        EnsureConv1DGather(gather_->length, gather_->gather, gather_->scatter, geometry, kernel_size_, stride_,
                           padding_, dilation_);
        // Per sample: [L_out, C*k] x [C*k, F] -> [L_out, F, N], the output layout.
        af::array output = af::matmulTN(Conv1DColumns(gather_->gather, input, geometry),
                                        Conv1DFilters(weights_, geometry, out_channels_));
        if (use_bias_) {
            output += af::tile(af::moddims(TensorToAf(bias_), 1, out_channels_, 1),
                               af::dim4(static_cast<dim_t>(geometry.output_length), 1,
                                        static_cast<dim_t>(geometry.batch_size)));
        }
        output.eval();
        Tensor result = Tensor::FromSemanticArray(output, output_shape);
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowConv1DDeviceError("Conv1DLayer::Forward", e);
    }
#else
    (void)geometry; (void)output_shape;
    throw std::runtime_error("Conv1D runs on ArrayFire, and this build has no ArrayFire");
#endif
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
    try {
        EnsureConv1DGather(gather_->length, gather_->gather, gather_->scatter, geometry, kernel_size_, stride_,
                           padding_, dilation_);
        const af::array columns = Conv1DColumns(gather_->gather, cached_input_, geometry);  // [C*k, L_out, N]
        const af::array filters = Conv1DFilters(weights_, geometry, out_channels_);  // [C*k, F, N]
        const af::array grad = TensorToAf(grad_output);                               // [L_out, F, N]

        // dW: sum over samples of columns . grad -> [C*k, F] -> [F, C, k]
        const af::array grad_filters = af::sum(af::matmul(columns, grad), 2);
        af::array grad_weight = af::moddims(af::transpose(grad_filters), static_cast<dim_t>(out_channels_),
                                            static_cast<dim_t>(in_channels_), static_cast<dim_t>(kernel_size_));
        grad_weight.eval();

        if (use_bias_) {
            af::array grad_bias = af::moddims(af::sum(af::sum(grad, 0), 2), out_channels_);
            grad_bias.eval();
            grad_bias_ = Tensor::FromSemanticArray(grad_bias, {static_cast<size_t>(out_channels_)});
        }

        // dX: each window's gradient [C*k, L_out, N], scattered back by G^T.
        const af::array grad_columns = af::moddims(
            af::matmulNT(filters, grad),
            static_cast<dim_t>(geometry.kernel_elements * geometry.output_length),
            static_cast<dim_t>(geometry.batch_size));
        af::array grad_input = af::moddims(af::matmul(gather_->scatter, grad_columns),
                                           static_cast<dim_t>(geometry.input_length),
                                           static_cast<dim_t>(geometry.in_channels),
                                           static_cast<dim_t>(geometry.batch_size));
        grad_input.eval();

        grad_weights_ = Tensor::FromSemanticArray(
            grad_weight, Conv1DWeightShape(out_channels_, in_channels_, kernel_size_));
        return Tensor::FromSemanticArray(grad_input, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowConv1DDeviceError("Conv1DLayer::Backward", e);
    }
#else
    (void)geometry;
    throw std::runtime_error("Conv1D runs on ArrayFire, and this build has no ArrayFire");
#endif
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
