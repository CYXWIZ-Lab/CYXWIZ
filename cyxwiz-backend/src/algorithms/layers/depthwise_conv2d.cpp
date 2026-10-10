// DepthwiseConv2D (TOFIX140 B): torch Conv2d(C, C*M, k, groups=C) on
// [H,W,C,N], on the ArrayFire device only: the input is zero-padded, each
// channel's windows unwrapped ([k*k, L, C, N]), weighted by that channel's M
// kernels and summed over the window; backward mirrors it, folds the window
// gradients back with wrap and crops the padding. The CPU is ArrayFire's CPU
// backend; a build without ArrayFire refuses the layer.
#include "cyxwiz/layers/convolution.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

namespace {

struct DepthwiseGeometry {
    size_t in_h;
    size_t in_w;
    size_t channels;
    size_t batch_size;
    size_t out_h;
    size_t out_w;
};

DepthwiseGeometry ValidateDepthwiseInput(const Tensor& input, int channels, int kernel, int stride,
                                         int padding) {
    ValidateSpatial4DInput(input, "DepthwiseConv2D");
    const auto& shape = input.Shape();
    if (shape[2] != static_cast<size_t>(channels)) {
        throw std::invalid_argument("DepthwiseConv2D input has " + std::to_string(shape[2]) +
                                    " channels, the layer " + std::to_string(channels));
    }
    const size_t padded_h = CheckedSpatialPaddedExtent(shape[0], padding, "DepthwiseConv2D");
    const size_t padded_w = CheckedSpatialPaddedExtent(shape[1], padding, "DepthwiseConv2D");
    const size_t k = static_cast<size_t>(kernel);
    if (padded_h < k || padded_w < k) {
        throw std::runtime_error("DepthwiseConv2D kernel is larger than the padded input");
    }
    const size_t s = static_cast<size_t>(stride);
    return {shape[0], shape[1], shape[2], shape[3], (padded_h - k) / s + 1, (padded_w - k) / s + 1};
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] void ThrowDepthwiseDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}

// Zero padding on H and W done explicitly, so unwrap needs none (unwrap
// refuses padding >= kernel; torch allows it).
af::array PadSpatial(const af::array& x, int padding) {
    if (padding == 0) return x;
    const dim_t p = padding;
    return af::pad(x, af::dim4(p, p, 0, 0), af::dim4(p, p, 0, 0), AF_PAD_ZERO);
}
#else
[[noreturn]] void ThrowDepthwiseNeedsArrayFire() {
    throw std::runtime_error("Depthwise Conv2D runs on ArrayFire, and this build has no ArrayFire");
}
#endif

}  // namespace

DepthwiseConv2DLayer::DepthwiseConv2DLayer(int channels, int depth_multiplier, int kernel_size,
                                           int stride, int padding, bool use_bias)
    : channels_(channels),
      multiplier_(depth_multiplier),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      use_bias_(use_bias) {
    if (channels <= 0 || depth_multiplier <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
        throw std::invalid_argument(
            "DepthwiseConv2D needs positive channels, depth multiplier, kernel and stride, and padding >= 0");
    }
    const size_t k = static_cast<size_t>(kernel_size);
    const size_t outputs = static_cast<size_t>(channels) * static_cast<size_t>(depth_multiplier);
    const std::vector<size_t> weight_shape{k, k, 1, outputs};
    // As Conv2D: Kaiming uniform over the fan-in (k*k: one input channel per group).
#ifdef CYXWIZ_HAS_ARRAYFIRE
    weights_ = AfToTensor(KaimingUniform(static_cast<int>(k * k),
                                         af::dim4(kernel_size, kernel_size, 1, static_cast<dim_t>(outputs))));
#else
    ThrowDepthwiseNeedsArrayFire();
#endif
    if (use_bias_) bias_ = Tensor::Zeros({outputs}, DataType::Float32);
    grad_weights_ = Tensor::Zeros(weight_shape, DataType::Float32);
    if (use_bias_) grad_bias_ = Tensor::Zeros({outputs}, DataType::Float32);
}

Tensor DepthwiseConv2DLayer::Forward(const Tensor& input) {
    has_forward_ = false;
    const DepthwiseGeometry g = ValidateDepthwiseInput(input, channels_, kernel_size_, stride_, padding_);
    const size_t m_count = static_cast<size_t>(multiplier_);
    const size_t outputs = g.channels * m_count;
    const std::vector<size_t> output_shape{g.out_h, g.out_w, outputs, g.batch_size};

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t kk = static_cast<dim_t>(kernel_size_) * kernel_size_;
        const dim_t positions = static_cast<dim_t>(g.out_h * g.out_w);
        const dim_t c = static_cast<dim_t>(g.channels);
        const dim_t n = static_cast<dim_t>(g.batch_size);
        const dim_t m = static_cast<dim_t>(m_count);
        // [k*k, L, C, N]: each channel's windows (unwrap works per dims 2, 3)
        const af::array columns = af::unwrap(PadSpatial(TensorToAf(input), padding_), kernel_size_, kernel_size_,
                                             stride_, stride_, 0, 0);
        // Weights [k,k,1,C*M] -> [k*k, M, C] (output channel c*M + m)
        const af::array kernels = af::moddims(TensorToAf(weights_), af::dim4(kk, m, c));
        af::array output(af::dim4(positions, m, c, n), f32);
        for (int j = 0; j < static_cast<int>(m); ++j) {  // af indexes take int
            const af::array kernel = af::tile(af::moddims(kernels(af::span, j, af::span), af::dim4(kk, 1, c, 1)),
                                              1, static_cast<unsigned>(positions), 1, static_cast<unsigned>(n));
            output(af::span, j, af::span, af::span) =
                af::moddims(af::sum(columns * kernel, 0), af::dim4(positions, 1, c, n));
        }
        // [L, M, C, N] -> [out_h, out_w, C*M, N]
        output = af::moddims(output, af::dim4(static_cast<dim_t>(g.out_h), static_cast<dim_t>(g.out_w),
                                              m * c, n));
        if (use_bias_) {
            output += af::tile(af::moddims(TensorToAf(bias_), af::dim4(1, 1, m * c, 1)),
                               static_cast<unsigned>(g.out_h), static_cast<unsigned>(g.out_w), 1,
                               static_cast<unsigned>(n));
        }
        output.eval();
        Tensor result = Tensor::FromSemanticArray(output, output_shape);
        cached_input_ = input;
        has_forward_ = true;
        return result;
    } catch (const af::exception& e) {
        ThrowDepthwiseDeviceError("DepthwiseConv2DLayer::Forward", e);
    }
#else
    (void)outputs; (void)output_shape;
    ThrowDepthwiseNeedsArrayFire();
#endif
}

Tensor DepthwiseConv2DLayer::Backward(const Tensor& grad_output) {
    if (!has_forward_) {
        throw std::logic_error("DepthwiseConv2DLayer::Backward requires a successful Forward call");
    }
    const DepthwiseGeometry g = ValidateDepthwiseInput(cached_input_, channels_, kernel_size_, stride_, padding_);
    const size_t m_count = static_cast<size_t>(multiplier_);
    const size_t outputs = g.channels * m_count;
    if (grad_output.GetDataType() != DataType::Float32 ||
        grad_output.Shape() != std::vector<size_t>{g.out_h, g.out_w, outputs, g.batch_size}) {
        throw std::runtime_error("DepthwiseConv2D backward gradient does not match the Forward output");
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t kk = static_cast<dim_t>(kernel_size_) * kernel_size_;
        const dim_t positions = static_cast<dim_t>(g.out_h * g.out_w);
        const dim_t c = static_cast<dim_t>(g.channels);
        const dim_t n = static_cast<dim_t>(g.batch_size);
        const dim_t m = static_cast<dim_t>(m_count);
        const dim_t padded_h = static_cast<dim_t>(g.in_h) + 2 * padding_;
        const dim_t padded_w = static_cast<dim_t>(g.in_w) + 2 * padding_;
        const af::array columns = af::unwrap(PadSpatial(TensorToAf(cached_input_), padding_), kernel_size_,
                                             kernel_size_, stride_, stride_, 0, 0);  // [k*k, L, C, N]
        const af::array kernels = af::moddims(TensorToAf(weights_), af::dim4(kk, m, c));
        const af::array grad = af::moddims(TensorToAf(grad_output), af::dim4(positions, m, c, n));
        af::array grad_kernels(af::dim4(kk, m, c), f32);
        af::array grad_columns = af::constant(0.0f, af::dim4(kk, positions, c, n));
        for (int j = 0; j < static_cast<int>(m); ++j) {  // af indexes take int
            // dY for output channels c*M + j, as [1, L, C, N] tiled over the window
            const af::array grad_j = af::tile(
                af::moddims(grad(af::span, j, af::span, af::span), af::dim4(1, positions, c, n)),
                static_cast<unsigned>(kk));
            grad_kernels(af::span, j, af::span) =
                af::moddims(af::sum(af::sum(columns * grad_j, 1), 3), af::dim4(kk, 1, c));
            const af::array kernel = af::tile(af::moddims(kernels(af::span, j, af::span), af::dim4(kk, 1, c, 1)),
                                              1, static_cast<unsigned>(positions), 1, static_cast<unsigned>(n));
            grad_columns += kernel * grad_j;
        }
        // Fold back into the padded input, then drop the padding.
        af::array grad_input = af::wrap(grad_columns, padded_h, padded_w, kernel_size_, kernel_size_, stride_,
                                        stride_, 0, 0);
        if (padding_ > 0) {
            grad_input = grad_input(af::seq(padding_, static_cast<double>(padding_ + g.in_h - 1)),
                                    af::seq(padding_, static_cast<double>(padding_ + g.in_w - 1)), af::span,
                                    af::span);
        }
        grad_input.eval();
        af::array grad_weight = af::moddims(grad_kernels, af::dim4(kernel_size_, kernel_size_, 1, m * c));
        grad_weight.eval();
        grad_weights_ = Tensor::FromSemanticArray(
            grad_weight, {static_cast<size_t>(kernel_size_), static_cast<size_t>(kernel_size_), 1, outputs});
        if (use_bias_) {
            af::array grad_bias = af::moddims(af::sum(af::sum(TensorToAf(grad_output), 0), 1), af::dim4(m * c, n));
            grad_bias = af::sum(grad_bias, 1);
            grad_bias.eval();
            grad_bias_ = Tensor::FromSemanticArray(grad_bias, {outputs});
        }
        return Tensor::FromSemanticArray(grad_input, cached_input_.Shape());
    } catch (const af::exception& e) {
        ThrowDepthwiseDeviceError("DepthwiseConv2DLayer::Backward", e);
    }
#else
    ThrowDepthwiseNeedsArrayFire();
#endif
}

std::map<std::string, Tensor> DepthwiseConv2DLayer::GetParameters() {
    std::map<std::string, Tensor> params;
    params["weights"] = weights_;
    params["grad_weights"] = grad_weights_;
    if (use_bias_) {
        params["bias"] = bias_;
        params["grad_bias"] = grad_bias_;
    }
    return params;
}

void DepthwiseConv2DLayer::SetParameters(const std::map<std::string, Tensor>& params) {
    bool changed = false;
    if (params.count("weights")) {
        weights_ = params.at("weights");
        changed = true;
    }
    if (params.count("bias") && use_bias_) {
        bias_ = params.at("bias");
        changed = true;
    }
    if (changed) has_forward_ = false;
}

}  // namespace cyxwiz
