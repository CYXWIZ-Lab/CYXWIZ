// DepthwiseConv2D (TOFIX140 B): torch Conv2d(C, C*M, k, groups=C) on
// [H,W,C,N]. ArrayFire path: unwrap each channel's windows ([k*k, L, C, N]),
// weight them by that channel's M kernels and sum over the window; backward
// mirrors it and folds the window gradients back with wrap. Native CPU loops
// otherwise (padding >= kernel, no ArrayFire, an ArrayFire error).
#include "cyxwiz/layers/convolution.h"
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
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
    weights_ = Tensor::Random(weight_shape, DataType::Float32);
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
    bool use_native_cpu = false;
    if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Forward", BackendFallbackReason::UnsupportedShape,
                                     "ArrayFire unwrap requires padding smaller than kernel size", input, "input");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting("DepthwiseConv2DLayer::Forward")) {
        RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Forward", "forced ArrayFire backend fallback test hook",
                                     input, "input");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const dim_t kk = static_cast<dim_t>(kernel_size_) * kernel_size_;
            const dim_t positions = static_cast<dim_t>(g.out_h * g.out_w);
            const dim_t c = static_cast<dim_t>(g.channels);
            const dim_t n = static_cast<dim_t>(g.batch_size);
            const dim_t m = static_cast<dim_t>(m_count);
            // [k*k, L, C, N]: each channel's windows (unwrap works per dims 2, 3)
            const af::array columns = af::unwrap(TensorToAf(input), kernel_size_, kernel_size_, stride_, stride_,
                                                 padding_, padding_);
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
            RecordLayerArrayFireFallbackObservation("DepthwiseConv2DLayer::Forward", "DepthwiseConv2D", e.what(),
                                                    input, "input");
        }
    }
#else
    RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Forward", BackendFallbackReason::BackendUnavailable,
                                 "ArrayFire support is not compiled", input, "input");
#endif

    const ScopedArrayFireHostSyncAttribution attribution(ArrayFireHostSyncCategory::LayerCpuPath,
                                                         "DepthwiseConv2DLayer::Forward");
    Tensor output(output_shape, DataType::Float32);
    const float* x = input.ReadData<float>();
    const float* w = weights_.ReadData<float>();
    const float* b = use_bias_ ? bias_.ReadData<float>() : nullptr;
    float* y = output.MutableData<float>();
    const long k = kernel_size_;
    for (size_t oh = 0; oh < g.out_h; ++oh) {
        for (size_t ow = 0; ow < g.out_w; ++ow) {
            for (size_t ch = 0; ch < g.channels; ++ch) {
                for (size_t j = 0; j < m_count; ++j) {
                    const size_t oc = ch * m_count + j;
                    for (size_t bn = 0; bn < g.batch_size; ++bn) {
                        float sum = b ? b[oc] : 0.0f;
                        for (long kh = 0; kh < k; ++kh) {
                            const long ih = static_cast<long>(oh) * stride_ - padding_ + kh;
                            if (ih < 0 || ih >= static_cast<long>(g.in_h)) continue;
                            for (long kw = 0; kw < k; ++kw) {
                                const long iw = static_cast<long>(ow) * stride_ - padding_ + kw;
                                if (iw < 0 || iw >= static_cast<long>(g.in_w)) continue;
                                sum += x[Pool4DIndex(static_cast<size_t>(ih), static_cast<size_t>(iw), ch, bn,
                                                     g.in_w, g.channels, g.batch_size)] *
                                       w[static_cast<size_t>(kh * k + kw) * outputs + oc];
                            }
                        }
                        y[Pool4DIndex(oh, ow, oc, bn, g.out_w, outputs, g.batch_size)] = sum;
                    }
                }
            }
        }
    }
    cached_input_ = input;
    has_forward_ = true;
    return output;
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
    bool use_native_cpu = false;
    if (padding_ >= kernel_size_) {
        RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Backward", BackendFallbackReason::UnsupportedShape,
                                     "ArrayFire wrap requires padding smaller than kernel size", grad_output,
                                     "grad_output");
        use_native_cpu = true;
    } else if (ShouldForceArrayFireBackendFallbackForTesting("DepthwiseConv2DLayer::Backward")) {
        RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Backward", "forced ArrayFire backend fallback test hook",
                                     grad_output, "grad_output");
        use_native_cpu = true;
    }
    if (!use_native_cpu) {
        try {
            const dim_t kk = static_cast<dim_t>(kernel_size_) * kernel_size_;
            const dim_t positions = static_cast<dim_t>(g.out_h * g.out_w);
            const dim_t c = static_cast<dim_t>(g.channels);
            const dim_t n = static_cast<dim_t>(g.batch_size);
            const dim_t m = static_cast<dim_t>(m_count);
            const af::array columns = af::unwrap(TensorToAf(cached_input_), kernel_size_, kernel_size_, stride_,
                                                 stride_, padding_, padding_);  // [k*k, L, C, N]
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
            af::array grad_input = af::wrap(grad_columns, static_cast<dim_t>(g.in_h), static_cast<dim_t>(g.in_w),
                                            kernel_size_, kernel_size_, stride_, stride_, padding_, padding_);
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
            RecordLayerArrayFireFallbackObservation("DepthwiseConv2DLayer::Backward", "DepthwiseConv2D", e.what(),
                                                    cached_input_, "input");
        }
    }
#else
    RecordLayerArrayFireFallback("DepthwiseConv2DLayer::Backward", BackendFallbackReason::BackendUnavailable,
                                 "ArrayFire support is not compiled", grad_output, "grad_output");
#endif

    const ScopedArrayFireHostSyncAttribution attribution(ArrayFireHostSyncCategory::LayerCpuPath,
                                                         "DepthwiseConv2DLayer::Backward");
    const std::vector<size_t> weight_shape{static_cast<size_t>(kernel_size_), static_cast<size_t>(kernel_size_), 1,
                                           outputs};
    Tensor grad_input(cached_input_.Shape(), DataType::Float32);
    Tensor grad_weights(weight_shape, DataType::Float32);
    Tensor grad_bias({outputs}, DataType::Float32);
    const float* x = cached_input_.ReadData<float>();
    const float* w = weights_.ReadData<float>();
    const float* dy = grad_output.ReadData<float>();
    float* dx = grad_input.MutableData<float>();
    float* dw = grad_weights.MutableData<float>();
    float* db = grad_bias.MutableData<float>();
    std::fill(dx, dx + grad_input.NumElements(), 0.0f);
    std::fill(dw, dw + grad_weights.NumElements(), 0.0f);
    std::fill(db, db + outputs, 0.0f);
    const long k = kernel_size_;
    for (size_t oh = 0; oh < g.out_h; ++oh) {
        for (size_t ow = 0; ow < g.out_w; ++ow) {
            for (size_t ch = 0; ch < g.channels; ++ch) {
                for (size_t j = 0; j < m_count; ++j) {
                    const size_t oc = ch * m_count + j;
                    for (size_t bn = 0; bn < g.batch_size; ++bn) {
                        const float d = dy[Pool4DIndex(oh, ow, oc, bn, g.out_w, outputs, g.batch_size)];
                        db[oc] += d;
                        for (long kh = 0; kh < k; ++kh) {
                            const long ih = static_cast<long>(oh) * stride_ - padding_ + kh;
                            if (ih < 0 || ih >= static_cast<long>(g.in_h)) continue;
                            for (long kw = 0; kw < k; ++kw) {
                                const long iw = static_cast<long>(ow) * stride_ - padding_ + kw;
                                if (iw < 0 || iw >= static_cast<long>(g.in_w)) continue;
                                const size_t xi = Pool4DIndex(static_cast<size_t>(ih), static_cast<size_t>(iw), ch, bn,
                                                              g.in_w, g.channels, g.batch_size);
                                const size_t wi = static_cast<size_t>(kh * k + kw) * outputs + oc;
                                dw[wi] += x[xi] * d;
                                dx[xi] += w[wi] * d;
                            }
                        }
                    }
                }
            }
        }
    }
    grad_weights_ = grad_weights;
    if (use_bias_) grad_bias_ = grad_bias;
    return grad_input;
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
