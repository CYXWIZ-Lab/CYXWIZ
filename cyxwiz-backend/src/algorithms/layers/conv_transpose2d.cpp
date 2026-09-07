#include "../arrayfire_backend_utils.h"
#include "conv_transpose2d_native.h"
#include "cyxwiz/layers/convolution.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyxwiz {
namespace {
constexpr const char *name = "ConvTranspose2D";

size_t OutputExtent(size_t input, int stride, int padding, int kernel,
                    int extra) {
  const size_t span = CheckedLayerProduct(
      input - 1, static_cast<size_t>(stride), name, "strided extent");
  const size_t tail = static_cast<size_t>(kernel) + static_cast<size_t>(extra);
  if (span > (std::numeric_limits<size_t>::max)() - tail) {
    throw std::overflow_error("ConvTranspose2D output extent overflow");
  }
  const size_t crop =
      CheckedLayerProduct(static_cast<size_t>(padding), 2, name, "padding");
  if (span + tail <= crop) {
    throw std::runtime_error("ConvTranspose2D output shape is not positive");
  }
  return span + tail - crop;
}

void ValidateBytes(const std::vector<size_t> &shape) {
  size_t elements = 1;
  for (size_t dim : shape) {
    elements = CheckedLayerProduct(elements, dim, name, "element count");
  }
  (void)CheckedLayerProduct(elements, sizeof(float), name, "byte count");
}

std::vector<size_t> WeightShape(int kernel, int outputs, int inputs) {
  return {static_cast<size_t>(kernel), static_cast<size_t>(kernel),
          static_cast<size_t>(outputs), static_cast<size_t>(inputs)};
}

struct Geometry {
  ConvTranspose2DNativeGeometry native;
  size_t kernel_area;
  size_t filter_rows;
  size_t positions;
  size_t columns;
  std::vector<size_t> output_shape;
};

Geometry Validate(const Tensor &input, const Tensor &weights,
                  const Tensor &bias, int inputs, int outputs, int kernel,
                  int stride, int padding, int extra, bool use_bias) {
  ValidateSpatial4DInput(input, name);
  if (weights.GetDataType() != DataType::Float32 ||
      (use_bias && bias.GetDataType() != DataType::Float32)) {
    throw std::runtime_error("ConvTranspose2D requires Float32 parameters");
  }
  if (weights.Shape() != WeightShape(kernel, outputs, inputs)) {
    throw std::runtime_error("ConvTranspose2D weight shape mismatch");
  }
  if (use_bias &&
      bias.Shape() != std::vector<size_t>{static_cast<size_t>(outputs)}) {
    throw std::runtime_error("ConvTranspose2D bias shape mismatch");
  }
  const auto &s = input.Shape();
  if (s[2] != static_cast<size_t>(inputs)) {
    throw std::runtime_error("ConvTranspose2D input channel mismatch");
  }
  const size_t h = OutputExtent(s[0], stride, padding, kernel, extra);
  const size_t w = OutputExtent(s[1], stride, padding, kernel, extra);
  const size_t area =
      CheckedLayerProduct(static_cast<size_t>(kernel),
                          static_cast<size_t>(kernel), name, "kernel area");
  const size_t rows = CheckedLayerProduct(area, static_cast<size_t>(outputs),
                                          name, "filter rows");
  const size_t positions = CheckedLayerProduct(s[0], s[1], name, "positions");
  const size_t columns = CheckedLayerProduct(positions, s[3], name, "columns");
  std::vector<size_t> output_shape{h, w, static_cast<size_t>(outputs), s[3]};
  ValidateBytes(s);
  ValidateBytes(output_shape);
  ValidateBytes({rows, columns});
#ifdef CYXWIZ_HAS_ARRAYFIRE
  for (size_t dim : {s[0], s[1], s[2], s[3], h, w, rows, positions, columns}) {
    (void)CheckedIntDim(dim, "ConvTranspose2D dimension");
  }
#endif
  return {{s[0], s[1], s[2], s[3], h, w},
          area,
          rows,
          positions,
          columns,
          output_shape};
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool UseNative(const char *operation, const Tensor &tensor, const Geometry &g,
               int kernel, int padding) {
  // wrap/unwrap require padding < window and a window fitting the image
  // plus one padding side. Other valid layer shapes retain native support.
  if (padding >= kernel ||
      g.native.out_h + static_cast<size_t>(padding) <
          static_cast<size_t>(kernel) ||
      g.native.out_w + static_cast<size_t>(padding) <
          static_cast<size_t>(kernel)) {
    RecordLayerArrayFireFallback(
        operation, BackendFallbackReason::UnsupportedShape,
        "ArrayFire wrap/unwrap window and padding limits", tensor, "tensor");
    return true;
  }
  if (ShouldForceArrayFireBackendFallbackForTesting(operation)) {
    RecordLayerArrayFireFallback(operation,
                                 "forced ArrayFire backend fallback test hook",
                                 tensor, "tensor");
    return true;
  }
  return false;
}

af::array InputColumns(const Tensor &input, const Geometry &g) {
  return af::moddims(af::reorder(input.GetSemanticArray(), 2, 0, 1, 3),
                     static_cast<dim_t>(g.native.in_channels),
                     static_cast<dim_t>(g.columns));
}
#endif
} // namespace

ConvTranspose2DLayer::ConvTranspose2DLayer(int in_channels, int out_channels,
                                           int kernel_size, int stride,
                                           int padding, int output_padding,
                                           bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      output_padding_(output_padding), use_bias_(use_bias) {
  if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0 ||
      stride <= 0 || padding < 0 || output_padding < 0 ||
      output_padding >= stride) {
    throw std::invalid_argument(
        "ConvTranspose2D requires positive channels/kernel/stride, "
        "non-negative padding, and output_padding < stride");
  }
  const auto shape = WeightShape(kernel_size, out_channels, in_channels);
  ValidateBytes(shape);
  const size_t fan = CheckedLayerProduct(
      CheckedLayerProduct(shape[0], shape[1], name, "kernel area"), shape[3],
      name, "fan-in");
  if (fan > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    throw std::overflow_error(
        "ConvTranspose2D fan-in exceeds initialization limit");
  }
#ifdef CYXWIZ_HAS_ARRAYFIRE
  weights_ = Tensor::FromSemanticArray(
      KaimingUniform(
          static_cast<int>(fan),
          af::dim4(kernel_size, kernel_size, out_channels, in_channels)),
      shape);
#else
  weights_ = Tensor::Random(shape);
#endif
  grad_weights_ = Tensor::Zeros(shape);
  if (use_bias_) {
    bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
  }
}

Tensor ConvTranspose2DLayer::Forward(const Tensor &input) {
  has_forward_ = false;
  const auto g =
      Validate(input, weights_, bias_, in_channels_, out_channels_,
               kernel_size_, stride_, padding_, output_padding_, use_bias_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
  if (!UseNative("ConvTranspose2DLayer::Forward", input, g, kernel_size_,
                 padding_)) {
    try {
      const af::array filters =
          af::moddims(weights_.GetSemanticArray(),
                      static_cast<dim_t>(g.filter_rows), in_channels_);
      // [K*K*Cout, Cin] * [Cin, H*W*N]: one output patch per input pixel.
      const af::array patches = af::reorder(
          af::moddims(af::matmul(filters, InputColumns(input, g)),
                      static_cast<dim_t>(g.kernel_area), out_channels_,
                      static_cast<dim_t>(g.positions),
                      static_cast<dim_t>(g.native.batch_size)),
          0, 2, 1, 3);
      af::array output =
          af::wrap(patches, static_cast<dim_t>(g.native.out_h),
                   static_cast<dim_t>(g.native.out_w), kernel_size_,
                   kernel_size_, stride_, stride_, padding_, padding_);
      if (use_bias_) {
        output += af::tile(
            af::moddims(bias_.GetSemanticArray(), 1, 1, out_channels_, 1),
            af::dim4(static_cast<dim_t>(g.native.out_h),
                     static_cast<dim_t>(g.native.out_w), 1,
                     static_cast<dim_t>(g.native.batch_size)));
      }
      output.eval();
      Tensor result = Tensor::FromSemanticArray(output, g.output_shape);
      cached_input_ = input;
      has_forward_ = true;
      return result;
    } catch (const af::exception &error) {
      RecordLayerArrayFireFallback("ConvTranspose2DLayer::Forward",
                                   error.what(), input, "input");
    }
  }
#else
  RecordLayerArrayFireFallback("ConvTranspose2DLayer::Forward",
                               BackendFallbackReason::BackendUnavailable,
                               "ArrayFire support is not compiled", input,
                               "input");
#endif
  Tensor output = ConvTranspose2DForwardNative(
      input, weights_, bias_, g.native,
      {static_cast<size_t>(out_channels_), kernel_size_, stride_, padding_,
       use_bias_});
  cached_input_ = input;
  has_forward_ = true;
  return output;
}

Tensor ConvTranspose2DLayer::Backward(const Tensor &grad_output) {
  if (!has_forward_) {
    throw std::logic_error(
        "ConvTranspose2DLayer::Backward requires a successful Forward call");
  }
  const auto g =
      Validate(cached_input_, weights_, bias_, in_channels_, out_channels_,
               kernel_size_, stride_, padding_, output_padding_, use_bias_);
  if (grad_output.GetDataType() != DataType::Float32 ||
      grad_output.Shape() != g.output_shape) {
    throw std::runtime_error(
        "ConvTranspose2D backward gradient dtype or shape mismatch");
  }
#ifdef CYXWIZ_HAS_ARRAYFIRE
  if (!UseNative("ConvTranspose2DLayer::Backward", grad_output, g, kernel_size_,
                 padding_)) {
    try {
      const af::array dy = grad_output.GetSemanticArray();
      const af::array columns = af::moddims(
          af::reorder(af::unwrap(dy, kernel_size_, kernel_size_, stride_,
                                 stride_, padding_, padding_),
                      0, 2, 1, 3),
          static_cast<dim_t>(g.filter_rows), static_cast<dim_t>(g.columns));
      const af::array filters =
          af::moddims(weights_.GetSemanticArray(),
                      static_cast<dim_t>(g.filter_rows), in_channels_);
      af::array dx =
          af::reorder(af::moddims(af::matmulTN(filters, columns), in_channels_,
                                  static_cast<dim_t>(g.native.in_h),
                                  static_cast<dim_t>(g.native.in_w),
                                  static_cast<dim_t>(g.native.batch_size)),
                      1, 2, 0, 3);
      af::array dw =
          af::moddims(af::matmulNT(columns, InputColumns(cached_input_, g)),
                      kernel_size_, kernel_size_, out_channels_, in_channels_);
      dx.eval();
      dw.eval();
      Tensor new_bias_gradient;
      if (use_bias_) {
        af::array db =
            af::moddims(af::sum(af::sum(af::sum(dy, 0), 1), 3), out_channels_);
        db.eval();
        new_bias_gradient =
            Tensor::FromSemanticArray(db, {static_cast<size_t>(out_channels_)});
      }
      Tensor result = Tensor::FromSemanticArray(dx, cached_input_.Shape());
      grad_weights_ = Tensor::FromSemanticArray(
          dw, WeightShape(kernel_size_, out_channels_, in_channels_));
      if (use_bias_)
        grad_bias_ = std::move(new_bias_gradient);
      return result;
    } catch (const af::exception &error) {
      RecordLayerArrayFireFallback("ConvTranspose2DLayer::Backward",
                                   error.what(), grad_output, "grad_output");
    }
  }
#else
  RecordLayerArrayFireFallback("ConvTranspose2DLayer::Backward",
                               BackendFallbackReason::BackendUnavailable,
                               "ArrayFire support is not compiled", grad_output,
                               "grad_output");
#endif
  return ConvTranspose2DBackwardNative(
      cached_input_, grad_output, weights_, grad_weights_, grad_bias_, g.native,
      {static_cast<size_t>(out_channels_), kernel_size_, stride_, padding_,
       use_bias_});
}

std::map<std::string, Tensor> ConvTranspose2DLayer::GetParameters() {
  std::map<std::string, Tensor> params{{"weights", weights_},
                                       {"grad_weights", grad_weights_}};
  if (use_bias_) {
    params["bias"] = bias_;
    params["grad_bias"] = grad_bias_;
  }
  return params;
}

void ConvTranspose2DLayer::SetParameters(
    const std::map<std::string, Tensor> &params) {
  if (params.count("weights")) {
    weights_ = params.at("weights");
    has_forward_ = false;
  }
  if (params.count("bias") && use_bias_) {
    bias_ = params.at("bias");
    has_forward_ = false;
  }
}
} // namespace cyxwiz
