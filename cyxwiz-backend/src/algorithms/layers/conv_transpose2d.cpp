// ConvTranspose2D on [H, W, C, N], on the ArrayFire device only. Each sample's
// k x k output patches come from one batched matmul ([H*W, Cin] x [Cin,
// k*k*Cout]); a sparse matrix built once per input size places them into the
// output, the padding crop folded in. Backward uses its transpose and batched
// matmuls. No wrap / unwrap / reorder: ArrayFire 3.10's CUDA backend fails to
// compile its unwrap and strided-copy kernels at image sizes. The CPU is
// ArrayFire's CPU backend; a build without ArrayFire refuses the layer.
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

// The input and output extents, as the former native path kept them.
struct Extents {
  size_t in_h, in_w, in_channels, batch_size, out_h, out_w;
};

struct Geometry {
  Extents native;
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
[[noreturn]] void ThrowDeviceError(const char *operation, const af::exception &error) {
  throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}
#endif
} // namespace

// Patch element (a, b) of input pixel p = ih + H*iw (column p + P*(a + k*b))
// lands on output (ih*stride + a - padding, iw*stride + b - padding), row
// h + H_out*w, when inside the output. place: [H_out*W_out, P*k*k]; lift is
// its transpose.
struct ConvTranspose2DLayer::DeviceScatter {
  size_t in_h = 0, in_w = 0;
#ifdef CYXWIZ_HAS_ARRAYFIRE
  af::array place;
  af::array lift;
#endif
};

ConvTranspose2DLayer::ConvTranspose2DLayer(int in_channels, int out_channels,
                                           int kernel_size, int stride,
                                           int padding, int output_padding,
                                           bool use_bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      output_padding_(output_padding), use_bias_(use_bias),
      scatter_(std::make_shared<DeviceScatter>()) {
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
  throw std::runtime_error("ConvTranspose2D runs on ArrayFire, and this build has no ArrayFire");
#endif
  grad_weights_ = Tensor::Zeros(shape);
  if (use_bias_) {
    bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
    grad_bias_ = Tensor::Zeros({static_cast<size_t>(out_channels)});
  }
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
namespace {

void EnsurePlacement(size_t &cached_h, size_t &cached_w, af::array &place, af::array &lift,
                     const Geometry &g, int kernel, int stride, int padding) {
  if (cached_h == g.native.in_h && cached_w == g.native.in_w && !place.isempty()) return;
  const size_t in_h = g.native.in_h, in_w = g.native.in_w;
  const size_t out_h = g.native.out_h, out_w = g.native.out_w;
  const size_t positions = in_h * in_w;
  const size_t k = static_cast<size_t>(kernel);
  const size_t columns = positions * k * k;
  const size_t rows = out_h * out_w;
  std::vector<int> target(columns, -1);  // the output row of each column, or -1
  for (size_t b = 0; b < k; ++b) {
    for (size_t a = 0; a < k; ++a) {
      for (size_t iw = 0; iw < in_w; ++iw) {
        const long long w = static_cast<long long>(iw) * stride + static_cast<long long>(b) - padding;
        for (size_t ih = 0; ih < in_h; ++ih) {
          const long long hh = static_cast<long long>(ih) * stride + static_cast<long long>(a) - padding;
          if (hh < 0 || w < 0 || hh >= static_cast<long long>(out_h) || w >= static_cast<long long>(out_w)) continue;
          const size_t p = ih + in_h * iw;
          target[p + positions * (a + k * b)] = static_cast<int>(static_cast<size_t>(hh) + out_h * static_cast<size_t>(w));
        }
      }
    }
  }
  // lift [P*k*k, H_out*W_out]: one entry per column of place, row = that column.
  std::vector<int> lift_offsets(columns + 1, 0), lift_columns;
  std::vector<int> place_offsets(rows + 1, 0);
  lift_columns.reserve(columns);
  for (size_t c = 0; c < columns; ++c) {
    lift_offsets[c + 1] = lift_offsets[c];
    if (target[c] < 0) continue;
    lift_columns.push_back(target[c]);
    ++lift_offsets[c + 1];
    ++place_offsets[static_cast<size_t>(target[c]) + 1];
  }
  for (size_t r = 0; r < rows; ++r) place_offsets[r + 1] += place_offsets[r];
  std::vector<int> place_columns(lift_columns.size());
  std::vector<int> fill(place_offsets.begin(), place_offsets.end() - 1);
  for (size_t c = 0; c < columns; ++c) {
    if (target[c] >= 0) place_columns[static_cast<size_t>(fill[static_cast<size_t>(target[c])]++)] = static_cast<int>(c);
  }
  const std::vector<float> ones(lift_columns.size(), 1.0f);
  const dim_t nnz = static_cast<dim_t>(ones.size());
  place = af::sparse(static_cast<dim_t>(rows), static_cast<dim_t>(columns), nnz, ones.data(), place_offsets.data(),
                     place_columns.data(), f32, AF_STORAGE_CSR, afHost);
  lift = af::sparse(static_cast<dim_t>(columns), static_cast<dim_t>(rows), nnz, ones.data(), lift_offsets.data(),
                    lift_columns.data(), f32, AF_STORAGE_CSR, afHost);
  cached_h = in_h;
  cached_w = in_w;
}

}  // namespace
#endif

Tensor ConvTranspose2DLayer::Forward(const Tensor &input) {
  has_forward_ = false;
  const auto g =
      Validate(input, weights_, bias_, in_channels_, out_channels_,
               kernel_size_, stride_, padding_, output_padding_, use_bias_);
#ifdef CYXWIZ_HAS_ARRAYFIRE
  try {
    EnsurePlacement(scatter_->in_h, scatter_->in_w, scatter_->place, scatter_->lift, g, kernel_size_, stride_,
                    padding_);
    const dim_t positions = static_cast<dim_t>(g.positions);
    const dim_t n = static_cast<dim_t>(g.native.batch_size);
    const dim_t area = static_cast<dim_t>(g.kernel_area);
    // Per sample [P, Cin] x [Cin, k*k*Cout] -> [P, k*k*Cout, N]: each pixel's patch.
    const af::array x = af::moddims(input.GetSemanticArray(), positions, in_channels_, n);
    const af::array filters_t = af::tile(
        af::transpose(af::moddims(weights_.GetSemanticArray(), static_cast<dim_t>(g.filter_rows), in_channels_)),
        1, 1, static_cast<unsigned>(n));
    const af::array patches = af::moddims(af::matmul(x, filters_t), positions * area, out_channels_ * n);
    af::array output = af::moddims(af::matmul(scatter_->place, patches), static_cast<dim_t>(g.native.out_h),
                                   static_cast<dim_t>(g.native.out_w), out_channels_, n);
    if (use_bias_) {
      output += af::tile(
          af::moddims(bias_.GetSemanticArray(), 1, 1, out_channels_, 1),
          af::dim4(static_cast<dim_t>(g.native.out_h),
                   static_cast<dim_t>(g.native.out_w), 1, n));
    }
    output.eval();
    Tensor result = Tensor::FromSemanticArray(output, g.output_shape);
    cached_input_ = input;
    has_forward_ = true;
    return result;
  } catch (const af::exception &error) {
    ThrowDeviceError("ConvTranspose2DLayer::Forward", error);
  }
#else
  (void)g;
  throw std::runtime_error("ConvTranspose2D runs on ArrayFire, and this build has no ArrayFire");
#endif
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
  try {
    EnsurePlacement(scatter_->in_h, scatter_->in_w, scatter_->place, scatter_->lift, g, kernel_size_, stride_,
                    padding_);
    const dim_t positions = static_cast<dim_t>(g.positions);
    const dim_t n = static_cast<dim_t>(g.native.batch_size);
    const af::array dy = grad_output.GetSemanticArray();
    // Each pixel's patch gradient: [P*k*k, Cout*N] -> [P, k*k*Cout, N].
    const af::array dy_flat = af::moddims(dy, static_cast<dim_t>(g.native.out_h * g.native.out_w), out_channels_ * n);
    const af::array dpatches = af::moddims(af::matmul(scatter_->lift, dy_flat), positions,
                                           static_cast<dim_t>(g.filter_rows), n);
    const af::array filters = af::moddims(weights_.GetSemanticArray(), static_cast<dim_t>(g.filter_rows), in_channels_);
    af::array dx = af::moddims(af::matmul(dpatches, af::tile(filters, 1, 1, static_cast<unsigned>(n))),
                               static_cast<dim_t>(g.native.in_h), static_cast<dim_t>(g.native.in_w), in_channels_, n);
    const af::array x = af::moddims(cached_input_.GetSemanticArray(), positions, in_channels_, n);
    // dW: sum over samples of x^T . dpatches -> [Cin, k*k*Cout] -> [k, k, Cout, Cin]
    af::array dw = af::moddims(af::transpose(af::sum(af::matmulTN(x, dpatches), 2)), kernel_size_, kernel_size_,
                               out_channels_, in_channels_);
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
    ThrowDeviceError("ConvTranspose2DLayer::Backward", error);
  }
#else
  (void)g;
  throw std::runtime_error("ConvTranspose2D runs on ArrayFire, and this build has no ArrayFire");
#endif
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
