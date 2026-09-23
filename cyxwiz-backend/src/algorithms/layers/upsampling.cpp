#include "cyxwiz/layers/upsampling.h"
#include "../arrayfire_backend_utils.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

namespace cyxwiz {
namespace {

std::vector<size_t> UpsampleOutputShape(const std::vector<size_t> &input_shape,
                                        int scale_factor) {
  auto output_shape = input_shape;
  const auto scale = static_cast<size_t>(scale_factor);
  output_shape[0] =
      CheckedLayerProduct(input_shape[0], scale, "Upsample2D", "output height");
  output_shape[1] =
      CheckedLayerProduct(input_shape[1], scale, "Upsample2D", "output width");
  size_t elements = 1;
  for (const size_t extent : output_shape) {
    elements =
        CheckedLayerProduct(elements, extent, "Upsample2D", "output elements");
  }
  CheckedLayerProduct(elements, sizeof(float), "Upsample2D", "output bytes");
  return output_shape;
}

Tensor NativeForward(const Tensor &input,
                     const std::vector<size_t> &output_shape, int scale_factor,
                     UpsampleMode mode) {
  const std::vector<size_t> &shape = input.Shape();
  const size_t in_h = shape[0];
  const size_t in_w = shape[1];
  const size_t channels = shape[2];
  const size_t batch_size = shape[3];
  const size_t scale = static_cast<size_t>(scale_factor);
  const size_t out_h = output_shape[0];
  const size_t out_w = output_shape[1];

  Tensor output(output_shape, DataType::Float32);
  const float *input_data = input.ReadData<float>();
  float *output_data = output.MutableData<float>();

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          const size_t out_index =
              Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size);
          if (mode == UpsampleMode::Nearest) {
            const size_t ih = oh / scale;
            const size_t iw = ow / scale;
            output_data[out_index] = input_data[Pool4DIndex(
                ih, iw, c, b, in_w, channels, batch_size)];
            continue;
          }

          const ResizeLinearSample h_sample =
              ComputeResizeLinearSample(oh, in_h, scale_factor);
          const ResizeLinearSample w_sample =
              ComputeResizeLinearSample(ow, in_w, scale_factor);
          const float h0_weight = 1.0f - h_sample.upper_weight;
          const float w0_weight = 1.0f - w_sample.upper_weight;
          const float v00 =
              input_data[Pool4DIndex(h_sample.lower, w_sample.lower, c, b, in_w,
                                     channels, batch_size)];
          const float v01 =
              input_data[Pool4DIndex(h_sample.lower, w_sample.upper, c, b, in_w,
                                     channels, batch_size)];
          const float v10 =
              input_data[Pool4DIndex(h_sample.upper, w_sample.lower, c, b, in_w,
                                     channels, batch_size)];
          const float v11 =
              input_data[Pool4DIndex(h_sample.upper, w_sample.upper, c, b, in_w,
                                     channels, batch_size)];
          output_data[out_index] =
              h0_weight * (w0_weight * v00 + w_sample.upper_weight * v01) +
              h_sample.upper_weight *
                  (w0_weight * v10 + w_sample.upper_weight * v11);
        }
      }
    }
  }

  return output;
}

Tensor NativeBackward(const Tensor &grad_output,
                      const std::vector<size_t> &input_shape, int scale_factor,
                      UpsampleMode mode) {
  const size_t in_h = input_shape[0];
  const size_t in_w = input_shape[1];
  const size_t channels = input_shape[2];
  const size_t batch_size = input_shape[3];
  const size_t scale = static_cast<size_t>(scale_factor);
  const auto &output_shape = grad_output.Shape();
  const size_t out_h = output_shape[0];
  const size_t out_w = output_shape[1];

  // This primitive executes its backward formula on native CPU. Construct
  // the host tensor with the semantic shape so trailing singleton channel
  // and batch dimensions are preserved; Tensor::Zeros may round-trip
  // through ArrayFire and collapse those dimensions.
  Tensor grad_input(input_shape, DataType::Float32);
  const float *grad_data = grad_output.ReadData<float>();
  float *grad_input_data = grad_input.MutableData<float>();

  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t c = 0; c < channels; ++c) {
      if (mode == UpsampleMode::Nearest) {
        for (size_t ih = 0; ih < in_h; ++ih) {
          for (size_t iw = 0; iw < in_w; ++iw) {
            float sum = 0.0f;
            for (int sh = 0; sh < scale_factor; ++sh) {
              for (int sw = 0; sw < scale_factor; ++sw) {
                const size_t oh = ih * scale + static_cast<size_t>(sh);
                const size_t ow = iw * scale + static_cast<size_t>(sw);
                sum += grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels,
                                             batch_size)];
              }
            }
            grad_input_data[Pool4DIndex(ih, iw, c, b, in_w, channels,
                                        batch_size)] = sum;
          }
        }
        continue;
      }

      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          const ResizeLinearSample h_sample =
              ComputeResizeLinearSample(oh, in_h, scale_factor);
          const ResizeLinearSample w_sample =
              ComputeResizeLinearSample(ow, in_w, scale_factor);
          const float h0_weight = 1.0f - h_sample.upper_weight;
          const float w0_weight = 1.0f - w_sample.upper_weight;
          const float grad_value =
              grad_data[Pool4DIndex(oh, ow, c, b, out_w, channels, batch_size)];
          grad_input_data[Pool4DIndex(h_sample.lower, w_sample.lower, c, b,
                                      in_w, channels, batch_size)] +=
              grad_value * h0_weight * w0_weight;
          grad_input_data[Pool4DIndex(h_sample.lower, w_sample.upper, c, b,
                                      in_w, channels, batch_size)] +=
              grad_value * h0_weight * w_sample.upper_weight;
          grad_input_data[Pool4DIndex(h_sample.upper, w_sample.lower, c, b,
                                      in_w, channels, batch_size)] +=
              grad_value * h_sample.upper_weight * w0_weight;
          grad_input_data[Pool4DIndex(h_sample.upper, w_sample.upper, c, b,
                                      in_w, channels, batch_size)] +=
              grad_value * h_sample.upper_weight * w_sample.upper_weight;
        }
      }
    }
  }

  return grad_input;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
// Explicit provider broadcasting avoids both a full-size weight tile and a
// process-wide gfor/batch-mode switch. The returned array owns the C API
// handle.
af::array ApplyInterpolationWeights(const af::array &values,
                                    const af::array &weights) {
  af_array result = nullptr;
  const af_err status = af_mul(&result, values.get(), weights.get(), true);
  if (status != AF_SUCCESS)
    throw af::exception("Upsample interpolation weights", __FILE__, __LINE__,
                        status);
  return af::array(result);
}

af::array BilinearAxis(af::array values, dim_t extent, int factor,
                       unsigned axis, bool backward) {
  if (factor == 1)
    return values;
  if (axis == 1)
    values = af::reorder(values, 1, 0, 2, 3);
  const auto dims = values.dims();
  if (extent == 1) {
    values = backward ? af::sum(values, 0) : af::tile(values, factor, 1, 1, 1);
  } else {
    const dim_t r = factor;
    const dim_t cn = dims[2] * dims[3];
    const int last = static_cast<int>(extent - 1); // Checked provider extent.
    // Integer upsampling: source = i + (t + 1/2)/r - 1/2 for t in [0,r).
    // Factor-local weights avoid loss of the fractional part at large i.
    const af::array delta =
        (af::range(af::dim4(r), 0, f32) + .5f) / static_cast<float>(factor) -
        .5f;
    const af::array previous_weight = af::max(-delta, 0.0f);
    const af::array next_weight = af::max(delta, 0.0f);
    const af::array center_weight = 1.0f - previous_weight - next_weight;
    if (backward) {
      values = af::moddims(values, r, extent, dims[1], cn);
      const af::array previous =
          af::sum(ApplyInterpolationWeights(values, previous_weight), 0);
      const af::array next =
          af::sum(ApplyInterpolationWeights(values, next_weight), 0);
      af::array from_right = af::shift(previous, 0, -1, 0, 0);
      af::array from_left = af::shift(next, 0, 1, 0, 0);
      from_right(af::span, last, af::span, af::span) = 0.0f;
      from_left(af::span, 0, af::span, af::span) = 0.0f;
      values = af::sum(ApplyInterpolationWeights(values, center_weight), 0) +
               from_right + from_left;
      // Clamped contributions stay on the first/last pixel rather than wrap.
      values(af::span, 0, af::span, af::span) +=
          previous(af::span, 0, af::span, af::span);
      values(af::span, last, af::span, af::span) +=
          next(af::span, last, af::span, af::span);
    } else {
      const af::array positions = af::range(af::dim4(extent), 0, s32);
      const af::array previous =
          af::lookup(values, af::max(positions - 1, 0.0), 0);
      const af::array next = af::lookup(
          values, af::min(positions + 1, static_cast<double>(last)), 0);
      values =
          ApplyInterpolationWeights(
              af::moddims(previous, 1, extent, dims[1], cn), previous_weight) +
          ApplyInterpolationWeights(af::moddims(values, 1, extent, dims[1], cn),
                                    center_weight) +
          ApplyInterpolationWeights(af::moddims(next, 1, extent, dims[1], cn),
                                    next_weight);
    }
    values = af::moddims(values, backward ? extent : r * extent, dims[1],
                         dims[2], dims[3]);
  }
  values.eval();
  return axis == 1 ? af::reorder(values, 1, 0, 2, 3) : values;
}
#endif

Tensor Execute(const Tensor &input, const std::vector<size_t> &low,
               const std::vector<size_t> &high, int factor, UpsampleMode mode,
               bool backward) {
  const char *operation =
      backward ? "Upsample2DLayer::Backward" : "Upsample2DLayer::Forward";
#ifdef CYXWIZ_HAS_ARRAYFIRE
  {
    const size_t channel_batches =
        CheckedLayerProduct(low[2], low[3], "Upsample2D", "channel batches");
    for (const size_t dimension :
         {low[0], low[1], low[2], low[3], high[0], high[1], channel_batches}) {
      (void)CheckedIntDim(dimension, "Upsample2D dimension");
    }
    if (ShouldForceArrayFireBackendFallbackForTesting(operation)) {
      RecordLayerArrayFireFallback(
          operation, "forced ArrayFire backend fallback test hook", input,
          "tensor");
    } else {
      try {
        const auto h = static_cast<dim_t>(low[0]);
        const auto w = static_cast<dim_t>(low[1]);
        const auto cn = static_cast<dim_t>(channel_batches);
        const auto r = static_cast<dim_t>(factor);
        af::array values = input.GetSemanticArray();
        if (mode == UpsampleMode::Bilinear) {
          if (backward) {
            values = BilinearAxis(values, h, factor, 0, true);
            values = BilinearAxis(values, w, factor, 1, true);
          } else {
            values = BilinearAxis(values, w, factor, 1, false);
            values = BilinearAxis(values, h, factor, 0, false);
          }
        } else if (backward) {
          // Sum [rh,h,rw,w,c,n] over the two repeated spatial axes.
          values = af::sum(af::moddims(values, r, h, r * w, cn), 0);
          values = af::sum(af::moddims(values, h, r, w, cn), 1);
        } else {
          values = af::tile(af::moddims(values, 1, h, w, cn), factor, 1, 1, 1);
          values =
              af::tile(af::moddims(values, r * h, 1, w, cn), 1, factor, 1, 1);
        }
        const auto &shape = backward ? low : high;
        values = af::moddims(
            values, static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]),
            static_cast<dim_t>(shape[2]), static_cast<dim_t>(shape[3]));
        values.eval();
        return Tensor::FromSemanticArray(values, shape);
      } catch (const af::exception &error) {
        RecordLayerArrayFireFallback(operation, error.what(), input, "tensor");
      }
    }
  }
#else
  RecordLayerArrayFireFallback(
      operation, BackendFallbackReason::BackendUnavailable,
      "ArrayFire support is not compiled", input, "tensor");
#endif
  const ScopedArrayFireHostSyncAttribution attribution(
      ArrayFireHostSyncCategory::LayerCpuPath, operation);
  return backward ? NativeBackward(input, low, factor, mode)
                  : NativeForward(input, high, factor, mode);
}
} // namespace

Upsample2DLayer::Upsample2DLayer(int scale_factor, UpsampleMode mode)
    : scale_factor_(scale_factor), mode_(mode) {
  if (scale_factor_ <= 0) {
    throw std::invalid_argument("Upsample2D scale_factor must be positive");
  }
  if (mode_ != UpsampleMode::Nearest && mode_ != UpsampleMode::Bilinear) {
    throw std::invalid_argument("Upsample2D mode must be Nearest or Bilinear");
  }
}

Tensor Upsample2DLayer::Forward(const Tensor &input) {
  cached_input_ = Tensor();
  ValidateSpatial4DInput(input, "Upsample2D");
  const auto output_shape = UpsampleOutputShape(input.Shape(), scale_factor_);
  // Backward needs geometry, never input values. The existing metadata-only
  // Tensor constructor avoids retaining storage and preserves class ABI.
  Tensor backward_context(input.Shape(), nullptr, DataType::Float32);
  Tensor output =
      Execute(input, input.Shape(), output_shape, scale_factor_, mode_, false);
  cached_input_ = std::move(backward_context);
  return output;
}

Tensor Upsample2DLayer::Backward(const Tensor &grad_output) {
  if (cached_input_.Shape().empty()) {
    throw std::runtime_error(
        "Upsample2D::Backward requires a successful Forward call");
  }
  if (grad_output.GetDataType() != DataType::Float32) {
    throw std::runtime_error(
        "Upsample2D backward requires Float32 grad_output");
  }
  const auto output_shape =
      UpsampleOutputShape(cached_input_.Shape(), scale_factor_);
  if (grad_output.Shape() != output_shape) {
    throw std::runtime_error("Upsample2D backward gradient shape mismatch");
  }
  return Execute(grad_output, cached_input_.Shape(), output_shape,
                 scale_factor_, mode_, true);
}

} // namespace cyxwiz
