#include "cyxwiz/layers/upsampling.h"
#include "layer_arrayfire_utils.h"
#include "layer_utils.h"

#include <stdexcept>
#include <vector>

namespace cyxwiz {
namespace {
constexpr const char *name = "PixelShuffle";

struct Geometry {
  std::vector<size_t> low;
  std::vector<size_t> high;
  size_t factor;
  size_t area;
  size_t positions;
  size_t channel_batches;
};

Geometry Validate(const Tensor &input, int factor, bool inverse) {
  ValidateSpatial4DInput(input, name);
  const auto &s = input.Shape();
  const size_t r = static_cast<size_t>(factor);
  const size_t area = CheckedLayerProduct(r, r, name, "factor square");
  Geometry g;
  g.factor = r;
  g.area = area;
  if (inverse) {
    if (s[0] % r != 0 || s[1] % r != 0) {
      throw std::runtime_error("PixelShuffle backward spatial shape must be "
                               "divisible by upscale_factor");
    }
    g.high = s;
    g.low = {s[0] / r, s[1] / r,
             CheckedLayerProduct(s[2], area, name, "input channels"), s[3]};
  } else {
    if (s[2] % area != 0) {
      throw std::runtime_error(
          "PixelShuffle input channels must be divisible by upscale_factor^2");
    }
    g.low = s;
    g.high = {CheckedLayerProduct(s[0], r, name, "output height"),
              CheckedLayerProduct(s[1], r, name, "output width"), s[2] / area,
              s[3]};
  }
  g.positions =
      CheckedLayerProduct(g.low[0], g.low[1], name, "spatial positions");
  g.channel_batches =
      CheckedLayerProduct(g.high[2], s[3], name, "channel batches");
  const size_t elements = CheckedLayerProduct(
      CheckedLayerProduct(g.positions, g.low[2], name, "element count"), s[3],
      name, "element count");
  (void)CheckedLayerProduct(elements, sizeof(float), name, "byte count");
#ifdef CYXWIZ_HAS_ARRAYFIRE
  for (size_t dim : {g.low[0], g.low[1], g.low[2], s[3], g.high[0], g.high[1],
                     g.high[2], r, g.positions, g.channel_batches}) {
    (void)CheckedIntDim(dim, "PixelShuffle dimension");
  }
#endif
  return g;
}

Tensor Execute(const Tensor &input, int factor, bool inverse) {
  const Geometry g = Validate(input, factor, inverse);
  const char *operation =
      inverse ? "PixelShuffleLayer::Backward" : "PixelShuffleLayer::Forward";
#ifdef CYXWIZ_HAS_ARRAYFIRE
  {
    try {
      const auto h = static_cast<dim_t>(g.low[0]);
      const auto w = static_cast<dim_t>(g.low[1]);
      const auto r = static_cast<dim_t>(g.factor);
      const auto positions = static_cast<dim_t>(g.positions);
      const auto cn = static_cast<dim_t>(g.channel_batches);
      af::array values = input.GetSemanticArray();
      if (inverse) {
        // [rh,h,rw,w,c,n] -> [h,w,rw,rh,c,n].
        values = af::reorder(af::moddims(values, r * h, r, w, cn), 0, 2, 1, 3);
        values =
            af::reorder(af::moddims(values, r, positions, r, cn), 1, 2, 0, 3);
      } else {
        // Split c as [rw,rh,c_out], then interleave spatial offsets.
        // Two reorders suffice; the number of launches is independent of r.
        values =
            af::reorder(af::moddims(values, positions, r, r, cn), 2, 0, 1, 3);
        values = af::reorder(af::moddims(values, r * h, w, r, cn), 0, 2, 1, 3);
      }
      const auto &shape = inverse ? g.low : g.high;
      values = af::moddims(
          values, static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[1]),
          static_cast<dim_t>(shape[2]), static_cast<dim_t>(shape[3]));
      values.eval();
      return Tensor::FromSemanticArray(values, shape);
    } catch (const af::exception &error) {
      throw std::runtime_error(std::string(operation) +
                               " failed on the ArrayFire device: " + error.what());
    }
  }
#else
  (void)g;
  (void)operation;
  throw std::runtime_error(
      "PixelShuffle runs on ArrayFire, and this build has no ArrayFire");
#endif
}
} // namespace

PixelShuffleLayer::PixelShuffleLayer(int upscale_factor)
    : upscale_factor_(upscale_factor) {
  if (upscale_factor_ <= 0) {
    throw std::invalid_argument("PixelShuffle upscale_factor must be positive");
  }
}

Tensor PixelShuffleLayer::Forward(const Tensor &input) {
  return Execute(input, upscale_factor_, false);
}

Tensor PixelShuffleLayer::Backward(const Tensor &grad_output) {
  // The inverse is stateless: no cached Forward input or full-data copy.
  return Execute(grad_output, upscale_factor_, true);
}
} // namespace cyxwiz
