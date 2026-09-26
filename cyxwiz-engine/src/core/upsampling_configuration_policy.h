#pragma once

#include "graph_model.h"
#include "spatial_sample_shape.h"
#include <charconv>
#include <map>
#include <optional>
#include <string>

namespace cyxwiz {

struct UpsamplingConfiguration {
  int factor = 2;
  int mode = 0; // Persisted contract: 0=nearest, 1=bilinear.
};

// Shared exact configuration boundary, following the existing header-only
// policies used by independently linked compiler/model test targets. Persisted
// keys override legacy CompiledLayer fields; explicit malformed values never
// fall back to a different model. Runtime Tensor validation stays in the
// layers.
inline std::optional<std::string> ResolveUpsamplingConfiguration(
    gui::NodeType type, const std::map<std::string, std::string> &parameters,
    UpsamplingConfiguration &result, int compiled_factor = 2,
    int compiled_mode = 0) {
  result = {};
  if (type != gui::NodeType::Upsample && type != gui::NodeType::PixelShuffle)
    return std::nullopt;

  const auto read = [&](const char *key, int fallback, int minimum, int maximum,
                        int &value) -> std::optional<std::string> {
    value = fallback;
    if (const auto it = parameters.find(key); it != parameters.end()) {
      const auto &text = it->second;
      const auto parsed =
          std::from_chars(text.data(), text.data() + text.size(), value);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        return std::string(key) + " must be an integer in [" +
               std::to_string(minimum) + ", " + std::to_string(maximum) + "].";
    }
    if (value < minimum || value > maximum)
      return std::string(key) + " must be an integer in [" +
             std::to_string(minimum) + ", " + std::to_string(maximum) + "].";
    return std::nullopt;
  };
  // Preserve the range already declared in the node property schema.
  if (auto error = read(type == gui::NodeType::Upsample ? "scale_factor"
                                                        : "upscale_factor",
                        compiled_factor, 1, 1048576, result.factor))
    return error;
  if (type == gui::NodeType::Upsample)
    return read("mode", compiled_mode, 0, 1, result.mode);
  return std::nullopt;
}

// Compiler per-sample inference, not runtime layout conversion. Reuse the
// sample contract and reject invalid geometry before multiplying dimensions.
inline std::vector<size_t>
InferUpsamplingSampleShape(gui::NodeType type,
                           const UpsamplingConfiguration &configuration,
                           const std::vector<size_t> &input_shape) {
  if (type != gui::NodeType::Upsample && type != gui::NodeType::PixelShuffle)
    throw std::invalid_argument(
        "Upsampling shape requires Upsample or PixelShuffle");
  UpsamplingConfiguration checked;
  if (auto error = ResolveUpsamplingConfiguration(
          type, {}, checked, configuration.factor, configuration.mode))
    throw std::invalid_argument(*error);
  SpatialSampleElements(input_shape);
  const auto factor = static_cast<size_t>(checked.factor);
  const auto axis_limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (input_shape[0] > axis_limit / factor ||
      input_shape[1] > axis_limit / factor)
    throw std::overflow_error(
        "Upsampling output height/width exceeds backend int range");
  auto output_shape = input_shape;
  output_shape[0] *= factor;
  output_shape[1] *= factor;
  if (type == gui::NodeType::PixelShuffle) {
    // Divide twice instead of forming factor^2, which can overflow size_t on
    // 32-bit platforms for otherwise representable persisted scale factors.
    if (input_shape[2] % factor != 0 || (input_shape[2] / factor) % factor != 0)
      throw std::invalid_argument("PixelShuffle input channels must be "
                                  "divisible by upscale_factor squared");
    output_shape[2] = input_shape[2] / factor / factor;
  }
  SpatialSampleElements(output_shape);
  return output_shape;
}

} // namespace cyxwiz
