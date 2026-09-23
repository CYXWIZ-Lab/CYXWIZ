#pragma once

#include "graph_compiler.h"
#include "upsampling_configuration_policy.h"

namespace cyxwiz {

struct SpatialSequentialHead {
  size_t flatten_index = 0;
  std::vector<size_t> sample_shape;
  size_t features = 0;
};

inline bool UsesSpatialSequentialInput(const TrainingConfiguration &config) {
  return !config.layers.empty() &&
         (config.layers.front().type == gui::NodeType::Upsample ||
          config.layers.front().type == gui::NodeType::PixelShuffle);
}

// Bounded direct-builder contract, not a Studio support decision. Only an
// explicit first spatial layer opts in; ordinary tabular/sequence paths are
// untouched. Until other spatial producers are qualified, do not guess their
// layouts or let rank-sensitive modules consume a batch-last tensor.
inline std::optional<SpatialSequentialHead>
ResolveSpatialSequentialHead(const TrainingConfiguration &config) {
  if (!UsesSpatialSequentialInput(config))
    return std::nullopt;

  auto shape = config.input_shape;
  if (!shape.empty()) {
    const auto elements = SpatialSampleElements(shape);
    if (config.input_size != 0 && config.input_size != elements)
      throw std::invalid_argument(
          "Spatial input_size must match input_shape [H,W,C]");
  }
  std::optional<SpatialSequentialHead> head;
  for (size_t i = 0; i < config.layers.size(); ++i) {
    const auto &layer = config.layers[i];
    if (layer.type == gui::NodeType::Upsample ||
        layer.type == gui::NodeType::PixelShuffle) {
      if (head)
        throw std::invalid_argument(
            "Spatial layers cannot follow the row Flatten head");
      UpsamplingConfiguration resolved;
      if (auto error = ResolveUpsamplingConfiguration(
              layer.type, layer.parameters, resolved, layer.scale_factor,
              layer.upsample_mode))
        throw std::invalid_argument(
            "invalid upsampling configuration at index " + std::to_string(i) +
            ": " + *error);
      if (!shape.empty())
        shape = InferUpsamplingSampleShape(layer.type, resolved, shape);
    } else if (!head && layer.type == gui::NodeType::Flatten) {
      if (shape.empty())
        throw std::invalid_argument(
            "Spatial Flatten requires explicit input_shape [H,W,C]");
      head = SpatialSequentialHead{i, shape, SpatialSampleElements(shape)};
    } else if (!head && layer.type != gui::NodeType::ReLU &&
               layer.type != gui::NodeType::Output) {
      throw std::invalid_argument(
          "Spatial head requires Flatten before layer index " +
          std::to_string(i) +
          "; only Upsample, PixelShuffle and ReLU are qualified before "
          "Flatten");
    }
  }
  return head;
}

} // namespace cyxwiz
