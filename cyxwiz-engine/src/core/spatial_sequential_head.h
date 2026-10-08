#pragma once

#include "graph_compiler.h"
#include "spatial_layer_shapes.h"

#include <optional>

namespace cyxwiz {

// The spatial section of a sequential model (TOFIX140 A1): the layers that
// run on [H,W,C,N] tensors, from the first spatial layer to the Flatten
// that turns the sample back into a row for Dense.
struct SpatialSequentialHead {
  size_t flatten_index = 0;                  // the Flatten that ends the section
  std::vector<size_t> sample_shape;          // [H,W,C] entering the Flatten
  size_t features = 0;                       // H*W*C
  // The [H,W,C] sample entering each layer before the Flatten (index = layer index).
  std::vector<std::vector<size_t>> input_shapes;
};

// A model whose first layer is spatial takes its rows as [H,W,C,N]
// (TrainingExecutor / TestExecutor wrap the batch with SpatialBatchFromRows).
inline bool UsesSpatialSequentialInput(const TrainingConfiguration &config) {
  return !config.layers.empty() &&
         spatial::IsSpatialLayer(config.layers.front().type);
}

// Resolves the section. Only an explicit first spatial layer opts in;
// ordinary tabular/sequence paths are untouched. Throws std::invalid_argument
// when the section is not well formed (a non-spatial layer before the
// Flatten, a kernel that does not fit, no input shape).
inline std::optional<SpatialSequentialHead>
ResolveSpatialSequentialHead(const TrainingConfiguration &config) {
  if (!UsesSpatialSequentialInput(config))
    return std::nullopt;

  auto shape = config.input_shape;
  if (shape.size() != 3)
    throw std::invalid_argument(
        "Spatial layers need an image input_shape [H,W,C]; the Data Input gives " +
        std::to_string(shape.size()) + " dimensions");
  const auto elements = SpatialSampleElements(shape);
  if (config.input_size != 0 && config.input_size != elements)
    throw std::invalid_argument(
        "Spatial input_size must match input_shape [H,W,C]");

  SpatialSequentialHead head;
  bool closed = false;
  for (size_t i = 0; i < config.layers.size() && !closed; ++i) {
    const auto &layer = config.layers[i];
    if (layer.type == gui::NodeType::Flatten) {
      head.flatten_index = i;
      head.sample_shape = shape;
      head.features = SpatialSampleElements(shape);
      closed = true;
      continue;
    }
    if (!spatial::IsSpatialLayer(layer.type) && !spatial::IsShapePreservingLayer(layer.type))
      throw std::invalid_argument(
          "Spatial head requires Flatten before layer index " + std::to_string(i) +
          " (" + layer.name + "); only convolution, pooling, normalisation, "
          "upsampling and activation layers run before Flatten");
    head.input_shapes.push_back(shape);
    shape = spatial::SampleShapeAfter(layer.type, layer.parameters, shape);
  }
  if (!closed)
    throw std::invalid_argument(
        "Spatial layers need a Flatten before the Dense head (or the Output)");
  return head;
}

} // namespace cyxwiz
