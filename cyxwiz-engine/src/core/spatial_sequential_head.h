#pragma once

#include "graph_compiler.h"
#include "spatial_layer_shapes.h"

#include <limits>
#include <optional>

namespace cyxwiz {

// The spatial section of a sequential model (TOFIX140 A1): the layers that
// run on [H,W,C,N] tensors, from the first spatial layer up to the Flatten or
// global pool (Global Avg / Max Pool) that turns the sample back into a row
// for Dense. A model may
// also stay spatial to its end (neither: it outputs [H,W,C,N]).
struct SpatialSequentialHead {
  static constexpr size_t kNoFlatten = std::numeric_limits<size_t>::max();
  size_t flatten_index = kNoFlatten;         // the Flatten / global pool that ends the section
  bool global_pool = false;                  // it is a Global Avg / Max Pool ([N,C] rows)
  std::vector<size_t> sample_shape;          // [H,W,C] entering it
  size_t features = 0;                       // row width after it: H*W*C, or C
  // The [H,W,C] sample entering each spatial-section layer (index = layer
  // index); empty entries when the model has no input_shape.
  std::vector<std::vector<size_t>> input_shapes;
};

// A model whose first layer is spatial takes its rows as [H,W,C,N]
// (TrainingExecutor / TestExecutor wrap the batch with SpatialBatchFromRows).
inline bool UsesSpatialSequentialInput(const TrainingConfiguration &config) {
  return !config.layers.empty() &&
         (spatial::IsSpatialLayer(config.layers.front().type) ||
          spatial::IsGlobalPoolLayer(config.layers.front().type));
}

// Resolves the section. Only an explicit first spatial layer opts in;
// ordinary tabular/sequence paths are untouched. Throws std::invalid_argument
// with the layer index when the section is not well formed.
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

  SpatialSequentialHead head;
  bool closed = false;
  for (size_t i = 0; i < config.layers.size(); ++i) {
    const auto &layer = config.layers[i];
    const bool spatial_layer = spatial::IsSpatialLayer(layer.type);
    if (closed) {
      if (spatial_layer || spatial::IsGlobalPoolLayer(layer.type))
        throw std::invalid_argument(
            "Spatial layers cannot follow the row Flatten head (index " +
            std::to_string(i) + ")");
      continue;
    }
    if (layer.type == gui::NodeType::Flatten || spatial::IsGlobalPoolLayer(layer.type)) {
      const bool global_pool = spatial::IsGlobalPoolLayer(layer.type);
      if (shape.empty())
        throw std::invalid_argument(
            std::string(global_pool ? "A global pool" : "Spatial Flatten") +
            " requires explicit input_shape [H,W,C]");
      head.flatten_index = i;
      head.global_pool = global_pool;
      head.sample_shape = shape;
      head.features = global_pool ? shape[2] : SpatialSampleElements(shape);
      closed = true;
      continue;
    }
    // PReLU's per-feature slopes act on dimension 1, which is W in [H,W,C,N]:
    // only the shared slope is element-wise there.
    const bool shared_prelu = layer.type == gui::NodeType::PReLU &&
                              spatial::ParseIntParam(layer.parameters, "num_parameters", 1) == 1;
    if (layer.type == gui::NodeType::PReLU && !shared_prelu)
      throw std::invalid_argument(
          "PReLU before Flatten needs one shared slope (num_parameters = 1); per-channel "
          "slopes are not supported on [H,W,C] samples (index " + std::to_string(i) + ")");
    if (!spatial_layer && !spatial::IsShapePreservingLayer(layer.type) && !shared_prelu &&
        layer.type != gui::NodeType::Output)
      throw std::invalid_argument(
          "Spatial head requires Flatten or a global pool before layer index " + std::to_string(i) +
          "; only convolution, pooling, normalisation, upsampling and "
          "activation layers run before Flatten");
    head.input_shapes.resize(i + 1);
    head.input_shapes[i] = shape;
    if (!spatial_layer) continue;
    const char *what = layer.type == gui::NodeType::Upsample ||
                               layer.type == gui::NodeType::PixelShuffle
                           ? "upsampling"
                           : "layer";
    try {
      if (shape.empty()) {
        // No sample shape yet: still validate the persisted parameters.
        if (layer.type == gui::NodeType::Upsample ||
            layer.type == gui::NodeType::PixelShuffle) {
          UpsamplingConfiguration resolved;
          if (auto error = ResolveUpsamplingConfiguration(
                  layer.type, layer.parameters, resolved, layer.scale_factor,
                  layer.upsample_mode))
            throw std::invalid_argument(*error);
        } else {
          spatial::ResolveGeometry(layer.type, layer.parameters, 0);
        }
      } else {
        shape = spatial::SampleShapeAfter(layer.type, layer.parameters, shape);
      }
    } catch (const std::invalid_argument &error) {
      throw std::invalid_argument(std::string("invalid ") + what +
                                  " configuration at index " +
                                  std::to_string(i) + ": " + error.what());
    }
  }
  return head;
}

} // namespace cyxwiz
