#pragma once

#include "graph_compiler.h"
#include "spatial_layer_shapes.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

// The 1-D (Conv1D) section of a sequential model (TOFIX140): the layers that
// run on [L,C,N] sequences, from the first Conv1D up to the Flatten or Global
// Avg / Max Pool that turns each sample back into a row for Dense. It opens either
// on the model's input rows (Conv1D first: time-series windows, audio
// features, table rows) or after an Embedding ([N,L,E] token vectors).
// The compiler reports its rules; ModelBuilder builds from it.
struct SequenceConvSection {
  static constexpr size_t kNone = std::numeric_limits<size_t>::max();
  size_t open_index = kNone;                  // the first Conv1D
  bool from_rows = false;                     // opens on the input rows, else after an Embedding
  size_t close_index = kNone;                 // the Flatten / global pool that ends it
  bool global_pool = false;                   // it is a Global Avg / Max Pool ([N,C] rows)
  std::vector<size_t> close_sample;           // [L,C] entering the close
  size_t features = 0;                        // row width after it: C*L, or C
  // The [L,C] sample entering each section layer (index = layer index);
  // empty when the input length is not known yet (audio before its probe).
  std::vector<std::vector<size_t>> input_shapes;
};

// The [L,C] sample the input rows hold when Conv1D is the first layer. The
// rows are channel-major, torch's [N,C,L]: time-series windows and audio
// features say so in `sequence_input_shape`; plain table rows are one channel.
inline std::vector<size_t> SequenceInputSample(const TrainingConfiguration &config) {
  if (!config.sequence_input_shape.empty()) return config.sequence_input_shape;
  if (config.input_shape.size() == 1 && config.preprocessing_domain != PreprocessingDomain::Audio)
    return {config.input_shape[0], 1};
  return {};
}

// Throws std::invalid_argument with the layer index when the section is not
// well formed; nullopt when the model has no Conv1D.
inline std::optional<SequenceConvSection>
ResolveSequenceConvSection(const TrainingConfiguration &config) {
  const auto &layers = config.layers;
  size_t open = SequenceConvSection::kNone;
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i].type == gui::NodeType::Conv1D) {
      open = i;
      break;
    }
  }
  if (open == SequenceConvSection::kNone) return std::nullopt;

  SequenceConvSection section;
  section.open_index = open;
  std::vector<size_t> shape;
  if (open == 0) {
    section.from_rows = true;
    shape = SequenceInputSample(config);
  } else {
    size_t before = open;
    while (before > 0 && spatial::IsShapePreservingLayer(layers[before - 1].type)) --before;
    if (before == 0 || layers[before - 1].type != gui::NodeType::Embedding)
      throw std::invalid_argument(
          "Conv1D runs on [L, C] sequences: make it the first model layer (time-series windows, audio "
          "features, table rows) or put it after an Embedding (index " + std::to_string(open) + ")");
    if (layers[before - 1].output_shape.size() == 2) shape = layers[before - 1].output_shape;
  }

  bool closed = false;
  for (size_t i = open; i < layers.size(); ++i) {
    const auto &layer = layers[i];
    if (closed) {
      if (layer.type == gui::NodeType::Conv1D)
        throw std::invalid_argument("Conv1D cannot follow the Flatten or global pool that ended the "
                                    "sequence section (index " + std::to_string(i) + ")");
      continue;
    }
    if (layer.type == gui::NodeType::Flatten || spatial::IsGlobalPoolLayer(layer.type)) {
      section.close_index = i;
      section.global_pool = spatial::IsGlobalPoolLayer(layer.type);
      section.close_sample = shape;
      if (shape.size() == 2) section.features = section.global_pool ? shape[1] : shape[0] * shape[1];
      closed = true;
      continue;
    }
    if (layer.type == gui::NodeType::Output) continue;
    section.input_shapes.resize(i + 1);
    section.input_shapes[i] = shape;
    if (spatial::IsShapePreservingLayer(layer.type)) continue;
    if (layer.type != gui::NodeType::Conv1D)
      throw std::invalid_argument(
          "The sequence section needs Flatten or a global pool before layer index " + std::to_string(i) +
          "; only Conv1D, activations and Dropout run on [L, C] sequences");
    try {
      if (shape.empty())
        spatial::ResolveGeometry(gui::NodeType::Conv1D, layer.parameters, 0);
      else
        shape = spatial::Conv1DSampleShapeAfter(layer.parameters, shape);
    } catch (const std::invalid_argument &error) {
      throw std::invalid_argument("invalid Conv1D configuration at index " + std::to_string(i) + ": " +
                                  error.what());
    }
  }
  if (!closed)
    throw std::invalid_argument(
        "End the sequence section with Flatten or a global pool before Dense and the loss (index " +
        std::to_string(open) + ")");
  return section;
}

} // namespace cyxwiz
