#pragma once

// The Studio Debugger's layer views (TOFIX178 6.2): what to show for a
// captured layer, picked by its tensors' roles and shapes, never by layer
// kind, so every layer (and an extension node) gets the views its tensors
// support. Pure data in and out: shapes, roles and host numbers the
// renderer read for the part it shows; no ImGui, no device.
//
// Layouts (row-major, as Tensor::Shape() reports them):
//   spatial activations [H, W, C, N]; conv weights [kh, kw, cin, cout];
//   Dense weights [out, in]; flat activations [N, features].

#include "debug_numerics.h"
#include "layer_capture.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::layer_view {

struct TensorInfo {
    CapturedTensorRole role = CapturedTensorRole::Output;
    std::string name;
    std::vector<size_t> shape;
};

struct LayerInfo {
    size_t module_index = 0;
    int node_id = -1;
    std::string name;
    std::string kind;  // the node type's display name ("Conv2D"), may be empty
    std::vector<TensorInfo> tensors;
};

// The layer's description from what was captured (no tensors kept).
LayerInfo DescribeCapturedLayer(const CapturedLayer& layer, std::string kind);

// "64 × 64 × 32"; "-" for no dimensions.
std::string ShapeText(const std::vector<size_t>& shape);
// The shape of one sample of an activation: [H, W, C, N] -> [H, W, C],
// [N, F] -> [F]; other ranks keep their dimensions after the first.
std::vector<size_t> PerSampleShape(const std::vector<size_t>& activation_shape);

struct LayerRow {
    std::string name;
    std::string kind;
    std::string output;  // per sample, "64 × 64 × 32"
};
LayerRow BuildLayerRow(const LayerInfo& layer);

enum class View { FeatureMaps, Values, Units, Weights, Gradients };

struct ViewTab {
    View view = View::Values;
    std::string label;
    std::string tensor;  // the parameter or gradient the view reads ("weights")
};
struct LaterTab {
    std::string label;
    std::string why;  // hover text: what it will show and when
};
struct LayerViews {
    std::vector<ViewTab> tabs;
    std::vector<LaterTab> later;
    std::string facts;  // "output 64 × 64 × 32, 42.6 % zeros" (zeros when known and any)
    std::string hint;   // the inspector's note for views without a channel
};
// `output_numerics` (optional) adds the share of zeros to the facts.
// `first_spatial_layer`: the first layer that takes the image, whose filters
// see colours (the later Filters view).
LayerViews PlanLayerViews(const LayerInfo& layer, const DebugTensorNumericSummary* output_numerics,
                          bool first_spatial_layer);

struct SpatialShape {
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;
    size_t samples = 0;
};
// [H, W, C, N] with a picture in it (H and W above 1).
std::optional<SpatialShape> AsSpatial(const std::vector<size_t>& activation_shape);

// Any tensor as rows x columns for a heatmap, and what a cell is.
struct MatrixLayout {
    size_t rows = 0;
    size_t columns = 0;
    std::string description;  // "64 output channels (rows) by 288 inputs (3 × 3 kernel × 32 input channels)"
};
MatrixLayout LayoutAsMatrix(const TensorInfo& tensor);
// The element a heatmap cell shows, in the tensor's own index:
// "[2, 1, 17, 40] = -0.0312". Row-major flat index = MatrixFlatIndex.
size_t MatrixFlatIndex(const TensorInfo& tensor, size_t row, size_t column);
std::string ElementText(const TensorInfo& tensor, size_t flat_index, float value);

// Channel numbers ordered strongest (largest mean) first; ties by number.
std::vector<size_t> StrongestFirst(const std::vector<float>& channel_means);
// Channels whose maximum is 0 everywhere (a dead ReLU channel).
size_t CountDeadChannels(const std::vector<float>& channel_maxima);

// Pixel values for a scale: sequential 0..high -> [0, 1] (clamped);
// diverging -limit..limit -> [-1, 1]. Non-finite values map to 0.
std::vector<float> ToSequential(const std::vector<float>& values, float high);
std::vector<float> ToDiverging(const std::vector<float>& values, float limit);

// The value at `fraction` (0.995) of a histogram over [low, high]: the top of
// the bin where the running count reaches it.
float HistogramQuantile(const std::vector<unsigned>& counts, float low, float high, double fraction);
// Bar heights in [0, 1]; log counts for activations (zeros dwarf the rest).
std::vector<float> HistogramBars(const std::vector<unsigned>& counts, bool log_scale);

// "min 0 · max 6.81 · mean 0.286", "42.6 % zeros" (when there are zeros and
// `show_zeros`), "131,072 values, no NaN or Inf".
std::vector<std::string> NumbersLines(const DebugTensorNumericSummary& numerics, bool show_zeros);

}  // namespace cyxwiz::layer_view
