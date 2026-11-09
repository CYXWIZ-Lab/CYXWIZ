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
#include "engine_config.h"
#include "layer_capture.h"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
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
// The row's kind says what the tensors show: "Conv2D 3 × 3, stride 2"
// (kernel from the weights, stride from the input and output sizes),
// "Dense, 257 parameters".
LayerRow BuildLayerRow(const LayerInfo& layer);

// The layer list's sections (6.2e): consecutive layers whose nodes share a
// canvas group, named by it; layers in no group form unnamed sections. No
// groups at all: one unnamed section.
struct LayerSection {
    std::string name;
    std::vector<size_t> layers;  // indices into the layer list
};
std::vector<LayerSection> SectionLayers(const std::vector<LayerInfo>& layers,
                                        const std::map<int, std::string>& group_of_node);

enum class View { FeatureMaps, Values, Units, Weights, Gradients, Filters, WhereItLooks };

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
// see colours (its Filters view). `takes_model_input`: the layer the model's
// input enters, which gets Where it looks (6.3b) for an image, tokens or
// features.
LayerViews PlanLayerViews(const LayerInfo& layer, const DebugTensorNumericSummary* output_numerics,
                          bool first_spatial_layer, bool takes_model_input = false);

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
// `kernel_position` (conv weights only, ky * kw + kx): one kernel position's
// slice, output channels by input channels, instead of all positions.
MatrixLayout LayoutAsMatrix(const TensorInfo& tensor, std::optional<size_t> kernel_position = std::nullopt);
// The element a heatmap cell shows, in the tensor's own index:
// "[2, 1, 17, 40] = -0.0312". Row-major flat index = MatrixFlatIndex.
size_t MatrixFlatIndex(const TensorInfo& tensor, size_t row, size_t column,
                       std::optional<size_t> kernel_position = std::nullopt);
// Conv weights' kernel positions (kh * kw) when there are several, else 0.
size_t KernelPositions(const TensorInfo& tensor);
// "(1, 2) of the 3 × 3 kernel".
std::string KernelPositionText(const TensorInfo& tensor, size_t kernel_position);
std::string ElementText(const TensorInfo& tensor, size_t flat_index, float value);

// The first layer's filters as pictures (6.3a), from conv weights
// [kh, kw, cin, cout] with 1 (grey) or 3 (RGB) input channels, read row-major.
// One picture per output channel: kh x kw pixels of cin values in [0, 1],
// each filter scaled to its own range over all its channels (so its colours
// keep their balance; a constant filter is mid grey), and its weight norm.
struct FilterPicture {
    std::vector<float> pixels;  // row-major [kh, kw, cin]
    float norm = 0.0f;          // the filter's L2 norm
};
struct FilterPictures {
    size_t height = 0;
    size_t width = 0;
    size_t channels = 0;  // 1 or 3
    std::vector<FilterPicture> filters;
};
// nullopt when the shape is not such weights or the values do not fill it.
std::optional<FilterPictures> BuildFilterPictures(const std::vector<float>& weights, const std::vector<size_t>& shape);
// "The 32 learned 3 × 3 RGB filters of the first layer, each scaled to its
// own range. ..."
std::string FiltersText(const FilterPictures& pictures);
// The Inspector's note for the Filters view.
const std::string& FiltersHint();

// Channel numbers in the chosen order: largest mean first (Strongest),
// largest maximum first (Peak) or by number; ties by number, NaN last.
std::vector<size_t> OrderChannels(const std::vector<float>& channel_means, const std::vector<float>& channel_maxima,
                                  core::ChannelSort sort);
// "strongest first", "highest peak first", "in channel order".
const char* ChannelSortText(core::ChannelSort sort);
// "The 16 strongest of 32 channels for this sample, strongest first.",
// "All 32 of 32 ...", "The first 16 of 64 channels ..., highest peak first."
std::string ChannelsShownText(size_t shown, size_t count, bool all, core::ChannelSort sort);
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

// How the model's output turns into an answer (6.2e), from the loss and the
// last layer: one logit through a sigmoid (BCE with logits), logits through a
// softmax (cross-entropy), outputs that already are probabilities (a Sigmoid
// or Softmax last layer), or values (regression).
enum class OutputRule { Unknown, Sigmoid, Softmax, Probability, Probabilities, Value };
const char* OutputRuleKey(OutputRule rule);  // "sigmoid", ... (saved with the run)
OutputRule OutputRuleFromKey(std::string_view key);

// "cat" for class 0 of {"cat", "dog"}; "class 3" when the name is unknown
// or only a number.
std::string ClassName(int index, const std::vector<std::string>& class_names);

struct Prediction {
    int class_index = -1;     // -1 for a value
    double confidence = 0.0;  // of the class, 0..1
    std::string text;         // "cat, 99.0 %", or the value "3.142"
    std::string detail;       // "logit -4.556 → sigmoid 0.0104 (dog is 1)"
};
// nullopt when the rule is Unknown or the output does not fit it.
std::optional<Prediction> Predict(const std::vector<float>& output, OutputRule rule,
                                  const std::vector<std::string>& class_names);

// Each item's label from a batch's labels ([N, classes] one-hot rows ->
// the hot class; [N] or [N, 1] -> the rounded value); -1 when not a class.
std::vector<int> ItemLabels(const std::vector<float>& values, const std::vector<size_t>& shape);

// The status line's sample: "cat.10599.jpg · label cat · predicted cat, 99.0 %",
// leaving out what is unknown.
std::string SampleHeadline(const std::string& item_name, const std::string& label,
                           const std::optional<Prediction>& prediction);

}  // namespace cyxwiz::layer_view
