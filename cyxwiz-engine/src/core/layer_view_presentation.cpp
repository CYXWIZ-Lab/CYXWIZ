#include "layer_view_presentation.h"

#include "count_text.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace cyxwiz::layer_view {

namespace {

constexpr const char* kTimes = " \xC3\x97 ";  // " × "
constexpr const char* kDot = " \xC2\xB7 ";    // " · "

size_t Elements(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

std::string Number(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4g", value);
    return buffer;
}

std::string Percent(double fraction) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f %%", fraction * 100.0);
    return buffer;
}

std::string Bracketed(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
    return text + "]";
}

bool IsSpatial(const std::vector<size_t>& shape) {
    return shape.size() == 4 && shape[0] > 1 && shape[1] > 1;
}

const TensorInfo* Find(const LayerInfo& layer, CapturedTensorRole role, const std::string& name = {}) {
    for (const auto& t : layer.tensors)
        if (t.role == role && (name.empty() || t.name == name)) return &t;
    return nullptr;
}

// The parameter the Weights view shows: the largest of rank 2 or more, else
// the largest (a BatchNorm's gamma).
const TensorInfo* MainParameter(const LayerInfo& layer) {
    const TensorInfo* best = nullptr;
    for (const auto& t : layer.tensors) {
        if (t.role != CapturedTensorRole::Parameter) continue;
        const bool better = !best || (t.shape.size() >= 2 && best->shape.size() < 2) ||
                            ((t.shape.size() >= 2) == (best->shape.size() >= 2) &&
                             Elements(t.shape) > Elements(best->shape));
        if (better) best = &t;
    }
    return best;
}

}  // namespace

LayerInfo DescribeCapturedLayer(const CapturedLayer& layer, std::string kind) {
    LayerInfo info;
    info.module_index = layer.module_index;
    info.node_id = layer.node_id;
    info.name = layer.node_name;
    info.kind = std::move(kind);
    for (const auto& t : layer.tensors) info.tensors.push_back({t.role, t.name, t.tensor.Shape()});
    return info;
}

std::string ShapeText(const std::vector<size_t>& shape) {
    if (shape.empty()) return "-";
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) text += (i ? kTimes : "") + std::to_string(shape[i]);
    return text;
}

std::vector<size_t> PerSampleShape(const std::vector<size_t>& shape) {
    if (shape.size() == 4) return {shape[0], shape[1], shape[2]};
    if (shape.size() >= 2) return std::vector<size_t>(shape.begin() + 1, shape.end());
    return shape;
}

LayerRow BuildLayerRow(const LayerInfo& layer) {
    LayerRow row;
    row.name = layer.name.empty() ? "Layer " + std::to_string(layer.module_index) : layer.name;
    row.kind = layer.kind;
    const TensorInfo* output = Find(layer, CapturedTensorRole::Output);
    row.output = output ? ShapeText(PerSampleShape(output->shape)) : "-";
    return row;
}

LayerViews PlanLayerViews(const LayerInfo& layer, const DebugTensorNumericSummary* output_numerics,
                          bool first_spatial_layer) {
    LayerViews views;
    const TensorInfo* output = Find(layer, CapturedTensorRole::Output);
    const TensorInfo* input = Find(layer, CapturedTensorRole::Input);
    const TensorInfo* weights = MainParameter(layer);

    if (output && IsSpatial(output->shape)) {
        views.tabs.push_back({View::FeatureMaps, "Feature maps", {}});
    } else if (output && output->shape.size() == 2 && output->shape[1] > 1) {
        views.tabs.push_back({View::Units, "Units", {}});
    }
    if (output) views.tabs.push_back({View::Values, "Values", {}});
    if (weights) {
        views.tabs.push_back({View::Weights, "Weights", weights->name});
        if (Find(layer, CapturedTensorRole::Gradient, weights->name))
            views.tabs.push_back({View::Gradients, "Gradients", weights->name});
    }

    if (first_spatial_layer && input && IsSpatial(input->shape)) {
        views.later.push_back({"Where it looks", "Occlusion: a grey patch slides over the image and maps the drop in "
                                                 "the true class's score (next step)"});
    }
    if (first_spatial_layer && weights && weights->shape.size() == 4 &&
        (weights->shape[2] == 1 || weights->shape[2] == 3)) {
        views.later.push_back({"Filters", "The first layer's filters drawn as small images: only it sees the "
                                          "colours (next step)"});
    }
    if (weights && weights->shape.size() == 2 && input && input->shape.size() == 2) {
        views.later.push_back({"Contributions", "How this sample's output was computed: w \xC3\x97 x per input, "
                                                "the largest first, then the bias and the activation"});
    }
    if (output && output->shape.size() == 2 && output->shape[1] > 1) {
        views.later.push_back({"Embedding", "This layer's output over many samples, projected to 2-D (Explore "
                                            "\xE2\x80\xBA Projection), coloured by class"});
    }

    std::string facts;
    if (weights) {
        size_t parameters = 0;
        std::string rest;
        for (const auto& t : layer.tensors) {
            if (t.role != CapturedTensorRole::Parameter) continue;
            parameters += Elements(t.shape);
            if (&t != weights) rest += " + " + t.name + " " + Bracketed(t.shape);
        }
        facts = weights->name + " " + Bracketed(weights->shape) + rest + ", " + CountText(parameters) +
                " parameters";
    }
    if (output) {
        if (!facts.empty()) facts += kDot;
        facts += "output " + ShapeText(PerSampleShape(output->shape));
        if (output_numerics && output_numerics->available && output_numerics->finite_count > 0) {
            facts += ", " + Percent(static_cast<double>(output_numerics->zero_count) /
                                    static_cast<double>(output_numerics->finite_count)) + " zeros";
        }
    }
    views.facts = facts;

    if (weights && weights->shape.size() >= 2) {
        views.hint = "Any tensor of 2 or more dimensions draws as a heatmap: Dense weights [out, in], conv weights "
                     "as [out, kernel \xC3\x97 in]. Hover a cell for its value and index.";
    } else if (output && !IsSpatial(output->shape)) {
        views.hint = "Values shows every number of this layer's output for the sample, with its range and the "
                     "share of zeros.";
    }
    return views;
}

std::optional<SpatialShape> AsSpatial(const std::vector<size_t>& shape) {
    if (!IsSpatial(shape)) return std::nullopt;
    return SpatialShape{shape[0], shape[1], shape[2], shape[3]};
}

MatrixLayout LayoutAsMatrix(const TensorInfo& tensor) {
    const auto& s = tensor.shape;
    MatrixLayout layout;
    const bool weight_like = tensor.role == CapturedTensorRole::Parameter ||
                             tensor.role == CapturedTensorRole::Gradient;
    if (s.size() <= 1) {
        layout.rows = 1;
        layout.columns = s.empty() ? 0 : s[0];
        layout.description = CountText(layout.columns) + " values";
    } else if (s.size() == 4 && weight_like) {
        layout.rows = s[3];
        layout.columns = s[0] * s[1] * s[2];
        layout.description = CountText(s[3]) + " output channels (rows) by " + CountText(layout.columns) +
                             " inputs (" + std::to_string(s[0]) + kTimes + std::to_string(s[1]) + " kernel" +
                             kTimes + CountText(s[2]) + " input channels)";
    } else if (s.size() == 2 && weight_like) {
        layout.rows = s[0];
        layout.columns = s[1];
        layout.description = CountText(s[0]) + " outputs (rows) by " + CountText(s[1]) + " inputs";
    } else {
        layout.rows = s[0];
        layout.columns = Elements(s) / std::max<size_t>(1, s[0]);
        layout.description = CountText(layout.rows) + " rows by " + CountText(layout.columns) + " values";
    }
    return layout;
}

size_t MatrixFlatIndex(const TensorInfo& tensor, size_t row, size_t column) {
    const auto& s = tensor.shape;
    const bool weight_like = tensor.role == CapturedTensorRole::Parameter ||
                             tensor.role == CapturedTensorRole::Gradient;
    // Conv weights [kh, kw, cin, cout]: the row is the output channel, the
    // column runs over (ky, kx, cin) in row-major order.
    if (s.size() == 4 && weight_like) return column * s[3] + row;
    return row * LayoutAsMatrix(tensor).columns + column;
}

std::string ElementText(const TensorInfo& tensor, size_t flat_index, float value) {
    std::vector<size_t> index(tensor.shape.size());
    size_t rest = flat_index;
    for (size_t i = tensor.shape.size(); i-- > 0;) {
        const size_t extent = std::max<size_t>(1, tensor.shape[i]);
        index[i] = rest % extent;
        rest /= extent;
    }
    return Bracketed(index) + " = " + Number(value);
}

std::vector<size_t> StrongestFirst(const std::vector<float>& means) {
    std::vector<size_t> order(means.size());
    std::iota(order.begin(), order.end(), size_t{0});
    const auto key = [&means](size_t i) { return std::isfinite(means[i]) ? means[i] : -INFINITY; };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key(a) > key(b); });
    return order;
}

size_t CountDeadChannels(const std::vector<float>& maxima) {
    return static_cast<size_t>(std::count(maxima.begin(), maxima.end(), 0.0f));
}

std::vector<float> ToSequential(const std::vector<float>& values, float high) {
    std::vector<float> out(values.size(), 0.0f);
    if (!(high > 0.0f)) return out;
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = std::isfinite(values[i]) ? std::clamp(values[i] / high, 0.0f, 1.0f) : 0.0f;
    return out;
}

std::vector<float> ToDiverging(const std::vector<float>& values, float limit) {
    std::vector<float> out(values.size(), 0.0f);
    if (!(limit > 0.0f)) return out;
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = std::isfinite(values[i]) ? std::clamp(values[i] / limit, -1.0f, 1.0f) : 0.0f;
    return out;
}

float HistogramQuantile(const std::vector<unsigned>& counts, float low, float high, double fraction) {
    const double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (counts.empty() || total <= 0.0) return high;
    const double target = std::clamp(fraction, 0.0, 1.0) * total;
    const double width = (static_cast<double>(high) - low) / static_cast<double>(counts.size());
    double running = 0.0;
    for (size_t i = 0; i < counts.size(); ++i) {
        running += counts[i];
        if (running >= target) return static_cast<float>(low + width * static_cast<double>(i + 1));
    }
    return high;
}

std::vector<float> HistogramBars(const std::vector<unsigned>& counts, bool log_scale) {
    std::vector<float> bars(counts.size(), 0.0f);
    double top = 0.0;
    for (unsigned c : counts) top = std::max(top, log_scale ? std::log10(c + 1.0) : static_cast<double>(c));
    if (top <= 0.0) return bars;
    for (size_t i = 0; i < counts.size(); ++i) {
        const double v = log_scale ? std::log10(counts[i] + 1.0) : static_cast<double>(counts[i]);
        bars[i] = static_cast<float>(v / top);
    }
    return bars;
}

std::vector<std::string> NumbersLines(const DebugTensorNumericSummary& n, bool show_zeros) {
    std::vector<std::string> lines;
    if (!n.available) {
        lines.push_back("No summary: the tensor is not Float32.");
        return lines;
    }
    if (n.finite_count > 0) {
        lines.push_back("min " + Number(n.minimum) + kDot + "max " + Number(n.maximum) + kDot + "mean " +
                        Number(n.mean));
        if (show_zeros)
            lines.push_back(Percent(static_cast<double>(n.zero_count) / static_cast<double>(n.finite_count)) +
                            " zeros");
    }
    std::string count = CountText(n.element_count) + " values, ";
    if (n.nan_count == 0 && n.inf_count == 0) {
        count += "no NaN or Inf";
    } else {
        count += CountText(n.nan_count) + " NaN, " + CountText(n.inf_count) + " Inf";
    }
    lines.push_back(count);
    return lines;
}

}  // namespace cyxwiz::layer_view
