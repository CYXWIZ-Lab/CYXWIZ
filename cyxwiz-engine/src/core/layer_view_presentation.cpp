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
    if (value == 0.0) value = 0.0;  // no "-0"
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

    // Kernel and parameters from the weights, stride from the sizes.
    const TensorInfo* weights = nullptr;
    size_t parameters = 0;
    for (const auto& t : layer.tensors) {
        if (t.role != CapturedTensorRole::Parameter) continue;
        parameters += Elements(t.shape);
        if (!weights || Elements(t.shape) > Elements(weights->shape)) weights = &t;
    }
    std::string detail;
    if (weights && weights->shape.size() == 4)
        detail += " " + std::to_string(weights->shape[0]) + kTimes + std::to_string(weights->shape[1]);
    const TensorInfo* input = Find(layer, CapturedTensorRole::Input);
    const auto in = input ? AsSpatial(input->shape) : std::nullopt;
    const auto out = output ? AsSpatial(output->shape) : std::nullopt;
    if (in && out && out->height < in->height && in->height % out->height == 0 && in->width % out->width == 0 &&
        in->height / out->height == in->width / out->width)
        detail += ", stride " + std::to_string(in->height / out->height);
    if (weights && weights->shape.size() == 2) detail += ", " + CountText(parameters) + " parameters";
    if (row.kind.empty() && !detail.empty()) detail.erase(0, detail[0] == ',' ? 2 : 1);
    row.kind += detail;
    return row;
}

std::vector<LayerSection> SectionLayers(const std::vector<LayerInfo>& layers,
                                        const std::map<int, std::string>& group_of_node) {
    std::vector<LayerSection> sections;
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto it = group_of_node.find(layers[i].node_id);
        const std::string name = it == group_of_node.end() ? std::string() : it->second;
        if (sections.empty() || sections.back().name != name) sections.push_back({name, {}});
        sections.back().layers.push_back(i);
    }
    return sections;
}

std::optional<Picture> SamplePicture(const std::vector<float>& values, const std::vector<size_t>& shape) {
    if (shape.size() != 3 || (shape[2] != 1 && shape[2] != 3) || values.size() != shape[0] * shape[1] * shape[2])
        return std::nullopt;
    float low = 0.0f, high = 0.0f;
    bool any = false;
    for (float v : values) {
        if (!std::isfinite(v)) continue;
        low = any ? std::min(low, v) : v;
        high = any ? std::max(high, v) : v;
        any = true;
    }
    const float span = std::max(1e-6f, high - low);
    Picture picture;
    picture.height = shape[0];
    picture.width = shape[1];
    const size_t channels = shape[2];
    picture.rgba.assign(picture.width * picture.height * 4, 255);
    for (size_t p = 0; p < picture.width * picture.height; ++p) {
        for (size_t ch = 0; ch < 3; ++ch) {
            const float v = values[p * channels + (channels == 3 ? ch : 0)];
            picture.rgba[p * 4 + ch] = static_cast<unsigned char>(
                std::lround(std::clamp((std::isfinite(v) ? v - low : 0.0f) / span, 0.0f, 1.0f) * 255.0f));
        }
    }
    return picture;
}

const std::string& TopImagesHint() {
    static const std::string hint =
        "Top images answer \"what makes this channel fire\": the items from the data where the channel is "
        "strongest. A channel whose top images all show the background looks at the background. With a Dense head "
        "right after the layer, its weight says which class a channel votes for. A Local Debug run's model has its "
        "starting weights; open a training snapshot to see a trained model.";
    return hint;
}

const std::string& EmbeddingHint() {
    static const std::string hint =
        "The embedding places each item from the data by this layer's values: items close together look alike to "
        "the layer. Classes that sit in their own clusters are already apart here, so the layers after it have an "
        "easy job. PCA keeps the large distances but lets clusters overlap; t-SNE and UMAP keep neighbours together, "
        "and the gaps between their clusters mean little. Colour by Predicted class rings the items the model gets wrong. "
        "Click a point to inspect that item.";
    return hint;
}

LayerViews PlanLayerViews(const LayerInfo& layer, const DebugTensorNumericSummary* output_numerics,
                          bool first_spatial_layer, bool takes_model_input, bool picture_input) {
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

    if (first_spatial_layer && weights && weights->shape.size() == 4 &&
        (weights->shape[2] == 1 || weights->shape[2] == 3)) {
        views.tabs.push_back({View::Filters, "Filters", weights->name});
    }

    if (takes_model_input && input && (IsSpatial(input->shape) || input->shape.size() == 2))
        views.tabs.push_back({View::WhereItLooks, "Where it looks", {}});
    if (weights && weights->shape.size() == 2 && input && input->shape.size() == 2)
        views.tabs.push_back({View::Contributions, "Contributions", weights->name});
    if (output && output->shape.size() == 2 && output->shape[1] > 1)
        views.tabs.push_back({View::Embedding, "Embedding", {}});
    if (output && (IsSpatial(output->shape) || (output->shape.size() == 2 && output->shape[1] > 1)))
        views.tabs.push_back({View::TopImages, picture_input ? "Top images" : "Top samples", {}});

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
        // Zeros only when there are any (a ReLU's dead share, not a conv's 0 %).
        if (output_numerics && output_numerics->available && output_numerics->finite_count > 0 &&
            output_numerics->zero_count > 0) {
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

MatrixLayout LayoutAsMatrix(const TensorInfo& tensor, std::optional<size_t> kernel_position) {
    const auto& s = tensor.shape;
    MatrixLayout layout;
    const bool weight_like = tensor.role == CapturedTensorRole::Parameter ||
                             tensor.role == CapturedTensorRole::Gradient;
    if (kernel_position && *kernel_position < KernelPositions(tensor)) {
        layout.rows = s[3];
        layout.columns = s[2];
        layout.description = CountText(s[3]) + " output channels (rows) by " + CountText(s[2]) +
                             " input channels at kernel position " + KernelPositionText(tensor, *kernel_position);
        return layout;
    }
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

size_t MatrixFlatIndex(const TensorInfo& tensor, size_t row, size_t column, std::optional<size_t> kernel_position) {
    const auto& s = tensor.shape;
    const bool weight_like = tensor.role == CapturedTensorRole::Parameter ||
                             tensor.role == CapturedTensorRole::Gradient;
    // Conv weights [kh, kw, cin, cout]: the row is the output channel, the
    // column runs over (ky, kx, cin) in row-major order; a slice's column is
    // the input channel at one kernel position.
    if (kernel_position && *kernel_position < KernelPositions(tensor)) column += *kernel_position * s[2];
    if (s.size() == 4 && weight_like) return column * s[3] + row;
    return row * LayoutAsMatrix(tensor).columns + column;
}

size_t KernelPositions(const TensorInfo& tensor) {
    const auto& s = tensor.shape;
    const bool weight_like = tensor.role == CapturedTensorRole::Parameter ||
                             tensor.role == CapturedTensorRole::Gradient;
    return s.size() == 4 && weight_like && s[0] * s[1] > 1 ? s[0] * s[1] : 0;
}

std::string KernelPositionText(const TensorInfo& tensor, size_t kernel_position) {
    const size_t kw = tensor.shape.size() == 4 ? std::max<size_t>(1, tensor.shape[1]) : 1;
    return "(" + std::to_string(kernel_position / kw) + ", " + std::to_string(kernel_position % kw) + ") of the " +
           std::to_string(tensor.shape.empty() ? 0 : tensor.shape[0]) + kTimes + std::to_string(kw) + " kernel";
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

std::optional<FilterPictures> BuildFilterPictures(const std::vector<float>& weights, const std::vector<size_t>& shape) {
    if (shape.size() != 4 || (shape[2] != 1 && shape[2] != 3) || shape[3] == 0 || Elements(shape) != weights.size())
        return std::nullopt;
    FilterPictures pictures;
    pictures.height = shape[0];
    pictures.width = shape[1];
    pictures.channels = shape[2];
    const size_t outputs = shape[3];
    const size_t per_filter = shape[0] * shape[1] * shape[2];
    pictures.filters.resize(outputs);
    for (size_t o = 0; o < outputs; ++o) {
        FilterPicture& picture = pictures.filters[o];
        picture.pixels.resize(per_filter);
        // Weights [kh, kw, cin, cout] row-major: filter o's value at
        // (y, x, c) sits at ((y * kw + x) * cin + c) * cout + o.
        for (size_t k = 0; k < per_filter; ++k) {
            const float v = weights[k * outputs + o];
            picture.pixels[k] = std::isfinite(v) ? v : 0.0f;
        }
        const auto [low, high] = std::minmax_element(picture.pixels.begin(), picture.pixels.end());
        const float lo = *low, span = *high - *low;
        double squares = 0.0;
        for (float& v : picture.pixels) {
            squares += static_cast<double>(v) * v;
            v = span > 0.0f ? (v - lo) / span : 0.5f;
        }
        picture.norm = static_cast<float>(std::sqrt(squares));
    }
    return pictures;
}

std::string FiltersText(const FilterPictures& pictures) {
    const size_t count = pictures.filters.size();
    return "The " + std::to_string(count) + " learned " + std::to_string(pictures.height) + kTimes +
           std::to_string(pictures.width) + (pictures.channels == 3 ? " RGB" : " grey") + " filter" +
           (count == 1 ? "" : "s") + " of the first layer, each scaled to its own range. " +
           (pictures.channels == 3 ? "Colour edges and blobs are what the first layer looks for."
                                   : "Edges and blobs are what the first layer looks for.");
}

const std::string& FiltersHint() {
    static const std::string hint =
        "A filter is drawn as the pattern it responds to: bright where its weights are high, dark where they are "
        "low, for each colour. The first layer slides it over the image; where a patch looks like the filter, that "
        "channel's feature map is bright. A larger weight norm means a stronger response.";
    return hint;
}

std::vector<size_t> OrderChannels(const std::vector<float>& means, const std::vector<float>& maxima,
                                  core::ChannelSort sort) {
    std::vector<size_t> order(means.size());
    std::iota(order.begin(), order.end(), size_t{0});
    if (sort == core::ChannelSort::Number) return order;
    const std::vector<float>& by = sort == core::ChannelSort::Peak && maxima.size() == means.size() ? maxima : means;
    const auto key = [&by](size_t i) { return std::isfinite(by[i]) ? by[i] : -INFINITY; };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key(a) > key(b); });
    return order;
}

const char* ChannelSortText(core::ChannelSort sort) {
    switch (sort) {
        case core::ChannelSort::Peak: return "highest peak first";
        case core::ChannelSort::Number: return "in channel order";
        case core::ChannelSort::Strongest: break;
    }
    return "strongest first";
}

std::string ChannelsShownText(size_t shown, size_t count, bool all, core::ChannelSort sort) {
    std::string text = all || shown >= count
        ? "All " + CountText(shown) + " of " + CountText(count)
        : (sort == core::ChannelSort::Strongest ? "The " + CountText(shown) + " strongest of "
                                                : "The first " + CountText(shown) + " of ") +
              CountText(count);
    return text + " channels for this sample, " + ChannelSortText(sort) + ".";
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
        if (show_zeros && n.zero_count > 0)
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

const char* OutputRuleKey(OutputRule rule) {
    switch (rule) {
        case OutputRule::Sigmoid: return "sigmoid";
        case OutputRule::Softmax: return "softmax";
        case OutputRule::Probability: return "probability";
        case OutputRule::Probabilities: return "probabilities";
        case OutputRule::Value: return "value";
        case OutputRule::Unknown: break;
    }
    return "";
}

OutputRule OutputRuleFromKey(std::string_view key) {
    for (OutputRule rule : {OutputRule::Sigmoid, OutputRule::Softmax, OutputRule::Probability,
                            OutputRule::Probabilities, OutputRule::Value})
        if (key == OutputRuleKey(rule)) return rule;
    return OutputRule::Unknown;
}

std::string ClassName(int index, const std::vector<std::string>& class_names) {
    // A name that is only a number (labels 0 / 1) reads as "class 1".
    if (index >= 0 && static_cast<size_t>(index) < class_names.size() && !class_names[index].empty() &&
        !std::all_of(class_names[index].begin(), class_names[index].end(),
                     [](char c) { return c >= '0' && c <= '9'; }))
        return class_names[index];
    return "class " + std::to_string(index);
}

std::optional<Prediction> Predict(const std::vector<float>& output, OutputRule rule,
                                  const std::vector<std::string>& class_names) {
    if (output.empty() || rule == OutputRule::Unknown) return std::nullopt;
    for (float v : output)
        if (!std::isfinite(v)) return std::nullopt;
    Prediction p;
    const auto percent = [](double share) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.1f %%", share * 100.0);
        return std::string(buffer);
    };
    if (rule == OutputRule::Value) {
        p.text = output.size() == 1 ? Number(output[0]) : CountText(output.size()) + " values";
        return p;
    }
    if (rule == OutputRule::Sigmoid || rule == OutputRule::Probability) {
        if (output.size() != 1) return std::nullopt;
        const double z = output[0];
        const double one = rule == OutputRule::Sigmoid ? 1.0 / (1.0 + std::exp(-z)) : std::clamp(z, 0.0, 1.0);
        p.class_index = one >= 0.5 ? 1 : 0;
        p.confidence = p.class_index == 1 ? one : 1.0 - one;
        char buffer[96];
        if (rule == OutputRule::Sigmoid)
            std::snprintf(buffer, sizeof(buffer), "logit %s \xE2\x86\x92 sigmoid %.4f", Number(output[0]).c_str(), one);
        else
            std::snprintf(buffer, sizeof(buffer), "output %.4f", one);
        p.detail = std::string(buffer) + " (" + ClassName(1, class_names) + " is 1)";
    } else {
        if (output.size() < 2) return std::nullopt;
        const size_t best = static_cast<size_t>(std::max_element(output.begin(), output.end()) - output.begin());
        double share = output[best];
        if (rule == OutputRule::Softmax) {
            double sum = 0.0;
            for (float v : output) sum += std::exp(static_cast<double>(v) - output[best]);
            share = 1.0 / sum;
            p.detail = "softmax over " + CountText(output.size()) + " classes, logit " + Number(output[best]);
        } else {
            p.detail = "the largest of " + CountText(output.size()) + " outputs";
        }
        p.class_index = static_cast<int>(best);
        p.confidence = share;
    }
    p.text = ClassName(p.class_index, class_names) + ", " + percent(p.confidence);
    return p;
}

std::vector<int> ItemLabels(const std::vector<float>& values, const std::vector<size_t>& shape) {
    if (shape.empty() || values.empty()) return {};
    const size_t items = shape[0];
    const size_t width = values.size() / std::max<size_t>(1, items);
    std::vector<int> labels(items, -1);
    for (size_t i = 0; i < items && (i + 1) * width <= values.size(); ++i) {
        const float* row = values.data() + i * width;
        if (width > 1) {
            labels[i] = static_cast<int>(std::max_element(row, row + width) - row);
        } else if (std::isfinite(row[0]) && row[0] >= 0.0f && std::abs(row[0] - std::round(row[0])) < 1e-4f) {
            labels[i] = static_cast<int>(std::lround(row[0]));
        }
    }
    return labels;
}

std::string SampleHeadline(const std::string& item_name, const std::string& label,
                           const std::optional<Prediction>& prediction) {
    std::string line = item_name;
    const auto add = [&line](const std::string& part) {
        if (part.empty()) return;
        line += (line.empty() ? "" : " \xC2\xB7 ") + part;
    };
    if (!label.empty()) add("label " + label);
    if (prediction) add("predicted " + prediction->text);
    return line;
}

}  // namespace cyxwiz::layer_view
