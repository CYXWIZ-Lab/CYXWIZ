#include "weight_match.h"

#include <algorithm>
#include <map>
#include <regex>
#include <set>

namespace cyxwiz::weight_match {
namespace {

// An LSTM / GRU / RNN parameter: "layer0_W_ih", "layer1.reverse.b_hh".
const std::regex& RecurrentKeyPattern() {
    static const std::regex pattern(R"(layer(\d+)(?:_|\.(forward|reverse)\.)(W|b)_(ih|hh))");
    return pattern;
}

bool IsRecurrentKey(const std::string& key) { return std::regex_match(key, RecurrentKeyPattern()); }

// The name PyTorch gives a parameter the Engine's module calls `key`.
std::string LeafFileSuffix(const std::string& key) {
    if (key == "weights") return "weight";  // the convolutions
    if (key == "gamma") return "weight";    // the normalisations
    if (key == "beta") return "bias";
    // Squeeze-Excitation: torchvision.ops.SqueezeExcitation's two 1x1
    // convolutions (fc1 squeezes, fc2 expands); their [out, in, 1, 1] kernels
    // arrive as the node's [out, in] weights.
    if (key == "reduce_weight") return "fc1.weight";
    if (key == "reduce_bias") return "fc1.bias";
    if (key == "expand_weight") return "fc2.weight";
    if (key == "expand_bias") return "fc2.bias";
    // LSTM / GRU / RNN: layer N's input / hidden weights and biases, the
    // reverse direction of a bidirectional one with "_reverse". The Engine keeps
    // torch's layout (gates stacked in torch's order), so only the name differs:
    // "layer0_W_ih" / "layer1.reverse.b_hh" are "weight_ih_l0" / "bias_hh_l1_reverse".
    std::smatch m;
    if (std::regex_match(key, m, RecurrentKeyPattern()))
        return std::string(m[3] == "W" ? "weight_" : "bias_") + m[4].str() + "_l" + m[1].str() +
               (m[2] == "reverse" ? "_reverse" : "");
    return key;                             // weight, bias, running_mean, running_var
}

// Where in a PyTorch file a parameter the Engine calls `key` is: the entry's
// name and, when torch packs several of the Engine's tensors into one, which
// of its `blocks` equal row blocks it is.
struct FileSource {
    std::string name;
    size_t block = 0;
    size_t blocks = 1;
};

FileSource SourceOf(const std::string& key) {
    if (IsRecurrentKey(key)) return {LeafFileSuffix(key)};
    // A module inside the layer (a transformer block's self_attn, linear1,
    // norm1): its own part of the name is kept, the last part is torch's.
    const size_t dot = key.rfind('.');
    const std::string head = dot == std::string::npos ? std::string() : key.substr(0, dot + 1);
    const std::string leaf = dot == std::string::npos ? key : key.substr(dot + 1);
    // nn.MultiheadAttention packs the query, key and value projections into
    // in_proj_weight [3d, d] / in_proj_bias [3d] (in that order); the Engine
    // keeps W_q / W_k / W_v in the same [out, in] layout.
    static const std::map<std::string, std::pair<std::string, size_t>> packed = {
        {"W_q", {"in_proj_weight", 0}}, {"W_k", {"in_proj_weight", 1}}, {"W_v", {"in_proj_weight", 2}},
        {"b_q", {"in_proj_bias", 0}},   {"b_k", {"in_proj_bias", 1}},   {"b_v", {"in_proj_bias", 2}}};
    if (const auto it = packed.find(leaf); it != packed.end()) return {head + it->second.first, it->second.second, 3};
    if (leaf == "W_o") return {head + "out_proj.weight"};
    if (leaf == "b_o") return {head + "out_proj.bias"};
    return {head + LeafFileSuffix(leaf)};
}

// The axes of the Engine's layout in terms of PyTorch's, for a layer's kernels;
// empty = the same order.
std::vector<size_t> KernelAxes(gui::NodeType type, const std::string& key, size_t rank) {
    if (key != "weights") return {};
    switch (type) {
        case gui::NodeType::Conv2D:           // [out, in, kh, kw] -> [kh, kw, in, out]
        case gui::NodeType::DepthwiseConv2D:  // [C x M, 1, k, k] -> [k, k, 1, C x M]
        case gui::NodeType::ConvTranspose2D:  // [in, out, kh, kw] -> [kh, kw, out, in]
            return rank == 4 ? std::vector<size_t>{2, 3, 1, 0} : std::vector<size_t>{};
        // Conv1D needs none: its kernel is [out, in, k], PyTorch's own order
        // (TOFIX223 0.15: it was permuted to [k, in, out]).
        default:
            return {};
    }
}

size_t Elements(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (const size_t d : shape) n *= d;
    return n;
}

// Row-major values of `shape` reordered so that output axis i is input axis axes[i].
std::vector<float> Permute(const std::vector<float>& values, const std::vector<size_t>& shape,
                           const std::vector<size_t>& axes) {
    const size_t rank = shape.size();
    std::vector<size_t> in_stride(rank, 1);
    for (size_t d = rank; d-- > 1;) in_stride[d - 1] = in_stride[d] * shape[d];
    std::vector<size_t> out_shape(rank);
    for (size_t i = 0; i < rank; ++i) out_shape[i] = shape[axes[i]];
    std::vector<float> out(values.size());
    std::vector<size_t> index(rank, 0);
    for (size_t n = 0; n < out.size(); ++n) {
        size_t source = 0;
        for (size_t i = 0; i < rank; ++i) source += index[i] * in_stride[axes[i]];
        out[n] = values[source];
        for (size_t d = rank; d-- > 0;) {
            if (++index[d] < out_shape[d]) break;
            index[d] = 0;
        }
    }
    return out;
}

// A [out, C x S] weight whose columns PyTorch orders channel first (c x S + s)
// reordered to the Engine's channel last (s x C + c).
std::vector<float> ChannelsLastColumns(const std::vector<float>& values, size_t rows, size_t channels, size_t spatial) {
    std::vector<float> out(values.size());
    const size_t width = channels * spatial;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t s = 0; s < spatial; ++s) out[r * width + s * channels + c] = values[r * width + c * spatial + s];
        }
    }
    return out;
}

std::string EntryText(const weight_file::Entry& entry) { return entry.name + " " + ShapeText(entry.shape); }

// "[64] x 4" when every parameter has the same shape, else the shapes in order.
std::string GraphShapes(const LayerSlot& layer) {
    if (layer.parameters.empty()) return {};
    const bool same = std::all_of(layer.parameters.begin(), layer.parameters.end(), [&](const auto& parameter) {
        return parameter.second == layer.parameters.front().second;
    });
    if (same && layer.parameters.size() > 1)
        return ShapeText(layer.parameters.front().second) + " x " + std::to_string(layer.parameters.size());
    std::string text;
    for (const auto& [key, shape] : layer.parameters) text += (text.empty() ? "" : ", ") + ShapeText(shape);
    return text;
}

}  // namespace

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text = "[";
    for (size_t d = 0; d < shape.size(); ++d) text += (d ? ", " : "") + std::to_string(shape[d]);
    return text + "]";
}

Report Match(const weight_file::WeightFile& file, const std::vector<LayerSlot>& layers) {
    Report report;
    report.format = file.format;
    report.file_entries = file.entries.size();
    std::map<std::string, const weight_file::Entry*> by_name;
    for (const auto& entry : file.entries) by_name.emplace(entry.name, &entry);
    std::set<std::string> used;
    std::set<std::string> node_names;
    // A layer whose shape did not fit: its entries are reported as such, not as "no node".
    std::set<std::string> offered;

    for (const auto& layer : layers) {
        if (layer.parameters.empty()) continue;
        node_names.insert(layer.node_name);
        LayerRow row;
        row.node_name = layer.node_name;
        row.type_label = layer.type_label;
        row.graph_shapes = GraphShapes(layer);
        std::vector<LoadedTensor> tensors;
        std::vector<std::string> taken;
        for (const auto& [key, shape] : layer.parameters) {
            const FileSource source = SourceOf(key);
            const std::string name = layer.node_name + "." + source.name;
            const auto found = by_name.find(name);
            if (found == by_name.end()) {
                if (row.reason.empty()) row.reason = "the file has no " + name;
                continue;
            }
            const weight_file::Entry& entry = *found->second;
            row.file_entries.push_back(EntryText(entry));
            offered.insert(name);
            if (!row.reason.empty()) continue;
            if (!entry.is_float) {
                row.reason = name + " is " + entry.dtype + ", not a float tensor";
                continue;
            }
            const auto axes = KernelAxes(layer.type, key, entry.shape.size());
            std::vector<size_t> arrives = entry.shape;
            const bool sliced = source.blocks > 1 && !arrives.empty() && arrives[0] % source.blocks == 0;
            if (sliced) {
                arrives[0] /= source.blocks;  // one row block of the packed entry
            } else if (!axes.empty()) {
                for (size_t i = 0; i < axes.size(); ++i) arrives[i] = entry.shape[axes[i]];
            } else if (arrives != shape && shape.size() == 2 && entry.shape.size() > 2 && entry.shape[0] == shape[0] &&
                       Elements(entry.shape) == Elements(shape)) {
                arrives = shape;  // kernels kept flat per filter: [out, in x k x k x k]
            }
            if (arrives != shape) {
                row.reason = "shape differs: the file has " + ShapeText(entry.shape) + ", the graph " + ShapeText(shape);
                continue;
            }
            LoadedTensor tensor;
            tensor.key = layer.key_prefix + key;
            tensor.shape = shape;
            const bool flattened_input = layer.flattened_channels > 1 && key == "weight" && shape.size() == 2 &&
                                         shape[1] % layer.flattened_channels == 0 &&
                                         shape[1] > layer.flattened_channels;
            if (entry.values.empty()) {
                // read without values: the report only
            } else if (sliced) {
                const size_t count = entry.values.size() / source.blocks;
                const auto first = entry.values.begin() + static_cast<std::ptrdiff_t>(source.block * count);
                tensor.values.assign(first, first + static_cast<std::ptrdiff_t>(count));
            } else if (!axes.empty()) {
                tensor.values = Permute(entry.values, entry.shape, axes);
            } else if (flattened_input) {
                tensor.values = ChannelsLastColumns(entry.values, shape[0], layer.flattened_channels,
                                                    shape[1] / layer.flattened_channels);
            } else {
                tensor.values = entry.values;
            }
            tensors.push_back(std::move(tensor));
            taken.push_back(name);
        }
        // Nothing of this layer is in the file: say that, not which entry was looked for first.
        if (row.file_entries.empty()) row.reason = "not in the file";
        row.loaded = row.reason.empty();
        if (row.loaded) {
            ++report.loaded;
            for (auto& tensor : tensors) report.tensors.push_back(std::move(tensor));
            used.insert(taken.begin(), taken.end());
        } else {
            ++report.fresh;
        }
        report.layers.push_back(std::move(row));
    }

    for (const auto& entry : file.entries) {
        if (used.count(entry.name)) continue;
        UnusedEntry unused;
        unused.name = EntryText(entry);
        const size_t dot = entry.name.rfind('.');
        const std::string owner = dot == std::string::npos ? std::string() : entry.name.substr(0, dot);
        const std::string suffix = dot == std::string::npos ? entry.name : entry.name.substr(dot + 1);
        if (suffix == "num_batches_tracked") unused.reason = "a counter PyTorch keeps, not a weight";
        else if (offered.count(entry.name)) unused.reason = "its layer starts fresh";
        else if (node_names.count(owner)) unused.reason = "the node " + owner + " has no such parameter";
        else unused.reason = owner.empty() ? "no node takes it" : "no node is named " + owner;
        report.unused.push_back(std::move(unused));
    }
    return report;
}

std::string Report::Summary() const {
    const size_t total = loaded + fresh;
    std::string text = std::to_string(loaded) + " of " + std::to_string(total) + (total == 1 ? " layer takes" : " layers take") +
                       " weights from the file";
    if (fresh) {
        text += "; " + std::to_string(fresh) + (fresh == 1 ? " starts fresh" : " start fresh") + ": ";
        size_t shown = 0;
        for (const auto& row : layers) {
            if (row.loaded) continue;
            if (shown == 3) {
                text += ", and " + std::to_string(fresh - shown) + " more";
                break;
            }
            text += (shown ? ", " : "") + row.node_name + " (" + row.reason + ")";
            ++shown;
        }
    }
    return text;
}

}  // namespace cyxwiz::weight_match
