#include "weight_match.h"

#include <algorithm>
#include <map>
#include <set>

namespace cyxwiz::weight_match {
namespace {

// The name PyTorch gives a parameter the Engine's module calls `key`.
std::string FileSuffix(const std::string& key) {
    if (key == "weights") return "weight";  // the convolutions
    if (key == "gamma") return "weight";    // the normalisations
    if (key == "beta") return "bias";
    return key;                             // weight, bias, running_mean, running_var
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
        case gui::NodeType::Conv1D:           // [out, in, k] -> [k, in, out]
            return rank == 3 ? std::vector<size_t>{2, 1, 0} : std::vector<size_t>{};
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
            const std::string name = layer.node_name + "." + FileSuffix(key);
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
            if (!axes.empty()) {
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
            tensor.values = axes.empty() ? entry.values : Permute(entry.values, entry.shape, axes);
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
