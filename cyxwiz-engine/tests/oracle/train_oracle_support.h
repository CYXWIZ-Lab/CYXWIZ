#pragma once

// What the training oracles share (TOFIX228 1.5b, 1.6): graphs built from the
// node factory, the PyTorch export and its module names, and the step check
// (the running loss and every parameter against PyTorch's, put in the
// Engine's layouts by the Engine's own weight matcher).

#include "../../src/core/code_export.h"
#include "../../src/core/executable_model.h"
#include "../../src/core/graph_node_factory.h"
#include "../../src/core/model_builder.h"
#include "../../src/core/sha256_digest.h"
#include "../../src/core/weight_match.h"
#include "../../src/core/weight_start.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz::test {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct TrainGraph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

class GraphBuilder {
public:
    // The Data node: one row of `shape` and its label (the legacy two-pin
    // form a compiled training graph reads; the data path is level 3's).
    int Data(const std::vector<size_t>& shape, const std::string& dataset = "oracle_train") {
        gui::MLNode node;
        node.id = next_node_++;
        node.type = gui::NodeType::DataInput;
        node.name = "Data";
        node.outputs = {Pin(gui::PinType::Tensor, "Data", false), Pin(gui::PinType::Labels, "Labels", false)};
        std::string text = "[";
        for (size_t i = 0; i < shape.size(); ++i) text += (i ? "," : "") + std::to_string(shape[i]);
        node.parameters = {{"dataset_name", dataset}, {"shape", text + "]"}};
        nodes_.push_back(node);
        return node.id;
    }
    // A model layer, loss or optimizer with the factory's pins.
    int Add(gui::NodeType type, const std::string& name, std::map<std::string, std::string> params = {}) {
        gui::MLNode node = gui::CreateGraphNode(type, name, next_node_, next_pin_);
        for (auto& [key, value] : params) node.parameters[key] = value;
        nodes_.push_back(node);
        return node.id;
    }
    // Output `out` of `from` to input `in` of `to` (pins by position).
    void Connect(int from, int to, size_t out = 0, size_t in = 0) {
        gui::NodeLink link;
        link.id = next_link_++;
        link.from_node = from;
        link.from_pin = Node(from).outputs.at(out).id;
        link.to_node = to;
        link.to_pin = Node(to).inputs.at(in).id;
        links_.push_back(link);
    }
    // from -> each of `chain` in turn; returns the last.
    int Chain(int from, const std::vector<int>& chain) {
        for (const int next : chain) {
            Connect(from, next);
            from = next;
        }
        return from;
    }
    // last layer -> loss (Predictions) <- Data's labels; loss -> optimizer.
    void Finish(int data, int last, int loss, int optimizer) {
        Connect(last, loss, 0, 0);
        Connect(data, loss, 1, 1);
        Connect(loss, optimizer, 0, 0);
    }
    TrainGraph Take() {
        TrainGraph graph;
        graph.nodes = std::move(nodes_);
        graph.links = std::move(links_);
        return graph;
    }

private:
    gui::NodePin Pin(gui::PinType type, const std::string& name, bool is_input) {
        gui::NodePin pin;
        pin.id = next_pin_++;
        pin.type = type;
        pin.name = name;
        pin.is_input = is_input;
        return pin;
    }
    const gui::MLNode& Node(int id) const {
        for (const auto& node : nodes_)
            if (node.id == id) return node;
        throw std::runtime_error("no node " + std::to_string(id));
    }
    std::vector<gui::MLNode> nodes_;
    std::vector<gui::NodeLink> links_;
    int next_node_ = 1;
    int next_pin_ = 1;
    int next_link_ = 1;
};

// The graph's PyTorch export, unstamped (the same graph, the same text).
inline cyxwiz::GraphCodeExport UnstampedPyTorchExport(const TrainGraph& graph) {
    cyxwiz::CodeExportOptions options;
    options.timestamp = false;
    return cyxwiz::ExportGraphCode(graph.nodes, graph.links, cyxwiz::CodeFramework::PyTorch, options);
}

// The graph with its parameter nodes named after the export's modules
// (self.layer3 -> node "layer3"): the PyTorch state-dict's keys.
inline TrainGraph NamedAfterModules(TrainGraph graph, const cyxwiz::GraphCodeExport& exported) {
    for (const auto& module : exported.modules)
        for (auto& node : graph.nodes)
            if (node.id == module.node_id) node.name = module.name;
    return graph;
}

inline std::string WithoutCarriageReturns(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    return text;
}

inline std::string ReadText(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return in ? WithoutCarriageReturns(text.str()) : std::string();
}

inline std::string Sha256Text(const std::string& text) {
    cyxwiz::Sha256Hasher hasher;
    std::string digest, error;
    if (!hasher.Update(text, error) || !hasher.Finish(digest, error)) return "sha256 failed: " + error;
    return digest;
}

// The step's torch values in the Engine's keys and layouts, by the matcher.
inline std::map<std::string, std::vector<float>> ExpectedInEngineLayout(const json& parameters,
                                                                  cyxwiz::IExecutableModel& model,
                                                                  const std::vector<cyxwiz::BuiltModuleProvenance>& provenance,
                                                                  std::string& problem) {
    cyxwiz::weight_file::WeightFile file;
    file.format = "fixture";
    for (const auto& [name, tensor] : parameters.items()) {
        cyxwiz::weight_file::Entry entry;
        entry.name = name;
        entry.shape = tensor.at("shape").get<std::vector<size_t>>();
        entry.dtype = "F32";
        entry.is_float = true;
        entry.values = tensor.at("values").get<std::vector<float>>();
        file.entries.push_back(std::move(entry));
    }
    const auto report = cyxwiz::weight_match::Match(file, cyxwiz::WeightSlots(model, provenance));
    if (report.fresh > 0) problem = report.Summary();
    std::map<std::string, std::vector<float>> out;
    for (const auto& tensor : report.tensors) out[tensor.key] = tensor.values;
    return out;
}

struct Difference {
    double abs = 0.0;
    double excess = 0.0;  // worst |a - e| / (atol + rtol |e|); <= 1 passes
    std::string where;
};

// `noise` (optional, the same size): elements only rounding noise moves (no
// gradient reaches them; see generate_train_trajectory_fixtures.py), held to
// `noise_atol` alone.
inline void Compare(const std::vector<float>& actual, const std::vector<float>& expected, double atol, double rtol,
             const std::string& where, Difference& worst, const std::vector<float>* noise = nullptr,
             double noise_atol = 0.0) {
    if (actual.size() != expected.size()) {
        worst.excess = 1e30;
        worst.where = where + " (" + std::to_string(actual.size()) + " values, the fixture " +
                      std::to_string(expected.size()) + ")";
        return;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = std::abs(static_cast<double>(actual[i]) - expected[i]);
        const bool noisy = noise && i < noise->size() && (*noise)[i] > 0.5f;
        const double bound = noisy ? noise_atol : atol + rtol * std::abs(expected[i]);
        const double excess = std::isfinite(d) ? d / bound : 1e30;
        if (excess > worst.excess) {
            worst.abs = d;
            worst.excess = excess;
            worst.where = where + "[" + std::to_string(i) + "]";
        }
    }
}

}  // namespace cyxwiz::test
