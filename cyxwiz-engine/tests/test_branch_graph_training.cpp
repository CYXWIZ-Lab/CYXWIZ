// Branched graphs through the Engine's own path (TOFIX140 A2), against
// PyTorch: Data Input -> Split -> two Dense branches -> Concatenate / Add ->
// MSE. The compiler gives every branch its own shape (Split's two outputs
// differ), ModelBuilder sizes each Dense from the pin that feeds it, and the
// GraphExecutableModel's forward pass and per-pin backward match
// fixtures/branch_graph_pytorch.json (generate_branch_graph_fixtures.py).
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_executable_model.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/model_builder.h"
#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "route_qualification_test_fixture.h"
#include "../src/core/arrow_dataset.h"
#include "../src/core/training_executor.h"
#include "../src/core/checkpoint_manager.h"
#include "../src/gui/node_import_guardrails.h"
#include "../src/core/formats/cyxmodel_format.h"
#include "../src/core/test_executor.h"
#include "../src/gui/loaders/data_loader.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// Graph-only: no file loaders, so datasets are not read.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string&) { return FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::NodePin Pin(int id, gui::PinType type, const std::string& name, bool is_input) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

// Pins: inputs id*100+1.., outputs id*100+11..
gui::MLNode Node(int id, gui::NodeType type, const std::string& name, int inputs, int outputs,
                 std::map<std::string, std::string> parameters = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    for (int i = 0; i < inputs; ++i)
        node.inputs.push_back(Pin(id * 100 + 1 + i, gui::PinType::Tensor, "Input " + std::to_string(i + 1), true));
    for (int i = 0; i < outputs; ++i)
        node.outputs.push_back(Pin(id * 100 + 11 + i, gui::PinType::Tensor, "Output " + std::to_string(i + 1), false));
    node.parameters = std::move(parameters);
    return node;
}

gui::NodeLink Link(int from_node, int from_output, int to_node, int to_input) {
    static int next_id = 1;
    gui::NodeLink link;
    link.id = next_id++;
    link.from_node = from_node;
    link.from_pin = from_node * 100 + 11 + from_output;
    link.to_node = to_node;
    link.to_pin = to_node * 100 + 1 + to_input;
    return link;
}

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

cyxwiz::Tensor ReadTensor(const json& fixture) {
    const auto shape = fixture.at("shape").get<std::vector<size_t>>();
    const auto values = fixture.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckClose(const cyxwiz::Tensor& actual, const json& expected, const json& tolerance,
                const std::string& what) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<float>>();
    Check(actual.Shape() == shape, what + ": shape " + ShapeText(actual.Shape()) + " vs PyTorch " + ShapeText(shape));
    const float absolute = tolerance.at("absolute").get<float>();
    const float relative = tolerance.at("relative").get<float>();
    const float* data = actual.ReadData<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        const float diff = std::fabs(data[i] - values[i]);
        Check(diff <= absolute + relative * std::fabs(values[i]),
              what + ": element " + std::to_string(i) + " is " + std::to_string(data[i]) +
                  ", PyTorch " + std::to_string(values[i]));
    }
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

// Data (1) -> Split (2) -> branches -> merge -> [Dense C] -> MSE (9) -> SGD (10).
Graph BuildGraph(const json& c) {
    Graph g;
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Rows";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    // The sample shape of the case's input (all but the batch axis).
    const auto input_shape = c.at("input").at("shape").get<std::vector<size_t>>();
    std::string sample_shape = "[";
    for (size_t i = 1; i < input_shape.size(); ++i) {
        sample_shape += (i > 1 ? ", " : "") + std::to_string(input_shape[i]);
    }
    data.parameters = {{"dataset_name", "branch_rows"}, {"shape", sample_shape + "]"}};
    gui::MLNode loss = Node(9, gui::NodeType::MSELoss, "MSE", 0, 0);
    loss.inputs = {Pin(901, gui::PinType::Tensor, "Predictions", true), Pin(902, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(911, gui::PinType::Loss, "Loss", false)};
    gui::MLNode sgd = Node(10, gui::NodeType::SGD, "SGD", 0, 0);
    sgd.inputs = {Pin(1001, gui::PinType::Loss, "Loss", true)};
    // Split comes from the node factory: its pins and defaults are the contract.
    int next_node_id = 2, next_pin_id = 0;
    auto split = gui::CreateGraphNode(gui::NodeType::Split, "Split", next_node_id, next_pin_id);
    Check(split.inputs.size() == 1 && split.outputs.size() == 2 && split.outputs[0].is_required &&
              !split.outputs[1].is_required && split.parameters["dim"] == "1",
          "Split: one input, Output 1 required, Output 2 optional, dim defaults to 1");
    split.id = 2;
    split.inputs[0].id = 201;
    split.outputs[0].id = 211;
    split.outputs[1].id = 212;
    split.parameters["split_size"] = std::to_string(c.at("split_size").get<int>());
    split.parameters["dim"] = std::to_string(c.at("dim").get<int>());
    const auto units = [&](const char* layer) {
        return std::to_string(c.at("layers").at(layer).at("bias").at("shape")[0].get<size_t>());
    };
    const std::string name = c.at("name").get<std::string>();
    g.nodes = {data, split, loss, sgd};
    g.links = {Link(1, 0, 2, 0), Link(9, 0, 10, 0)};
    gui::NodeLink labels;
    labels.id = 900; labels.from_node = 1; labels.from_pin = 112; labels.to_node = 9; labels.to_pin = 902;
    g.links.push_back(labels);
    const auto to_loss = [&](int from) {
        gui::NodeLink l;
        l.id = 901; l.from_node = from; l.from_pin = from * 100 + 11; l.to_node = 9; l.to_pin = 901;
        g.links.push_back(l);
    };
    if (name == "fanout_residual_concat") {
        // No Split: the data pin feeds three branches directly.
        g.nodes = {data, loss, sgd};
        g.links = {Link(9, 0, 10, 0), labels};
        const auto dense = [&](int id, const char* layer) {
            g.nodes.push_back(Node(id, gui::NodeType::Dense, layer, 1, 1, {{"units", units(layer)}}));
        };
        dense(3, "Dense A1");
        g.nodes.push_back(Node(4, gui::NodeType::ReLU, "ReLU A", 1, 1));
        dense(5, "Dense A2");
        dense(6, "Projection");
        g.nodes.push_back(Node(7, gui::NodeType::Add, "Residual Add", 2, 1));
        g.nodes.push_back(Node(8, gui::NodeType::ReLU, "ReLU", 1, 1));
        dense(11, "Wide");
        g.nodes.push_back(Node(12, gui::NodeType::Tanh, "Tanh", 1, 1));
        g.nodes.push_back(Node(13, gui::NodeType::Concatenate, "Concat", 2, 1, {{"dim", "1"}}));
        dense(14, "Dense C");
        for (const auto& [from, out, to, in] : std::vector<std::tuple<int, int, int, int>>{
                 {1, 0, 3, 0}, {3, 0, 4, 0}, {4, 0, 5, 0}, {1, 0, 6, 0}, {5, 0, 7, 0}, {6, 0, 7, 1}, {7, 0, 8, 0},
                 {1, 0, 11, 0}, {11, 0, 12, 0}, {8, 0, 13, 0}, {12, 0, 13, 1}, {13, 0, 14, 0}}) {
            gui::NodeLink link = Link(from, out, to, in);
            if (from == 1) link.from_pin = 111;  // the data node's Data pin
            g.links.push_back(link);
        }
        to_loss(14);
        return g;
    }
    if (name == "split_bmm") {
        // A = Output 1, B = Output 2: Batch MatMul -> Flatten -> Dense C.
        g.nodes.push_back(Node(5, gui::NodeType::TensorBatchMatMul, "Batch MatMul", 2, 1));
        g.nodes.push_back(Node(6, gui::NodeType::Flatten, "Flatten", 1, 1));
        g.nodes.push_back(Node(7, gui::NodeType::Dense, "Dense C", 1, 1, {{"units", units("Dense C")}}));
        g.links.push_back(Link(2, 0, 5, 0));
        g.links.push_back(Link(2, 1, 5, 1));
        g.links.push_back(Link(5, 0, 6, 0));
        g.links.push_back(Link(6, 0, 7, 0));
        to_loss(7);
        return g;
    }
    g.nodes.push_back(Node(3, gui::NodeType::Dense, "Dense A", 1, 1, {{"units", units("Dense A")}}));
    g.links.push_back(Link(2, 0, 3, 0));
    if (name == "split_one_branch") {
        to_loss(3);
        return g;
    }
    g.nodes.push_back(Node(4, gui::NodeType::Dense, "Dense B", 1, 1, {{"units", units("Dense B")}}));
    g.links.push_back(Link(2, 1, 4, 0));
    if (name == "split_concat") {
        g.nodes.push_back(Node(5, gui::NodeType::Concatenate, "Concat", 2, 1, {{"dim", "1"}}));
        g.nodes.push_back(Node(6, gui::NodeType::Dense, "Dense C", 1, 1, {{"units", units("Dense C")}}));
        g.links.push_back(Link(3, 0, 5, 0));
        g.links.push_back(Link(4, 0, 5, 1));
        g.links.push_back(Link(5, 0, 6, 0));
        to_loss(6);
    } else {
        g.nodes.push_back(Node(5, gui::NodeType::Add, "Add", 2, 1));
        g.links.push_back(Link(3, 0, 5, 0));
        g.links.push_back(Link(4, 0, 5, 1));
        to_loss(5);
    }
    return g;
}

const cyxwiz::CompiledLayer& LayerNamed(const cyxwiz::TrainingConfiguration& config, const std::string& name,
                                        size_t* index = nullptr) {
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (config.layers[i].name == name) {
            if (index) *index = i;
            return config.layers[i];
        }
    }
    Check(false, "compiled layer missing: " + name);
    return config.layers.front();
}

void PrintIssues(const cyxwiz::TrainingConfiguration& config) {
    for (const auto& issue : config.issues) {
        std::cerr << "  issue(" << (issue.level == cyxwiz::IssueLevel::Error ? "error" : "note") << "): "
                  << issue.node_name << ": " << issue.message << "\n";
    }
}

// One batch: the rows a test reads (TOFIX157, Run Test on a graph model).
class OneBatch final : public cyxwiz::IBatcher {
public:
    OneBatch(cyxwiz::Tensor data, cyxwiz::Tensor labels) : data_(std::move(data)), labels_(std::move(labels)) {}
    cyxwiz::Batch GetNextBatch() override {
        cyxwiz::Batch batch;
        if (done_) return batch;
        done_ = true;
        batch.data = data_;
        batch.labels = labels_;
        batch.size = data_.Shape().front();
        return batch;
    }
    void Reset() override { done_ = false; }
    bool IsEpochComplete() const override { return done_; }
    size_t GetNumBatches() const override { return 1; }
    size_t GetNumSamples() const override { return data_.Shape().front(); }
    void SetNormalization(float, float) override {}
    void SetOneHotEncoding(size_t) override {}
    void SetFlatten(bool) override {}

private:
    cyxwiz::Tensor data_, labels_;
    bool done_ = false;
};

// Run Test on the trained graph (TOFIX157): TestExecutor runs the graph model
// it is handed (it ran only a SequentialModel: "No model is active").
void CheckTestRunsGraphModel(const std::string& name, const cyxwiz::TrainingConfiguration& config,
                             std::shared_ptr<cyxwiz::IExecutableModel> model, const cyxwiz::Tensor& input) {
    model->SetTraining(false);
    const cyxwiz::Tensor expected = model->Forward(input);
    const auto out_shape = expected.Shape();
    const size_t rows = out_shape.front();
    const size_t width = expected.NumElements() / rows;
    const float* out = expected.ReadData<float>();
    cyxwiz::ExternalTestSource source;
    source.make_batcher = [input, out_shape](const cyxwiz::TrainingConfiguration&, int) -> std::unique_ptr<cyxwiz::IBatcher> {
        return std::make_unique<OneBatch>(input, cyxwiz::Tensor::Zeros(out_shape));
    };
    cyxwiz::TestExecutor test(config, source);
    test.SetModel(model);
    test.Test(static_cast<int>(rows));
    const auto m = test.GetMetrics();
    Check(m.is_complete && m.total_samples == static_cast<int>(rows),
          name + ": Run Test runs the graph model (" + m.status_message + ")");
    if (m.regression_mode) {
        Check(m.regression_predictions.size() == rows * width, name + ": a prediction per output value");
        for (size_t i = 0; i < m.regression_predictions.size() && i < rows * width; ++i)
            Check(std::abs(m.regression_predictions[i] - out[i]) < 1e-4f, name + ": the test's output is the model's");
    } else {
        Check(m.predictions.size() == rows, name + ": a prediction per row");
        for (size_t r = 0; r < m.predictions.size() && r < rows; ++r) {
            size_t best = 0;
            for (size_t k = 1; k < width; ++k)
                if (out[r * width + k] > out[r * width + best]) best = k;
            Check(m.predictions[r] == static_cast<int>(best), name + ": the test's class is the model's argmax");
        }
    }
    std::cout << "  " << name << ": Run Test runs the graph model (" << rows << " rows)\n";
}

// Save / Export a graph with branches (TOFIX157): the CyxWiz package holds its
// weights and its graph, marked GraphModel; without the graph, or as ONNX, it
// is refused.
void CheckGraphModelExports(const std::string& name, cyxwiz::IExecutableModel& model) {
    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_branch_graph_export";
    std::filesystem::create_directories(dir);
    const auto path = dir / (name + ".cyxmodel");
    std::filesystem::remove(path);
    const std::string graph_json = R"({"nodes":[],"links":[],"case":")" + name + R"("})";
    cyxwiz::ModelExporter exporter;
    cyxwiz::ExportOptions options;
    options.model_name = name;
    const auto exported = exporter.ExportCyxModel(model, nullptr, nullptr, graph_json, path.string(), options);
    Check(exported.success, name + ": a graph model exports to .cyxmodel: " + exported.error_message);
    Check(exported.num_layers == static_cast<int>(model.LayerCount()), name + ": the package counts its layers");

    cyxwiz::formats::CyxModelFormat format;
    cyxwiz::ModelManifest manifest;
    std::string read_graph;
    cyxwiz::TrainingConfig config;
    std::map<std::string, std::vector<uint8_t>> weights;
    std::map<std::string, std::vector<int64_t>> shapes;
    Check(format.Extract(path.string(), manifest, read_graph, config, nullptr, weights, shapes, nullptr,
                         cyxwiz::ImportOptions{}),
          name + ": the package reads back: " + format.GetLastError());
    Check(manifest.model_type == "GraphModel" && manifest.has_graph && read_graph == graph_json,
          name + ": marked GraphModel, with its graph (" + manifest.model_type + ")");
    const auto params = model.GetParameters();
    bool same = weights.size() == params.size();
    for (const auto& [key, tensor] : params) {
        const auto found = weights.find(key);
        same = same && found != weights.end() && found->second.size() == tensor.NumBytes() &&
               std::memcmp(found->second.data(), tensor.ReadData(), tensor.NumBytes()) == 0;
    }
    Check(same, name + ": every weight is in the package, byte for byte");

    auto no_graph = options;
    no_graph.include_graph = false;
    const auto refused = exporter.ExportCyxModel(model, nullptr, nullptr, graph_json, (dir / "no_graph.cyxmodel").string(),
                                                 no_graph);
    Check(!refused.success && refused.error_message.find("needs its graph") != std::string::npos,
          name + ": without its graph the package is refused: " + refused.error_message);
    auto onnx = options;
    onnx.format = cyxwiz::ModelFormat::ONNX;
    const auto onnx_refused = exporter.Export(model, nullptr, nullptr, graph_json, (dir / "model.onnx").string(), onnx);
    Check(!onnx_refused.success && onnx_refused.error_message.find("graph with branches") != std::string::npos,
          name + ": ONNX is refused for a graph with branches: " + onnx_refused.error_message);
    std::filesystem::remove_all(dir);
    std::cout << "  " << name << ": saved as a CyxWiz package with its graph\n";
}

// A graph with branches, written as the editor saves it (factory nodes, links
// by pin index), goes into a CyxWiz package and comes back as the same graph
// model: the same outputs for the same rows. As a layer stack the package is
// refused (it built the layers alone before, TOFIX157).
std::string EditorGraphJson(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links) {
    // The header the editor writes (the current data boundary).
    json document = gui::detail::CreateSerializedGraphDocument("1.0");
    document["nodes"] = json::array();
    document["links"] = json::array();
    const auto pin_index = [](const std::vector<gui::NodePin>& pins, int id) {
        for (size_t i = 0; i < pins.size(); ++i)
            if (pins[i].id == id) return static_cast<int>(i);
        return -1;
    };
    const auto node_of = [&](int id) -> const gui::MLNode& {
        return *std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
    };
    for (const auto& node : nodes)
        document["nodes"].push_back({{"id", node.id}, {"type", static_cast<int>(node.type)}, {"name", node.name},
                                     {"parameters", node.parameters}});
    for (const auto& link : links)
        document["links"].push_back({{"id", link.id}, {"from_node", link.from_node}, {"from_pin", link.from_pin},
                                     {"to_node", link.to_node}, {"to_pin", link.to_pin},
                                     {"link_type", static_cast<int>(link.type)},
                                     {"from_pin_index", pin_index(node_of(link.from_node).outputs, link.from_pin)},
                                     {"to_pin_index", pin_index(node_of(link.to_node).inputs, link.to_pin)}});
    return document.dump(4);
}

// Data Input -> Data Split -> Data Loader -> {Dense a, Dense b} -> Add -> Dense head (3) -> loss <- labels;
// SGD (the data path of the branching example graph), as the editor saves it.
std::string ResidualGraphJson(gui::NodeType loss_type, const std::string& dataset_name) {
    // The nodes get the pins the saved-graph loader gives them (the node list
    // is loaded first), then links are made between those pins.
    int next_link = 1;
    json listed = gui::detail::CreateSerializedGraphDocument("1.0");
    listed["nodes"] = json::array();
    listed["links"] = json::array();
    const auto make = [&](gui::NodeType type, const std::string& name,
                          std::map<std::string, std::string> parameters = {}) -> int {
        const int id = static_cast<int>(listed["nodes"].size()) + 1;
        listed["nodes"].push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name},
                                   {"parameters", parameters}});
        return id;
    };
    const int data = make(gui::NodeType::DataInput, "Rows", {{"dataset_name", dataset_name}, {"shape", "[8]"}});
    const int split = make(gui::NodeType::DataSplit, "Split");
    const int loader = make(gui::NodeType::DataLoader, "Loader");
    const int a = make(gui::NodeType::Dense, "Dense a", {{"units", "6"}});
    const int b = make(gui::NodeType::Dense, "Dense b", {{"units", "6"}});
    const int add = make(gui::NodeType::Add, "Add");
    const int head = make(gui::NodeType::Dense, "Head", {{"units", "3"}});
    const int loss = make(loss_type, "Loss");
    const int sgd = make(gui::NodeType::SGD, "SGD");
    cyxwiz::GraphDocument listed_document;
    std::string list_error;
    const bool listed_ok = cyxwiz::ParseGraphDocument(listed.dump(), listed_document, list_error);
    Check(listed_ok, "the node list loads: " + list_error);
    std::vector<gui::MLNode> nodes = listed_document.nodes;
    std::vector<gui::NodeLink> links;
    const auto node_of = [&](int id) -> gui::MLNode& {
        return *std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
    };
    const auto pin = [&](int node, bool input, gui::PinType type, int nth = 0) {
        const auto& pins = input ? node_of(node).inputs : node_of(node).outputs;
        for (const auto& p : pins)
            if (p.type == type && nth-- == 0) return p.id;
        std::string have;
        for (const auto& p : pins) have += " " + p.name + "(" + std::to_string(static_cast<int>(p.type)) + ")";
        Check(false, node_of(node).name + ": no " + (input ? "input" : "output") + " pin of type " +
                         std::to_string(static_cast<int>(type)) + "; pins:" + have);
        return -1;
    };
    const auto connect = [&](int from, gui::PinType from_type, int to, gui::PinType to_type, int to_nth = 0) {
        gui::NodeLink link;
        link.id = next_link++;
        link.from_node = from;
        link.from_pin = pin(from, false, from_type);
        link.to_node = to;
        link.to_pin = pin(to, true, to_type, to_nth);
        links.push_back(link);
    };
    // Each output of a data node to the next node's input of the same type.
    const auto pass_through = [&](int from, int to) {
        auto& inputs = node_of(to).inputs;
        std::map<gui::PinType, int> seen;
        for (const auto& out : node_of(from).outputs) {
            const int nth = seen[out.type]++;
            int match = nth;
            for (const auto& in : inputs) {
                if (in.type != out.type || match-- != 0) continue;
                gui::NodeLink link;
                link.id = next_link++;
                link.from_node = from;
                link.from_pin = out.id;
                link.to_node = to;
                link.to_pin = in.id;
                links.push_back(link);
                break;
            }
        }
    };
    pass_through(data, split);
    pass_through(split, loader);
    connect(loader, gui::PinType::Tensor, a, gui::PinType::Tensor);
    connect(loader, gui::PinType::Tensor, b, gui::PinType::Tensor);
    connect(a, gui::PinType::Tensor, add, gui::PinType::Tensor, 0);
    connect(b, gui::PinType::Tensor, add, gui::PinType::Tensor, 1);
    connect(add, gui::PinType::Tensor, head, gui::PinType::Tensor);
    connect(head, gui::PinType::Tensor, loss, gui::PinType::Tensor);
    connect(loader, gui::PinType::Labels, loss, gui::PinType::Labels);
    connect(loss, gui::PinType::Loss, sgd, gui::PinType::Loss);
    return EditorGraphJson(nodes, links);
}

void CheckGraphPackageRoundTrip() {
    const std::string graph_json = ResidualGraphJson(gui::NodeType::MSELoss, "branch_rows");

    // What Import does with the file: the shared loader, then ModelImport.
    cyxwiz::GraphDocument document;
    std::string error;
    const bool parsed = cyxwiz::ParseGraphDocument(graph_json, document, error);
    Check(parsed, "the editor-format graph parses: " + error);
    cyxwiz::GraphCompiler compiler;
    const auto config = compiler.Compile(document.nodes, document.links, true,
                                         cyxwiz::GraphCompiler::Purpose::ModelImport);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid && !config.graph_op_node_ids.empty(), "the residual graph compiles with branches");
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), "the residual graph builds: " + built.error_message);
    if (!built.ok()) return;
    std::shared_ptr<cyxwiz::IExecutableModel> trained(std::move(built.model));

    const size_t rows = 5, width = 8;
    std::vector<float> values(rows * width);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(0.37f * static_cast<float>(i));
    const cyxwiz::Tensor input({rows, width}, values.data());
    trained->SetTraining(false);
    const cyxwiz::Tensor expected = trained->Forward(input);

    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_branch_graph_roundtrip";
    std::filesystem::create_directories(dir);
    const auto path = dir / "residual.cyxmodel";
    std::filesystem::remove(path);
    cyxwiz::ModelExporter exporter;
    const auto exported = exporter.ExportCyxModel(*trained, nullptr, nullptr, graph_json, path.string(),
                                                  cyxwiz::ExportOptions{});
    Check(exported.success, "the residual graph model exports: " + exported.error_message);

    cyxwiz::ModelImporter importer;
    std::shared_ptr<cyxwiz::IExecutableModel> loaded;
    const auto imported = importer.ImportExecutableModel(path.string(), loaded, cyxwiz::ImportOptions{});
    Check(imported.success && loaded, "the package imports: " + imported.error_message);
    Check(dynamic_cast<cyxwiz::GraphExecutableModel*>(loaded.get()) != nullptr, "it comes back as a graph model");
    if (loaded) {
        loaded->SetTraining(false);
        const cyxwiz::Tensor output = loaded->Forward(input);
        bool same = output.Shape() == expected.Shape();
        for (size_t i = 0; same && i < expected.NumElements(); ++i)
            same = std::abs(output.ReadData<float>()[i] - expected.ReadData<float>()[i]) < 1e-6f;
        Check(same, "the imported graph model gives the same outputs");
    }

    cyxwiz::SequentialModel stack;
    const auto as_stack = importer.ImportCyxModel(path.string(), stack, cyxwiz::ImportOptions{});
    Check(!as_stack.success && as_stack.error_message.find("graph with branches") != std::string::npos,
          "as a layer stack the package is refused: " + as_stack.error_message);

    // A checkpoint of the graph model loads into the model its graph builds
    // (Load Checkpoint for Testing does the same): the same outputs.
    cyxwiz::CheckpointManager saving((dir / "checkpoints").string());
    cyxwiz::TrainingMetrics metrics;
    metrics.current_epoch = 1;
    const std::string saved = saving.SaveCheckpoint(*trained, nullptr, metrics, "graph_epoch_1");
    Check(!saved.empty(), "a graph model's checkpoint saves: " + saving.GetLastError());
    auto rebuilt = cyxwiz::BuildExecutableFromConfig(config);
    Check(rebuilt.ok(), "the graph builds again: " + rebuilt.error_message);
    if (rebuilt.ok() && !saved.empty()) {
        cyxwiz::CheckpointManager loading((dir / "checkpoints").string());
        const auto metadata = loading.LoadCheckpoint(*rebuilt.model, nullptr, "graph_epoch_1");
        Check(metadata.has_value(), "the checkpoint loads into the graph model: " + loading.GetLastError());
        rebuilt.model->SetTraining(false);
        const cyxwiz::Tensor output = rebuilt.model->Forward(input);
        bool same = output.Shape() == expected.Shape();
        for (size_t i = 0; same && i < expected.NumElements(); ++i)
            same = std::abs(output.ReadData<float>()[i] - expected.ReadData<float>()[i]) < 1e-6f;
        Check(same, "the graph model from the checkpoint gives the same outputs");
    }
    std::filesystem::remove_all(dir);
    std::cout << "  residual graph: exported, imported and checkpointed as the same graph model\n";
}

// Training a graph with branches with "save best checkpoint" (TOFIX157): the
// best epoch's checkpoint is written and restored when validation stops
// improving, as for a layer stack (it was skipped for graph models).
void CheckGraphTrainingKeepsBestCheckpoint() {
    constexpr const char* kDataset = "branch_train_rows";
    const size_t rows = 24;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int f = 0; f < 8; ++f) {
        arrow::FloatBuilder builder;
        for (size_t r = 0; r < rows; ++r) (void)builder.Append(std::sin(0.3f * static_cast<float>(r * 8 + f)));
        std::shared_ptr<arrow::Array> column;
        (void)builder.Finish(&column);
        columns.push_back(column);
        fields.push_back(arrow::field("x" + std::to_string(f), arrow::float32()));
    }
    arrow::Int32Builder labels;
    for (size_t r = 0; r < rows; ++r) (void)labels.Append(static_cast<int32_t>(r % 3));
    std::shared_ptr<arrow::Array> label_column;
    (void)labels.Finish(&label_column);
    columns.push_back(label_column);
    fields.push_back(arrow::field("label", arrow::int32()));
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(arrow::Table::Make(arrow::schema(fields), columns),
                                                          kDataset);

    cyxwiz::GraphDocument document;
    std::string error;
    const bool parsed = cyxwiz::ParseGraphDocument(ResidualGraphJson(gui::NodeType::CrossEntropyLoss, kDataset),
                                                   document, error);
    Check(parsed, "the training graph parses: " + error);
    cyxwiz::GraphCompiler compiler;
    auto config = compiler.Compile(document.nodes, document.links, true);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid && !config.graph_op_node_ids.empty(), "the training graph compiles with branches");

    const auto dir = std::filesystem::temp_directory_path() / "cyxwiz_branch_graph_best";
    std::filesystem::remove_all(dir);
    config.dataset_name = kDataset;
    config.epochs = 5;
    config.batch_size = 4;
    config.learning_rate = 0.0f;  // validation loss stays put: the run stops at epoch 2, restoring epoch 1
    config.validation_freq = 1;
    config.early_stopping_patience = 1;
    config.save_best_checkpoint = true;
    config.checkpoint_dir = dir.string();
    config.shuffle = false;
    config.num_workers = 0;
    config.log_interval = 0;

    cyxwiz::test::InstallQualifiedRouteSnapshot();  // a test run trains on this PC's devices
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(config.epochs, config.batch_size, nullptr, nullptr,
                   [&](const cyxwiz::TrainingMetrics& metrics) { final_metrics = metrics; });
    Check(final_metrics.is_complete, "the graph model trains: " + final_metrics.status_message);
    Check(!final_metrics.checkpoint_used.empty() && final_metrics.restored_checkpoint_epoch == 1 &&
              final_metrics.active_model_provenance == "restored_best_checkpoint",
          "the graph model's best checkpoint is kept and restored (" + final_metrics.active_model_provenance +
              ", epoch " + std::to_string(final_metrics.restored_checkpoint_epoch) + ")");
    std::filesystem::remove_all(dir);
    std::cout << "  residual graph: training keeps and restores its best checkpoint\n";
}

void RunCase(const json& c) {
    const std::string name = c.at("name").get<std::string>();
    const auto graph = BuildGraph(c);
    cyxwiz::GraphCompiler compiler;
    const auto config = compiler.Compile(graph.nodes, graph.links, true);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, name + ": the branched graph compiles");
    if (name != "fanout_residual_concat")
        Check(config.graph_op_node_ids.size() >= 1, name + ": Split runs as a graph op");

    // Every Dense sees its own branch: Split's outputs have different widths.
    for (const auto& [layer_name, layer] : c.at("layers").items()) {
        const auto weight_shape = layer.at("weight").at("shape").get<std::vector<size_t>>();
        const auto& compiled = LayerNamed(config, layer_name);
        Check(compiled.input_shape == std::vector<size_t>{weight_shape[1]},
              name + ": " + layer_name + " input " + ShapeText(compiled.input_shape) + ", PyTorch [" +
                  std::to_string(weight_shape[1]) + "]");
        Check(compiled.output_shape == std::vector<size_t>{weight_shape[0]},
              name + ": " + layer_name + " output " + ShapeText(compiled.output_shape));
    }
    Check(config.output_size == c.at("output").at("shape")[1].get<size_t>(),
          name + ": output size is the width that feeds the loss: " + std::to_string(config.output_size));

    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds the branches: " + built.error_message);
    auto* model = dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(model != nullptr, name + ": a graph executable");

    std::map<std::string, cyxwiz::Tensor> parameters;
    for (const auto& [layer_name, layer] : c.at("layers").items()) {
        size_t index = 0;
        LayerNamed(config, layer_name, &index);
        parameters["layer" + std::to_string(index) + ".weight"] = ReadTensor(layer.at("weight"));
        parameters["layer" + std::to_string(index) + ".bias"] = ReadTensor(layer.at("bias"));
    }
    model->SetParameters(parameters);
    model->SetTraining(true);

    const json& tolerance = c.at("tolerance");
    const cyxwiz::Tensor output = model->Forward(ReadTensor(c.at("input")));
    CheckClose(output, c.at("output"), tolerance, name + " forward");
    const cyxwiz::Tensor grad_input = model->Backward(ReadTensor(c.at("grad_output")));
    CheckClose(grad_input, c.at("grad_input"), tolerance, name + " input gradient");

    const auto gradients = model->GetGradients();
    for (const auto& [layer_name, layer] : c.at("layers").items()) {
        size_t index = 0;
        LayerNamed(config, layer_name, &index);
        const std::string prefix = "layer" + std::to_string(index) + ".";
        const auto weight = gradients.find(prefix + "weight");
        const auto bias = gradients.find(prefix + "bias");
        Check(weight != gradients.end() && bias != gradients.end(), name + ": gradients of " + layer_name);
        CheckClose(weight->second, layer.at("grad_weight"), tolerance, name + " " + layer_name + " weight gradient");
        CheckClose(bias->second, layer.at("grad_bias"), tolerance, name + " " + layer_name + " bias gradient");
    }
    std::cout << "  " << name << ": forward, input and layer gradients match PyTorch\n";

    // The same branches behind Data Split and Data Loader, as a graph built in
    // the Engine has them: the loader hands the batch to all three branches
    // (it stopped with "missing cached input tensor", 2026-10-15).
    if (name == "fanout_residual_concat") {
        auto through = graph;
        // Data Input -> Data Split -> Data Loader, as the Engine builds them:
        // the Data Input's one pin is a Dataset (no Tensor "Data" pin).
        for (auto& node : through.nodes) {
            if (node.id == 1) node.outputs = {Pin(113, gui::PinType::Dataset, "Dataset", false)};
        }
        gui::MLNode data_split;
        data_split.id = 20;
        data_split.type = gui::NodeType::DataSplit;
        data_split.name = "Split";
        data_split.inputs = {Pin(2001, gui::PinType::Dataset, "Training Dataset", true)};
        data_split.outputs = {Pin(2011, gui::PinType::Dataset, "Partitions", false)};
        gui::MLNode loader;
        loader.id = 21;
        loader.type = gui::NodeType::DataLoader;
        loader.name = "Loader";
        loader.inputs = {Pin(2101, gui::PinType::Dataset, "Partitions", true)};
        loader.outputs = {Pin(2111, gui::PinType::Tensor, "Data", false),
                          Pin(2112, gui::PinType::Labels, "Labels", false)};
        for (auto& link : through.links) {
            if (link.from_node == 1 && link.from_pin == 111) {
                link.from_node = loader.id;
                link.from_pin = 2111;
            } else if (link.from_node == 1 && link.from_pin == 112) {
                link.from_node = loader.id;
                link.from_pin = 2112;
            }
        }
        const auto add_link = [&](int id, int from_node, int from_pin, int to_node, int to_pin) {
            gui::NodeLink link;
            link.id = id;
            link.from_node = from_node;
            link.from_pin = from_pin;
            link.to_node = to_node;
            link.to_pin = to_pin;
            through.links.push_back(link);
        };
        add_link(2000, 1, 113, data_split.id, 2001);
        add_link(2001, data_split.id, 2011, loader.id, 2101);
        through.nodes.push_back(data_split);
        through.nodes.push_back(loader);
        cyxwiz::GraphCompiler through_compiler;
        const auto through_config = through_compiler.Compile(through.nodes, through.links, true);
        if (!through_config.is_valid) PrintIssues(through_config);
        Check(through_config.is_valid, name + ": the graph with Data Split and Data Loader compiles");
        auto through_built = cyxwiz::BuildExecutableFromConfig(through_config);
        Check(through_built.ok(), name + ": with a loader, the branches build: " + through_built.error_message);
        through_built.model->SetParameters(parameters);
        through_built.model->SetTraining(true);
        CheckClose(through_built.model->Forward(ReadTensor(c.at("input"))), c.at("output"), tolerance,
                   name + " forward through the loader");
        CheckClose(through_built.model->Backward(ReadTensor(c.at("grad_output"))), c.at("grad_input"), tolerance,
                   name + " input gradient through the loader");
        const auto through_gradients = through_built.model->GetGradients();
        for (const auto& [layer_name, layer] : c.at("layers").items()) {
            size_t index = 0;
            LayerNamed(through_config, layer_name, &index);
            const auto weight = through_gradients.find("layer" + std::to_string(index) + ".weight");
            Check(weight != through_gradients.end(), name + ": gradient of " + layer_name + " through the loader");
            CheckClose(weight->second, layer.at("grad_weight"), tolerance,
                       name + " " + layer_name + " weight gradient through the loader");
        }
        std::cout << "  " << name << ": the same through a Data Loader\n";
    }

    // A merge takes its inputs in pin order, as the compiler's shapes do, not
    // in the order its links were made: reversing the links into Concatenate
    // or Batch MatMul (A x B, not B x A) changes nothing.
    if (name == "split_concat" || name == "split_bmm") {
        auto reversed = graph;
        std::vector<gui::NodeLink> into_concat;
        std::erase_if(reversed.links, [&](const gui::NodeLink& link) {
            if (link.to_node != 5) return false;
            into_concat.push_back(link);
            return true;
        });
        reversed.links.insert(reversed.links.end(), into_concat.rbegin(), into_concat.rend());
        cyxwiz::GraphCompiler reversed_compiler;
        const auto reversed_config = reversed_compiler.Compile(reversed.nodes, reversed.links, true);
        Check(reversed_config.is_valid, name + ": the graph with reversed merge links compiles");
        auto rebuilt = cyxwiz::BuildExecutableFromConfig(reversed_config);
        Check(rebuilt.ok(), name + ": reversed links build: " + rebuilt.error_message);
        rebuilt.model->SetParameters(parameters);
        rebuilt.model->SetTraining(true);
        CheckClose(rebuilt.model->Forward(ReadTensor(c.at("input"))), c.at("output"), tolerance,
                   name + " forward with the merge links made in reverse");
        CheckClose(rebuilt.model->Backward(ReadTensor(c.at("grad_output"))), c.at("grad_input"), tolerance,
                   name + " input gradient with the merge links made in reverse");
        std::cout << "  " << name << ": reversed merge links give the same result\n";
    }
    std::shared_ptr<cyxwiz::IExecutableModel> trained(std::move(built.model));
    CheckTestRunsGraphModel(name, config, trained, ReadTensor(c.at("input")));
    CheckGraphModelExports(name, *trained);
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "branch_graph_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_BRANCH_GRAPH_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "branch_graph_pytorch.json is readable");
    const json fixture = json::parse(in);
    for (const auto& c : fixture.at("cases")) {
        try {
            RunCase(c);
        } catch (const std::exception& e) {
            Check(false, c.at("name").get<std::string>() + " threw: " + e.what());
        }
    }
    try {
        CheckGraphPackageRoundTrip();
    } catch (const std::exception& e) {
        Check(false, std::string("the graph package round trip threw: ") + e.what());
    }
    try {
        CheckGraphTrainingKeepsBestCheckpoint();
    } catch (const std::exception& e) {
        Check(false, std::string("the graph training with a best checkpoint threw: ") + e.what());
    }

    // The compiler refuses the splits the runtime cannot run, on the node.
    const auto& first = fixture.at("cases").at(0);
    const auto refused = [&](const std::map<std::string, std::string>& params, const std::string& text) {
        auto graph = BuildGraph(first);
        for (auto& node : graph.nodes) {
            if (node.type == gui::NodeType::Split) {
                for (const auto& [key, value] : params) node.parameters[key] = value;
            }
        }
        cyxwiz::GraphCompiler compiler;
        const auto config = compiler.Compile(graph.nodes, graph.links, true);
        bool reported = false;
        for (const auto& issue : config.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error && issue.node_name == "Split" &&
                issue.message.find(text) != std::string::npos) reported = true;
        }
        if (!reported) PrintIssues(config);
        Check(!config.is_valid && reported, "Split refused with '" + text + "'");
    };
    refused({{"dim", "0"}}, "batch");
    refused({{"split_size", "6"}}, "must be less than");
    refused({{"dim", "3"}}, "outside the input");

    // Concatenate inputs must agree outside dim: a mismatch is reported.
    {
        auto graph = BuildGraph(first);
        for (auto& node : graph.nodes) {
            if (node.type == gui::NodeType::Concatenate) node.parameters["dim"] = "-2";
        }
        cyxwiz::GraphCompiler compiler;
        const auto config = compiler.Compile(graph.nodes, graph.links, true);
        Check(!config.is_valid, "Concatenate on a dim outside the rows is refused");
    }

    // Tensor Reshape is retired (Reshape replaced it): a saved graph with one
    // fails to load with a message naming Reshape.
    {
        const json saved = {
            {"nodes", json::array({{{"id", 1}, {"type", static_cast<int>(gui::NodeType::TensorReshape)},
                                    {"name", "Old reshape"}, {"parameters", json::object()}}})},
            {"links", json::array()}};
        cyxwiz::GraphDocument graph;
        std::string error;
        Check(!cyxwiz::ParseGraphDocument(saved.dump(), graph, error) &&
                  error.find("Reshape node replaced") != std::string::npos,
              "a saved Tensor Reshape node is refused with a message naming Reshape: " + error);
    }

    // Bidirectional and Self Attention are retired (LSTM/GRU/RNN's
    // bidirectional setting and Multi-Head Attention replaced them), as are
    // Shared Encoder and Siamese Branch (metric-learning losses run the model
    // once over the stacked batch): a saved graph with one fails to load with
    // a message naming the replacement.
    for (const auto& [retired, name, replacement] :
         {std::tuple{gui::NodeType::Bidirectional, "Old bidirectional", "bidirectional = true"},
          std::tuple{gui::NodeType::SelfAttention, "Old self attention", "Multi-Head Attention replaced"},
          std::tuple{gui::NodeType::SharedEncoder, "Old shared encoder",
                     "connect the encoder layers straight to the loss"},
          std::tuple{gui::NodeType::SiameseBranch, "Old siamese branch",
                     "connect the encoder layers straight to the loss"},
          // TOFIX140 image transforms: the bundle nodes the single transforms replaced.
          std::tuple{gui::NodeType::Augmentation, "Old augmentation", "Augmentation Preset replaced"},
          std::tuple{gui::NodeType::GeometricTransform, "Old geometric", "Random Crop, Image Rotate"},
          std::tuple{gui::NodeType::ColorTransform, "Old color", "Color Jitter replaced"},
          std::tuple{gui::NodeType::ImagePreprocessor, "Old preprocessor", "Resize and Normalize replaced"}}) {
        const json saved = {
            {"nodes", json::array({{{"id", 1}, {"type", static_cast<int>(retired)},
                                    {"name", name}, {"parameters", json::object()}}})},
            {"links", json::array()}};
        cyxwiz::GraphDocument graph;
        std::string error;
        Check(!cyxwiz::ParseGraphDocument(saved.dump(), graph, error) &&
                  error.find(replacement) != std::string::npos,
              std::string("a saved ") + name + " node is refused with a message naming its replacement: " + error);
    }

    std::cout << "Branched graphs match PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
