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
#include "../src/gui/loaders/data_loader.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
    data.parameters = {{"dataset_name", "branch_rows"}, {"shape", "[6]"}};
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

void RunCase(const json& c) {
    const std::string name = c.at("name").get<std::string>();
    const auto graph = BuildGraph(c);
    cyxwiz::GraphCompiler compiler;
    const auto config = compiler.Compile(graph.nodes, graph.links, true);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, name + ": the branched graph compiles");
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

    // A merge takes its inputs in pin order, as the compiler's shapes do, not
    // in the order its links were made: reversing the links into Concatenate
    // changes nothing.
    if (name == "split_concat") {
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
        Check(reversed_config.is_valid, name + ": the graph with reversed Concatenate links compiles");
        auto rebuilt = cyxwiz::BuildExecutableFromConfig(reversed_config);
        Check(rebuilt.ok(), name + ": reversed links build: " + rebuilt.error_message);
        rebuilt.model->SetParameters(parameters);
        rebuilt.model->SetTraining(true);
        CheckClose(rebuilt.model->Forward(ReadTensor(c.at("input"))), c.at("output"), tolerance,
                   name + " forward with Concatenate links made in reverse");
        CheckClose(rebuilt.model->Backward(ReadTensor(c.at("grad_output"))), c.at("grad_input"), tolerance,
                   name + " input gradient with Concatenate links made in reverse");
        std::cout << "  " << name << ": reversed Concatenate links give the same result\n";
    }
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
    for (const auto& c : fixture.at("cases")) RunCase(c);

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
                     "connect the encoder layers straight to the loss"}}) {
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
