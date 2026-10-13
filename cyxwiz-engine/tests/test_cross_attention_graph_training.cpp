// Cross Attention through the Engine's own path (TOFIX140 Group C), against
// PyTorch: Data Input [T, E] -> Split (and a second Split) -> Cross Attention
// (Query over Key / Value from other branches, different lengths) -> Flatten
// -> Dense -> MSE. The compiler gives every input pin its own shape, the
// GraphExecutableModel feeds the three inputs by pin (links are made in the
// order Value, Key, Query on purpose) and returns one gradient per pin, and
// forward, input gradient and every parameter gradient match
// torch.nn.MultiheadAttention(q, k, v): fixtures/cross_attention_graph_pytorch.json
// (generate_cross_attention_graph_fixtures.py). Also: the compiler's refusals.
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
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

// A node from the node factory (its pins and defaults are the contract); pins
// renumbered: inputs id*100+1.., outputs id*100+11..
gui::MLNode FactoryNode(gui::NodeType type, int id, const std::string& name) {
    int next_node_id = id, next_pin_id = 0;
    auto node = gui::CreateGraphNode(type, name, next_node_id, next_pin_id);
    node.id = id;
    for (size_t i = 0; i < node.inputs.size(); ++i) node.inputs[i].id = id * 100 + 1 + static_cast<int>(i);
    for (size_t i = 0; i < node.outputs.size(); ++i) node.outputs[i].id = id * 100 + 11 + static_cast<int>(i);
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
              what + ": element " + std::to_string(i) + " is " + std::to_string(data[i]) + ", PyTorch " +
                  std::to_string(values[i]));
    }
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

// Data (1) -> Split (2) [-> Split (3)] -> Cross Attention (5) -> Flatten (6)
// -> Dense (7) -> MSE (9) -> SGD (10).
Graph BuildGraph(const json& c) {
    Graph g;
    const auto input_shape = c.at("input").at("shape").get<std::vector<size_t>>();
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Sequences";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "sequence_rows"},
                       {"shape", "[" + std::to_string(input_shape[1]) + ", " + std::to_string(input_shape[2]) + "]"}};
    gui::MLNode loss;
    loss.id = 9;
    loss.type = gui::NodeType::MSELoss;
    loss.name = "MSE";
    loss.inputs = {Pin(901, gui::PinType::Tensor, "Predictions", true), Pin(902, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(911, gui::PinType::Loss, "Loss", false)};
    gui::MLNode sgd;
    sgd.id = 10;
    sgd.type = gui::NodeType::SGD;
    sgd.name = "SGD";
    sgd.inputs = {Pin(1001, gui::PinType::Loss, "Loss", true)};

    const auto splits = c.at("splits").get<std::vector<int>>();
    auto split = FactoryNode(gui::NodeType::Split, 2, "Split");
    split.parameters["split_size"] = std::to_string(splits[0]);
    split.parameters["dim"] = "1";

    auto attention = FactoryNode(gui::NodeType::CrossAttention, 5, "Cross Attention");
    Check(attention.inputs.size() == 3 && attention.inputs[0].name == "Query" && attention.inputs[1].name == "Key" &&
              attention.inputs[2].name == "Value" && attention.outputs.size() == 1,
          "Cross Attention: Query, Key, Value in, Output out");
    attention.parameters["embed_dim"] = std::to_string(c.at("embed_dim").get<int>());
    attention.parameters["num_heads"] = std::to_string(c.at("num_heads").get<int>());

    gui::MLNode flatten;
    flatten.id = 6;
    flatten.type = gui::NodeType::Flatten;
    flatten.name = "Flatten";
    flatten.inputs = {Pin(601, gui::PinType::Tensor, "Input", true)};
    flatten.outputs = {Pin(611, gui::PinType::Tensor, "Output", false)};
    gui::MLNode dense;
    dense.id = 7;
    dense.type = gui::NodeType::Dense;
    dense.name = "Head";
    dense.inputs = {Pin(701, gui::PinType::Tensor, "Input", true)};
    dense.outputs = {Pin(711, gui::PinType::Tensor, "Output", false)};
    dense.parameters = {{"units", std::to_string(c.at("head").at("bias").at("shape")[0].get<size_t>())}};

    g.nodes = {data, split, attention, flatten, dense, loss, sgd};
    g.links = {Link(1, 0, 2, 0)};
    // Value, Key, Query: the runtime must route by pin, not by link order.
    if (c.at("kv_separate").get<bool>()) {
        auto rest = FactoryNode(gui::NodeType::Split, 3, "Split K / V");
        rest.parameters["split_size"] = std::to_string(splits[1]);
        rest.parameters["dim"] = "1";
        g.nodes.push_back(rest);
        g.links.push_back(Link(2, 1, 3, 0));
        g.links.push_back(Link(3, 1, 5, 2));
        g.links.push_back(Link(3, 0, 5, 1));
    } else {
        g.links.push_back(Link(2, 1, 5, 2));
        g.links.push_back(Link(2, 1, 5, 1));
    }
    g.links.push_back(Link(2, 0, 5, 0));
    g.links.push_back(Link(5, 0, 6, 0));
    g.links.push_back(Link(6, 0, 7, 0));
    gui::NodeLink to_loss;
    to_loss.id = 900; to_loss.from_node = 7; to_loss.from_pin = 711; to_loss.to_node = 9; to_loss.to_pin = 901;
    gui::NodeLink labels;
    labels.id = 901; labels.from_node = 1; labels.from_pin = 112; labels.to_node = 9; labels.to_pin = 902;
    gui::NodeLink to_sgd;
    to_sgd.id = 902; to_sgd.from_node = 9; to_sgd.from_pin = 911; to_sgd.to_node = 10; to_sgd.to_pin = 1001;
    g.links.push_back(to_loss);
    g.links.push_back(labels);
    g.links.push_back(to_sgd);
    return g;
}

size_t LayerIndex(const cyxwiz::TrainingConfiguration& config, gui::NodeType type) {
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (config.layers[i].type == type) return i;
    }
    Check(false, "compiled layer missing");
    return 0;
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
    Check(config.is_valid, name + ": the Cross Attention graph compiles");
    Check(config.multi_input_layer_node_ids == std::vector<int>{5}, name + ": Cross Attention runs as a multi-input layer");

    const size_t attention_index = LayerIndex(config, gui::NodeType::CrossAttention);
    const size_t head_index = LayerIndex(config, gui::NodeType::Dense);
    const size_t query_length = static_cast<size_t>(c.at("splits")[0].get<int>());
    const size_t embed_dim = c.at("embed_dim").get<size_t>();
    Check(config.layers[attention_index].input_shape == std::vector<size_t>{query_length, embed_dim} &&
              config.layers[attention_index].output_shape == std::vector<size_t>{query_length, embed_dim},
          name + ": Cross Attention keeps the Query's shape: " +
              ShapeText(config.layers[attention_index].output_shape));

    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds the graph: " + built.error_message);
    auto* model = dynamic_cast<cyxwiz::GraphExecutableModel*>(built.model.get());
    Check(model != nullptr, name + ": a graph executable");

    std::map<std::string, cyxwiz::Tensor> parameters;
    const std::string attention_prefix = "layer" + std::to_string(attention_index) + ".";
    const std::string head_prefix = "layer" + std::to_string(head_index) + ".";
    for (const auto& [key, value] : c.at("attention").items()) parameters[attention_prefix + key] = ReadTensor(value);
    parameters[head_prefix + "weight"] = ReadTensor(c.at("head").at("weight"));
    parameters[head_prefix + "bias"] = ReadTensor(c.at("head").at("bias"));
    model->SetParameters(parameters);
    model->SetTraining(true);

    const json& tolerance = c.at("tolerance");
    const cyxwiz::Tensor output = model->Forward(ReadTensor(c.at("input")));
    CheckClose(output, c.at("output"), tolerance, name + " forward");
    const cyxwiz::Tensor grad_input = model->Backward(ReadTensor(c.at("grad_output")));
    CheckClose(grad_input, c.at("grad_input"), tolerance, name + " input gradient (Query and Key / Value parts)");

    const auto gradients = model->GetGradients();
    for (const auto& [key, value] : c.at("attention_grad").items()) {
        const auto found = gradients.find(attention_prefix + key);
        Check(found != gradients.end(), name + ": gradient of " + key);
        CheckClose(found->second, value, tolerance, name + " " + key + " gradient");
    }
    CheckClose(gradients.at(head_prefix + "weight"), c.at("head").at("grad_weight"), tolerance, name + " head weight gradient");
    CheckClose(gradients.at(head_prefix + "bias"), c.at("head").at("grad_bias"), tolerance, name + " head bias gradient");
    std::cout << "  " << name << ": forward, input and parameter gradients match PyTorch\n";
}

void CheckRefused(Graph graph, const std::string& text, const std::string& what) {
    cyxwiz::GraphCompiler compiler;
    const auto config = compiler.Compile(graph.nodes, graph.links, true);
    bool reported = false;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error && issue.message.find(text) != std::string::npos) reported = true;
    }
    if (!reported) PrintIssues(config);
    Check(!config.is_valid && reported, what + ": refused with '" + text + "'");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "cross_attention_graph_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_CROSS_ATTENTION_GRAPH_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "cross_attention_graph_pytorch.json is readable");
    const json fixture = json::parse(in);
    for (const auto& c : fixture.at("cases")) RunCase(c);

    const json& shared = fixture.at("cases").at(0);
    const auto with_attention = [&](const std::string& key, const std::string& value) {
        auto graph = BuildGraph(shared);
        for (auto& node : graph.nodes) {
            if (node.type == gui::NodeType::CrossAttention) node.parameters[key] = value;
        }
        return graph;
    };
    CheckRefused(with_attention("embed_dim", "8"), "embed_dim", "an embed_dim that is not the input width");
    CheckRefused(with_attention("num_heads", "3"), "divide evenly", "heads that do not divide embed_dim");
    {
        auto graph = BuildGraph(shared);
        std::erase_if(graph.links, [](const gui::NodeLink& link) { return link.to_node == 5 && link.to_pin == 503; });
        CheckRefused(graph, "Value", "an unconnected Value");
    }

    std::cout << "Cross Attention matches PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
