// Linear Attention through the Engine's own path (TOFIX140 Group C), against
// PyTorch: Data Input [T, E] -> Linear Attention -> Flatten -> Dense -> MSE.
// Forward, input gradient and every parameter gradient match a PyTorch
// reference that computes the full T x T kernel matrix (the Engine uses the
// O(T d^2) key/value summary when not causal): fixtures/linear_attention_pytorch.json
// (generate_linear_attention_fixtures.py). Cases cover elu + 1 and relu,
// causal and not, one to four heads, with and without bias. Also: the
// compiler's refusals.
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
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

gui::NodeLink Link(int id, int from_node, int from_pin, int to_node, int to_pin) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_pin;
    link.to_node = to_node;
    link.to_pin = to_pin;
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

// Data (1) -> Linear Attention (5) -> Flatten (6) -> Dense (7) -> MSE (9) -> SGD (10).
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

    // From the node factory: its pins and defaults are the contract.
    int next_node_id = 5, next_pin_id = 500;
    auto attention = gui::CreateGraphNode(gui::NodeType::LinearAttention, "Linear Attention", next_node_id, next_pin_id);
    attention.id = 5;
    Check(attention.inputs.size() == 1 && attention.inputs[0].name == "Input" && attention.outputs.size() == 1,
          "Linear Attention: one Input in, one Output out");
    attention.inputs[0].id = 501;
    attention.outputs[0].id = 511;
    attention.parameters["embed_dim"] = std::to_string(c.at("embed_dim").get<int>());
    attention.parameters["num_heads"] = std::to_string(c.at("num_heads").get<int>());
    attention.parameters["feature_map"] = c.at("feature_map").get<std::string>();
    attention.parameters["causal"] = c.at("causal").get<bool>() ? "true" : "false";
    attention.parameters["use_bias"] = c.at("use_bias").get<bool>() ? "true" : "false";

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

    g.nodes = {data, attention, flatten, dense, loss, sgd};
    g.links = {Link(1, 1, 111, 5, 501), Link(2, 5, 511, 6, 601), Link(3, 6, 611, 7, 701),
               Link(4, 7, 711, 9, 901), Link(5, 1, 112, 9, 902), Link(6, 9, 911, 10, 1001)};
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
    Check(config.is_valid, name + ": the Linear Attention graph compiles");

    const size_t attention_index = LayerIndex(config, gui::NodeType::LinearAttention);
    const size_t head_index = LayerIndex(config, gui::NodeType::Dense);
    const auto input_shape = c.at("input").at("shape").get<std::vector<size_t>>();
    const std::vector<size_t> sample{input_shape[1], input_shape[2]};
    Check(config.layers[attention_index].input_shape == sample &&
              config.layers[attention_index].output_shape == sample,
          name + ": Linear Attention keeps the [length, embed_dim] shape: " +
              ShapeText(config.layers[attention_index].output_shape));

    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds the graph: " + built.error_message);

    std::map<std::string, cyxwiz::Tensor> parameters;
    const std::string attention_prefix = "layer" + std::to_string(attention_index) + ".";
    const std::string head_prefix = "layer" + std::to_string(head_index) + ".";
    for (const auto& [key, value] : c.at("attention").items()) parameters[attention_prefix + key] = ReadTensor(value);
    parameters[head_prefix + "weight"] = ReadTensor(c.at("head").at("weight"));
    parameters[head_prefix + "bias"] = ReadTensor(c.at("head").at("bias"));
    built.model->SetParameters(parameters);
    built.model->SetTraining(true);
    const auto loaded = built.model->GetParameters();
    for (const auto& [key, value] : c.at("attention").items()) {
        Check(loaded.count(attention_prefix + key) == 1, name + ": the model has parameter " + key);
        CheckClose(loaded.at(attention_prefix + key), value, c.at("tolerance"), name + " loaded " + key);
    }

    const json& tolerance = c.at("tolerance");
    const cyxwiz::Tensor output = built.model->Forward(ReadTensor(c.at("input")));
    CheckClose(output, c.at("output"), tolerance, name + " forward");
    const cyxwiz::Tensor grad_input = built.model->Backward(ReadTensor(c.at("grad_output")));
    CheckClose(grad_input, c.at("grad_input"), tolerance, name + " input gradient");

    const auto gradients = built.model->GetGradients();
    for (const auto& [key, value] : c.at("attention_grad").items()) {
        const auto found = gradients.find(attention_prefix + key);
        Check(found != gradients.end(), name + ": gradient of " + key);
        CheckClose(found->second, value, tolerance, name + " " + key + " gradient");
    }
    CheckClose(gradients.at(head_prefix + "weight"), c.at("head").at("grad_weight"), tolerance, name + " head weight gradient");
    CheckClose(gradients.at(head_prefix + "bias"), c.at("head").at("grad_bias"), tolerance, name + " head bias gradient");
    std::cout << "  " << name << ": forward, input and parameter gradients match PyTorch\n";
}

void CheckRefused(const Graph& graph, const std::string& text, const std::string& what) {
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
                        "linear_attention_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_LINEAR_ATTENTION_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "linear_attention_pytorch.json is readable");
    const json fixture = json::parse(in);
    for (const auto& c : fixture.at("cases")) RunCase(c);

    const json& first = fixture.at("cases").at(0);
    const auto with_attention = [&](const std::string& key, const std::string& value) {
        auto graph = BuildGraph(first);
        for (auto& node : graph.nodes) {
            if (node.type == gui::NodeType::LinearAttention) node.parameters[key] = value;
        }
        return graph;
    };
    CheckRefused(with_attention("embed_dim", "8"), "embed_dim", "an embed_dim that is not the input width");
    CheckRefused(with_attention("num_heads", "3"), "divide evenly", "heads that do not divide embed_dim");
    // The retired random-feature option is refused, not silently replaced.
    CheckRefused(with_attention("feature_map", "favor+"), "feature_map", "the retired favor+ feature map");
    CheckRefused(with_attention("eps", "0"), "eps", "a zero eps");

    std::cout << "Linear Attention matches PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
