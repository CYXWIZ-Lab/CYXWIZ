// Conv3D through the Engine's own path (TOFIX140 Group C), against PyTorch:
// Data Input [D, H, W, C] -> Conv3D (-> ReLU -> Conv3D) -> Flatten -> Dense
// -> MSE. The compiled shapes, the forward output, the input gradient and
// every parameter gradient match torch.nn.Conv3d: fixtures/conv3d_pytorch.json
// (generate_conv3d_fixtures.py). Cases cover one and several channels,
// 'same' and 'valid' padding, stride 2, a 1x1x1 kernel and two stacked
// Conv3D layers. Also: the compiler's refusals (the volume section rules).
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

gui::MLNode Simple(int id, gui::NodeType type, const std::string& name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 11, gui::PinType::Tensor, "Output", false)};
    return node;
}

// Data (1) -> [Conv3D (5), ReLU (6), Conv3D (7)] -> Flatten (8) -> Dense (9)
// -> MSE (10) -> SGD (11); with_flatten = false drops the Flatten.
Graph BuildGraph(const json& c, bool with_flatten = true) {
    Graph g;
    const auto sample = c.at("sample").get<std::vector<size_t>>();
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Volumes";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "volume_rows"}, {"shape", ShapeText(sample)}};
    g.nodes.push_back(data);

    int previous_node = 1, previous_pin = 111, link_id = 1, conv_id = 5;
    const auto chain = [&](const gui::MLNode& node) {
        g.links.push_back(Link(link_id++, previous_node, previous_pin, node.id, node.inputs[0].id));
        g.nodes.push_back(node);
        previous_node = node.id;
        previous_pin = node.outputs[0].id;
    };
    const auto& convs = c.at("convs");
    for (size_t i = 0; i < convs.size(); ++i) {
        if (i > 0 && c.at("relu_between").get<bool>()) chain(Simple(6, gui::NodeType::ReLU, "ReLU"));
        // From the node factory: its pins and defaults are the contract.
        int next_node_id = conv_id, next_pin_id = conv_id * 100;
        auto conv = gui::CreateGraphNode(gui::NodeType::Conv3D, "Conv3D " + std::to_string(i + 1), next_node_id,
                                         next_pin_id);
        Check(conv.inputs.size() == 1 && conv.inputs[0].name == "Input" && conv.outputs.size() == 1,
              "Conv3D: one Input in, one Output out");
        conv.id = conv_id;
        conv.inputs[0].id = conv_id * 100 + 1;
        conv.outputs[0].id = conv_id * 100 + 11;
        conv.parameters["filters"] = std::to_string(convs[i].at("filters").get<int>());
        conv.parameters["kernel_size"] = std::to_string(convs[i].at("kernel_size").get<int>());
        conv.parameters["stride"] = std::to_string(convs[i].at("stride").get<int>());
        conv.parameters["padding"] = convs[i].at("padding").get<std::string>();
        chain(conv);
        conv_id += 2;
    }
    if (with_flatten) chain(Simple(8, gui::NodeType::Flatten, "Flatten"));
    auto dense = Simple(9, gui::NodeType::Dense, "Head");
    dense.parameters = {{"units", std::to_string(c.at("head").at("bias").at("shape")[0].get<size_t>())}};
    chain(dense);

    gui::MLNode loss;
    loss.id = 10;
    loss.type = gui::NodeType::MSELoss;
    loss.name = "MSE";
    loss.inputs = {Pin(1001, gui::PinType::Tensor, "Predictions", true), Pin(1002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(1011, gui::PinType::Loss, "Loss", false)};
    gui::MLNode sgd;
    sgd.id = 11;
    sgd.type = gui::NodeType::SGD;
    sgd.name = "SGD";
    sgd.inputs = {Pin(1101, gui::PinType::Loss, "Loss", true)};
    g.nodes.push_back(loss);
    g.nodes.push_back(sgd);
    g.links.push_back(Link(link_id++, 9, 911, 10, 1001));
    g.links.push_back(Link(link_id++, 1, 112, 10, 1002));
    g.links.push_back(Link(link_id++, 10, 1011, 11, 1101));
    return g;
}

std::vector<size_t> LayerIndices(const cyxwiz::TrainingConfiguration& config, gui::NodeType type) {
    std::vector<size_t> indices;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (config.layers[i].type == type) indices.push_back(i);
    }
    return indices;
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
    Check(config.is_valid, name + ": the Conv3D graph compiles");

    const auto conv_indices = LayerIndices(config, gui::NodeType::Conv3D);
    const auto head_indices = LayerIndices(config, gui::NodeType::Dense);
    Check(conv_indices.size() == c.at("convs").size() && head_indices.size() == 1, name + ": compiled layers");
    Check(config.layers[conv_indices.front()].input_shape == c.at("sample").get<std::vector<size_t>>(),
          name + ": the first Conv3D takes the Data Input's [D, H, W, C] sample");
    const auto expected_out = c.at("output_sample").get<std::vector<size_t>>();
    Check(config.layers[conv_indices.back()].output_shape == expected_out,
          name + ": the last Conv3D gives PyTorch's " + ShapeText(expected_out) + ", compiled " +
              ShapeText(config.layers[conv_indices.back()].output_shape));

    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds the graph: " + built.error_message);

    std::map<std::string, cyxwiz::Tensor> parameters;
    const auto& convs = c.at("convs");
    for (size_t i = 0; i < convs.size(); ++i) {
        const std::string prefix = "layer" + std::to_string(conv_indices[i]) + ".";
        parameters[prefix + "weight"] = ReadTensor(convs[i].at("weight"));
        parameters[prefix + "bias"] = ReadTensor(convs[i].at("bias"));
    }
    const std::string head_prefix = "layer" + std::to_string(head_indices[0]) + ".";
    parameters[head_prefix + "weight"] = ReadTensor(c.at("head").at("weight"));
    parameters[head_prefix + "bias"] = ReadTensor(c.at("head").at("bias"));
    built.model->SetParameters(parameters);
    built.model->SetTraining(true);
    const auto loaded = built.model->GetParameters();
    const json& tolerance = c.at("tolerance");
    for (size_t i = 0; i < convs.size(); ++i) {
        const std::string key = "layer" + std::to_string(conv_indices[i]) + ".weight";
        Check(loaded.count(key) == 1, name + ": the model has " + key);
        CheckClose(loaded.at(key), convs[i].at("weight"), tolerance, name + " loaded " + key);
    }

    const cyxwiz::Tensor output = built.model->Forward(ReadTensor(c.at("input")));
    CheckClose(output, c.at("output"), tolerance, name + " forward");
    const cyxwiz::Tensor grad_input = built.model->Backward(ReadTensor(c.at("grad_output")));
    CheckClose(grad_input, c.at("grad_input"), tolerance, name + " input gradient");

    const auto gradients = built.model->GetGradients();
    for (size_t i = 0; i < convs.size(); ++i) {
        const std::string prefix = "layer" + std::to_string(conv_indices[i]) + ".";
        CheckClose(gradients.at(prefix + "weight"), convs[i].at("grad_weight"), tolerance,
                   name + " Conv3D " + std::to_string(i + 1) + " weight gradient");
        CheckClose(gradients.at(prefix + "bias"), convs[i].at("grad_bias"), tolerance,
                   name + " Conv3D " + std::to_string(i + 1) + " bias gradient");
    }
    CheckClose(gradients.at(head_prefix + "weight"), c.at("head").at("grad_weight"), tolerance, name + " head weight gradient");
    CheckClose(gradients.at(head_prefix + "bias"), c.at("head").at("grad_bias"), tolerance, name + " head bias gradient");
    std::cout << "  " << name << ": shapes, forward, input and parameter gradients match PyTorch\n";
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
                        "conv3d_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_CONV3D_FIXTURE);
}

}  // namespace

int main(int, char** argv) {
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "conv3d_pytorch.json is readable");
    const json fixture = json::parse(in);
    for (const auto& c : fixture.at("cases")) RunCase(c);

    const json& first = fixture.at("cases").at(0);
    const auto with_conv = [&](const std::string& key, const std::string& value) {
        auto graph = BuildGraph(first);
        for (auto& node : graph.nodes) {
            if (node.type == gui::NodeType::Conv3D) node.parameters[key] = value;
        }
        return graph;
    };
    // 'same' pads any odd kernel to fit, so the oversized kernel is 'valid'.
    auto oversized = with_conv("kernel_size", "5");
    for (auto& node : oversized.nodes) {
        if (node.type == gui::NodeType::Conv3D) node.parameters["padding"] = "valid";
    }
    CheckRefused(oversized, "does not fit", "a kernel larger than the 4-deep volume");
    CheckRefused(with_conv("kernel_size", "2"), "odd kernel", "'same' padding with an even kernel");
    CheckRefused(with_conv("filters", "0"), "filters must be positive", "zero filters");

    auto flat_rows = BuildGraph(first);
    flat_rows.nodes[0].parameters["shape"] = "[120]";
    CheckRefused(flat_rows, "[D, H, W, C] volume", "a Data Input without a volume shape");
    CheckRefused(BuildGraph(first, false), "end the section with Flatten", "Dense straight after Conv3D");

    std::cout << "Conv3D matches PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
