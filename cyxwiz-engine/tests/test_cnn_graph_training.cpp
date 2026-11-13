// A CNN graph through the Engine's own path (TOFIX140 A1): image Data Input
// -> Resize 64 -> Conv2D 16 -> ReLU -> MaxPool2D -> Conv2D 32 -> ReLU ->
// MaxPool2D -> Flatten -> Dense 2 -> CrossEntropy, Adam. The compiler must
// give every layer the PyTorch shapes, ModelBuilder must build the modules,
// and a forward pass on an HWC-flattened batch must produce [N, 2] logits.
#include "core/weight_start.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/model_builder.h"
#include "../src/core/spatial_batch_layout.h"
#include "../src/core/spatial_sequential_head.h"
#include "../src/gui/loaders/data_loader.h"
#include "computation_truth/test_device_selection.h"

#include <sstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

// Graph-only: no file loaders, so datasets are not read.
namespace cyxwiz::loaders {
DataLoader* GetByCategory(FileCategory) { return nullptr; }
DataLoader* GetByRegisteredDataset(const std::string&) { return nullptr; }
FileCategory FileCategoryFromString(const std::string& text) { return text == "image" ? FileCategory::Image : FileCategory::Tabular; }
}  // namespace cyxwiz::loaders

namespace {

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

gui::MLNode Layer(int id, gui::NodeType type, const std::string& name,
                  std::map<std::string, std::string> parameters = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 2, gui::PinType::Tensor, "Output", false)};
    node.parameters = std::move(parameters);
    return node;
}

gui::NodeLink Link(int id, int from_node, int to_node) {
    gui::NodeLink link;
    link.id = id;
    link.from_node = from_node;
    link.from_pin = from_node * 100 + 2;
    link.to_node = to_node;
    link.to_pin = to_node * 100 + 1;
    return link;
}

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    return s + "]";
}

// Takes a view: a reference returned for a temporary std::string argument looks
// dangling to GCC (-Wdangling-reference).
const cyxwiz::CompiledLayer& LayerNamed(const cyxwiz::TrainingConfiguration& config, std::string_view name) {
    for (const auto& layer : config.layers) {
        if (layer.name == name) return layer;
    }
    Check(false, "compiled layer missing: " + std::string(name));
    return config.layers.front();
}

size_t LayerIndex(const cyxwiz::TrainingConfiguration& config, const std::string& name) {
    for (size_t i = 0; i < config.layers.size(); ++i) {
        if (config.layers[i].name == name) return i;
    }
    Check(false, "compiled layer missing: " + name);
    return 0;
}

cyxwiz::Tensor ReadTensor(const nlohmann::json& value) {
    const auto shape = value.at("shape").get<std::vector<size_t>>();
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckClose(const cyxwiz::Tensor& actual, const nlohmann::json& expected, const nlohmann::json& tolerance,
                const std::string& what) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    Check(actual.Shape() == shape, what + ": shape " + ShapeText(actual.Shape()) + ", PyTorch " + ShapeText(shape));
    if (actual.Shape() != shape) return;
    const auto values = expected.at("values").get<std::vector<float>>();
    const float* data = actual.ReadData<float>();
    const float atol = tolerance.at("atol").get<float>();
    const float rtol = tolerance.at("rtol").get<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::fabs(data[i] - values[i]) > atol + rtol * std::fabs(values[i])) {
            Check(false, what + ": element " + std::to_string(i) + " expected " + std::to_string(values[i]) +
                             ", got " + std::to_string(data[i]));
            return;
        }
    }
}

// The PyTorch residual CNN (computation_truth/fixtures/cnn_graph_pytorch.json,
// generate_cnn_graph_fixtures.py) built from Engine nodes: a stem, one ResNet
// downsampling block with a projection shortcut, Global Avg Pool and a Dense
// head; one training step's forward, gradients and BatchNorm running statistics.
void CheckGraphAgainstPyTorch(cyxwiz::GraphCompiler& compiler, const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links, const nlohmann::json& c);

void CheckResidualCnnAgainstPyTorch(cyxwiz::GraphCompiler& compiler) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(0);

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "residual_fixture"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/residual_fixture"}, {"image_layout", "0"}};
    const auto conv = [](int id, const std::string& n, int filters, int kernel, int stride, int pad) {
        return Layer(id, gui::NodeType::Conv2D, n,
                     {{"filters", std::to_string(filters)}, {"kernel_size", std::to_string(kernel)},
                      {"stride", std::to_string(stride)}, {"padding", std::to_string(pad)}});
    };
    const auto bn = [](int id, const std::string& n) {
        return Layer(id, gui::NodeType::BatchNorm, n, {{"eps", "1e-5"}, {"momentum", "0.1"}});
    };
    gui::MLNode add = Layer(30, gui::NodeType::Add, "Block add");
    add.inputs.push_back(Pin(3003, gui::PinType::Tensor, "Input 2", true));
    gui::MLNode loss;
    loss.id = 40;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(4001, gui::PinType::Tensor, "Predictions", true), Pin(4002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(4003, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 41;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(4101, gui::PinType::Loss, "Loss", true)};
    gui::MLNode output;
    output.id = 42;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};
    std::vector<gui::MLNode> nodes = {
        data, Layer(2, gui::NodeType::Resize, "Resize 8", {{"width", "8"}, {"height", "8"}, {"mode", "exact"}}),
        conv(3, "Stem conv", 8, 3, 1, 1), bn(4, "Stem BN"), Layer(5, gui::NodeType::ReLU, "Stem ReLU"),
        conv(10, "Main conv", 16, 3, 2, 1), bn(11, "Main BN"), Layer(12, gui::NodeType::ReLU, "Main ReLU"),
        conv(13, "Main conv 2", 16, 3, 1, 1), bn(14, "Main BN 2"),
        conv(20, "Shortcut conv", 16, 1, 2, 0), bn(21, "Shortcut BN"),
        add, Layer(31, gui::NodeType::ReLU, "Block ReLU"), Layer(32, gui::NodeType::GlobalAvgPool, "GAP"),
        Layer(33, gui::NodeType::Dense, "Head", {{"units", "2"}}), loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 4), Link(4, 4, 5),
                                        Link(5, 5, 10), Link(6, 10, 11), Link(7, 11, 12), Link(8, 12, 13),
                                        Link(9, 13, 14), Link(10, 5, 20), Link(11, 20, 21), Link(12, 14, 30),
                                        Link(14, 30, 31), Link(15, 31, 32), Link(16, 32, 33)};
    gui::NodeLink second = Link(13, 21, 30);
    second.to_pin = 3003;
    links.push_back(second);
    gui::NodeLink to_loss = Link(17, 33, 40);
    to_loss.to_pin = 4001;
    gui::NodeLink labels;
    labels.id = 18; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 40; labels.to_pin = 4002;
    gui::NodeLink to_adam;
    to_adam.id = 19; to_adam.from_node = 40; to_adam.from_pin = 4003; to_adam.to_node = 41; to_adam.to_pin = 4101;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);

    CheckGraphAgainstPyTorch(compiler, nodes, links, c);
    std::cout << "residual CNN matches PyTorch: logits, input and " << c.at("layers").size()
              << " layers' gradients, BatchNorm running statistics\n";
}

// A graph of Engine nodes against a fixture case: PyTorch's weights go in by
// node name; one training step's logits, input gradient, every layer's
// gradients and BatchNorm's running statistics are compared.
void CheckGraphAgainstPyTorch(cyxwiz::GraphCompiler& compiler, const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links, const nlohmann::json& c) {
    const std::string name = c.at("name").get<std::string>();
    const auto config = compiler.Compile(nodes, links, true);
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error)
            std::cerr << "  issue: " << issue.node_name << ": " << issue.message << "\n";
    }
    Check(config.is_valid, name + ": the graph compiles");
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds it: " + built.error_message);
    if (!built.ok()) return;

    // PyTorch's weights and running statistics, by node name.
    std::map<std::string, cyxwiz::Tensor> parameters;
    for (const auto& [layer_name, layer] : c.at("layers").items()) {
        const std::string prefix = "layer" + std::to_string(LayerIndex(config, layer_name)) + ".";
        for (const auto& [key, value] : layer.items()) {
            if (key.rfind("grad_", 0) != 0) parameters[prefix + key] = ReadTensor(value);
        }
    }
    for (const auto& [layer_name, stats] : c.at("state").items()) {
        const std::string prefix = "layer" + std::to_string(LayerIndex(config, layer_name)) + ".";
        for (const auto& [key, value] : stats.items()) parameters[prefix + key] = ReadTensor(value);
    }
    built.model->SetParameters(parameters);
    built.model->SetTraining(true);

    const auto& tolerance = c.at("tolerance");
    const cyxwiz::Tensor logits = built.model->Forward(ReadTensor(c.at("input")));
    CheckClose(logits, c.at("output"), tolerance, name + " logits");
    const cyxwiz::Tensor grad_input = built.model->Backward(ReadTensor(c.at("grad_output")));
    CheckClose(grad_input, c.at("grad_input"), tolerance, name + " input gradient");
    const auto gradients = built.model->GetGradients();
    const auto state = built.model->GetParameters();
    for (const auto& [layer_name, layer] : c.at("layers").items()) {
        const std::string prefix = "layer" + std::to_string(LayerIndex(config, layer_name)) + ".";
        for (const auto& [key, value] : layer.items()) {
            if (key.rfind("grad_", 0) != 0) continue;
            const auto it = gradients.find(prefix + key.substr(5));
            Check(it != gradients.end(), name + ": no gradient " + prefix + key.substr(5) + " (" + layer_name + ")");
            if (it != gradients.end()) CheckClose(it->second, value, tolerance, name + " " + layer_name + " " + key);
        }
    }
    for (const auto& [layer_name, stats] : c.at("state_after").items()) {
        const std::string prefix = "layer" + std::to_string(LayerIndex(config, layer_name)) + ".";
        for (const auto& [key, value] : stats.items()) {
            const auto it = state.find(prefix + key);
            Check(it != state.end(), name + ": no state " + prefix + key);
            if (it != state.end()) CheckClose(it->second, value, tolerance, name + " " + layer_name + " " + key);
        }
    }
}

// The Squeeze-Excitation node after a stem, against PyTorch: its gate's two
// Linear layers are the node's reduce_* and expand_* parameters.
void CheckSqueezeExcitationAgainstPyTorch(cyxwiz::GraphCompiler& compiler, size_t case_index, const std::string& case_name,
                                          const std::map<std::string, std::string>& se_settings) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(case_index);
    Check(c.at("name").get<std::string>() == case_name, "fixture case " + std::to_string(case_index) + " is " + case_name);

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "se_fixture"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/se_fixture"}, {"image_layout", "0"}};
    gui::MLNode loss;
    loss.id = 40;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(4001, gui::PinType::Tensor, "Predictions", true), Pin(4002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(4003, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 41;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(4101, gui::PinType::Loss, "Loss", true)};
    gui::MLNode output;
    output.id = 42;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};
    const std::vector<gui::MLNode> nodes = {
        data, Layer(2, gui::NodeType::Resize, "Resize 8", {{"width", "8"}, {"height", "8"}, {"mode", "exact"}}),
        Layer(3, gui::NodeType::Conv2D, "Stem conv",
              {{"filters", "8"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "1"}}),
        Layer(4, gui::NodeType::BatchNorm, "Stem BN", {{"eps", "1e-5"}, {"momentum", "0.1"}}),
        Layer(5, gui::NodeType::ReLU, "Stem ReLU"),
        Layer(10, gui::NodeType::SqueezeExcitation, "SE", se_settings),
        Layer(32, gui::NodeType::GlobalAvgPool, "GAP"),
        Layer(33, gui::NodeType::Dense, "Head", {{"units", "2"}}), loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 4), Link(4, 4, 5),
                                        Link(5, 5, 10), Link(6, 10, 32), Link(7, 32, 33)};
    gui::NodeLink to_loss = Link(17, 33, 40);
    to_loss.to_pin = 4001;
    gui::NodeLink labels;
    labels.id = 18; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 40; labels.to_pin = 4002;
    gui::NodeLink to_adam;
    to_adam.id = 19; to_adam.from_node = 40; to_adam.from_pin = 4003; to_adam.to_node = 41; to_adam.to_pin = 4101;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);
    CheckGraphAgainstPyTorch(compiler, nodes, links, c);
    std::cout << "Squeeze-Excitation node (" << case_name << ") matches PyTorch: logits, input and " << c.at("layers").size()
              << " layers' gradients\n";
}

// ResNet's opening against PyTorch: a 7x7 stride-2 convolution with its Bias
// setting off, BatchNorm, ReLU6 and a padded max pool (size 3, stride 2,
// padding 1), the settings a pretrained ResNet or MobileNet needs.
void CheckResNetStemAgainstPyTorch(cyxwiz::GraphCompiler& compiler) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(3);
    Check(c.at("name").get<std::string>() == "resnet_stem", "fixture case 3 is resnet_stem");

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "stem_fixture"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/stem_fixture"}, {"image_layout", "0"}};
    gui::MLNode loss;
    loss.id = 40;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(4001, gui::PinType::Tensor, "Predictions", true), Pin(4002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(4003, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 41;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(4101, gui::PinType::Loss, "Loss", true)};
    gui::MLNode output;
    output.id = 42;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};
    const std::vector<gui::MLNode> nodes = {
        data, Layer(2, gui::NodeType::Resize, "Resize 16", {{"width", "16"}, {"height", "16"}, {"mode", "exact"}}),
        Layer(3, gui::NodeType::Conv2D, "Stem conv",
              {{"filters", "8"}, {"kernel_size", "7"}, {"stride", "2"}, {"padding", "same"}, {"use_bias", "false"}}),
        Layer(4, gui::NodeType::BatchNorm, "Stem BN", {{"eps", "1e-5"}, {"momentum", "0.1"}}),
        Layer(5, gui::NodeType::ReLU6, "Stem ReLU6"),
        Layer(6, gui::NodeType::MaxPool2D, "Stem pool", {{"pool_size", "3"}, {"stride", "2"}, {"padding", "1"}}),
        Layer(32, gui::NodeType::GlobalAvgPool, "GAP"),
        Layer(33, gui::NodeType::Dense, "Head", {{"units", "2"}}), loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 4), Link(4, 4, 5),
                                        Link(5, 5, 6), Link(6, 6, 32), Link(7, 32, 33)};
    gui::NodeLink to_loss = Link(17, 33, 40);
    to_loss.to_pin = 4001;
    gui::NodeLink labels;
    labels.id = 18; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 40; labels.to_pin = 4002;
    gui::NodeLink to_adam;
    to_adam.id = 19; to_adam.from_node = 40; to_adam.from_pin = 4003; to_adam.to_node = 41; to_adam.to_pin = 4101;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);
    CheckGraphAgainstPyTorch(compiler, nodes, links, c);

    // The convolution holds its kernels only, and the pool's sample is PyTorch's.
    const auto config = compiler.Compile(nodes, links, true);
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), "resnet_stem builds again");
    const std::string conv_prefix = "layer" + std::to_string(LayerIndex(config, "Stem conv")) + ".";
    size_t conv_tensors = 0;
    for (const auto& [name, tensor] : built.model->GetParameters()) {
        if (name.rfind(conv_prefix, 0) == 0) ++conv_tensors;
    }
    Check(conv_tensors == 1, "a convolution with Bias off holds its kernels only");
    for (const auto& layer : config.layers) {
        if (layer.name == "Stem pool")
            Check(layer.output_shape == std::vector<size_t>({4, 4, 8}), "the padded pool gives a 4 x 4 x 8 sample");
    }

    // A pool padded by more than half its window is refused, with the reason.
    auto wide = nodes;
    for (auto& node : wide) {
        if (node.name == "Stem pool") node.parameters["padding"] = "2";
    }
    const auto refused = compiler.Compile(wide, links, true);
    bool said_why = false;
    for (const auto& issue : refused.issues) {
        said_why = said_why || issue.message.find("more than half the pool size") != std::string::npos;
    }
    Check(!refused.is_valid && said_why, "padding 2 on a pool of size 3 is refused with the reason");

    // The same weights arrive from a PyTorch state-dict named after the nodes:
    // layouts converted, running statistics included, and the model then gives
    // PyTorch's logits.
    const std::filesystem::path weights = std::filesystem::path(CYXWIZ_WEIGHT_FILE_FIXTURES) / "resnet_stem.pth";
    {
        auto fresh = cyxwiz::BuildExecutableFromConfig(config);
        Check(fresh.ok(), "resnet_stem builds for the weights file");
        cyxwiz::weight_match::Report report;
        std::string error;
        const bool started = cyxwiz::StartFromWeightsFile(weights, *fresh.model, fresh.module_provenance, report, error);
        Check(started, "the graph starts from the weights file: " + error);
        Check(report.loaded == 3 && report.fresh == 0, "all three layers take their weights: " + report.Summary());
        Check(report.unused.size() == 2, "two entries are left over");
        bool counter = false, other = false;
        for (const auto& unused : report.unused) {
            counter = counter || (unused.name.rfind("Stem BN.num_batches_tracked", 0) == 0 &&
                                  unused.reason.find("counter") != std::string::npos);
            other = other || (unused.name.rfind("aux.weight", 0) == 0 && unused.reason == "no node is named aux");
        }
        Check(counter && other, "the left-over entries say why: BatchNorm's counter, and a layer the graph lacks");
        const auto loaded = fresh.model->GetParameters();
        for (const auto& [layer_name, layer] : c.at("layers").items()) {
            const std::string prefix = "layer" + std::to_string(LayerIndex(config, layer_name)) + ".";
            for (const auto& [key, value] : layer.items()) {
                if (key.rfind("grad_", 0) == 0) continue;
                const auto it = loaded.find(prefix + key);
                Check(it != loaded.end(), "weights file: " + prefix + key + " is set");
                if (it != loaded.end()) CheckClose(it->second, value, c.at("tolerance"), "weights file " + layer_name + " " + key);
            }
        }
        for (const auto& [key, value] : c.at("state").at("Stem BN").items()) {
            const auto it = loaded.find("layer" + std::to_string(LayerIndex(config, "Stem BN")) + "." + key);
            Check(it != loaded.end(), "weights file: Stem BN " + key + " is set");
            if (it != loaded.end()) CheckClose(it->second, value, c.at("tolerance"), "weights file Stem BN " + key);
        }
        fresh.model->SetTraining(true);
        CheckClose(fresh.model->Forward(ReadTensor(c.at("input"))), c.at("output"), c.at("tolerance"),
                   "logits after starting from the weights file");
    }
    // A head with another class count starts fresh and keeps its own weights;
    // the rest still loads.
    {
        auto three = nodes;
        for (auto& node : three) {
            if (node.name == "Head") node.parameters["units"] = "3";
            if (node.type == gui::NodeType::Output) node.parameters["num_classes"] = "3";
        }
        const auto config3 = compiler.Compile(three, links, true);
        auto other = cyxwiz::BuildExecutableFromConfig(config3);
        Check(config3.is_valid && other.ok(), "the three-class graph builds");
        const std::string head = "layer" + std::to_string(LayerIndex(config3, "Head")) + ".weight";
        const cyxwiz::Tensor before = other.model->GetParameters().at(head).Clone();
        cyxwiz::weight_match::Report report;
        std::string error;
        Check(cyxwiz::StartFromWeightsFile(weights, *other.model, other.module_provenance, report, error),
              "a graph with another head still starts from the file: " + error);
        Check(report.loaded == 2 && report.fresh == 1, "two layers load, the head starts fresh: " + report.Summary());
        bool head_row = false;
        for (const auto& row : report.layers) {
            head_row = head_row || (row.node_name == "Head" && !row.loaded &&
                                    row.reason == "shape differs: the file has [2], the graph [3]");  // its bias is looked at first
        }
        Check(head_row, "the report says the head's shape differs");
        Check(report.Summary().find("2 of 3 layers take weights from the file; 1 starts fresh: Head") == 0,
              "the summary names the fresh layer: " + report.Summary());
        const cyxwiz::Tensor after = other.model->GetParameters().at(head);
        bool unchanged = before.Shape() == after.Shape();
        for (size_t i = 0; unchanged && i < before.NumElements(); ++i)
            unchanged = before.ReadData<float>()[i] == after.ReadData<float>()[i];
        Check(unchanged, "the fresh head keeps the weights it was built with");
    }
    // No layer matches (other node names): refused, the model untouched.
    {
        auto renamed = nodes;
        for (auto& node : renamed) {
            if (node.name == "Stem conv") node.name = "First conv";
            if (node.name == "Stem BN") node.name = "First BN";
            if (node.name == "Head") node.name = "Classifier";
        }
        const auto config_renamed = compiler.Compile(renamed, links, true);
        auto other = cyxwiz::BuildExecutableFromConfig(config_renamed);
        Check(config_renamed.is_valid && other.ok(), "the renamed graph builds");
        cyxwiz::weight_match::Report report;
        std::string error;
        Check(!cyxwiz::StartFromWeightsFile(weights, *other.model, other.module_provenance, report, error) &&
                  error.find("no layer of this graph matches the file") == 0,
              "a file that matches no layer is refused: " + error);
        Check(report.loaded == 0 && report.fresh == 3, "the report still lists the three layers");
    }
    std::cout << "ResNet stem (7x7 convolution without bias, ReLU6, padded max pool) matches PyTorch\n";
}

// A Dense that reads a flattened feature map (how VGG and LeNet end): the
// graph matches PyTorch with the Engine's own column order, and a PyTorch
// state-dict, whose columns are channel first, is reordered on loading.
void CheckFlattenHeadAgainstPyTorch(cyxwiz::GraphCompiler& compiler) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(4);
    Check(c.at("name").get<std::string>() == "flatten_head", "fixture case 4 is flatten_head");

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "flatten_fixture"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/flatten_fixture"}, {"image_layout", "0"}};
    gui::MLNode loss;
    loss.id = 40;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(4001, gui::PinType::Tensor, "Predictions", true), Pin(4002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(4003, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 41;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(4101, gui::PinType::Loss, "Loss", true)};
    gui::MLNode output;
    output.id = 42;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};
    const std::vector<gui::MLNode> nodes = {
        data, Layer(2, gui::NodeType::Resize, "Resize 6", {{"width", "6"}, {"height", "6"}, {"mode", "exact"}}),
        Layer(3, gui::NodeType::Conv2D, "Stem conv",
              {{"filters", "4"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}}),
        Layer(5, gui::NodeType::ReLU, "Stem ReLU"),
        Layer(6, gui::NodeType::Flatten, "Flatten"),
        Layer(33, gui::NodeType::Dense, "Head", {{"units", "2"}}), loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 5), Link(4, 5, 6), Link(5, 6, 33)};
    gui::NodeLink to_loss = Link(17, 33, 40);
    to_loss.to_pin = 4001;
    gui::NodeLink labels;
    labels.id = 18; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 40; labels.to_pin = 4002;
    gui::NodeLink to_adam;
    to_adam.id = 19; to_adam.from_node = 40; to_adam.from_pin = 4003; to_adam.to_node = 41; to_adam.to_pin = 4101;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);
    CheckGraphAgainstPyTorch(compiler, nodes, links, c);

    const auto config = compiler.Compile(nodes, links, true);
    auto fresh = cyxwiz::BuildExecutableFromConfig(config);
    Check(config.is_valid && fresh.ok(), "flatten_head builds for the weights file");
    cyxwiz::weight_match::Report report;
    std::string error;
    const auto weights = std::filesystem::path(CYXWIZ_WEIGHT_FILE_FIXTURES) / "flatten_head.pth";
    Check(cyxwiz::StartFromWeightsFile(weights, *fresh.model, fresh.module_provenance, report, error),
          "the Flatten graph starts from the weights file: " + error);
    Check(report.loaded == 2 && report.fresh == 0, "both layers take their weights: " + report.Summary());
    const auto loaded = fresh.model->GetParameters();
    const auto head = loaded.find("layer" + std::to_string(LayerIndex(config, "Head")) + ".weight");
    Check(head != loaded.end(), "the head's weight is set");
    if (head != loaded.end())
        CheckClose(head->second, c.at("layers").at("Head").at("weight"), c.at("tolerance"),
                   "the head's columns are in the Engine's order");
    fresh.model->SetTraining(true);
    CheckClose(fresh.model->Forward(ReadTensor(c.at("input"))), c.at("output"), c.at("tolerance"),
               "logits of the Flatten graph after starting from the weights file");
    std::cout << "Dense after Flatten matches PyTorch, from the Engine's layout and from a PyTorch state-dict\n";
}

// Frozen layers against PyTorch (requires_grad off): the stem's convolution
// and BatchNorm are frozen by their node setting and the Head trains. With
// "keep" the BatchNorm runs on its running statistics; with "update" it
// normalises with the batch and its statistics move.
void CheckFrozenStemAgainstPyTorch(cyxwiz::GraphCompiler& compiler, size_t case_index, const std::string& case_name,
                                   const std::string& statistics) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(case_index);
    Check(c.at("name").get<std::string>() == case_name, "fixture case " + std::to_string(case_index) + " is " + case_name);

    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Images";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "frozen_fixture"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/frozen_fixture"}, {"image_layout", "0"}};
    gui::MLNode loss;
    loss.id = 40;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(4001, gui::PinType::Tensor, "Predictions", true), Pin(4002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(4003, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 41;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(4101, gui::PinType::Loss, "Loss", true)};
    gui::MLNode output;
    output.id = 42;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};
    const std::vector<gui::MLNode> nodes = {
        data, Layer(2, gui::NodeType::Resize, "Resize 16", {{"width", "16"}, {"height", "16"}, {"mode", "exact"}}),
        Layer(3, gui::NodeType::Conv2D, "Stem conv",
              {{"filters", "8"}, {"kernel_size", "7"}, {"stride", "2"}, {"padding", "same"}, {"use_bias", "false"},
               {"frozen", "true"}}),
        Layer(4, gui::NodeType::BatchNorm, "Stem BN",
              {{"eps", "1e-5"}, {"momentum", "0.1"}, {"frozen", "true"}, {"frozen_statistics", statistics}}),
        Layer(5, gui::NodeType::ReLU6, "Stem ReLU6"),
        Layer(6, gui::NodeType::MaxPool2D, "Stem pool", {{"pool_size", "3"}, {"stride", "2"}, {"padding", "1"}}),
        Layer(32, gui::NodeType::GlobalAvgPool, "GAP"),
        Layer(33, gui::NodeType::Dense, "Head", {{"units", "2"}}), loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 4), Link(4, 4, 5),
                                        Link(5, 5, 6), Link(6, 6, 32), Link(7, 32, 33)};
    gui::NodeLink to_loss = Link(17, 33, 40);
    to_loss.to_pin = 4001;
    gui::NodeLink labels;
    labels.id = 18; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 40; labels.to_pin = 4002;
    gui::NodeLink to_adam;
    to_adam.id = 19; to_adam.from_node = 40; to_adam.from_pin = 4003; to_adam.to_node = 41; to_adam.to_pin = 4101;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);
    // Logits, the input gradient, the Head's gradients and the running
    // statistics after the step (unchanged with "keep", moved with "update").
    CheckGraphAgainstPyTorch(compiler, nodes, links, c);

    // The frozen layers report no gradient and one optimizer step leaves them
    // bit for bit as they were, while the Head moves by lr x its gradient.
    const auto config = compiler.Compile(nodes, links, true);
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(config.is_valid && built.ok(), case_name + " builds again");
    const std::string conv = "layer" + std::to_string(LayerIndex(config, "Stem conv")) + ".";
    const std::string norm = "layer" + std::to_string(LayerIndex(config, "Stem BN")) + ".";
    const std::string head = "layer" + std::to_string(LayerIndex(config, "Head")) + ".";
    const std::set<std::string> frozen_names = {conv + "weights", norm + "gamma", norm + "beta",
                                                norm + "running_mean", norm + "running_var"};
    Check(built.model->GetFrozenParameterNames() == frozen_names, case_name + ": the frozen parameters are the stem's");
    built.model->SetTraining(true);
    built.model->Forward(ReadTensor(c.at("input")));
    built.model->Backward(ReadTensor(c.at("grad_output")));
    for (const auto& [name, gradient] : built.model->GetGradients())
        Check(name.rfind(head, 0) == 0, case_name + ": a gradient only for the Head, got " + name);
    const auto values = [](const cyxwiz::Tensor& tensor) {
        const float* data = tensor.ReadData<float>();
        return std::vector<float>(data, data + tensor.NumElements());
    };
    std::map<std::string, std::vector<float>> before;
    for (const auto& [name, tensor] : built.model->GetParameters()) before[name] = values(tensor);
    const auto head_gradient = values(built.model->GetGradients().at(head + "weight"));
    cyxwiz::SGDOptimizer optimizer(0.5);
    built.model->UpdateParameters(&optimizer);
    bool frozen_same = true, head_moved = true;
    for (const auto& [name, tensor] : built.model->GetParameters()) {
        const auto now = values(tensor);
        // "update" lets the running statistics move in the forward pass; that happened before `before` was taken.
        if (frozen_names.count(name)) frozen_same = frozen_same && now == before.at(name);
        if (name == head + "weight") {
            for (size_t i = 0; i < now.size(); ++i)
                head_moved = head_moved && std::abs(now[i] - (before.at(name)[i] - 0.5f * head_gradient[i])) < 1e-6f;
        }
    }
    Check(frozen_same, case_name + ": an optimizer step leaves the frozen layers bit for bit");
    Check(head_moved, case_name + ": the Head moves by the learning rate times its gradient");
    std::cout << "Frozen stem (" << case_name << ") matches PyTorch with requires_grad off\n";
}

// A whole published network as scripts/examples/torch_to_cyxgraph.py writes it
// (nodes named as PyTorch names its modules), filled from a PyTorch
// state-dict and compared with PyTorch's logits. Returns false when the graph
// or the weights could not be used.
bool CheckNetworkGraph(cyxwiz::GraphCompiler& compiler, const std::filesystem::path& graph_path,
                        const std::filesystem::path& weights, const nlohmann::json& expected, const std::string& label,
                        bool training_mode_too) {
    std::ifstream in(graph_path, std::ios::binary);
    Check(in.good(), label + ": cannot open " + graph_path.string());
    std::stringstream text;
    text << in.rdbuf();
    cyxwiz::GraphDocument document;
    std::string error;
    const bool parsed = cyxwiz::ParseGraphDocument(text.str(), document, error);
    Check(parsed, label + ": the graph file loads: " + error);
    if (!parsed) return false;
    const auto config = compiler.Compile(document.nodes, document.links, true);
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.node_name << ": " << issue.message << "\n";
    }
    Check(config.is_valid, label + ": the graph compiles");
    if (!config.is_valid) return false;

    const auto run = [&](bool training, const char* output_key) {
        auto built = cyxwiz::BuildExecutableFromConfig(config);
        Check(built.ok(), label + ": the model builds: " + built.error_message);
        if (!built.ok()) return false;
        cyxwiz::weight_match::Report report;
        std::string load_error;
        const bool started = cyxwiz::StartFromWeightsFile(weights, *built.model, built.module_provenance, report, load_error);
        Check(started, label + ": starts from " + weights.filename().string() + ": " + load_error);
        if (!started) return false;
        // Every layer with weights is filled by name.
        const size_t layers = expected.at("layers_with_weights").get<size_t>();
        Check(report.loaded == layers && report.fresh == 0,
              label + ": all " + std::to_string(layers) + " layers take their weights: " + report.Summary());
        for (const auto& unused : report.unused)
            Check(unused.reason.find("counter") != std::string::npos, label + ": left over: " + unused.name + " (" + unused.reason + ")");
        built.model->SetTraining(training);
        CheckClose(built.model->Forward(ReadTensor(expected.at("input"))), expected.at(output_key), expected.at("tolerance"),
                   label + (training ? " logits in training mode" : " logits in evaluation"));
        return true;
    };
    if (!run(false, "output_eval")) return false;
    return !training_mode_too || run(true, "output_train");
}

// ResNet-18, MobileNetV2 and MobileNetV3 small against PyTorch.
void CheckPublishedNetworksAgainstPyTorch(cyxwiz::GraphCompiler& compiler) {
    const std::filesystem::path fixtures = std::filesystem::path(CYXWIZ_WEIGHT_FILE_FIXTURES).parent_path();
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream in(path);
        Check(in.good(), "cannot open " + path.string());
        return nlohmann::json::parse(in);
    };
    const char* const networks[] = {"resnet18", "mobilenet_v2", "mobilenet_v3_small"};
    // Always: small versions of each, with weights PyTorch wrote.
    for (const char* network : networks) {
        const std::string name = std::string(network) + "_thin";
        if (CheckNetworkGraph(compiler, fixtures / (name + ".cyxgraph"), fixtures / "weight_files" / (name + ".pth"),
                              read(fixtures / (name + "_pytorch.json")), name, true))
            std::cout << name << " graph matches PyTorch in evaluation and in training mode\n";
    }
    // With the published files at hand: the full networks against torchvision.
    const char* folder = std::getenv("CYXWIZ_PRETRAINED_WEIGHTS_DIR");
    if (!folder || !*folder) {
        std::cout << "Networks with the published ImageNet weights: skipped (set CYXWIZ_PRETRAINED_WEIGHTS_DIR to a "
                     "folder with resnet18-f37072fd.pth, mobilenet_v2-b0353104.pth and "
                     "mobilenet_v3_small-047dcff4.pth to run them)\n";
        return;
    }
    for (const char* network : networks) {
        const std::string name = std::string(network) + "_imagenet";
        const auto expected = read(fixtures / (name + "_pytorch.json"));
        const auto weights = std::filesystem::path(folder) / expected.at("weights_file").get<std::string>();
        if (!std::filesystem::exists(weights)) {
            std::cout << name << ": skipped, " << weights.string() << " not found\n";
            continue;
        }
        if (CheckNetworkGraph(compiler, fixtures / (name + ".cyxgraph"), weights, expected, name, false))
            std::cout << name << " graph with the published weights matches torchvision's logits\n";
    }
}

}  // namespace

int main() {
    // CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu picks the device (the oracle's convention).
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    // The graph (ids 1..13); the Data Input is an image folder with a Resize.
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Cats and dogs";
    data.outputs = {Pin(102, gui::PinType::Tensor, "Data", false), Pin(103, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "catdog_small"}, {"file_category", "image"},
                       {"folder_path", "D:/tmp/gui129/catdog_small"}, {"image_layout", "0"}};
    auto resize = Layer(2, gui::NodeType::Resize, "Resize 64", {{"width", "64"}, {"height", "64"}, {"mode", "exact"}});
    auto conv1 = Layer(3, gui::NodeType::Conv2D, "Conv 16", {{"filters", "16"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}});
    auto relu1 = Layer(4, gui::NodeType::ReLU, "ReLU");
    auto pool1 = Layer(5, gui::NodeType::MaxPool2D, "Pool", {{"pool_size", "2"}, {"stride", "2"}});
    auto conv2 = Layer(6, gui::NodeType::Conv2D, "Conv 32", {{"filters", "32"}, {"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}});
    auto relu2 = Layer(7, gui::NodeType::ReLU, "ReLU 2");
    auto pool2 = Layer(8, gui::NodeType::MaxPool2D, "Pool 2", {{"pool_size", "2"}, {"stride", "2"}});
    auto flatten = Layer(9, gui::NodeType::Flatten, "Flatten");
    auto dense = Layer(10, gui::NodeType::Dense, "Dense 2", {{"units", "2"}});
    gui::MLNode loss;
    loss.id = 11;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Cross entropy";
    loss.inputs = {Pin(1101, gui::PinType::Tensor, "Predictions", true), Pin(1102, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(1103, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 12;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(1201, gui::PinType::Loss, "Loss", true)};
    adam.parameters = {{"learning_rate", "0.001"}};
    gui::MLNode output;
    output.id = 13;
    output.type = gui::NodeType::Output;
    output.name = "Output";
    output.parameters = {{"num_classes", "2"}};

    std::vector<gui::MLNode> nodes = {data, resize, conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense, loss, adam, output};
    std::vector<gui::NodeLink> links = {Link(1, 1, 2), Link(2, 2, 3), Link(3, 3, 4), Link(4, 4, 5), Link(5, 5, 6),
                                        Link(6, 6, 7), Link(7, 7, 8), Link(8, 8, 9), Link(9, 9, 10)};
    gui::NodeLink to_loss;
    to_loss.id = 10; to_loss.from_node = 10; to_loss.from_pin = 1002; to_loss.to_node = 11; to_loss.to_pin = 1101;
    gui::NodeLink labels;
    labels.id = 11; labels.from_node = 1; labels.from_pin = 103; labels.to_node = 11; labels.to_pin = 1102;
    gui::NodeLink to_adam;
    to_adam.id = 12; to_adam.from_node = 11; to_adam.from_pin = 1103; to_adam.to_node = 12; to_adam.to_pin = 1201;
    links.push_back(to_loss);
    links.push_back(labels);
    links.push_back(to_adam);

    // The Engine's image loader tells the compiler the domain of an image
    // Data Input; here the hook does, labels come from the class folders.
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.preprocessing_domain = [](const std::string& category) -> std::optional<cyxwiz::PreprocessingDomain> {
        if (category == "image") return cyxwiz::PreprocessingDomain::Image;
        return std::nullopt;
    };
    hooks.labels_from_structure = [](const std::string& category) -> std::optional<bool> {
        if (category == "image") return true;
        return std::nullopt;
    };
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    // ---- Compile: the shapes are PyTorch's ----------------------------------
    cyxwiz::GraphCompiler compiler;
    auto config = compiler.Compile(nodes, links, true);
    for (const auto& issue : config.issues) {
        std::cerr << "  issue(" << (issue.level == cyxwiz::IssueLevel::Error ? "error" : "note") << "): "
                  << issue.node_name << ": " << issue.message << "\n";
    }
    Check(config.input_shape == std::vector<size_t>{64, 64, 3}, "image input shape [64, 64, 3], got " + ShapeText(config.input_shape));
    Check(config.input_size == 64 * 64 * 3, "input size 12288");
    const auto& c1 = LayerNamed(config, "Conv 16");
    Check(c1.input_shape == std::vector<size_t>{64, 64, 3} && c1.output_shape == std::vector<size_t>{64, 64, 16},
          "Conv 16 keeps 64x64 with padding same: " + ShapeText(c1.output_shape));
    Check(c1.padding == 1 && c1.kernel_size == 3 && c1.filters == 16, "Conv 16 geometry resolved ('same' = 1)");
    Check(LayerNamed(config, "Pool").output_shape == std::vector<size_t>{32, 32, 16}, "Pool halves to 32x32x16");
    Check(LayerNamed(config, "Conv 32").output_shape == std::vector<size_t>{32, 32, 32}, "Conv 32 -> 32x32x32");
    Check(LayerNamed(config, "Pool 2").output_shape == std::vector<size_t>{16, 16, 32}, "Pool 2 -> 16x16x32");
    Check(LayerNamed(config, "Flatten").output_shape == std::vector<size_t>{16 * 16 * 32}, "Flatten -> 8192");
    Check(LayerNamed(config, "Dense 2").output_shape == std::vector<size_t>{2}, "Dense -> 2");
    bool unsupported = false;
    for (const auto& issue : config.issues) {
        if (issue.message.find("not supported") != std::string::npos ||
            issue.message.find("blocked") != std::string::npos) unsupported = true;
    }
    Check(!unsupported, "no layer of the CNN stack is reported unsupported");

    // ---- The spatial head sees the same shapes --------------------------------
    const auto head = cyxwiz::ResolveSpatialSequentialHead(config);
    Check(head.has_value(), "the model opens with a spatial layer");
    Check(head->sample_shape == std::vector<size_t>{16, 16, 32} && head->features == 8192,
          "spatial head ends at Flatten with [16, 16, 32] = 8192 features");
    Check(head->input_shapes.size() == 6 && head->input_shapes[5] == std::vector<size_t>{32, 32, 32},
          "the head tracks every layer's input sample");

    // ---- Build and run a forward pass --------------------------------------------
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), "ModelBuilder builds the CNN: " + built.error_message);
    long long parameters = 0;
    for (const auto& [name, tensor] : built.model->GetParameters()) parameters += static_cast<long long>(tensor.NumElements());
    // Conv 16: 3*3*3*16 + 16 = 448; Conv 32: 3*3*16*32 + 32 = 4640; Dense: 8192*2 + 2 = 16386
    Check(parameters == 448 + 4640 + 16386, "learnable parameters 21,474, got " + std::to_string(parameters));

    const size_t batch = 4;
    std::vector<float> rows(batch * config.input_size);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (float& v : rows) v = uniform(rng);
    const cyxwiz::Tensor input({batch, config.input_size}, rows.data(), cyxwiz::DataType::Float32);
    const cyxwiz::Tensor spatial = cyxwiz::SpatialBatchFromRows(input, config.input_shape);
    Check(spatial.Shape() == std::vector<size_t>{64, 64, 3, batch}, "rows unpack to [64, 64, 3, N]");
    const cyxwiz::Tensor logits = built.model->Forward(spatial);
    Check(logits.Shape() == std::vector<size_t>{batch, 2}, "forward gives [N, 2] logits, got " + ShapeText(logits.Shape()));
    const float* values = logits.ReadData<float>();
    for (size_t i = 0; i < batch * 2; ++i) Check(std::isfinite(values[i]), "logits are finite");

    // ---- PReLU (shared slope) and SELU inside the conv section (TOFIX140 A2) ----
    {
        auto variant = nodes;
        for (auto& n : variant) {
            if (n.name == "ReLU") {
                n.type = gui::NodeType::PReLU;
                n.parameters = {{"num_parameters", "1"}, {"init", "0.25"}};
            }
            if (n.name == "ReLU 2") n.type = gui::NodeType::SELU;
        }
        const auto vconfig = compiler.Compile(variant, links, true);
        Check(vconfig.is_valid, "a CNN with PReLU and SELU compiles");
        auto vbuilt = cyxwiz::BuildExecutableFromConfig(vconfig);
        Check(vbuilt.ok(), "ModelBuilder builds PReLU and SELU: " + vbuilt.error_message);
        long long vparams = 0;
        for (const auto& [name, tensor] : vbuilt.model->GetParameters()) vparams += static_cast<long long>(tensor.NumElements());
        Check(vparams == parameters + 1, "PReLU adds one learnable slope: " + std::to_string(vparams));
        const cyxwiz::Tensor vlogits = vbuilt.model->Forward(spatial);
        Check(vlogits.Shape() == std::vector<size_t>{batch, 2}, "PReLU/SELU CNN forward gives [N, 2]");

        // Per-channel slopes act on dimension 1 (W in [H,W,C,N]): refused there.
        for (auto& n : variant) {
            if (n.type == gui::NodeType::PReLU) n.parameters["num_parameters"] = "16";
        }
        const auto bad = compiler.Compile(variant, links, true);
        bool reported = false;
        for (const auto& issue : bad.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error && issue.message.find("shared slope") != std::string::npos &&
                issue.node_name == "ReLU") reported = true;  // the node keeps its name "ReLU" in this variant
        }
        Check(!bad.is_valid && reported, "the compiler reports the per-channel PReLU on its node");
        const auto bad_built = cyxwiz::BuildExecutableFromConfig(bad);
        Check(!bad_built.ok() && bad_built.error_message.find("shared slope") != std::string::npos,
              "per-channel PReLU before Flatten is refused with the reason: " + bad_built.error_message);
    }

    // ---- Conv -> BatchNorm -> ReLU (TOFIX178 1.1): BatchNorm before Flatten is
    // BatchNorm2D, per channel (it used to stop the compile: "Spatial head requires
    // Flatten or a global pool before ...") ----
    {
        auto variant = nodes;
        variant.push_back(Layer(30, gui::NodeType::BatchNorm, "BN 16", {{"eps", "1e-5"}, {"momentum", "0.1"}}));
        variant.push_back(Layer(31, gui::NodeType::BatchNorm, "BN 32", {{"eps", "1e-5"}, {"momentum", "0.1"}}));
        auto blinks = links;
        int next_link = 900;
        for (auto& l : blinks) {
            if (l.from_node == 3 && l.to_node == 4) l = Link(l.id, 3, 30);
            if (l.from_node == 6 && l.to_node == 7) l = Link(l.id, 6, 31);
        }
        blinks.push_back(Link(next_link++, 30, 4));
        blinks.push_back(Link(next_link++, 31, 7));
        const auto bconfig = compiler.Compile(variant, blinks, true);
        for (const auto& issue : bconfig.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
        }
        Check(bconfig.is_valid, "a CNN with Conv -> BatchNorm -> ReLU compiles");
        Check(LayerNamed(bconfig, "BN 16").output_shape == std::vector<size_t>{64, 64, 16},
              "BatchNorm keeps [64, 64, 16]");
        auto bbuilt = cyxwiz::BuildExecutableFromConfig(bconfig);
        Check(bbuilt.ok(), "ModelBuilder builds BatchNorm2D: " + bbuilt.error_message);
        long long bparams = 0;
        for (const auto& [name, tensor] : bbuilt.model->GetParameters()) bparams += static_cast<long long>(tensor.NumElements());
        // Per channel: gamma, beta, running mean, running var = 4 x 16 and 4 x 32 (torch BatchNorm2d state).
        Check(bparams == parameters + 4 * 16 + 4 * 32,
              "BatchNorm2D holds 4 values per channel, got " + std::to_string(bparams - parameters));
        const cyxwiz::Tensor blogits = bbuilt.model->Forward(spatial);
        Check(blogits.Shape() == std::vector<size_t>{batch, 2}, "Conv-BN CNN forward gives [N, 2]");
        std::vector<float> ones(batch * 2, 1.0f);
        const cyxwiz::Tensor bgrad = bbuilt.model->Backward(cyxwiz::Tensor({batch, 2}, ones.data(), cyxwiz::DataType::Float32));
        Check(bgrad.Shape() == std::vector<size_t>{64, 64, 3, batch}, "backward through BatchNorm2D reaches the input");
        // Eval uses the running statistics: the same batch gives different logits than in training.
        bbuilt.model->SetTraining(false);
        const cyxwiz::Tensor eval_logits = bbuilt.model->Forward(spatial);
        const float* train_values = blogits.ReadData<float>();
        const float* eval_values = eval_logits.ReadData<float>();
        bool differs = false;
        for (size_t i = 0; i < batch * 2; ++i) {
            Check(std::isfinite(eval_values[i]), "eval logits are finite");
            differs = differs || std::fabs(train_values[i] - eval_values[i]) > 1e-6f;
        }
        Check(differs, "eval mode uses the running statistics, not the batch's");
    }

    // ---- Residual blocks inside the conv section (TOFIX178 1.2) ----
    // A two-input Add (Input 1 = id*100+1, Input 2 = id*100+3).
    const auto add_node = [](int id, const std::string& name) {
        gui::MLNode node = Layer(id, gui::NodeType::Add, name);
        node.inputs.push_back(Pin(id * 100 + 3, gui::PinType::Tensor, "Input 2", true));
        return node;
    };
    const auto to_second = [](int id, int from_node, int to_node) {
        gui::NodeLink link = Link(id, from_node, to_node);
        link.to_pin = to_node * 100 + 3;
        return link;
    };
    const auto conv = [](int id, const std::string& name, int filters, int kernel, int stride, const std::string& pad) {
        return Layer(id, gui::NodeType::Conv2D, name,
                     {{"filters", std::to_string(filters)}, {"kernel_size", std::to_string(kernel)},
                      {"stride", std::to_string(stride)}, {"padding", pad}});
    };
    const auto bn = [](int id, const std::string& name) {
        return Layer(id, gui::NodeType::BatchNorm, name, {{"eps", "1e-5"}, {"momentum", "0.1"}});
    };
    const auto build_and_run = [&](const std::vector<gui::MLNode>& rnodes, const std::vector<gui::NodeLink>& rlinks,
                                   const std::string& what) {
        const auto rconfig = compiler.Compile(rnodes, rlinks, true);
        for (const auto& issue : rconfig.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error)
                std::cerr << "  issue: " << issue.node_name << ": " << issue.message << "\n";
        }
        Check(rconfig.is_valid, what + " compiles");
        auto rbuilt = cyxwiz::BuildExecutableFromConfig(rconfig);
        Check(rbuilt.ok(), what + " builds: " + rbuilt.error_message);
        const cyxwiz::Tensor rlogits = rbuilt.model->Forward(spatial);
        Check(rlogits.Shape() == std::vector<size_t>{batch, 2}, what + " forward gives [N, 2], got " + ShapeText(rlogits.Shape()));
        std::vector<float> rones(batch * 2, 1.0f);
        const cyxwiz::Tensor rgrad = rbuilt.model->Backward(cyxwiz::Tensor({batch, 2}, rones.data(), cyxwiz::DataType::Float32));
        Check(rgrad.Shape() == std::vector<size_t>{64, 64, 3, batch}, what + " backward reaches the input as [64, 64, 3, N]");
        return rconfig;
    };
    {
        // Identity block after the first ReLU: ReLU -> Conv 16 -> BN -> Add(ReLU, BN) -> ReLU -> Pool.
        auto rnodes = nodes;
        rnodes.push_back(conv(40, "Res conv", 16, 3, 1, "same"));
        rnodes.push_back(bn(41, "Res BN"));
        rnodes.push_back(add_node(42, "Res add"));
        rnodes.push_back(Layer(43, gui::NodeType::ReLU, "Res ReLU"));
        std::vector<gui::NodeLink> rlinks;
        for (const auto& l : links)
            if (!(l.from_node == 4 && l.to_node == 5)) rlinks.push_back(l);
        rlinks.push_back(Link(901, 4, 40));
        rlinks.push_back(Link(902, 40, 41));
        rlinks.push_back(Link(903, 4, 42));          // the shortcut: the block's input
        rlinks.push_back(to_second(904, 41, 42));
        rlinks.push_back(Link(905, 42, 43));
        rlinks.push_back(Link(906, 43, 5));
        const auto rconfig = build_and_run(rnodes, rlinks, "an identity residual block");
        Check(LayerNamed(rconfig, "Res conv").input_shape == std::vector<size_t>{64, 64, 16}, "the block conv takes [64, 64, 16]");
    }
    {
        // Downsampling block (ResNet): main Conv 32 stride 2 -> BN -> ReLU -> Conv 32 -> BN, shortcut
        // Conv 1x1 32 stride 2 -> BN, Add -> ReLU, then ReLU 2 -> Pool 2 -> Flatten. The shortcut conv
        // takes the block's input [64, 64, 16], not the main path's [32, 32, 32].
        std::vector<gui::MLNode> rnodes;
        for (const auto& n : nodes)
            if (n.id != 5 && n.id != 6) rnodes.push_back(n);
        rnodes.push_back(conv(50, "Down conv", 32, 3, 2, "1"));
        rnodes.push_back(bn(51, "Down BN"));
        rnodes.push_back(Layer(52, gui::NodeType::ReLU, "Down ReLU"));
        rnodes.push_back(conv(53, "Down conv 2", 32, 3, 1, "same"));
        rnodes.push_back(bn(54, "Down BN 2"));
        rnodes.push_back(conv(55, "Shortcut conv", 32, 1, 2, "0"));
        rnodes.push_back(bn(56, "Shortcut BN"));
        rnodes.push_back(add_node(57, "Down add"));
        rnodes.push_back(Layer(58, gui::NodeType::ReLU, "Down out"));
        std::vector<gui::NodeLink> rlinks;
        for (const auto& l : links)
            if (l.from_node != 4 && l.from_node != 5 && l.from_node != 6) rlinks.push_back(l);
        rlinks.push_back(Link(911, 4, 50));
        rlinks.push_back(Link(912, 50, 51));
        rlinks.push_back(Link(913, 51, 52));
        rlinks.push_back(Link(914, 52, 53));
        rlinks.push_back(Link(915, 53, 54));
        rlinks.push_back(Link(916, 4, 55));
        rlinks.push_back(Link(917, 55, 56));
        rlinks.push_back(Link(918, 54, 57));
        rlinks.push_back(to_second(919, 56, 57));
        rlinks.push_back(Link(920, 57, 58));
        rlinks.push_back(Link(921, 58, 7));
        const auto rconfig = build_and_run(rnodes, rlinks, "a downsampling residual block");
        Check(LayerNamed(rconfig, "Shortcut conv").input_shape == std::vector<size_t>{64, 64, 16},
              "the shortcut conv takes the block input [64, 64, 16], got " + ShapeText(LayerNamed(rconfig, "Shortcut conv").input_shape));
        Check(LayerNamed(rconfig, "Down out").output_shape == std::vector<size_t>{32, 32, 32}, "the block gives [32, 32, 32]");
    }

    // ---- Global Avg Pool ends the conv section instead of Flatten (TOFIX140 A4b) ----
    {
        auto variant = nodes;
        for (auto& n : variant) {
            if (n.name == "Flatten") {
                n.type = gui::NodeType::GlobalAvgPool;
                n.name = "GAP";
            }
        }
        const auto gconfig = compiler.Compile(variant, links, true);
        for (const auto& issue : gconfig.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
        }
        Check(gconfig.is_valid, "a CNN ending in Global Avg Pool compiles");
        Check(LayerNamed(gconfig, "GAP").output_shape == std::vector<size_t>{32}, "Global Avg Pool [16,16,32] -> [32]");
        const auto ghead = cyxwiz::ResolveSpatialSequentialHead(gconfig);
        Check(ghead && ghead->global_pool && ghead->features == 32, "the spatial head ends at Global Avg Pool, 32 features");
        auto gbuilt = cyxwiz::BuildExecutableFromConfig(gconfig);
        Check(gbuilt.ok(), "ModelBuilder builds Global Avg Pool: " + gbuilt.error_message);
        long long gparams = 0;
        for (const auto& [name, tensor] : gbuilt.model->GetParameters()) gparams += static_cast<long long>(tensor.NumElements());
        // Dense now takes 32 inputs: 32*2 + 2 = 66 (torch: Linear(32, 2))
        Check(gparams == 448 + 4640 + 66, "learnable parameters 5,154, got " + std::to_string(gparams));
        const cyxwiz::Tensor glogits = gbuilt.model->Forward(spatial);
        Check(glogits.Shape() == std::vector<size_t>{batch, 2}, "Global Avg Pool CNN forward gives [N, 2], got " +
                                                                    ShapeText(glogits.Shape()));
        std::vector<float> ones(batch * 2, 1.0f);
        const cyxwiz::Tensor ggrad = gbuilt.model->Backward(cyxwiz::Tensor({batch, 2}, ones.data(), cyxwiz::DataType::Float32));
        Check(ggrad.Shape() == std::vector<size_t>{64, 64, 3, batch}, "backward reaches the input as [64, 64, 3, N]");

        // After Flatten it has no [H,W,C] sample to average: refused on its node.
        auto after_flatten = nodes;
        auto gap = Layer(14, gui::NodeType::GlobalAvgPool, "GAP");
        after_flatten.push_back(gap);
        auto relinked = links;
        for (auto& link : relinked) {
            if (link.from_node == 9 && link.to_node == 10) link = Link(link.id, 9, 14);
        }
        relinked.push_back(Link(20, 14, 10));
        const auto bad = compiler.Compile(after_flatten, relinked, true);
        bool reported = false;
        for (const auto& issue : bad.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error && issue.node_name == "GAP" &&
                issue.message.find("pools each channel") != std::string::npos) reported = true;
        }
        Check(!bad.is_valid && reported, "Global Avg Pool after Flatten is refused on its node");
    }

    // ---- Adaptive Avg Pool inside the section: [16,16,32] -> [4,4,32] -> 512 ----
    {
        auto variant = nodes;
        auto adaptive = Layer(14, gui::NodeType::AdaptiveAvgPool, "Adaptive", {{"output_size", "4"}});
        variant.push_back(adaptive);
        auto relinked = links;
        for (auto& link : relinked) {
            if (link.from_node == 8 && link.to_node == 9) link = Link(link.id, 8, 14);
        }
        relinked.push_back(Link(21, 14, 9));
        const auto aconfig = compiler.Compile(variant, relinked, true);
        for (const auto& issue : aconfig.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
        }
        Check(aconfig.is_valid, "a CNN with Adaptive Avg Pool compiles");
        Check(LayerNamed(aconfig, "Adaptive").output_shape == std::vector<size_t>{4, 4, 32},
              "Adaptive Avg Pool [16,16,32] -> [4,4,32], got " + ShapeText(LayerNamed(aconfig, "Adaptive").output_shape));
        Check(LayerNamed(aconfig, "Flatten").output_shape == std::vector<size_t>{512}, "Flatten -> 512");
        auto abuilt = cyxwiz::BuildExecutableFromConfig(aconfig);
        Check(abuilt.ok(), "ModelBuilder builds Adaptive Avg Pool: " + abuilt.error_message);
        long long aparams = 0;
        for (const auto& [name, tensor] : abuilt.model->GetParameters()) aparams += static_cast<long long>(tensor.NumElements());
        // Dense now takes 512 inputs: 512*2 + 2 = 1026 (torch: Linear(512, 2))
        Check(aparams == 448 + 4640 + 1026, "learnable parameters 6,114, got " + std::to_string(aparams));
        const cyxwiz::Tensor alogits = abuilt.model->Forward(spatial);
        Check(alogits.Shape() == std::vector<size_t>{batch, 2}, "Adaptive Avg Pool CNN forward gives [N, 2]");
    }

    // ---- Depthwise Conv2D in place of Conv 32: 16 channels x 2 = 32 ------------
    {
        auto variant = nodes;
        for (auto& n : variant) {
            if (n.name == "Conv 32") {
                n.type = gui::NodeType::DepthwiseConv2D;
                n.name = "Depthwise";
                n.parameters = {{"kernel_size", "3"}, {"stride", "1"}, {"padding", "same"}, {"depth_multiplier", "2"}};
            }
        }
        const auto dconfig = compiler.Compile(variant, links, true);
        for (const auto& issue : dconfig.issues) {
            if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
        }
        Check(dconfig.is_valid, "a CNN with Depthwise Conv2D compiles");
        Check(LayerNamed(dconfig, "Depthwise").output_shape == std::vector<size_t>{32, 32, 32},
              "Depthwise x2 over [32,32,16] -> [32,32,32], got " + ShapeText(LayerNamed(dconfig, "Depthwise").output_shape));
        auto dbuilt = cyxwiz::BuildExecutableFromConfig(dconfig);
        Check(dbuilt.ok(), "ModelBuilder builds Depthwise Conv2D: " + dbuilt.error_message);
        long long dparams = 0;
        for (const auto& [name, tensor] : dbuilt.model->GetParameters()) dparams += static_cast<long long>(tensor.NumElements());
        // Depthwise: 3*3*1*32 + 32 = 320 (torch Conv2d(16, 32, 3, groups=16)) in place of Conv 32's 4,640
        Check(dparams == 448 + 320 + 16386, "learnable parameters 17,154, got " + std::to_string(dparams));
        const cyxwiz::Tensor dlogits = dbuilt.model->Forward(spatial);
        Check(dlogits.Shape() == std::vector<size_t>{batch, 2}, "Depthwise CNN forward gives [N, 2]");
    }

    // ---- Global Max Pool the same way ----------------------------------------
    {
        auto variant = nodes;
        for (auto& n : variant) {
            if (n.name == "Flatten") {
                n.type = gui::NodeType::GlobalMaxPool;
                n.name = "GMP";
            }
        }
        const auto mconfig = compiler.Compile(variant, links, true);
        Check(mconfig.is_valid, "a CNN ending in Global Max Pool compiles");
        Check(LayerNamed(mconfig, "GMP").output_shape == std::vector<size_t>{32}, "Global Max Pool [16,16,32] -> [32]");
        auto mbuilt = cyxwiz::BuildExecutableFromConfig(mconfig);
        Check(mbuilt.ok(), "ModelBuilder builds Global Max Pool: " + mbuilt.error_message);
        const cyxwiz::Tensor mlogits = mbuilt.model->Forward(spatial);
        Check(mlogits.Shape() == std::vector<size_t>{batch, 2}, "Global Max Pool CNN forward gives [N, 2]");
    }

    CheckResidualCnnAgainstPyTorch(compiler);
    CheckSqueezeExcitationAgainstPyTorch(compiler, 1, "squeeze_excitation_block", {{"reduction", "2"}});
    // The node's other choices: a typed width, Hardswish inside, a Hardsigmoid gate, no biases.
    CheckResNetStemAgainstPyTorch(compiler);
    CheckFlattenHeadAgainstPyTorch(compiler);
    CheckFrozenStemAgainstPyTorch(compiler, 5, "frozen_stem_keep", "keep");
    CheckFrozenStemAgainstPyTorch(compiler, 6, "frozen_stem_update", "update");
    CheckPublishedNetworksAgainstPyTorch(compiler);
    CheckSqueezeExcitationAgainstPyTorch(compiler, 2, "squeeze_excitation_hard",
                                         {{"width_mode", "exact"}, {"squeeze_channels", "3"}, {"inner_activation", "hardswish"},
                                          {"gate", "hardsigmoid"}, {"use_bias", "false"}});
    std::cout << "CNN graph compiles, builds and runs forward: " << parameters << " parameters\n";
    return 0;
}
