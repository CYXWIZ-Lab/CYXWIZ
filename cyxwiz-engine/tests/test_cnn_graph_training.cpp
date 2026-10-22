// A CNN graph through the Engine's own path (TOFIX140 A1): image Data Input
// -> Resize 64 -> Conv2D 16 -> ReLU -> MaxPool2D -> Conv2D 32 -> ReLU ->
// MaxPool2D -> Flatten -> Dense 2 -> CrossEntropy, Adam. The compiler must
// give every layer the PyTorch shapes, ModelBuilder must build the modules,
// and a forward pass on an HWC-flattened batch must produce [N, 2] logits.
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/model_builder.h"
#include "../src/core/spatial_batch_layout.h"
#include "../src/core/spatial_sequential_head.h"
#include "../src/gui/loaders/data_loader.h"
#include "computation_truth/test_device_selection.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <iostream>
#include <random>
#include <string>
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

const cyxwiz::CompiledLayer& LayerNamed(const cyxwiz::TrainingConfiguration& config, const std::string& name) {
    for (const auto& layer : config.layers) {
        if (layer.name == name) return layer;
    }
    Check(false, "compiled layer missing: " + name);
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
void CheckResidualCnnAgainstPyTorch(cyxwiz::GraphCompiler& compiler) {
    std::ifstream in(CYXWIZ_CNN_GRAPH_FIXTURE);
    Check(in.good(), std::string("cannot open ") + CYXWIZ_CNN_GRAPH_FIXTURE);
    const auto fixture = nlohmann::json::parse(in);
    const auto& c = fixture.at("cases").at(0);
    const std::string name = c.at("name").get<std::string>();

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

    const auto config = compiler.Compile(nodes, links, true);
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error)
            std::cerr << "  issue: " << issue.node_name << ": " << issue.message << "\n";
    }
    Check(config.is_valid, name + ": the residual CNN compiles");
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
    std::cout << "residual CNN matches PyTorch: logits, input and " << c.at("layers").size()
              << " layers' gradients, BatchNorm running statistics\n";
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
    std::cout << "CNN graph compiles, builds and runs forward: " << parameters << " parameters\n";
    return 0;
}
