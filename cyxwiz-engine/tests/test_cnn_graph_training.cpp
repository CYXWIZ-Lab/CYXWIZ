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

#include <cstdlib>
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

}  // namespace

int main() {
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

    std::cout << "CNN graph compiles, builds and runs forward: " << parameters << " parameters\n";
    return 0;
}
