#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace cyxwiz::loaders {

DataLoader* GetByCategory(FileCategory) {
    return nullptr;
}

DataLoader* GetByRegisteredDataset(const std::string&) {
    return nullptr;
}

FileCategory FileCategoryFromString(const std::string&) {
    return FileCategory::Tabular;
}

} // namespace cyxwiz::loaders

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

gui::NodePin Pin(int id,
                 gui::PinType type,
                 const std::string& name,
                 bool is_input,
                 bool required = true) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    pin.is_required = required;
    return pin;
}

gui::MLNode Node(int id,
                 gui::NodeType type,
                 const std::string& name,
                 std::vector<gui::NodePin> inputs,
                 std::vector<gui::NodePin> outputs) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = std::move(inputs);
    node.outputs = std::move(outputs);
    return node;
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

bool HasErrorText(const cyxwiz::TrainingConfiguration& config,
                  const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error &&
            issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Token features",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "causal_lm_shape_dataset";
    data.parameters["data_loaded"] = "true";
    data.parameters["file_category"] = "tabular";
    data.parameters["shape"] = "[4]";
    data.parameters["create_causal_lm_targets"] = "true";
    data.parameters["max_sequence_length"] = "7";

    auto decoder = Node(2,
                        gui::NodeType::TransformerDecoder,
                        "Decoder",
                        {Pin(201, gui::PinType::Tensor, "Input", true)},
                        {Pin(202, gui::PinType::Tensor, "Output", false)});
    decoder.parameters["d_model"] = "4";
    decoder.parameters["num_heads"] = "2";
    decoder.parameters["dim_feedforward"] = "8";
    decoder.parameters["dropout"] = "0";

    auto head = Node(3,
                     gui::NodeType::TimeDistributed,
                     "Token Head",
                     {Pin(301, gui::PinType::Tensor, "Input", true)},
                     {Pin(302, gui::PinType::Tensor, "Output", false)});
    head.parameters["units"] = "6";

    auto output = Node(4,
                       gui::NodeType::Output,
                       "Sequence Logits",
                       {Pin(401, gui::PinType::Tensor, "Input", true)},
                       {});
    output.parameters["num_classes"] = "6";

    auto loss = Node(5,
                     gui::NodeType::CrossEntropyLoss,
                     "Token Loss",
                     {Pin(501, gui::PinType::Tensor, "Predictions", true),
                      Pin(502, gui::PinType::Labels, "Targets", true)},
                     {Pin(503, gui::PinType::Tensor, "Loss", false, false)});

    auto optimizer = Node(6,
                          gui::NodeType::Adam,
                          "Adam",
                          {},
                          {Pin(601, gui::PinType::Optimizer, "Optimizer", false)});

    const std::vector<gui::MLNode> nodes = {
        data, decoder, head, output, loss, optimizer};
    const std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 3, 302, 4, 401),
        Link(4, 3, 302, 5, 501),
        Link(5, 1, 102, 5, 502),
        Link(6, 6, 601, 5, 503),
    };

    cyxwiz::GraphCompiler compiler;
    const cyxwiz::TrainingConfiguration config =
        compiler.Compile(nodes, links, true);

    Check(config.is_valid,
          "causal LM graph should compile without shape errors: " +
              config.error_message);
    Check(config.input_shape == std::vector<size_t>({7, 4}),
          "causal LM DatasetInput should preserve [seq, features] shape");
    Check(config.input_size == 4,
          "causal LM input_size should be per-token feature width");
    Check(config.layers.size() == 2,
          "causal LM graph should compile decoder plus token head");
    Check(config.layers[0].input_shape == std::vector<size_t>({7, 4}),
          "decoder should receive sequence-shaped input");
    Check(config.layers[0].output_shape == std::vector<size_t>({7, 4}),
          "decoder should preserve sequence-shaped output");
    Check(config.layers[1].input_shape == std::vector<size_t>({7, 4}),
          "TimeDistributed should receive sequence-shaped input");
    Check(config.layers[1].output_shape == std::vector<size_t>({7, 6}),
          "TimeDistributed should produce [seq, vocab] output shape");
    Check(!HasErrorText(config, "TimeDistributed requires sequence input shape"),
          "causal LM graph should not report TimeDistributed shape error");

    std::cout << "Graph compiler causal LM shape test passed\n";
    return 0;
}
