#include "../src/core/graph_compiler.h"
#include "../src/gui/loaders/data_loader.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
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
                 bool is_input) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
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

bool HasIssueText(const cyxwiz::TrainingConfiguration& config,
                  const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Data",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "deferred_guard_dataset";

    auto dense = Node(2,
                      gui::NodeType::Dense,
                      "Dense",
                      {Pin(201, gui::PinType::Tensor, "Input", true)},
                      {Pin(202, gui::PinType::Tensor, "Output", false)});
    dense.parameters["units"] = "2";

    auto dot = Node(3,
                    gui::NodeType::TensorDot,
                    "Deferred Dot",
                    {Pin(301, gui::PinType::Tensor, "A", true),
                     Pin(302, gui::PinType::Tensor, "B", true)},
                    {Pin(303, gui::PinType::Tensor, "Output", false)});

    auto loss = Node(4,
                     gui::NodeType::MSELoss,
                     "Loss",
                     {Pin(401, gui::PinType::Tensor, "Predictions", true),
                      Pin(402, gui::PinType::Labels, "Targets", true)},
                     {Pin(403, gui::PinType::Loss, "Loss", false)});

    auto optimizer = Node(5,
                          gui::NodeType::Adam,
                          "Adam",
                          {Pin(501, gui::PinType::Loss, "Loss", true)},
                          {});

    std::vector<gui::MLNode> nodes = {data, dense, dot, loss, optimizer};
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 3, 301),
        Link(3, 2, 202, 3, 302),
        Link(4, 3, 303, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    cyxwiz::GraphCompiler compiler;
    auto config = compiler.Compile(nodes, links, true);

    Check(!config.is_valid, "training path with template TensorDot must be invalid");
    Check(HasIssueText(config, "template/deferred"),
          "compile should report template/deferred status");
    Check(HasIssueText(config, "Deferred Dot"),
          "compile issue should name the deferred node");

    auto side_dot = dot;
    side_dot.id = 6;
    side_dot.name = "Disconnected Deferred Dot";
    side_dot.inputs = {Pin(601, gui::PinType::Tensor, "A", true),
                       Pin(602, gui::PinType::Tensor, "B", true)};
    side_dot.outputs = {Pin(603, gui::PinType::Tensor, "Output", false)};

    auto side_output = Node(7,
                            gui::NodeType::Output,
                            "Side Output",
                            {Pin(701, gui::PinType::Tensor, "Input", true)},
                            {});

    nodes = {data, dense, loss, optimizer, side_dot, side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 2, 202, 6, 601),
        Link(6, 2, 202, 6, 602),
        Link(7, 6, 603, 7, 701),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "deferred node outside selected training path should not block compile");
    Check(!HasIssueText(config, "Disconnected Deferred Dot"),
          "compile should not report side deferred node");

    std::cout << "Graph compiler deferred node guard passed\n";
    return 0;
}
