#include "../src/core/graph_compiler.h"
#include "../src/core/pipeline_runtime_capabilities.h"
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

bool HasPlanNode(const cyxwiz::CompiledGraphPlan& plan, int node_id) {
    for (const auto& node : plan.nodes) {
        if (node.node_id == node_id) {
            return true;
        }
    }
    return false;
}

bool HasPlanEdge(const cyxwiz::CompiledGraphPlan& plan,
                 int from_node,
                 int from_pin,
                 int to_node,
                 int to_pin) {
    for (const auto& edge : plan.edges) {
        if (edge.from_node_id == from_node &&
            edge.from_pin_id == from_pin &&
            edge.to_node_id == to_node &&
            edge.to_pin_id == to_pin) {
            return true;
        }
    }
    return false;
}

bool HasGraphOpId(const cyxwiz::TrainingConfiguration& config, int node_id) {
    for (int graph_op_node_id : config.graph_op_node_ids) {
        if (graph_op_node_id == node_id) {
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

    auto batch_matmul = Node(3,
                             gui::NodeType::TensorBatchMatMul,
                             "Deferred BatchMatMul",
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

    std::vector<gui::MLNode> nodes = {data, dense, batch_matmul, loss, optimizer};
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

    Check(!config.is_valid, "training path with template TensorBatchMatMul must be invalid");
    Check(HasIssueText(config, "template/deferred"),
          "compile should report template/deferred status");
    Check(HasIssueText(config, "Deferred BatchMatMul"),
          "compile issue should name the deferred node");

    auto side_dot = batch_matmul;
    side_dot.id = 6;
    side_dot.name = "Disconnected Deferred BatchMatMul";
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
    Check(!HasIssueText(config, "Disconnected Deferred BatchMatMul"),
          "compile should not report side deferred node");
    Check(config.layers.size() == 1,
          "linear selected path should still compile one sequential layer");

    const auto& plan = config.graph_plan;
    Check(plan.available, "valid selected training path should produce graph plan");
    Check(plan.data_node_id == 1, "graph plan should record selected data node");
    Check(plan.loss_node_id == 4, "graph plan should record selected loss node");
    Check(plan.optimizer_node_id == 5, "graph plan should record selected optimizer node");
    Check(plan.data_pin_id == 101, "graph plan should record data output pin");
    Check(plan.label_pin_id == 102, "graph plan should record label output pin");
    Check(plan.prediction_pin_id == 401, "graph plan should record loss prediction pin");
    Check(plan.label_target_pin_id == 402, "graph plan should record loss target pin");
    Check(plan.loss_output_pin_id == 403, "graph plan should record loss output pin");

    Check(HasPlanNode(plan, 1), "graph plan should include data node");
    Check(HasPlanNode(plan, 2), "graph plan should include dense node");
    Check(HasPlanNode(plan, 4), "graph plan should include loss node");
    Check(HasPlanNode(plan, 5), "graph plan should include optimizer node");
    Check(!HasPlanNode(plan, 6), "graph plan should exclude disconnected deferred node");
    Check(!HasPlanNode(plan, 7), "graph plan should exclude disconnected side output");
    Check(plan.nodes.size() == 4, "graph plan should contain only selected path nodes");

    Check(HasPlanEdge(plan, 1, 101, 2, 201),
          "graph plan should include data-to-dense edge");
    Check(HasPlanEdge(plan, 2, 202, 4, 401),
          "graph plan should include dense-to-loss prediction edge");
    Check(HasPlanEdge(plan, 1, 102, 4, 402),
          "graph plan should include labels-to-loss target edge");
    Check(HasPlanEdge(plan, 4, 403, 5, 501),
          "graph plan should include loss-to-optimizer edge");
    Check(plan.edges.size() == 4, "graph plan should contain only selected path edges");

    struct OptimizerCase {
        gui::NodeType node_type;
        cyxwiz::OptimizerType optimizer_type;
        const char* name;
    };

    const std::vector<OptimizerCase> optimizer_cases = {
        {gui::NodeType::SGD, cyxwiz::OptimizerType::SGD, "SGD"},
        {gui::NodeType::Adam, cyxwiz::OptimizerType::Adam, "Adam"},
        {gui::NodeType::AdamW, cyxwiz::OptimizerType::AdamW, "AdamW"},
        {gui::NodeType::RMSprop, cyxwiz::OptimizerType::RMSprop, "RMSprop"},
        {gui::NodeType::Adagrad, cyxwiz::OptimizerType::AdaGrad, "Adagrad"},
        {gui::NodeType::NAdam, cyxwiz::OptimizerType::NAdam, "NAdam"},
    };

    for (const auto& optimizer_case : optimizer_cases) {
        auto selected_optimizer = optimizer;
        selected_optimizer.type = optimizer_case.node_type;
        selected_optimizer.name = optimizer_case.name;
        selected_optimizer.parameters["learning_rate"] = "0.002";

        nodes = {data, dense, loss, selected_optimizer};
        links = {
            Link(1, 1, 101, 2, 201),
            Link(2, 2, 202, 4, 401),
            Link(3, 1, 102, 4, 402),
            Link(4, 4, 403, 5, 501),
        };

        config = compiler.Compile(nodes, links, true);
        Check(config.is_valid,
              std::string("compiler should accept optimizer ") + optimizer_case.name);
        Check(config.optimizer_node_id == 5,
              std::string("compiler should select optimizer ") + optimizer_case.name);
        Check(config.GetOptimizerType() == optimizer_case.optimizer_type,
              std::string("compiler should map backend type for ") + optimizer_case.name);
        Check(config.GetOptimizerName() == optimizer_case.name,
              std::string("compiler should preserve optimizer name for ") + optimizer_case.name);
        Check(config.learning_rate == 0.002f,
              std::string("compiler should extract learning rate for ") + optimizer_case.name);
    }

    for (const auto& unsupported_case :
         cyxwiz::GetPipelineUnsupportedSequentialModelLayerCapabilities()) {
        auto unsupported = Node(17,
                                unsupported_case.node_type,
                                "UnsupportedLayer",
                                {Pin(1701, gui::PinType::Tensor, "Input", true)},
                                {Pin(1702, gui::PinType::Tensor, "Output", false)});

        nodes = {data, unsupported, loss, optimizer};
        links = {
            Link(1, 1, 101, 17, 1701),
            Link(2, 17, 1702, 4, 401),
            Link(3, 1, 102, 4, 402),
            Link(4, 4, 403, 5, 501),
        };

        config = compiler.Compile(nodes, links, true);
        Check(!config.is_valid,
              "unsupported layer should block compile");
        Check(HasIssueText(config, unsupported_case.reason),
              "unsupported layer should report backend gap");
    }

    auto encoded_ner = Node(20,
                            gui::NodeType::Dense,
                            "Sentence Sequences",
                            {Pin(2001, gui::PinType::Tensor, "Input", true)},
                            {Pin(2002, gui::PinType::Tensor, "Output", false)});
    encoded_ner.parameters["units"] = "128";
    encoded_ner.parameters["bio_scheme"] = "BIO";
    encoded_ner.parameters["token_column"] = "tokens";
    encoded_ner.parameters["tag_column"] = "ner_tags";

    nodes = {data, encoded_ner, loss, optimizer};
    links = {
        Link(1, 1, 101, 20, 2001),
        Link(2, 20, 2002, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "Dense-encoded NER target-design node should block compile");
    Check(HasIssueText(config, "encoded as Dense"),
          "Dense-encoded NER node should report Dense encoding mismatch");
    Check(HasIssueText(config, "first-class sequence/NER nodes"),
          "Dense-encoded NER node should report missing NER contract");

    auto encoded_side_output = Node(21,
                                    gui::NodeType::Output,
                                    "Encoded Side Output",
                                    {Pin(2101, gui::PinType::Tensor, "Input", true)},
                                    {});

    nodes = {data, dense, loss, optimizer, encoded_ner, encoded_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 20, 2001),
        Link(6, 20, 2002, 21, 2101),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side Dense-encoded target-design node should not block selected path");
    Check(!HasIssueText(config, "Sentence Sequences"),
          "side Dense-encoded target-design node should not be reported");

    for (const auto& scheduler_case :
         cyxwiz::GetPipelineUnsupportedTrainingControlCapabilities()) {
        auto scheduler = Node(18,
                              scheduler_case.node_type,
                              "UnsupportedTrainingControl",
                              {},
                              {});

        nodes = {data, dense, loss, optimizer, scheduler};
        links = {
            Link(1, 1, 101, 2, 201),
            Link(2, 2, 202, 4, 401),
            Link(3, 1, 102, 4, 402),
            Link(4, 4, 403, 5, 501),
        };

        config = compiler.Compile(nodes, links, true);
        Check(!config.is_valid,
              "unsupported training control should block compile");
        Check(HasIssueText(config, scheduler_case.reason),
              "unsupported training control should report execution gap");
    }

    auto abs = Node(8,
                    gui::NodeType::TensorAbs,
                    "Abs",
                    {Pin(801, gui::PinType::Tensor, "Input", true)},
                    {Pin(802, gui::PinType::Tensor, "Output", false)});

    auto add = Node(9,
                    gui::NodeType::Add,
                    "Runtime Add",
                    {Pin(901, gui::PinType::Tensor, "Input 1", true),
                     Pin(902, gui::PinType::Tensor, "Input 2", true)},
                    {Pin(903, gui::PinType::Tensor, "Output", false)});

    nodes = {data, abs, add, loss, optimizer};
    links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 9, 901),
        Link(3, 1, 101, 9, 902),
        Link(4, 9, 903, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected training path with backend-supported Add should compile");
    Check(!HasIssueText(config, "Runtime Add"),
          "supported graph-runtime Add should not be reported as deferred");
    Check(config.layers.size() == 1,
          "selected Add graph should still extract the unary tensor layer");
    Check(config.graph_op_node_ids.size() == 1,
          "selected Add graph should record one graph runtime op");
    Check(HasGraphOpId(config, 9),
          "selected Add graph should record the Add node id");
    Check(config.graph_plan.available,
          "selected Add graph should still produce a graph plan");
    Check(HasPlanNode(config.graph_plan, 9),
          "graph plan should include selected Add node");
    Check(HasPlanEdge(config.graph_plan, 8, 802, 9, 901),
          "graph plan should include Abs-to-Add edge");
    Check(HasPlanEdge(config.graph_plan, 1, 101, 9, 902),
          "graph plan should include Data-to-Add second input edge");
    Check(HasPlanEdge(config.graph_plan, 9, 903, 4, 401),
          "graph plan should include Add-to-loss prediction edge");

    auto concat = Node(10,
                       gui::NodeType::Concatenate,
                       "Runtime Concat",
                       {Pin(1001, gui::PinType::Tensor, "Input 1", true),
                        Pin(1002, gui::PinType::Tensor, "Input 2", true)},
                       {Pin(1003, gui::PinType::Tensor, "Output", false)});

    nodes = {data, abs, concat, loss, optimizer};
    links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 10, 1001),
        Link(3, 1, 101, 10, 1002),
        Link(4, 10, 1003, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected training path with backend-supported Concatenate should compile");
    Check(!HasIssueText(config, "Runtime Concat"),
          "supported graph-runtime Concatenate should not be reported as deferred");
    Check(config.graph_op_node_ids.size() == 1,
          "selected Concatenate graph should record one graph runtime op");
    Check(HasGraphOpId(config, 10),
          "selected Concatenate graph should record the Concatenate node id");
    Check(HasPlanNode(config.graph_plan, 10),
          "graph plan should include selected Concatenate node");
    Check(HasPlanEdge(config.graph_plan, 8, 802, 10, 1001),
          "graph plan should include Abs-to-Concatenate edge");
    Check(HasPlanEdge(config.graph_plan, 1, 101, 10, 1002),
          "graph plan should include Data-to-Concatenate second input edge");
    Check(HasPlanEdge(config.graph_plan, 10, 1003, 4, 401),
          "graph plan should include Concatenate-to-loss prediction edge");

    auto runtime_dot = Node(13,
                            gui::NodeType::TensorDot,
                            "Runtime Dot",
                            {Pin(1301, gui::PinType::Tensor, "A", true),
                             Pin(1302, gui::PinType::Tensor, "B", true)},
                            {Pin(1303, gui::PinType::Tensor, "Output", false)});

    nodes = {data, abs, runtime_dot, loss, optimizer};
    links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 13, 1301),
        Link(3, 1, 101, 13, 1302),
        Link(4, 13, 1303, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected training path with backend-supported TensorDot should compile");
    Check(!HasIssueText(config, "Runtime Dot"),
          "supported graph-runtime TensorDot should not be reported as deferred");
    Check(config.graph_op_node_ids.size() == 1,
          "selected TensorDot graph should record one graph runtime op");
    Check(HasGraphOpId(config, 13),
          "selected TensorDot graph should record the TensorDot node id");
    Check(config.layers.size() == 1,
          "selected TensorDot graph should still extract only the unary tensor layer");
    Check(HasPlanNode(config.graph_plan, 13),
          "graph plan should include selected TensorDot node");

    auto compare = Node(11,
                        gui::NodeType::TensorCompare,
                        "Runtime Compare",
                        {Pin(1101, gui::PinType::Tensor, "A", true),
                         Pin(1102, gui::PinType::Tensor, "B", true)},
                        {Pin(1103, gui::PinType::Tensor, "Mask", false)});
    compare.parameters["op"] = "==";

    nodes = {data, abs, compare, loss, optimizer};
    links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 11, 1101),
        Link(3, 1, 101, 11, 1102),
        Link(4, 11, 1103, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected training path with two-input TensorCompare should compile");
    Check(!HasIssueText(config, "disconnect input B"),
          "two-input TensorCompare should not use scalar-only compiler error");
    Check(config.graph_op_node_ids.size() == 1,
          "selected TensorCompare graph should record one graph runtime op");
    Check(HasGraphOpId(config, 11),
          "selected TensorCompare graph should record the TensorCompare node id");
    Check(config.layers.size() == 1,
          "selected TensorCompare graph should still extract only the unary tensor layer");
    Check(HasPlanNode(config.graph_plan, 11),
          "graph plan should include selected TensorCompare node");

    auto logical = Node(12,
                        gui::NodeType::TensorLogicalMask,
                        "Runtime Logical",
                        {Pin(1201, gui::PinType::Tensor, "A", true),
                         Pin(1202, gui::PinType::Tensor, "B", true)},
                        {Pin(1203, gui::PinType::Tensor, "Mask", false)});
    logical.parameters["op"] = "and";

    nodes = {data, abs, logical, loss, optimizer};
    links = {
        Link(1, 1, 101, 8, 801),
        Link(2, 8, 802, 12, 1201),
        Link(3, 1, 101, 12, 1202),
        Link(4, 12, 1203, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected training path with two-input TensorLogicalMask should compile");
    Check(!HasIssueText(config, "disconnect input B"),
          "two-input TensorLogicalMask should not use unary-only compiler error");
    Check(config.graph_op_node_ids.size() == 1,
          "selected TensorLogicalMask graph should record one graph runtime op");
    Check(HasGraphOpId(config, 12),
          "selected TensorLogicalMask graph should record the TensorLogicalMask node id");
    Check(config.layers.size() == 1,
          "selected TensorLogicalMask graph should still extract only the unary tensor layer");

    auto binary_loss = Node(14,
                            gui::NodeType::BCELoss,
                            "Binary Loss",
                            {Pin(1401, gui::PinType::Tensor, "Predictions", true),
                             Pin(1402, gui::PinType::Labels, "Targets", true)},
                            {Pin(1403, gui::PinType::Loss, "Loss", false)});

    auto binary_dense = dense;
    binary_dense.parameters["units"] = "2";

    nodes = {data, binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 14, 1401),
        Link(3, 1, 102, 14, 1402),
        Link(4, 14, 1403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "BCE loss with two prediction outputs should be invalid");
    Check(HasIssueText(config, "requires a single prediction output"),
          "BCE loss should report binary output-size mismatch");

    binary_dense.parameters["units"] = "1";
    nodes = {data, binary_dense, binary_loss, optimizer};

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "BCE loss with one prediction output should compile");
    Check(!HasIssueText(config, "requires a single prediction output"),
          "BCE loss should not report output-size mismatch for one output");

    auto class_loss = Node(15,
                           gui::NodeType::CrossEntropyLoss,
                           "Class Loss",
                           {Pin(1501, gui::PinType::Tensor, "Predictions", true),
                            Pin(1502, gui::PinType::Labels, "Targets", true)},
                           {Pin(1503, gui::PinType::Loss, "Loss", false)});

    auto output = Node(16,
                       gui::NodeType::Output,
                       "Output",
                       {Pin(1601, gui::PinType::Tensor, "Input", true)},
                       {Pin(1602, gui::PinType::Tensor, "Predictions", false)});
    output.parameters["classes"] = "3";

    auto class_dense = dense;
    class_dense.parameters["units"] = "2";

    nodes = {data, class_dense, output, class_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 16, 1601),
        Link(3, 16, 1602, 15, 1501),
        Link(4, 1, 102, 15, 1502),
        Link(5, 15, 1503, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "CrossEntropy class count mismatch should be invalid");
    Check(HasIssueText(config, "class count"),
          "CrossEntropy loss should report class/output mismatch");

    auto resize = Node(17,
                       gui::NodeType::Resize,
                       "Image Resize",
                       {Pin(1701, gui::PinType::Tensor, "Input", true)},
                       {Pin(1702, gui::PinType::Tensor, "Output", false)});
    resize.parameters["width"] = "64";
    resize.parameters["height"] = "64";

    nodes = {data, resize, dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 17, 1701),
        Link(2, 17, 1702, 2, 201),
        Link(3, 2, 202, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "image preprocessing on a tabular data path should be invalid");
    Check(HasIssueText(config, "is for image data"),
          "compile should report preprocessing/domain mismatch");

    std::cout << "Graph compiler deferred node guard and graph plan passed\n";
    return 0;
}
