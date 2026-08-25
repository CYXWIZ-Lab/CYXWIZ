#include "../src/core/graph_compiler.h"
#include "../src/core/error_codes.h"
#include "../src/core/data_registry.h"
#include "../src/core/pipeline_runtime_capabilities.h"
#include "../src/gui/loaders/data_loader.h"

#include <core/arrow_dataset.h>
#include <core/parquet_backed_dataset.h>

#include <arrow/api.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
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

bool HasIssueText(const cyxwiz::TrainingConfiguration& config,
                  cyxwiz::IssueLevel level,
                  const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.level == level &&
            issue.message.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool HasIssueCode(const cyxwiz::TrainingConfiguration& config,
                  const std::string& code) {
    for (const auto& issue : config.issues) {
        if (issue.error_code == code) {
            return true;
        }
    }
    return false;
}

bool AllIssuesHaveCodes(const cyxwiz::TrainingConfiguration& config) {
    for (const auto& issue : config.issues) {
        if (issue.error_code.empty()) {
            return false;
        }
    }
    return true;
}

bool HasMetricLearningBlocker(const cyxwiz::TrainingConfiguration& config,
                              const std::string& text) {
    for (const auto& blocker : config.metric_learning_graph.blockers) {
        if (blocker.find(text) != std::string::npos) {
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

const cyxwiz::BackendPlacementEntry* FindPlacement(
    const cyxwiz::TrainingConfiguration& config,
    int node_id) {
    for (const auto& placement : config.backend_placements) {
        if (placement.node_id == node_id) {
            return &placement;
        }
    }
    return nullptr;
}

} // namespace

int main() {
    {
        cyxwiz::GraphCompiler compiler;
        const auto empty_config = compiler.Compile({}, {}, true);
        Check(!empty_config.is_valid,
              "empty graph should fail compile");
        Check(HasIssueCode(empty_config,
                           cyxwiz::errors::Compiler::MissingTrainingPathNode),
              "empty graph should expose stable missing-training-path code");
        Check(AllIssuesHaveCodes(empty_config),
              "empty graph compile should code every issue");
    }

    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Data",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "deferred_guard_dataset";

    auto dev_data = Node(6, gui::NodeType::DataInput, "Development Data", {}, {});
    dev_data.parameters["dataset_name"] = "deferred_guard_dev";
    dev_data.parameters["dataset_role"] = "dev";

    auto test_data = Node(7, gui::NodeType::DataInput, "Test Data", {}, {});
    test_data.parameters["dataset_name"] = "deferred_guard_test";
    test_data.parameters["dataset_role"] = "test";

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

    std::vector<gui::MLNode> nodes = {data, dev_data, test_data, dense, batch_matmul, loss, optimizer};
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

    Check(config.dataset_roles.train.dataset_name == "deferred_guard_dataset",
          "the selected DataInput must resolve as the Train dataset role");
    Check(config.dataset_roles.train.source_node_id == 1,
          "the Train dataset role must retain its source node id");
    Check(config.dataset_roles.train.externally_supplied,
          "the connected physical Train source must retain external origin");
    Check(!config.dataset_roles.dev.IsSupplied() &&
              !config.dataset_roles.test.IsSupplied(),
          "disconnected legacy role hints must not affect resolved partitions");

    Check(!config.is_valid, "training path with template TensorBatchMatMul must be invalid");
    Check(HasIssueText(config, "template/deferred"),
          "compile should report template/deferred status");
    Check(HasIssueText(config, "Deferred BatchMatMul"),
          "compile issue should name the deferred node");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::UnsupportedTrainingNode),
          "selected deferred training node should expose unsupported-node code");
    Check(AllIssuesHaveCodes(config),
          "deferred-node compile should code every issue");

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
    Check(AllIssuesHaveCodes(config),
          "valid compile warnings should still carry codes");
    Check(!HasIssueText(config, "Disconnected Deferred BatchMatMul"),
          "compile should not report side deferred node");
    Check(config.layers.size() == 1,
          "linear selected path should still compile one sequential layer");

    auto configured_optimizer = optimizer;
    configured_optimizer.type = gui::NodeType::AdamW;
    configured_optimizer.name = "Configured AdamW";
    configured_optimizer.parameters = {
        {"learning_rate", "0.004"},
        {"lr", "0.009"},
        {"beta1", "0.8"},
        {"beta2", "0.95"},
        {"epsilon", "2e-7"},
        {"weight_decay", "0.03"},
    };
    auto optimizer_config = compiler.Compile(
        {data, dense, loss, configured_optimizer},
        {Link(1, 1, 101, 2, 201),
         Link(2, 2, 202, 4, 401),
         Link(3, 1, 102, 4, 402),
         Link(4, 4, 403, 5, 501)},
        true);
    Check(optimizer_config.is_valid,
          "valid AdamW hyperparameters should compile");
    Check(std::fabs(optimizer_config.learning_rate - 0.004f) < 1e-7f &&
              std::fabs(optimizer_config.beta1 - 0.8f) < 1e-7f &&
              std::fabs(optimizer_config.beta2 - 0.95f) < 1e-7f &&
              std::fabs(optimizer_config.epsilon - 2e-7f) < 1e-10f &&
              std::fabs(optimizer_config.weight_decay - 0.03f) < 1e-7f,
          "compiler should retain every AdamW backend setting");
    Check(HasIssueText(optimizer_config,
                       "learning_rate is authoritative"),
          "canonical learning_rate should win over a conflicting legacy lr");

    configured_optimizer.parameters["beta1"] = "not-a-number";
    optimizer_config = compiler.Compile(
        {data, dense, loss, configured_optimizer},
        {Link(1, 1, 101, 2, 201),
         Link(2, 2, 202, 4, 401),
         Link(3, 1, 102, 4, 402),
         Link(4, 4, 403, 5, 501)},
        true);
    Check(!optimizer_config.is_valid,
          "malformed optimizer hyperparameters should fail without throwing");
    Check(HasIssueCode(optimizer_config,
                       cyxwiz::errors::Compiler::InvalidParameter),
          "malformed optimizer hyperparameters should expose invalid-parameter code");

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

    auto stale_generated_dense = dense;
    stale_generated_dense.name = "Dense (128)";
    stale_generated_dense.parameters["units"] = "512";
    nodes = {data, stale_generated_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "compiler should accept a Dense node with a stale generated name");
    Check(config.layers.size() == 1 && config.layers.front().units == 512,
          "Dense execution width should come from the units parameter");
    Check(config.layers.front().name == "Dense (512)",
          "compiled Dense name should report the configured execution width");
    const auto* dense_placement = FindPlacement(config, stale_generated_dense.id);
    Check(dense_placement != nullptr &&
              dense_placement->node_name == "Dense (512)",
          "Dense placement should report the configured execution width");

    auto custom_named_dense = stale_generated_dense;
    custom_named_dense.name = "Classifier Head";
    custom_named_dense.parameters["units"] = "256";
    config = compiler.Compile({data, custom_named_dense, loss, optimizer},
                              links,
                              true);
    Check(config.is_valid && config.layers.front().name == "Classifier Head",
          "compiler should preserve custom Dense names");

    gui::RefreshGeneratedNodeName(stale_generated_dense);
    Check(stale_generated_dense.name == "Dense (512)",
          "property edits should refresh generated Dense names");

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
        if (unsupported_case.node_type == gui::NodeType::Conv2D) {
            // Saved graphs use this legacy design value. A blocked node must
            // report capability truth instead of attempting numeric parsing.
            unsupported.parameters["padding"] = "same";
        }

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
        Check(!HasIssueText(config, "Graph must have at least one model layer"),
              "unsupported sequential layer should count as a model layer for validation");
        const auto placement_summary = config.SummarizeBackendPlacements();
        Check(placement_summary.unknown == 0,
              "unsupported layer should not report unknown backend placement");
        Check(placement_summary.unsupported == 1,
              "unsupported layer should report unsupported backend placement");
        Check(!config.backend_placements.empty() &&
                  config.backend_placements.front().status ==
                      cyxwiz::BackendPlacementStatus::Unsupported,
              "unsupported layer placement status should be unsupported");
        Check(!config.backend_placements.empty() &&
                  config.backend_placements.front().reason_code ==
                      cyxwiz::BackendPlacementReason::
                          UnsupportedSequentialModelLayer,
              "unsupported layer placement should use central reason code");
        Check(HasIssueCode(config,
                           cyxwiz::errors::Compiler::UnsupportedTrainingNode),
              "unsupported sequential layer should expose unsupported-node code");
        if (unsupported_case.node_type == gui::NodeType::PolicyNetwork ||
            unsupported_case.node_type == gui::NodeType::ValueNetwork) {
            Check(HasIssueText(config, "reinforcement-learning training"),
                  "RL layer should report missing RL training contract");
        } else {
            Check(HasIssueText(config, unsupported_case.reason),
                  "unsupported layer should report backend gap");
        }
    }

    auto standalone_mha = Node(37,
                               gui::NodeType::MultiHeadAttention,
                               "Standalone MHA",
                               {Pin(3701, gui::PinType::Tensor, "Query", true),
                                Pin(3702, gui::PinType::Tensor, "Key", true)},
                               {Pin(3703, gui::PinType::Tensor, "Output", false)});
    standalone_mha.parameters["embed_dim"] = "4";
    standalone_mha.parameters["num_heads"] = "2";

    nodes = {data, standalone_mha, loss, optimizer};
    links = {
        Link(1, 1, 101, 37, 3701),
        Link(2, 1, 101, 37, 3702),
        Link(3, 37, 3703, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "connected-key MultiHeadAttention should stay blocked until cross-attention contract exists");
    Check(!HasIssueText(config, "Graph must have at least one model layer"),
          "standalone MultiHeadAttention should count as a model layer for validation");
    Check(HasIssueText(config, "Standalone MHA"),
          "standalone MultiHeadAttention blocker should name the graph node");
    Check(HasIssueText(config, "supports only single-input self-attention"),
          "connected-key MultiHeadAttention blocker should report cross-attention gap");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::UnsupportedTrainingNode),
          "standalone MultiHeadAttention should expose unsupported-node code");
    {
        const auto* mha_placement = FindPlacement(config, 37);
        Check(mha_placement != nullptr,
              "standalone MultiHeadAttention should produce backend placement truth");
        Check(mha_placement->node_type == "MultiHeadAttention",
              "standalone MultiHeadAttention placement should name the node type");
        Check(mha_placement->status ==
                  cyxwiz::BackendPlacementStatus::Cpu,
              "self-attention-capable MultiHeadAttention placement should be CPU-backed");
        Check(mha_placement->reason_code ==
                  cyxwiz::BackendPlacementReason::
                      GraphRuntimeCpuBacked,
              "standalone MultiHeadAttention placement should use CPU-backed reason");
    }

    auto masked_mha = Node(371,
                           gui::NodeType::MultiHeadAttention,
                           "Masked MHA",
                           {Pin(3711, gui::PinType::Tensor, "Query", true),
                            Pin(3712, gui::PinType::Tensor, "Mask", true)},
                           {Pin(3713, gui::PinType::Tensor, "Output", false)});
    masked_mha.parameters["embed_dim"] = "4";
    masked_mha.parameters["num_heads"] = "2";

    nodes = {data, masked_mha, loss, optimizer};
    links = {
        Link(1, 1, 101, 371, 3711),
        Link(2, 1, 101, 371, 3712),
        Link(3, 371, 3713, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "connected-mask MultiHeadAttention should fail closed until mask "
          "execution exists");
    Check(HasIssueText(config, "Key/Value/Context/Mask"),
          "connected-mask MultiHeadAttention should report the unsupported "
          "input contract");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::UnsupportedTrainingNode),
          "connected-mask MultiHeadAttention should expose unsupported-node code");

    auto self_mha = Node(38,
                         gui::NodeType::MultiHeadAttention,
                         "Self MHA",
                         {Pin(3801, gui::PinType::Tensor, "Query", true)},
                         {Pin(3803, gui::PinType::Tensor, "Output", false)});
    self_mha.parameters["embed_dim"] = "4";
    self_mha.parameters["num_heads"] = "2";

    auto self_mha_data = data;
    self_mha_data.parameters["input_shape"] = "[2,4]";
    nodes = {self_mha_data, self_mha, loss, optimizer};
    links = {
        Link(1, 1, 101, 38, 3801),
        Link(2, 38, 3803, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "single-input MultiHeadAttention self-attention should compile");
    Check(config.layers.size() == 1,
          "single-input MultiHeadAttention should produce one compiled layer");
    Check(config.layers.front().type == gui::NodeType::MultiHeadAttention,
          "single-input MultiHeadAttention compiled layer should preserve node type");

    auto cpu_loader = Node(39,
                           gui::NodeType::DataLoader,
                           "CPU DataLoader",
                           {Pin(3901, gui::PinType::Dataset, "Dataset", true)},
                           {Pin(3902, gui::PinType::Tensor, "Batch", false)});
    cpu_loader.parameters["batch_size"] = "8";
    cpu_loader.parameters["pin_memory"] = "true";

    nodes = {self_mha_data, cpu_loader, self_mha, loss, optimizer};
    links = {
        Link(1, 1, 101, 39, 3901),
        Link(2, 39, 3902, 38, 3801),
        Link(3, 38, 3803, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "CPU-only pin_memory graph should compile with a warning");
    Check(HasIssueText(config, "pin_memory=true is not applicable"),
          "CPU-only pin_memory request should surface a not-applicable warning");
    Check(config.pin_memory_transfer.requested,
          "CPU-only pin_memory request should be preserved");
    Check(config.pin_memory_transfer.backend == "CPU",
          "CPU-only pin_memory status should identify CPU backend");
    Check(config.pin_memory_transfer.effective_mode ==
              cyxwiz::PinMemoryTransferMode::PinnedRequestedButNotApplicable,
          "CPU-only pin_memory status should expose not-applicable mode");
    Check(config.pin_memory_transfer.reason_code ==
              cyxwiz::PinMemoryTransferReason::CpuBackendNotApplicable,
          "CPU-only pin_memory status should expose not-applicable reason");
    Check(config.pin_memory_transfer.NeedsUserWarning(),
          "CPU-only pin_memory status should require user warning");

    auto sequence_data = data;
    sequence_data.name = "NER Sentence CSV";
    sequence_data.parameters["file_category"] = "sequence_text";
    sequence_data.parameters["token_column"] = "tokens";
    sequence_data.parameters["tag_column"] = "ner_tags";

    nodes = {sequence_data, dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected sequence DataInput contract should compile");
    Check(!HasIssueText(config, "single tensor/label batch contract"),
          "sequence DataInput contract should not report the old batch limitation");
    Check(config.sequence_batch.enabled,
          "sequence DataInput should populate the sequence batch contract");
    Check(config.sequence_batch.token_column == "tokens",
          "sequence batch contract should capture token column");
    Check(config.sequence_batch.tag_column == "ner_tags",
          "sequence batch contract should capture tag column");
    Check(config.sequence_batch.ignore_index == -100,
          "sequence batch contract should default ignore_index to -100");

    auto sequence_loader = Node(19,
                                gui::NodeType::DataLoader,
                                "Sequence DataLoader",
                                {Pin(1901, gui::PinType::Tensor, "Input", true)},
                                {Pin(1902, gui::PinType::Tensor, "Batch", false)});
    sequence_loader.parameters["batch_size"] = "32";
    sequence_loader.parameters["epochs"] = "11";
    sequence_loader.parameters["num_workers"] = "0";
    sequence_loader.parameters["prefetch_factor"] = "3";
    sequence_loader.parameters["log_interval"] = "4";
    sequence_loader.parameters["validation_freq"] = "2";
    sequence_loader.parameters["seed"] = "1234";
    sequence_loader.parameters["grad_accum_steps"] = "5";
    sequence_loader.parameters["save_best_checkpoint"] = "false";
    sequence_loader.parameters["early_stopping_patience"] = "7";
    sequence_loader.parameters["checkpoint_dir"] = "runs/compiler_contract";
    sequence_loader.parameters["batch_layout"] = "batch_first";
    sequence_loader.parameters["pin_memory"] = "true";

    nodes = {data, sequence_loader, dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 19, 1901),
        Link(2, 19, 1902, 2, 201),
        Link(3, 2, 202, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected sequence DataLoader contract should compile");
    Check(!HasIssueText(config, "single tensor/label batch contract"),
          "sequence DataLoader contract should not report the old batch limitation");
    Check(config.sequence_batch.enabled,
          "sequence DataLoader should populate the sequence batch contract");
    Check(config.sequence_batch.batch_first,
          "sequence DataLoader batch_first layout should be captured");
    Check(config.sequence_batch.ignore_index == -100,
          "sequence DataLoader contract should default ignore_index to -100");
    Check(config.batch_size == 32,
          "DataLoader should own compiled batch size");
    Check(config.epochs == 11,
          "DataLoader should own compiled epoch count");
    Check(config.num_workers == 0,
          "DataLoader should preserve explicit num_workers");
    Check(config.prefetch_factor == 3,
          "DataLoader should preserve prefetch_factor");
    Check(config.log_interval == 4,
          "DataLoader should preserve log_interval");
    Check(config.validation_freq == 2,
          "DataLoader should preserve validation_freq");
    Check(config.dataloader_seed == 1234,
          "DataLoader should preserve seed");
    Check(config.grad_accum_steps == 5,
          "DataLoader should preserve grad_accum_steps");
    Check(!config.save_best_checkpoint,
          "DataLoader should preserve save_best_checkpoint policy");
    Check(config.early_stopping_patience == 7,
          "DataLoader should preserve early stopping patience");
    Check(config.checkpoint_dir == "runs/compiler_contract",
          "DataLoader should preserve checkpoint directory");
    Check(HasIssueText(config, "pin_memory=true is unsupported"),
          "DataLoader should surface unsupported pin_memory as a compiler warning");
    Check(config.pin_memory_transfer.requested,
          "DataLoader should preserve pin_memory as a runtime capability request");
    Check(config.pin_memory_transfer.node_id == 19,
          "pin_memory transfer status should identify the DataLoader node");
    Check(config.pin_memory_transfer.batch_size == 32,
          "pin_memory transfer status should include compiled batch size");
    Check(config.pin_memory_transfer.effective_mode ==
              cyxwiz::PinMemoryTransferMode::PinnedRequestedButUnsupported,
          "pin_memory transfer status should expose unsupported effective mode");
    Check(config.pin_memory_transfer.reason_code ==
              cyxwiz::PinMemoryTransferReason::BackendUnavailable,
          "pin_memory transfer status should expose unsupported reason code");
    Check(config.pin_memory_transfer.NeedsUserWarning(),
          "unsupported pin_memory transfer status should require user warning");

    auto sequence_builder = Node(22,
                                 gui::NodeType::NERSequenceBuilder,
                                 "NER Sequence Builder",
                                 {Pin(2201, gui::PinType::Dataset, "Rows", true)},
                                 {Pin(2202, gui::PinType::Tensor, "Sequence Samples", false)});
    sequence_builder.parameters["token_column"] = "tokens";
    sequence_builder.parameters["pos_column"] = "pos_tags";
    sequence_builder.parameters["tag_column"] = "ner_tags";
    sequence_builder.parameters["sentence_id_column"] = "sentence_id";
    sequence_builder.parameters["max_sequence_length"] = "8";
    sequence_builder.parameters["ignore_index"] = "-100";
    sequence_builder.parameters["create_attention_mask"] = "true";

    nodes = {data, sequence_builder, dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 22, 2201),
        Link(2, 22, 2202, 2, 201),
        Link(3, 2, 202, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "selected first-class NERSequenceBuilder should compile");
    Check(!HasIssueText(config, "single tensor/label batch contract"),
          "first-class NERSequenceBuilder should not report the old batch limitation");
    Check(config.sequence_batch.enabled,
          "first-class NERSequenceBuilder should populate sequence batch contract");
    Check(config.sequence_batch.token_column == "tokens",
          "first-class NERSequenceBuilder should capture token column");
    Check(config.sequence_batch.pos_column == "pos_tags",
          "first-class NERSequenceBuilder should capture POS column");
    Check(config.sequence_batch.tag_column == "ner_tags",
          "first-class NERSequenceBuilder should capture tag column");
    Check(config.sequence_batch.sentence_id_column == "sentence_id",
          "first-class NERSequenceBuilder should capture sentence id column");
    Check(config.sequence_batch.create_attention_mask,
          "first-class NERSequenceBuilder should capture attention mask setting");
    Check(config.sequence_batch.max_sequence_length == 8,
          "first-class NERSequenceBuilder should capture max sequence length");
    Check(config.sequence_batch.ignore_index == -100,
          "first-class NERSequenceBuilder should capture ignore_index");

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

    auto decoder = Node(26,
                        gui::NodeType::TransformerDecoder,
                        "Decoder LM",
                        {Pin(2601, gui::PinType::Tensor, "Input", true),
                         Pin(2602, gui::PinType::Tensor, "Memory", true)},
                        {Pin(2603, gui::PinType::Tensor, "Output", false)});
    decoder.parameters["d_model"] = "2";
    decoder.parameters["nhead"] = "1";
    decoder.parameters["num_layers"] = "1";

    nodes = {data, dense, decoder, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 26, 2601),
        Link(3, 2, 202, 26, 2602),
        Link(4, 26, 2603, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected TransformerDecoder path with Memory input should not compile without seq2seq contract");
    Check(HasIssueText(config, "connected Memory input"),
          "TransformerDecoder Memory path should report missing seq2seq/cross-attention contract");
    Check(HasIssueText(config, "decoder-only causal self-attention"),
          "TransformerDecoder Memory path should name the supported decoder-only contract");

    auto decoder_only = decoder;
    decoder_only.name = "Decoder Only LM";
    decoder_only.inputs[1].is_required = false;

    nodes = {data, dense, decoder_only, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 26, 2601),
        Link(3, 26, 2603, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "decoder-only TransformerDecoder path should compile under current engine support");
    Check(config.SummarizeBackendPlacements().cpu == 1,
          "decoder-only TransformerDecoder path should carry one CPU-backed placement");
    Check(config.SummarizeBackendPlacements().unknown == 0,
          "decoder-only TransformerDecoder path should not carry unknown backend placement");

    auto decoder_side_output = Node(27,
                                    gui::NodeType::Output,
                                    "Decoder Side Output",
                                    {Pin(2701, gui::PinType::Tensor, "Input", true)},
                                    {});

    nodes = {data, dense, loss, optimizer, decoder, decoder_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 2, 202, 26, 2601),
        Link(6, 2, 202, 26, 2602),
        Link(7, 26, 2603, 27, 2701),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side TransformerDecoder outside selected training path should not block compile");
    Check(!HasIssueText(config, "Decoder LM"),
          "side TransformerDecoder should not be reported");

    auto causal_dense = dense;
    causal_dense.name = "Causal Dense Sketch";
    causal_dense.parameters["causal"] = "true";

    nodes = {data, causal_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected causal/generative objective sketch should be invalid");
    Check(HasIssueText(config, "causal"),
          "causal sketch should report the matched generative parameter");
    Check(HasIssueText(config, "causal language-model contract"),
          "causal sketch should report missing causal language-model contract");

    auto pretrained = Node(28,
                           gui::NodeType::PretrainedMobileNet,
                           "Imported MobileNet",
                           {Pin(2801, gui::PinType::Tensor, "Image", true)},
                           {Pin(2802, gui::PinType::Tensor, "Features", false)});

    nodes = {data, pretrained, loss, optimizer};
    links = {
        Link(1, 1, 101, 28, 2801),
        Link(2, 28, 2802, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected pretrained model training path should be invalid");
    Check(HasIssueText(config, "imported/pretrained fine-tuning"),
          "pretrained path should report missing import-to-training contract");
    Check(HasIssueText(config, "parameter mapping"),
          "pretrained path should report missing parameter mapping contract");

    auto pretrained_side_output = Node(29,
                                       gui::NodeType::Output,
                                       "Imported Side Output",
                                       {Pin(2901, gui::PinType::Tensor, "Input", true)},
                                       {});

    nodes = {data, dense, loss, optimizer, pretrained, pretrained_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 28, 2801),
        Link(6, 28, 2802, 29, 2901),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side pretrained model outside selected training path should not block compile");
    Check(!HasIssueText(config, "Imported MobileNet"),
          "side pretrained model should not be reported");

    auto fine_tune_dense = dense;
    fine_tune_dense.name = "Fine Tune Dense Sketch";
    fine_tune_dense.parameters["fine_tune"] = "true";
    fine_tune_dense.parameters["pretrained_model_path"] = "bert-base-uncased";

    nodes = {data, fine_tune_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected imported/fine-tune sketch should be invalid");
    Check(HasIssueText(config, "fine_tune"),
          "fine-tune sketch should report the matched parameter");
    Check(HasIssueText(config, "freeze/unfreeze ownership"),
          "fine-tune sketch should report missing freeze contract");

    auto policy = Node(30,
                       gui::NodeType::PolicyNetwork,
                       "Policy Sketch",
                       {Pin(3001, gui::PinType::Tensor, "Observation", true)},
                       {Pin(3002, gui::PinType::Tensor, "Action", false)});
    policy.parameters["hidden_sizes"] = "8";

    nodes = {data, policy, loss, optimizer};
    links = {
        Link(1, 1, 101, 30, 3001),
        Link(2, 30, 3002, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected RL policy training path should be invalid");
    Check(HasIssueText(config, "reinforcement-learning training"),
          "RL policy path should report missing RL training contract");
    Check(HasIssueText(config, "environment stepping loop"),
          "RL policy path should report missing environment loop");

    auto policy_side_output = Node(31,
                                   gui::NodeType::Output,
                                   "Policy Side Output",
                                   {Pin(3101, gui::PinType::Tensor, "Input", true)},
                                   {});

    nodes = {data, dense, loss, optimizer, policy, policy_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 30, 3001),
        Link(6, 30, 3002, 31, 3101),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side RL policy outside selected training path should not block compile");
    Check(!HasIssueText(config, "Policy Sketch"),
          "side RL policy should not be reported");

    auto rl_dense = dense;
    rl_dense.name = "RL Dense Sketch";
    rl_dense.parameters["rl_training"] = "true";
    rl_dense.parameters["reward_column"] = "reward";

    nodes = {data, rl_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected RL objective sketch should be invalid");
    Check(HasIssueText(config, "rl_training"),
          "RL sketch should report the matched parameter");
    Check(HasIssueText(config, "policy/value loss contracts"),
          "RL sketch should report missing policy/value loss contract");

    auto detector = Node(32,
                         gui::NodeType::DNNDetect,
                         "Detector Sketch",
                         {Pin(3201, gui::PinType::Tensor, "Image", true)},
                         {Pin(3202, gui::PinType::Tensor, "Detections", false)});
    detector.parameters["confidence"] = "0.5";

    nodes = {data, detector, loss, optimizer};
    links = {
        Link(1, 1, 101, 32, 3201),
        Link(2, 32, 3202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected detection training path should be invalid");
    Check(HasIssueText(config, "detection/segmentation training"),
          "detection path should report missing detection training contract");
    Check(HasIssueText(config, "box/mask/class target materialization"),
          "detection path should report missing target materialization");

    auto detector_side_output = Node(33,
                                     gui::NodeType::Output,
                                     "Detector Side Output",
                                     {Pin(3301, gui::PinType::Tensor, "Input", true)},
                                     {});

    nodes = {data, dense, loss, optimizer, detector, detector_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 32, 3201),
        Link(6, 32, 3202, 33, 3301),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side detection node outside selected training path should not block compile");
    Check(!HasIssueText(config, "Detector Sketch"),
          "side detection node should not be reported");

    auto detection_dense = dense;
    detection_dense.name = "Detection Dense Sketch";
    detection_dense.parameters["bbox_column"] = "boxes";
    detection_dense.parameters["mask_column"] = "masks";

    nodes = {data, detection_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected detection target sketch should be invalid");
    Check(HasIssueText(config, "bbox_column"),
          "detection sketch should report the matched parameter");
    Check(HasIssueText(config, "multi-head loss contract"),
          "detection sketch should report missing multi-head loss contract");

    auto time_distributed = Node(34,
                                 gui::NodeType::TimeDistributed,
                                 "Token Head Sketch",
                                 {Pin(3401, gui::PinType::Tensor, "Input", true)},
                                 {Pin(3402, gui::PinType::Tensor, "Output", false)});

    nodes = {data, dense, time_distributed, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 34, 3401),
        Link(3, 34, 3402, 4, 401),
        Link(4, 1, 102, 4, 402),
        Link(5, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected TimeDistributed path over non-sequence tensor should be invalid");
    Check(HasIssueText(config, "TimeDistributed"),
          "TimeDistributed path should report the wrapper");
    Check(HasIssueText(config, "sequence input shape"),
          "TimeDistributed path should report the required sequence shape");

    auto time_distributed_side_output = Node(35,
                                             gui::NodeType::Output,
                                             "Token Head Side Output",
                                             {Pin(3501, gui::PinType::Tensor, "Input", true)},
                                             {});

    nodes = {data, dense, loss, optimizer, time_distributed,
             time_distributed_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 2, 202, 34, 3401),
        Link(6, 34, 3402, 35, 3501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side TimeDistributed outside selected training path should not block compile");
    Check(!HasIssueText(config, "Token Head Sketch"),
          "side TimeDistributed should not be reported");

    auto per_token_dense = dense;
    per_token_dense.name = "Per Token Dense Sketch";
    per_token_dense.parameters["per_token_head"] = "true";

    nodes = {data, per_token_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected per-token head sketch should be invalid");
    Check(HasIssueText(config, "per_token_head"),
          "per-token sketch should report the matched parameter");
    Check(HasIssueText(config, "per-token metrics"),
          "per-token sketch should report missing metrics contract");

    auto vae_sketch = dense;
    vae_sketch.id = 36;
    vae_sketch.name = "VAE";
    vae_sketch.inputs = {Pin(3601, gui::PinType::Tensor, "Input", true)};
    vae_sketch.outputs = {Pin(3602, gui::PinType::Tensor, "Output", false)};

    nodes = {data, vae_sketch, loss, optimizer};
    links = {
        Link(1, 1, 101, 36, 3601),
        Link(2, 36, 3602, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected VAE-style training sketch should be invalid");
    Check(HasIssueText(config, "VAE"),
          "VAE sketch should report the matched sketch name");
    Check(HasIssueText(config, "latent KL-loss contracts"),
          "VAE sketch should report missing latent KL contract");

    auto vae_side_output = Node(37,
                                gui::NodeType::Output,
                                "VAE Side Output",
                                {Pin(3701, gui::PinType::Tensor, "Input", true)},
                                {});

    nodes = {data, dense, loss, optimizer, vae_sketch, vae_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 36, 3601),
        Link(6, 36, 3602, 37, 3701),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side VAE sketch outside selected training path should not block compile");
    Check(!HasIssueText(config, "latent KL-loss contracts"),
          "side VAE sketch should not be reported");

    auto gan_dense = dense;
    gan_dense.name = "GAN Dense Sketch";
    gan_dense.parameters["generator_loss"] = "non_saturating";
    gan_dense.parameters["diffusion_timestep"] = "t";

    nodes = {data, gan_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected GAN/diffusion marker sketch should be invalid");
    Check(HasIssueText(config, "generator_loss"),
          "GAN/diffusion sketch should report the matched parameter");
    Check(HasIssueText(config, "alternating optimizer"),
          "GAN/diffusion sketch should report missing training-step contract");

    auto siamese_sketch = dense;
    siamese_sketch.id = 38;
    siamese_sketch.name = "SharedEncoder";
    siamese_sketch.inputs = {Pin(3801, gui::PinType::Tensor, "Input", true)};
    siamese_sketch.outputs = {Pin(3802, gui::PinType::Tensor, "Output", false)};

    nodes = {data, siamese_sketch, loss, optimizer};
    links = {
        Link(1, 1, 101, 38, 3801),
        Link(2, 38, 3802, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected metric-learning sketch should be invalid");
    Check(HasIssueText(config, "SharedEncoder"),
          "metric-learning sketch should report the matched sketch name");
    Check(HasIssueText(config, "shared-weight graph contract"),
          "metric-learning sketch should report missing shared encoder contract");
    Check(config.metric_learning_graph.detected,
          "metric-learning contract should detect selected SharedEncoder sketch");
    Check(!config.metric_learning_graph.executable,
          "metric-learning graph contract should remain non-executable");
    Check(config.metric_learning_graph.kind ==
              cyxwiz::MetricLearningGraphKind::None,
          "lone SharedEncoder sketch should not infer a training kind");
    Check(config.metric_learning_graph.shared_encoder_node_ids.size() == 1 &&
              config.metric_learning_graph.shared_encoder_node_ids[0] == 38,
          "metric-learning contract should record the shared encoder node");
    Check(HasMetricLearningBlocker(config, "visual shared-encoder graph execution"),
          "metric-learning contract should record shared-encoder execution blocker");

    auto typed_pair_score_output = Node(
        42,
        gui::NodeType::PairScoreOutput,
        "Renamed Pair Scorer",
        {Pin(4201, gui::PinType::Tensor, "Embedding A", true),
         Pin(4202, gui::PinType::Tensor, "Embedding B", true)},
        {Pin(4203, gui::PinType::Dataset, "Pair Scores", false)});

    nodes = {data, dense, typed_pair_score_output, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 42, 4201),
        Link(3, 2, 202, 42, 4202),
        Link(4, 42, 4203, 4, 401),
        Link(5, 1, 102, 4, 402),
        Link(6, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "typed metric-learning output node should be invalid");
    Check(HasIssueText(config, "PairScoreOutput"),
          "typed metric-learning node should report its enum contract");
    Check(config.metric_learning_graph.detected,
          "metric-learning contract should detect typed PairScoreOutput");
    Check(config.metric_learning_graph.kind ==
              cyxwiz::MetricLearningGraphKind::PairScoring,
          "PairScoreOutput should infer pair-scoring contract kind");
    Check(config.metric_learning_graph.pair_score_output_node_ids.size() == 1 &&
              config.metric_learning_graph.pair_score_output_node_ids[0] == 42,
          "metric-learning contract should record pair-score output node");
    Check(HasMetricLearningBlocker(config, "visual graph/runtime routing"),
          "pair-score contract should report missing visual graph output routing");

    auto siamese_side_output = Node(39,
                                    gui::NodeType::Output,
                                    "Siamese Side Output",
                                    {Pin(3901, gui::PinType::Tensor, "Input", true)},
                                    {});

    nodes = {data, dense, loss, optimizer, siamese_sketch,
             siamese_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 38, 3801),
        Link(6, 38, 3802, 39, 3901),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side metric-learning sketch outside selected path should not block compile");
    Check(!HasIssueText(config, "shared-weight graph contract"),
          "side metric-learning sketch should not be reported");

    auto triplet_dense = dense;
    triplet_dense.name = "Triplet Dense Sketch";
    triplet_dense.parameters["anchor_column"] = "anchor_text";
    triplet_dense.parameters["positive_column"] = "positive_text";
    triplet_dense.parameters["negative_column"] = "negative_text";

    nodes = {data, triplet_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected triplet marker sketch should be invalid");
    Check(HasIssueText(config, "anchor_column"),
          "triplet sketch should report the matched parameter");
    Check(HasIssueText(config, "pair/triplet batch payloads"),
          "triplet sketch should report missing batch contract");
    Check(config.metric_learning_graph.detected,
          "metric-learning contract should detect triplet column sketch");
    Check(config.metric_learning_graph.kind ==
              cyxwiz::MetricLearningGraphKind::TripletTraining,
          "triplet columns should infer triplet-training contract kind");
    Check(config.metric_learning_graph.triplet_dataset_builder_node_ids.size() == 1 &&
              config.metric_learning_graph.triplet_dataset_builder_node_ids[0] == 2,
          "metric-learning contract should record triplet batch source sketch");

    auto gnn_sketch = dense;
    gnn_sketch.id = 40;
    gnn_sketch.name = "GATConv";
    gnn_sketch.inputs = {Pin(4001, gui::PinType::Tensor, "Input", true)};
    gnn_sketch.outputs = {Pin(4002, gui::PinType::Tensor, "Output", false)};

    nodes = {data, gnn_sketch, loss, optimizer};
    links = {
        Link(1, 1, 101, 40, 4001),
        Link(2, 40, 4002, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected GNN sketch should be invalid");
    Check(HasIssueText(config, "GATConv"),
          "GNN sketch should report the matched sketch name");
    Check(HasIssueText(config, "edge-index/adjacency routing"),
          "GNN sketch should report missing edge-index contract");

    auto gnn_side_output = Node(41,
                                gui::NodeType::Output,
                                "GNN Side Output",
                                {Pin(4101, gui::PinType::Tensor, "Input", true)},
                                {});

    nodes = {data, dense, loss, optimizer, gnn_sketch, gnn_side_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 1, 101, 40, 4001),
        Link(6, 40, 4002, 41, 4101),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side GNN sketch outside selected path should not block compile");
    Check(!HasIssueText(config, "edge-index/adjacency routing"),
          "side GNN sketch should not be reported");

    auto edge_index_dense = dense;
    edge_index_dense.name = "Edge Index Dense Sketch";
    edge_index_dense.parameters["edge_index_column"] = "edge_index";
    edge_index_dense.parameters["node_features_column"] = "node_features";

    nodes = {data, edge_index_dense, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "selected GNN marker sketch should be invalid");
    Check(HasIssueText(config, "edge_index_column"),
          "GNN marker sketch should report the matched parameter");
    Check(HasIssueText(config, "message-passing kernels"),
          "GNN marker sketch should report missing message-passing contract");

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
    const auto* abs_placement = FindPlacement(config, 8);
    Check(abs_placement != nullptr,
          "selected Add graph should report unary layer backend placement");
    Check(abs_placement->status == cyxwiz::BackendPlacementStatus::Gpu,
          "selected unary tensor layer should remain ArrayFire-capable");
    Check(abs_placement->explanation.find("supported dtype/shape paths") != std::string::npos,
          "unary tensor layer placement should avoid blanket GPU wording");
    Check(abs_placement->explanation.find("CPU fallback") != std::string::npos,
          "unary tensor layer placement should explain CPU fallback");
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
    const auto* add_placement = FindPlacement(config, 9);
    Check(add_placement != nullptr,
          "selected Add graph should report graph-op backend placement");
    Check(add_placement->status == cyxwiz::BackendPlacementStatus::Mixed,
          "selected Add graph op should be reported as mixed backend");
    Check(add_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeArrayFireMixed,
          "selected Add graph op should use mixed graph-runtime reason");
    Check(add_placement->fallback_backend == "CPU",
          "selected Add graph op should report CPU fallback");
    Check(add_placement->explanation.find("row-major 2D elementwise addition") != std::string::npos,
          "selected Add graph op should explain concrete ArrayFire coverage");
    Check(add_placement->explanation.find("CPU fallback") != std::string::npos,
          "selected Add graph op should describe fallback as normal behavior");
    Check(add_placement->suggested_action.find("No correctness action needed") != std::string::npos,
          "selected Add graph op should avoid noisy fallback warnings");

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
    const auto* concat_placement = FindPlacement(config, 10);
    Check(concat_placement != nullptr,
          "selected Concatenate graph should report graph-op backend placement");
    Check(concat_placement->status == cyxwiz::BackendPlacementStatus::Mixed,
          "selected Concatenate graph op should be reported as mixed");
    Check(concat_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeArrayFireMixed,
          "selected Concatenate graph op should use mixed graph-runtime reason");

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
    const auto* dot_placement = FindPlacement(config, 13);
    Check(dot_placement != nullptr,
          "selected TensorDot graph should report graph-op backend placement");
    Check(dot_placement->status == cyxwiz::BackendPlacementStatus::Mixed,
          "selected TensorDot graph op should be reported as mixed backend");
    Check(dot_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeArrayFireMixed,
          "selected TensorDot graph op should use mixed graph-runtime reason");
    Check(dot_placement->explanation.find("graph backward") != std::string::npos,
          "selected TensorDot graph op should keep backward policy visible");
    Check(dot_placement->suggested_action.find("focused benchmarks") != std::string::npos,
          "selected TensorDot graph op should require benchmarks before speed claims");

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
    const auto* compare_placement = FindPlacement(config, 11);
    Check(compare_placement != nullptr,
          "selected TensorCompare graph should report graph-op backend placement");
    Check(compare_placement->status == cyxwiz::BackendPlacementStatus::Mixed,
          "selected TensorCompare graph op should be reported as mixed");
    Check(compare_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeArrayFireMixed,
          "selected TensorCompare graph op should use mixed graph reason");

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
    const auto* logical_placement = FindPlacement(config, 12);
    Check(logical_placement != nullptr,
          "selected TensorLogicalMask graph should report graph-op backend placement");
    Check(logical_placement->status == cyxwiz::BackendPlacementStatus::Mixed,
          "selected TensorLogicalMask graph op should be reported as mixed");
    Check(logical_placement->reason_code ==
              cyxwiz::BackendPlacementReason::GraphRuntimeArrayFireMixed,
          "selected TensorLogicalMask graph op should use mixed graph-runtime reason");

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

    std::shared_ptr<arrow::Array> sensor_a;
    std::shared_ptr<arrow::Array> sensor_b;
    std::shared_ptr<arrow::Array> class_label;
    arrow::FloatBuilder sensor_a_builder;
    arrow::Int64Builder sensor_b_builder;
    arrow::StringBuilder class_builder;
    for (int row = 0; row < 64; ++row) {
        (void)sensor_a_builder.Append(static_cast<float>(row));
        (void)sensor_b_builder.Append(row);
        (void)class_builder.Append("negative");
    }
    Check(sensor_a_builder.Finish(&sensor_a).ok() &&
              sensor_b_builder.Finish(&sensor_b).ok() &&
              class_builder.Finish(&class_label).ok(),
          "schema-contract arrays should build");
    auto schema_contract_table = arrow::Table::Make(
        arrow::schema({
            arrow::field("sensor_a", arrow::float32()),
            arrow::field("class", arrow::utf8()),
            arrow::field("sensor_b", arrow::int64()),
        }),
        {sensor_a, class_label, sensor_b});
    cyxwiz::DataRegistry::Instance().RegisterArrowTable(
        schema_contract_table, "compiler_schema_contract");

    const std::string schema_contract_name = "compiler_schema_contract";
    const std::string parquet_contract_name =
        schema_contract_name + "_parquet";
    auto schema_contract_dataset =
        cyxwiz::DataRegistry::Instance().GetArrowDataset(
            schema_contract_name);
    Check(schema_contract_dataset != nullptr, schema_contract_name);
    std::string parquet_file_name = parquet_contract_name;
    parquet_file_name += std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    parquet_file_name += ".parquet";
    const auto parquet_manifest_path =
        std::filesystem::temp_directory_path() / parquet_file_name;
    Check(schema_contract_dataset->ExportParquet(
              parquet_manifest_path.string()),
          parquet_contract_name);
    auto parquet_manifest_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_manifest_path.string(), parquet_contract_name);
    Check(parquet_manifest_dataset != nullptr, parquet_contract_name);
    cyxwiz::DataRegistry::Instance().RegisterParquetBacked(
        parquet_contract_name, parquet_manifest_dataset);

    auto schema_data = data;
    schema_data.parameters["dataset_name"] = "compiler_schema_contract";
    schema_data.parameters["file_category"] = "tabular";
    schema_data.parameters["label_column"] = "";
    nodes = {schema_data, binary_dense, binary_loss, optimizer};
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "a registered tabular schema with a conventional label should compile");
    Check(config.input_size == 2 && config.input_shape == std::vector<size_t>{2},
          "compiler input width must match the two numeric batch features");
    Check(config.dataset_roles.train.label_column == "class",
          "compiler should auto-resolve the tabular class label");
    Check(!HasIssueText(config, cyxwiz::IssueLevel::Warning,
                        "No label column selected"),
          "an auto-resolved schema label should not emit the missing-label warning");

    auto generated_target_table = arrow::Table::Make(
        arrow::schema({arrow::field("MT_320", arrow::float32())}),
        {sensor_a});
    cyxwiz::DataRegistry::Instance().RegisterArrowTable(
        generated_target_table, "compiler_generated_target");

    auto forecast_data = Node(
        30,
        gui::NodeType::DataInput,
        "Forecast Source",
        {},
        {Pin(3001, gui::PinType::Dataset, "Dataset", false)});
    forecast_data.parameters["dataset_name"] = "compiler_generated_target";
    forecast_data.parameters["file_category"] = "timeseries";
    forecast_data.parameters["label_column"] = "";

    auto forecast_window = Node(
        31,
        gui::NodeType::TimeSeriesWindow,
        "Forecast Window",
        {Pin(3101, gui::PinType::Dataset, "Data", true)},
        {Pin(3102, gui::PinType::Dataset, "Windowed", false)});
    forecast_window.parameters["value_col"] = "MT_320";
    forecast_window.parameters["input_width"] = "2";
    forecast_window.parameters["label_width"] = "2";

    auto forecast_loader = Node(
        32,
        gui::NodeType::DataLoader,
        "Forecast Loader",
        {Pin(3201, gui::PinType::Dataset, "Dataset", true)},
        {Pin(3202, gui::PinType::Tensor, "Features", false),
         Pin(3203, gui::PinType::Labels, "Labels", false)});

    auto forecast_dense = Node(
        33,
        gui::NodeType::Dense,
        "Forecast Head",
        {Pin(3301, gui::PinType::Tensor, "Input", true)},
        {Pin(3302, gui::PinType::Tensor, "Output", false)});
    forecast_dense.parameters["units"] = "2";

    auto forecast_loss = Node(
        34,
        gui::NodeType::MSELoss,
        "Forecast Loss",
        {Pin(3401, gui::PinType::Tensor, "Predictions", true),
         Pin(3402, gui::PinType::Labels, "Targets", true)},
        {Pin(3403, gui::PinType::Loss, "Loss", false)});

    auto forecast_optimizer = Node(
        35,
        gui::NodeType::Adam,
        "Forecast Adam",
        {Pin(3501, gui::PinType::Loss, "Loss", true)},
        {});

    nodes = {forecast_data, forecast_window, forecast_loader, forecast_dense,
             forecast_loss, forecast_optimizer};
    links = {
        Link(300, 30, 3001, 31, 3101),
        Link(301, 31, 3102, 32, 3201),
        Link(302, 32, 3202, 33, 3301),
        Link(303, 33, 3302, 34, 3401),
        Link(304, 32, 3203, 34, 3402),
        Link(305, 34, 3403, 35, 3501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.target.required_by_objective,
          "MSE objective should declare that it requires targets");
    Check(config.target.origin == cyxwiz::TargetOrigin::GraphGenerated,
          "TimeSeriesWindow should resolve graph-generated target provenance");
    Check(config.target.producer_node_id == 31 &&
              config.target.primary_column == "y" &&
              config.target.width == 2,
          "generated target contract should retain producer, column, and width");
    Check(!HasIssueCode(config,
                        cyxwiz::errors::Data::RequiredLabelColumnMissing),
          "generated forecast targets should suppress raw-label errors");

    auto target_scaler = Node(
        36,
        gui::NodeType::StandardScaler,
        "Forecast Target Scaler",
        {Pin(3601, gui::PinType::Dataset, "Data", true)},
        {Pin(3602, gui::PinType::Dataset, "Scaled", false)});
    target_scaler.parameters["columns"] = "y,y_1";
    target_scaler.parameters["label_col"] = "";
    target_scaler.parameters["exclude_columns"] = "";
    target_scaler.parameters["with_mean"] = "true";
    target_scaler.parameters["with_std"] = "true";
    target_scaler.parameters["transform_role"] = "regression_target";
    target_scaler.parameters["operation_mode"] = "transform_only";
    target_scaler.parameters["state_path"] = "fitted-target-state.json";

    nodes = {forecast_data, forecast_window, target_scaler, forecast_loader,
             forecast_dense, forecast_loss, forecast_optimizer};
    links = {
        Link(300, 30, 3001, 31, 3101),
        Link(301, 31, 3102, 36, 3601),
        Link(306, 36, 3602, 32, 3201),
        Link(302, 32, 3202, 33, 3301),
        Link(303, 33, 3302, 34, 3401),
        Link(304, 32, 3203, 34, 3402),
        Link(305, 34, 3403, 35, 3501),
    };
    config = compiler.Compile(nodes, links, true);
    Check(!HasIssueText(config, "Regression target scaler Columns"),
          "ordered Train-fitted target columns should satisfy compilation");
    Check(config.regression_target_transform.enabled &&
              config.regression_target_transform.target_columns ==
                  std::vector<std::string>({"y", "y_1"}) &&
              config.regression_target_transform.state_path ==
                  "fitted-target-state.json",
          "compiler should preserve the target inverse-transform contract");
    Check(HasIssueText(config, cyxwiz::IssueLevel::Info,
                       "original target units"),
          "compiler should disclose original-unit regression metrics");

    target_scaler.parameters["columns"] = "y_1,y";
    nodes = {forecast_data, forecast_window, target_scaler, forecast_loader,
             forecast_dense, forecast_loss, forecast_optimizer};
    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "reordered target scaler columns should fail compilation");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::LabelOutputShapeMismatch),
          "target scaler order mismatch should expose the shape error code");

    schema_data.parameters["label_column"] = "missing_label";
    nodes = {schema_data, binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 14, 1401),
        Link(3, 1, 102, 14, 1402),
        Link(4, 14, 1403, 5, 501),
    };
    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "an explicit label absent from the registered schema should block training");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Data::RequiredLabelColumnMissing),
          "missing explicit label should expose the stable data error code");

    auto second_head = Node(19,
                            gui::NodeType::Dense,
                            "Second Head",
                            {Pin(1901, gui::PinType::Tensor, "Input", true)},
                            {Pin(1902, gui::PinType::Tensor, "Output", false)});
    second_head.parameters["units"] = "1";

    auto second_loss = Node(22,
                            gui::NodeType::MSELoss,
                            "Auxiliary Loss",
                            {Pin(2201, gui::PinType::Tensor, "Predictions", true),
                             Pin(2202, gui::PinType::Labels, "Targets", true)},
                            {Pin(2203, gui::PinType::Loss, "Loss", false)});

    nodes = {data, dense, second_head, loss, second_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 1, 101, 19, 1901),
        Link(3, 2, 202, 4, 401),
        Link(4, 19, 1902, 22, 2201),
        Link(5, 1, 102, 4, 402),
        Link(6, 1, 102, 22, 2202),
        Link(7, 4, 403, 5, 501),
        Link(8, 22, 2203, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "multi-head graph with two dataset-reachable losses should be invalid");
    Check(HasIssueText(config, "exactly one dataset-reachable loss node"),
          "multi-loss graph should report single-loss training contract");
    Check(HasIssueText(config, "loss aggregation"),
          "multi-loss graph should point to missing aggregation contract");

    auto second_data = Node(23,
                            gui::NodeType::DataInput,
                            "Positive Data",
                            {},
                            {Pin(2301, gui::PinType::Tensor, "Data", false),
                             Pin(2302, gui::PinType::Labels, "Labels", false)});
    second_data.parameters["dataset_name"] = "positive_dataset";

    nodes = {data, second_data, dense, second_head, add, loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 23, 2301, 19, 1901),
        Link(3, 2, 202, 9, 901),
        Link(4, 19, 1902, 9, 902),
        Link(5, 9, 903, 4, 401),
        Link(6, 1, 102, 4, 402),
        Link(7, 23, 2302, 4, 402),
        Link(8, 4, 403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "multi-input graph with two dataset sources feeding one loss should be invalid");
    Check(HasIssueText(config, "exactly one dataset source"),
          "multi-input graph should report single dataset-source training contract");
    Check(HasIssueText(config, "typed named-batch contract"),
          "multi-input graph should point to missing named-batch contract");

    auto role_train = Node(
        60,
        gui::NodeType::DataInput,
        "Role Train Data",
        {},
        {Pin(6001, gui::PinType::Dataset, "Dataset", false)});
    role_train.parameters["dataset_name"] = "role_train_dataset";
    role_train.parameters["dataset_role"] = "test";
    role_train.parameters["label_column"] = "label";
    role_train.parameters["shape"] = "[1]";

    auto role_test = Node(
        61,
        gui::NodeType::DataInput,
        "Role Test Data",
        {},
        {Pin(6101, gui::PinType::Dataset, "Dataset", false)});
    role_test.parameters["dataset_name"] = "role_test_dataset";
    role_test.parameters["dataset_role"] = "train";
    role_test.parameters["label_column"] = "label";
    role_test.parameters["shape"] = "[1]";

    auto role_split = Node(
        62,
        gui::NodeType::DataSplit,
        "Role-Aware Split",
        {Pin(6201, gui::PinType::Dataset, "Training Dataset", true),
         Pin(6202, gui::PinType::Dataset, "Validation Dataset", false),
         Pin(6203, gui::PinType::Dataset, "Test Dataset", false)},
        {Pin(6204, gui::PinType::Dataset, "Partitions", false)});
    role_split.parameters["train_ratio"] = "0.8";
    role_split.parameters["val_ratio"] = "0.1";
    role_split.parameters["test_ratio"] = "0.1";
    role_split.inputs[1].is_required = false;
    role_split.inputs[2].is_required = false;

    auto role_loader = Node(
        63,
        gui::NodeType::DataLoader,
        "Role-Aware Loader",
        {Pin(6301, gui::PinType::Dataset, "Partitions", true)},
        {Pin(6302, gui::PinType::Tensor, "Data", false),
         Pin(6303, gui::PinType::Labels, "Labels", false)});

    nodes = {role_train, role_test, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 60, 6001, 62, 6201),
        Link(2, 61, 6101, 62, 6203),
        Link(3, 62, 6204, 63, 6301),
        Link(4, 63, 6302, 2, 201),
        Link(5, 2, 202, 14, 1401),
        Link(6, 63, 6303, 14, 1402),
        Link(7, 14, 1403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "Train plus role-matched external Test should compile");
    Check(!HasIssueText(config, "exactly one dataset source"),
          "role-matched external Test must not be treated as a second Train source");
    Check(config.dataset_roles.train.dataset_name == "role_train_dataset" &&
              config.dataset_roles.test.dataset_name == "role_test_dataset" &&
              config.dataset_roles.test.externally_supplied,
          "named Data Split pins must override stale Data Input role hints");
    Check(std::fabs(config.train_ratio - 0.9f) < 0.0001f &&
              std::fabs(config.val_ratio - 0.1f) < 0.0001f &&
              std::fabs(config.test_ratio) < 0.0001f,
          "external Test must reclaim its derived split share for Train while preserving Validation");
    Check(config.dataset_roles.policy.seed == 42 &&
              config.dataset_roles.policy.stratified == false &&
              config.dataset_roles.policy.train_ratio == config.train_ratio &&
              config.dataset_roles.manifest.test_origin ==
                  cyxwiz::PartitionOrigin::External &&
              config.dataset_roles.manifest.dev_origin ==
                  cyxwiz::PartitionOrigin::Derived,
          "typed partition policy and manifest origins must reflect topology");

    auto role_transform = Node(
        65,
        gui::NodeType::StandardScaler,
        "Training Dataset Transform",
        {Pin(6501, gui::PinType::Dataset, "Data", true)},
        {Pin(6502, gui::PinType::Dataset, "Scaled", false)});

    nodes = {role_train, role_transform, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 60, 6001, 65, 6501),
        Link(2, 65, 6502, 62, 6201),
        Link(3, 62, 6204, 63, 6301),
        Link(4, 63, 6302, 2, 201),
        Link(5, 2, 202, 14, 1401),
        Link(6, 63, 6303, 14, 1402),
        Link(7, 14, 1403, 5, 501),
    };
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "Dataset-preserving preprocessing before Data Split should compile");
    Check(config.dataset_roles.train.dataset_name == "role_train_dataset" &&
              config.dataset_roles.train.source_node_id == role_train.id,
          "Data Split should resolve Data Input through a Dataset transform");
    Check(!HasIssueText(config, "Training role must originate"),
          "valid Dataset preprocessing must not obscure the Training source");

    auto false_dataset_transform = Node(
        67,
        gui::NodeType::DataOutput,
        "Non-materializer Dataset Node",
        {Pin(6701, gui::PinType::Dataset, "Data", true)},
        {Pin(6702, gui::PinType::Dataset, "Dataset", false)});
    nodes = {role_train, false_dataset_transform, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links[0] = Link(1, 60, 6001, 67, 6701);
    links[1] = Link(2, 67, 6702, 62, 6201);
    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid &&
              HasIssueText(config, "Training role must originate"),
          "Dataset-shaped nodes without a materializer owner must not hide "
          "the Data Split source contract");

    auto role_dev = Node(
        64,
        gui::NodeType::DataInput,
        "Role Validation Data",
        {},
        {Pin(6401, gui::PinType::Dataset, "Dataset", false)});
    role_dev.parameters["dataset_name"] = "role_dev_dataset";
    role_dev.parameters["dataset_role"] = "train";
    role_dev.parameters["label_column"] = "label";
    role_dev.parameters["shape"] = "[1]";

    nodes = {role_train, role_dev, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 60, 6001, 62, 6201),
        Link(2, 64, 6401, 62, 6202),
        Link(3, 62, 6204, 63, 6301),
        Link(4, 63, 6302, 2, 201),
        Link(5, 2, 202, 14, 1401),
        Link(6, 63, 6303, 14, 1402),
        Link(7, 14, 1403, 5, 501),
    };
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid && config.dataset_roles.dev.IsSupplied() &&
              !config.dataset_roles.test.IsSupplied() &&
              std::fabs(config.train_ratio - 0.9f) < 0.0001f &&
              std::fabs(config.val_ratio) < 0.0001f &&
              std::fabs(config.test_ratio - 0.1f) < 0.0001f,
          "Train plus external Dev must preserve Dev and derive only Test from Train");

    nodes = {role_train, role_dev, role_test, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links.insert(links.begin() + 2, Link(8, 61, 6101, 62, 6203));
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid && config.dataset_roles.dev.IsSupplied() &&
              config.dataset_roles.test.IsSupplied() &&
              std::fabs(config.train_ratio - 1.0f) < 0.0001f &&
              std::fabs(config.val_ratio) < 0.0001f &&
              std::fabs(config.test_ratio) < 0.0001f &&
              config.dataset_roles.policy.method ==
                  cyxwiz::PartitionSplitMethod::None,
          "external Train, Dev, and Test must be preserved without internal splitting");

    nodes = {role_train, role_test, role_split, role_loader,
             binary_dense, binary_loss, optimizer};
    links = {
        Link(1, 60, 6001, 62, 6201),
        Link(2, 61, 6101, 62, 6203),
        Link(3, 62, 6204, 63, 6301),
        Link(4, 63, 6302, 2, 201),
        Link(5, 2, 202, 14, 1401),
        Link(6, 63, 6303, 14, 1402),
        Link(7, 14, 1403, 5, 501),
    };

    links[1] = Link(2, 61, 6101, 62, 6201);
    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "Test role wired to the Training Dataset pin should be invalid");
    Check(HasIssueText(config, "exactly one dataset source"),
          "misrouted Test role must retain the multi-source safety guard");

    links[1] = Link(2, 61, 6101, 62, 6203);
    links.push_back(Link(8, 61, 6101, 2, 201));
    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "Test role with an extra model-input branch should be invalid");
    Check(HasIssueText(config, "exactly one dataset source"),
          "every selected-path branch from Test must enter its named split pin");

    auto side_data = second_data;
    side_data.id = 24;
    side_data.name = "Side Data";
    side_data.outputs = {Pin(2401, gui::PinType::Tensor, "Data", false),
                         Pin(2402, gui::PinType::Labels, "Labels", false)};
    side_data.outputs[1].is_required = false;

    auto side_data_output = Node(25,
                                 gui::NodeType::Output,
                                 "Side Data Output",
                                 {Pin(2501, gui::PinType::Tensor, "Input", true)},
                                 {});

    nodes = {data, dense, loss, optimizer, side_data, side_data_output};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 4, 401),
        Link(3, 1, 102, 4, 402),
        Link(4, 4, 403, 5, 501),
        Link(5, 24, 2401, 25, 2501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "side dataset source outside selected loss path should not block compile");
    Check(!HasIssueText(config, "exactly one dataset source"),
          "side dataset source should not report multi-input training contract");

    config = compiler.Compile({data, dense, loss, optimizer},
                              {Link(1, 1, 101, 2, 201),
                               Link(2, 2, 202, 4, 401),
                               Link(3, 1, 102, 4, 402),
                               Link(4, 4, 403, 5, 501)},
                              true);
    Check(HasIssueText(config, "No pre-train data inspection node found"),
          "training graph without reachable inspection should warn before training");

    auto profiler = Node(26,
                         gui::NodeType::DataProfiler,
                         "Data Profiler",
                         {Pin(2601, gui::PinType::Tensor, "Data", true)},
                         {});

    config = compiler.Compile({data, dense, loss, optimizer, profiler},
                              {Link(1, 1, 101, 2, 201),
                               Link(2, 2, 202, 4, 401),
                               Link(3, 1, 102, 4, 402),
                               Link(4, 4, 403, 5, 501),
                               Link(5, 1, 101, 26, 2601)},
                              true);
    Check(!HasIssueText(config, "No pre-train data inspection node found"),
          "reachable DataProfiler should satisfy the pre-train inspection warning");

    auto stratified_split = Node(27,
                                 gui::NodeType::DataSplit,
                                 "Stratified Split",
                                 {Pin(2701, gui::PinType::Tensor, "Data", true),
                                  Pin(2702, gui::PinType::Labels, "Labels", true)},
                                 {Pin(2703, gui::PinType::Tensor, "Train Data", false),
                                  Pin(2704, gui::PinType::Labels, "Train Labels", false)});
    stratified_split.parameters["train_ratio"] = "0.8";
    stratified_split.parameters["val_ratio"] = "0.1";
    stratified_split.parameters["test_ratio"] = "0.1";
    stratified_split.parameters["stratified"] = "true";

    auto balanced_loader = Node(31,
                                gui::NodeType::DataLoader,
                                "Balanced Loader",
                                {Pin(3101, gui::PinType::Tensor, "Input", true),
                                 Pin(3102, gui::PinType::Labels, "Labels", true)},
                                {Pin(3103, gui::PinType::Tensor, "Batch", false),
                                 Pin(3104, gui::PinType::Labels, "Labels", false)});
    balanced_loader.parameters["batch_size"] = "16";
    balanced_loader.parameters["balance_classes"] = "true";
    balanced_loader.parameters["balance_mode"] = "weighted_sampler";
    balanced_loader.parameters["balance_target"] = "max";

    auto weighted_loss = Node(32,
                              gui::NodeType::CrossEntropyLoss,
                              "Weighted Loss",
                              {Pin(3201, gui::PinType::Tensor, "Predictions", true),
                               Pin(3202, gui::PinType::Labels, "Targets", true)},
                              {Pin(3203, gui::PinType::Loss, "Loss", false)});
    weighted_loss.parameters["class_weight"] = "manual";
    weighted_loss.parameters["class_weights"] = "[1.0, 3.0]";
    weighted_loss.parameters["label_smoothing"] = "0.1";

    nodes = {data, stratified_split, balanced_loader, dense, weighted_loss, optimizer};
    links = {
        Link(1, 1, 101, 27, 2701),
        Link(2, 1, 102, 27, 2702),
        Link(3, 27, 2703, 31, 3101),
        Link(4, 27, 2704, 31, 3102),
        Link(5, 31, 3103, 2, 201),
        Link(6, 2, 202, 32, 3201),
        Link(7, 31, 3104, 32, 3202),
        Link(8, 32, 3203, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "supported imbalance parameters should compile");
    Check(config.stratified,
          "compiler should preserve DataSplit.stratified=true in training config");
    Check(HasIssueText(config,
                       cyxwiz::IssueLevel::Info,
                       "resolves its named Dataset inputs"),
          "compiler should describe the modern Data Split partition contract");
    Check(!HasIssueText(config, "DataSplit.stratified=true is not implemented"),
          "compiler should not warn that implemented DataSplit stratification is ignored");
    Check(config.balance_classes,
          "compiler should preserve DataLoader balance_classes=true");
    Check(config.balance_mode == "weighted_sampler",
          "compiler should preserve DataLoader balance_mode");
    Check(config.balance_target == "max",
          "compiler should preserve DataLoader balance_target");
    Check(!HasIssueText(config, "DataLoader class-balancing parameters are present"),
          "compiler should not warn that implemented DataLoader balancing is ignored");
    Check(config.loss_params.at("class_weight") == "manual",
          "compiler should preserve CrossEntropy class_weight mode");
    Check(config.loss_params.at("class_weights") == "[1.0, 3.0]",
          "compiler should preserve CrossEntropy manual class weights");
    Check(config.loss_params.at("label_smoothing") == "0.1",
          "compiler should preserve CrossEntropy label_smoothing");
    Check(!HasIssueText(config, "Loss weighting parameters are present"),
          "compiler should not warn that supported loss weights are ignored");

    auto parquet_stratified_data = data;
    parquet_stratified_data.parameters["dataset_name"] =
        parquet_contract_name;
    nodes = {parquet_stratified_data, stratified_split, balanced_loader,
             dense, weighted_loss, optimizer};
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "supported Parquet imbalance graph should compile");
    Check(config.stratified,
          "requested Parquet stratification should remain visible");
    Check(!config.dataset_roles.policy.stratified &&
              config.dataset_roles.policy.method ==
                  cyxwiz::PartitionSplitMethod::Random &&
              !config.dataset_roles.manifest.stratified &&
              config.dataset_roles.manifest.split_method ==
                  cyxwiz::PartitionSplitMethod::Random,
          "Parquet manifest must record executed ratio slicing");
    cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(
        parquet_contract_name);
    std::error_code parquet_remove_error;
    std::filesystem::remove(parquet_manifest_path, parquet_remove_error);

    auto invalid_weighted_loss = weighted_loss;
    invalid_weighted_loss.parameters["class_weights"] = "[1.0, 2.0, 3.0]";
    nodes = {data, dense, invalid_weighted_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 32, 3201),
        Link(3, 1, 102, 32, 3202),
        Link(4, 32, 3203, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "CrossEntropy manual class_weights length mismatch should be invalid");
    Check(HasIssueText(config, "class_weights size"),
          "class_weights length mismatch should report a clear diagnostic");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::LabelOutputShapeMismatch),
          "class_weights length mismatch should expose label/output code");

    auto invalid_smoothed_loss = weighted_loss;
    invalid_smoothed_loss.parameters["label_smoothing"] = "1.0";
    nodes = {data, dense, invalid_smoothed_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 32, 3201),
        Link(3, 1, 102, 32, 3202),
        Link(4, 32, 3203, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(!config.is_valid,
          "CrossEntropy label_smoothing >= 1 should be invalid");
    Check(HasIssueText(config, "label_smoothing"),
          "invalid label_smoothing should report a clear diagnostic");
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::InvalidParameter),
          "invalid label_smoothing should expose invalid-parameter code");

    auto focal_loss = Node(33,
                           gui::NodeType::FocalLoss,
                           "Focal Loss",
                           {Pin(3301, gui::PinType::Tensor, "Predictions", true),
                            Pin(3302, gui::PinType::Labels, "Targets", true)},
                           {Pin(3303, gui::PinType::Loss, "Loss", false)});
    focal_loss.parameters["alpha"] = "0.75";
    focal_loss.parameters["gamma"] = "1.5";
    nodes = {data, dense, focal_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 33, 3301),
        Link(3, 1, 102, 33, 3302),
        Link(4, 33, 3303, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "FocalLoss graph should compile as a supported classification loss");
    Check(config.loss_type == gui::NodeType::FocalLoss,
          "compiler should preserve FocalLoss type");
    Check(config.loss_params.at("alpha") == "0.75",
          "compiler should preserve FocalLoss alpha");
    Check(config.loss_params.at("gamma") == "1.5",
          "compiler should preserve FocalLoss gamma");
    Check(!config.preprocessing.has_onehot,
          "FocalLoss should keep integer class labels instead of one-hot labels");

    auto dice_loss = Node(34,
                          gui::NodeType::SoftDiceLoss,
                          "Soft Dice Loss",
                          {Pin(3401, gui::PinType::Tensor, "Predictions", true),
                           Pin(3402, gui::PinType::Tensor, "Targets", true)},
                          {Pin(3403, gui::PinType::Loss, "Loss", false)});
    dice_loss.parameters["smooth"] = "0.5";
    nodes = {data, dense, dice_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 34, 3401),
        Link(3, 1, 102, 34, 3402),
        Link(4, 34, 3403, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "SoftDiceLoss graph should compile as a supported probability-mask loss");
    Check(config.loss_type == gui::NodeType::SoftDiceLoss,
          "compiler should preserve SoftDiceLoss type");
    Check(config.loss_params.at("smooth") == "0.5",
          "compiler should preserve SoftDice smooth");

    auto tversky_loss = Node(35,
                             gui::NodeType::TverskyLoss,
                             "Tversky Loss",
                             {Pin(3501, gui::PinType::Tensor, "Predictions", true),
                              Pin(3502, gui::PinType::Tensor, "Targets", true)},
                             {Pin(3503, gui::PinType::Loss, "Loss", false)});
    tversky_loss.parameters["alpha"] = "0.3";
    tversky_loss.parameters["beta"] = "0.7";
    tversky_loss.parameters["smooth"] = "0.5";
    nodes = {data, dense, tversky_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 35, 3501),
        Link(3, 1, 102, 35, 3502),
        Link(4, 35, 3503, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "TverskyLoss graph should compile as a supported probability-mask loss");
    Check(config.loss_type == gui::NodeType::TverskyLoss,
          "compiler should preserve TverskyLoss type");
    Check(config.loss_params.at("alpha") == "0.3",
          "compiler should preserve Tversky alpha");
    Check(config.loss_params.at("beta") == "0.7",
          "compiler should preserve Tversky beta");
    Check(config.loss_params.at("smooth") == "0.5",
          "compiler should preserve Tversky smooth");

    auto jaccard_loss = Node(36,
                             gui::NodeType::JaccardLoss,
                             "Jaccard Loss",
                             {Pin(3601, gui::PinType::Tensor, "Predictions", true),
                              Pin(3602, gui::PinType::Tensor, "Targets", true)},
                             {Pin(3603, gui::PinType::Loss, "Loss", false)});
    jaccard_loss.parameters["smooth"] = "0.5";
    nodes = {data, dense, jaccard_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 36, 3601),
        Link(3, 1, 102, 36, 3602),
        Link(4, 36, 3603, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "JaccardLoss graph should compile as a supported probability-mask loss");
    Check(config.loss_type == gui::NodeType::JaccardLoss,
          "compiler should preserve JaccardLoss type");
    Check(config.loss_params.at("smooth") == "0.5",
          "compiler should preserve Jaccard smooth");

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
    Check(HasIssueCode(config,
                       cyxwiz::errors::Compiler::LabelOutputShapeMismatch),
          "CrossEntropy class/output mismatch should expose stable code");

    output.parameters.clear();
    output.parameters["num_classes"] = "2";
    nodes = {data, class_dense, output, class_loss, optimizer};
    config = compiler.Compile(nodes, links, true);
    Check(config.is_valid,
          "Output num_classes should be accepted as classes fallback");
    Check(config.preprocessing.num_classes == 2,
          "Output num_classes fallback should drive CrossEntropy class count");
    Check(!HasIssueText(config, "class count"),
          "Output num_classes fallback should satisfy class-count validation");

    auto sequence_tag_output = Node(
        18,
        gui::NodeType::SequenceTagOutput,
        "Sequence Tag Output",
        {Pin(1801, gui::PinType::Tensor, "Token Logits", true)},
        {Pin(1802, gui::PinType::Tensor, "Predictions", false)});
    sequence_tag_output.parameters["num_tags"] = "4";
    sequence_tag_output.parameters["decode_scheme"] = "BIO";
    sequence_tag_output.parameters["tag_vocab_file"] =
        "examples/cyxgraph/NER/generated/ner_tag_vocab.txt";

    class_dense.parameters["units"] = "4";
    nodes = {data, class_dense, sequence_tag_output, class_loss, optimizer};
    links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 2, 202, 18, 1801),
        Link(3, 18, 1802, 15, 1501),
        Link(4, 1, 102, 15, 1502),
        Link(5, 15, 1503, 5, 501),
    };

    config = compiler.Compile(nodes, links, true);
    Check(config.preprocessing.num_classes == 4,
          "SequenceTagOutput num_tags should drive CrossEntropy class count");
    Check(!HasIssueText(config, "class count"),
          "SequenceTagOutput num_tags should satisfy class-count validation");

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
    Check(HasIssueCode(config,
                       cyxwiz::errors::Data::ColumnTypeMismatch),
          "preprocessing/domain mismatch should expose data contract code");

    std::cout << "Graph compiler deferred node guard and graph plan passed\n";
    return 0;
}
