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

bool HasWarningText(const cyxwiz::TrainingConfiguration& config,
                    const std::string& text) {
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Warning &&
            issue.message.find(text) != std::string::npos) {
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

cyxwiz::TrainingConfiguration CompileRecurrentGraph(gui::NodeType recurrent_type,
                                                    int hidden_size,
                                                    bool bidirectional) {
    auto data = Node(1,
                     gui::NodeType::DataInput,
                     "Data",
                     {},
                     {Pin(101, gui::PinType::Tensor, "Data", false),
                      Pin(102, gui::PinType::Labels, "Labels", false)});
    data.parameters["dataset_name"] = "placement_test_dataset";
    data.parameters["data_loaded"] = "true";
    data.parameters["file_category"] = "tabular";
    data.parameters["label_column"] = "label";
    data.parameters["shape"] = "[64]";

    auto loader = Node(2,
                       gui::NodeType::DataLoader,
                       "Loader",
                       {Pin(201, gui::PinType::Tensor, "Data", true),
                        Pin(202, gui::PinType::Labels, "Labels", true)},
                       {Pin(203, gui::PinType::Tensor, "Data", false),
                        Pin(204, gui::PinType::Labels, "Labels", false)});
    loader.parameters["batch_size"] = "64";
    loader.parameters["epochs"] = "1";

    auto embedding = Node(3,
                          gui::NodeType::Embedding,
                          "Embedding",
                          {Pin(301, gui::PinType::Tensor, "Indices", true)},
                          {Pin(302, gui::PinType::Tensor, "Embeddings", false)});
    embedding.parameters["num_embeddings"] = "1000";
    embedding.parameters["embedding_dim"] = "64";

    auto recurrent = Node(4,
                          recurrent_type,
                          recurrent_type == gui::NodeType::GRU ? "GRU 32" : "LSTM 8",
                          {Pin(401, gui::PinType::Tensor, "Input", true)},
                          {Pin(402, gui::PinType::Tensor, "Output", false),
                           Pin(403, gui::PinType::Tensor, "Hidden", false, false)});
    recurrent.parameters["input_size"] = "64";
    recurrent.parameters["hidden_size"] = std::to_string(hidden_size);
    recurrent.parameters["num_layers"] = "1";
    recurrent.parameters["bidirectional"] = bidirectional ? "true" : "false";
    recurrent.parameters["return_sequences"] = "false";

    auto dense = Node(5,
                      gui::NodeType::Dense,
                      "Classifier",
                      {Pin(501, gui::PinType::Tensor, "Input", true)},
                      {Pin(502, gui::PinType::Tensor, "Output", false)});
    dense.parameters["units"] = "2";

    auto loss = Node(6,
                     gui::NodeType::CrossEntropyLoss,
                     "Loss",
                     {Pin(601, gui::PinType::Tensor, "Predictions", true),
                      Pin(602, gui::PinType::Labels, "Targets", true)},
                     {Pin(603, gui::PinType::Loss, "Loss", false)});

    auto optimizer = Node(7,
                          gui::NodeType::Adam,
                          "Adam",
                          {Pin(701, gui::PinType::Loss, "Loss", true)},
                          {});

    std::vector<gui::MLNode> nodes = {
        data, loader, embedding, recurrent, dense, loss, optimizer,
    };
    std::vector<gui::NodeLink> links = {
        Link(1, 1, 101, 2, 201),
        Link(2, 1, 102, 2, 202),
        Link(3, 2, 203, 3, 301),
        Link(4, 3, 302, 4, 401),
        Link(5, 4, 402, 5, 501),
        Link(6, 5, 502, 6, 601),
        Link(7, 2, 204, 6, 602),
        Link(8, 6, 603, 7, 701),
    };

    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(nodes, links, true);
}

} // namespace

int main() {
    auto gru_config = CompileRecurrentGraph(gui::NodeType::GRU, 32, false);
    Check(gru_config.is_valid, "GRU placement graph should compile");
    Check(gru_config.backend_placements.size() == 1,
          "GRU graph should produce one backend placement entry");

    const auto* gru_placement = FindPlacement(gru_config, 4);
    Check(gru_placement != nullptr, "GRU placement entry should reference node 4");
    Check(gru_placement->node_type == "GRU", "GRU placement should name the layer");
    Check(gru_placement->expected_backend == "CPU",
          "GRU should be conservatively placed on CPU");
    Check(gru_placement->status == "cpu", "GRU placement status should be cpu");
    Check(gru_placement->reason_code == "gru_arrayfire_cuda_probe_required",
          "GRU placement should use the shared reason code");
    Check(gru_placement->explanation.find("batch_size=64") != std::string::npos,
          "GRU placement explanation should include compiled batch size");
    Check(gru_placement->explanation.find("seq_len=64") != std::string::npos,
          "GRU placement explanation should include inferred sequence length");
    Check(HasWarningText(gru_config, "gru_arrayfire_cuda_probe_required"),
          "GRU CPU placement should surface as a compiler warning");

    auto lstm_config = CompileRecurrentGraph(gui::NodeType::LSTM, 8, false);
    Check(lstm_config.is_valid, "small LSTM placement graph should compile");
    Check(lstm_config.backend_placements.size() == 1,
          "LSTM graph should produce one backend placement entry");

    const auto* lstm_placement = FindPlacement(lstm_config, 4);
    Check(lstm_placement != nullptr, "LSTM placement entry should reference node 4");
    Check(lstm_placement->node_type == "LSTM", "LSTM placement should name the layer");
    Check(lstm_placement->expected_backend == "ArrayFire CUDA",
          "small single-direction LSTM should remain GPU-eligible");
    Check(lstm_placement->status == "gpu", "LSTM placement status should be gpu");
    Check(lstm_placement->reason_code == "arrayfire_cuda_allowed_by_estimator",
          "LSTM placement should use the shared allow reason code");
    Check(!HasWarningText(lstm_config, "arrayfire_cuda_allowed_by_estimator"),
          "GPU-eligible LSTM placement should not create a warning");

    std::cout << "Recurrent backend placement tests passed\n";
    return 0;
}
