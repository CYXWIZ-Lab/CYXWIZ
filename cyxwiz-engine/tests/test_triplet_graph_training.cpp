// Triplet metric learning through the Engine's own path (TOFIX140 A5), against
// PyTorch: Data Input -> Triplet Dataset Builder -> Dense -> ReLU -> Dense ->
// Triplet Loss -> SGD. The builder stacks [anchors; positives; negatives]
// (TripletBatchSampler), the encoder runs once over the stack, the Triplet
// Loss splits it, and the trained parameters match torch's from the same
// start and the same picks: fixtures/triplet_graph_pytorch.json
// (generate_triplet_graph_fixtures.py). The start is the Engine's own seeded
// initialisation, which the fixture records and the test checks first.
// Also: the sampler's picks, the stacked loss, and the compiler's refusals.
#include "../src/core/arrow_dataset.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/metric_learning_sampling.h"
#include "../src/core/stacked_metric_loss.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_resume_checkpoint.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Parameters = std::map<std::string, std::vector<double>>;

constexpr int kModelSeed = 52;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside =
        std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "triplet_graph_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_TRIPLET_GRAPH_FIXTURE;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset(const json& fixture) {
    const auto x = fixture.at("x").get<std::vector<std::vector<float>>>();
    const auto label = fixture.at("label").get<std::vector<float>>();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    for (size_t feature = 0; feature <= x.front().size(); ++feature) {
        const bool is_label = feature == x.front().size();
        arrow::FloatBuilder builder;
        for (size_t row = 0; row < x.size(); ++row) {
            Check(builder.Append(is_label ? label[row] : x[row][feature]).ok(), "append");
        }
        std::shared_ptr<arrow::Array> array;
        Check(builder.Finish(&array).ok(), "finish");
        fields.push_back(arrow::field(is_label ? "label" : "x" + std::to_string(feature), arrow::float32()));
        columns.push_back(array);
    }
    auto table = arrow::Table::Make(arrow::schema(fields), columns, static_cast<int64_t>(x.size()));
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "triplet_rows");
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

// A node from the node factory: its pins and defaults are the contract. Node
// `id` has input pins id*100+1.. and output pins id*100+11..
gui::MLNode FactoryNode(gui::NodeType type, int id, const std::string& name) {
    int next_node_id = id, next_pin_id = 0;
    auto node = gui::CreateGraphNode(type, name, next_node_id, next_pin_id);
    node.id = id;
    for (size_t i = 0; i < node.inputs.size(); ++i) node.inputs[i].id = id * 100 + 1 + static_cast<int>(i);
    for (size_t i = 0; i < node.outputs.size(); ++i) node.outputs[i].id = id * 100 + 11 + static_cast<int>(i);
    return node;
}

gui::MLNode Layer(int id, gui::NodeType type, const std::string& name, std::map<std::string, std::string> params) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 11, gui::PinType::Tensor, "Output", false)};
    node.parameters = std::move(params);
    return node;
}

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

// Data -> [Builder] -> Dense 4 -> ReLU -> Dense 2 -> loss -> SGD; the labels go
// to the loss.
Graph BuildGraph(bool builder, gui::NodeType loss_type, const std::string& margin, double learning_rate) {
    Graph g;
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "triplet_rows"}, {"shape", "[3]"}};
    g.nodes.push_back(data);
    int from_node = 1, from_pin = 111;
    if (builder) {
        auto triplets = FactoryNode(gui::NodeType::TripletDatasetBuilder, 2, "Triplets");
        Check(triplets.inputs.size() == 1 && triplets.inputs[0].type == gui::PinType::Tensor &&
                  triplets.outputs.size() == 1 && triplets.outputs[0].type == gui::PinType::Tensor,
              "Triplet Dataset Builder: one Tensor input, one Tensor output");
        g.nodes.push_back(triplets);
        g.links.push_back(Link(1, 1, 111, 2, 201));
        from_node = 2;
        from_pin = 211;
    }
    g.nodes.push_back(Layer(3, gui::NodeType::Dense, "Dense 4", {{"units", "4"}}));
    g.nodes.push_back(Layer(4, gui::NodeType::ReLU, "ReLU", {}));
    g.nodes.push_back(Layer(5, gui::NodeType::Dense, "Embedding", {{"units", "2"}}));
    g.links.push_back(Link(2, from_node, from_pin, 3, 301));
    g.links.push_back(Link(3, 3, 311, 4, 401));
    g.links.push_back(Link(4, 4, 411, 5, 501));

    auto loss = FactoryNode(loss_type, 9, "Loss");
    Check(loss.inputs.size() == 2 && loss.inputs[0].type == gui::PinType::Tensor &&
              loss.inputs[1].type == gui::PinType::Labels && loss.outputs.size() == 1 &&
              loss.outputs[0].type == gui::PinType::Loss,
          "loss node: Tensor + Labels in, Loss out");
    if (!margin.empty()) {
        Check(loss.parameters.count("margin") == 1, "Triplet Loss has a margin");
        loss.parameters["margin"] = margin;
    }
    g.nodes.push_back(loss);
    g.links.push_back(Link(5, 5, 511, 9, 901));
    g.links.push_back(Link(6, 1, 112, 9, 902));

    gui::MLNode sgd;
    sgd.id = 10;
    sgd.type = gui::NodeType::SGD;
    sgd.name = "SGD";
    sgd.inputs = {Pin(1001, gui::PinType::Loss, "Loss", true)};
    sgd.outputs = {Pin(1011, gui::PinType::Optimizer, "State", false)};
    sgd.outputs[0].is_required = false;  // as the SGD node's metadata says
    sgd.parameters = {{"learning_rate", std::to_string(learning_rate)}, {"momentum", "0"}};
    g.nodes.push_back(sgd);
    g.links.push_back(Link(7, 9, 911, 10, 1001));
    return g;
}

void PrintIssues(const cyxwiz::TrainingConfiguration& config) {
    for (const auto& issue : config.issues) {
        std::cerr << "  issue(" << (issue.level == cyxwiz::IssueLevel::Error ? "error" : "note") << "): "
                  << issue.node_name << ": " << issue.message << "\n";
    }
}

cyxwiz::TrainingConfiguration Compile(const Graph& graph) {
    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(graph.nodes, graph.links, true);
}

cyxwiz::TrainingConfiguration CompileValid(const Graph& graph, const json& fixture, const std::string& what) {
    auto config = Compile(graph);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, what + " compiles");
    // Every row trains, in order, in batches of batch_size.
    config.dataset_name = "triplet_rows";
    config.batch_size = fixture.at("batch_size").get<int>();
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.model_seed = kModelSeed;
    config.dataloader_seed = fixture.at("dataloader_seed").get<int>();
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.checkpoint_dir =
        (std::filesystem::temp_directory_path() / "cyxwiz_triplet_graph_checkpoints").string();
    return config;
}

void CheckRefused(const Graph& graph, const std::string& text, const std::string& what) {
    const auto config = Compile(graph);
    bool reported = false;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error && issue.message.find(text) != std::string::npos) {
            reported = true;
        }
    }
    if (!reported) PrintIssues(config);
    Check(!config.is_valid && reported, what + ": refused with '" + text + "'");
}

void SelectArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(devices.begin(), devices.end(), [](const cyxwiz::DeviceInfo& device) {
        return device.type == cyxwiz::DeviceType::CPU;
    });
    Check(cpu != devices.end(), "an ArrayFire CPU route is required");
    cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
}

struct Trained {
    Parameters parameters;
    std::vector<double> batch_losses;
    cyxwiz::TrainingMetrics metrics;
};

Trained Train(const cyxwiz::TrainingConfiguration& config, const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
              int epochs, const std::string& what) {
    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    Trained trained;
    bool completed = false;
    const auto on_batch = [&](int, int, int, float loss, float) { trained.batch_losses.push_back(loss); };
    executor.Train(epochs, config.batch_size, on_batch, nullptr, [&](const cyxwiz::TrainingMetrics& metrics) {
        trained.metrics = metrics;
        completed = true;
    });
    if (!completed) trained.metrics = executor.GetMetrics();
    Check(trained.metrics.terminal_status == "completed",
          what + " completes (" + trained.metrics.terminal_status + ": " + trained.metrics.terminal_reason + ")");
    Check(executor.GetModel() != nullptr, what + ": the trained model");
    for (const auto& [name, tensor] : executor.GetModel()->GetParameters()) {
        const float* values = tensor.ReadData<float>();
        trained.parameters[name].assign(values, values + tensor.NumElements());
    }
    return trained;
}

std::string Describe(const Parameters& parameters) {
    std::string text;
    for (const auto& [name, values] : parameters) {
        text += "    \"" + name + "\": [";
        for (size_t i = 0; i < values.size(); ++i) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%s%.9g", i ? ", " : "", values[i]);
            text += buffer;
        }
        text += "],\n";
    }
    return text;
}

void CheckParameters(const Parameters& actual, const json& expected, double tolerance, const std::string& what) {
    bool same = actual.size() == expected.size();
    for (const auto& [name, values] : actual) {
        if (!same || !expected.contains(name)) {
            same = false;
            break;
        }
        const auto reference = expected.at(name).get<std::vector<double>>();
        if (reference.size() != values.size()) {
            same = false;
            break;
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::abs(values[i] - reference[i]) > tolerance) same = false;
        }
    }
    if (!same) {
        std::cerr << what << ": Engine parameters\n" << Describe(actual) << "expected\n" << expected.dump(2) << "\n";
    }
    Check(same, what);
}

void CheckSampler(const json& fixture) {
    const auto label = fixture.at("label").get<std::vector<int64_t>>();
    const size_t batch = fixture.at("batch_size").get<size_t>();
    const std::vector<int64_t> first(label.begin(), label.begin() + static_cast<std::ptrdiff_t>(batch));
    const auto key = cyxwiz::TrainingEpochSeed(fixture.at("dataloader_seed").get<uint64_t>(), 1);
    const auto triplets = cyxwiz::SelectBatchTriplets(first, key, 0);
    const auto expected = fixture.at("first_batch_triplets").get<std::vector<std::array<int, 3>>>();
    Check(triplets == expected, "the sampler's first-batch picks are the fixture's");
    for (const auto& t : triplets) {
        Check(first[t[0]] == first[t[1]] && t[0] != t[1] && first[t[0]] != first[t[2]],
              "a triplet's positive shares the anchor's class and its negative does not");
    }
    // A class with one row in the batch gives no triplet; one class gives none.
    Check(cyxwiz::SelectBatchTriplets({0, 1, 1}, key, 0).size() == 2, "a lone class row is not an anchor");
    Check(cyxwiz::SelectBatchTriplets({3, 3, 3}, key, 0).empty(), "one class: no negatives, no triplets");
    Check(cyxwiz::SelectBatchTriplets(first, key, 0) == cyxwiz::SelectBatchTriplets(first, key, 0),
          "the picks replay");
}

void CheckStackedLoss() {
    // T = 1, D = 2: a = (0, 0), p = (3, 4) -> d_ap = 5; n = (0, 1) -> d_an = 1.
    const float values[] = {0, 0, 3, 4, 0, 1};
    const cyxwiz::Tensor embeddings({3, 2}, values, cyxwiz::DataType::Float32);
    const cyxwiz::Tensor ids = cyxwiz::Tensor::Zeros({1}, cyxwiz::DataType::Float32);
    cyxwiz::StackedTripletLoss loss(1.0f);
    const float value = loss.Forward(embeddings, ids).ReadData<float>()[0];
    Check(std::abs(value - 5.0f) < 1e-4f, "max(0, 5 - 1 + 1) = 5, got " + std::to_string(value));
    const cyxwiz::Tensor gradient = loss.Backward(embeddings, ids);
    Check(gradient.Shape() == std::vector<size_t>{3, 2}, "the gradient is stacked like the embeddings");
    bool refused = false;
    try {
        (void)loss.Forward(cyxwiz::Tensor::Zeros({4, 2}, cyxwiz::DataType::Float32), ids);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("[3T, D]") != std::string::npos;
    }
    Check(refused, "4 rows is not a stack of triplets");
}

void CheckRefusals() {
    CheckRefused(BuildGraph(false, gui::NodeType::TripletLoss, "", 0.1), "Triplet Dataset Builder",
                 "Triplet Loss without the builder");
    CheckRefused(BuildGraph(true, gui::NodeType::MSELoss, "", 0.1), "Triplet Loss",
                 "the builder with another loss");
    CheckRefused(BuildGraph(true, gui::NodeType::TripletLoss, "-1", 0.1), "margin",
                 "a negative margin");
}

}  // namespace

int main(int, char** argv) {
    namespace fs = std::filesystem;
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_triplet_graph_training";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(work_dir / "debug_runs");

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "triplet_graph_pytorch.json is readable");
    const json fixture = json::parse(in);
    const double tolerance = fixture.at("tolerance").get<double>();
    const double learning_rate = fixture.at("learning_rate").get<double>();
    const int epochs = fixture.at("epochs").get<int>();

    CheckSampler(fixture);
    CheckStackedLoss();
    CheckRefusals();
    const auto dataset = MakeDataset(fixture);

    // The start: the Engine's seeded initialisation (learning rate 0 leaves
    // it untouched) is the one torch starts from.
    auto still = CompileValid(BuildGraph(true, gui::NodeType::TripletLoss, "1.0", learning_rate), fixture,
                              "the start");
    still.learning_rate = 0.0f;
    const auto start = Train(still, dataset, 1, "the start");
    // 9 significant digits round-trip a float exactly.
    CheckParameters(start.parameters, fixture.at("initial_parameters"), 1.0e-8,
                    "the Engine's seed " + std::to_string(kModelSeed) +
                        " initialisation is the fixture's (if it changed, regenerate the fixture from these values)");

    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        char margin[32];
        std::snprintf(margin, sizeof(margin), "%g", c.at("margin").get<double>());
        const auto config =
            CompileValid(BuildGraph(true, gui::NodeType::TripletLoss, margin, learning_rate), fixture, name);
        Check(config.loss_type == gui::NodeType::TripletLoss, name + ": the loss is Triplet");
        const auto trained = Train(config, dataset, epochs, name);
        // The batch callback reports the epoch's running mean loss.
        const auto batch_losses = c.at("batch_losses").get<std::vector<double>>();
        const size_t per_epoch = batch_losses.size() / static_cast<size_t>(epochs);
        std::vector<double> losses;
        double epoch_sum = 0.0;
        for (size_t i = 0; i < batch_losses.size(); ++i) {
            if (i % per_epoch == 0) epoch_sum = 0.0;
            epoch_sum += batch_losses[i];
            losses.push_back(epoch_sum / static_cast<double>(i % per_epoch + 1));
        }
        bool same_losses = trained.batch_losses.size() == losses.size();
        for (size_t i = 0; same_losses && i < losses.size(); ++i) {
            same_losses = std::abs(trained.batch_losses[i] - losses[i]) <= tolerance;
        }
        if (!same_losses) {
            std::cerr << name << ": Engine batch losses";
            for (double loss : trained.batch_losses) std::cerr << " " << loss;
            std::cerr << "\nexpected";
            for (double loss : losses) std::cerr << " " << loss;
            std::cerr << "\n";
        }
        Check(same_losses, name + ": every batch loss matches torch");
        CheckParameters(trained.parameters, c.at("parameters_after"), tolerance,
                        name + ": trained parameters match torch");
        const float accuracy = trained.metrics.train_accuracy;
        Check(accuracy >= 0.0f && accuracy <= 1.0f,
              name + ": triplet accuracy (d(a,p) < d(a,n)) is a fraction, got " + std::to_string(accuracy));
    }

    fs::remove_all(work_dir);
    std::cout << "Triplet metric learning matches PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
