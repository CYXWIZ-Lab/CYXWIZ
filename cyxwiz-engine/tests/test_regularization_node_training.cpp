// Regularization nodes through the Engine's own path (TOFIX140 A4), against
// PyTorch: Data Input -> Dense -> MSE -> penalty node -> SGD. GraphCompiler
// turns the node into TrainingConfiguration::regularization_l1/l2,
// TrainingExecutor adds the penalty's gradient to every optimizer step, and the
// trained parameters match torch's (loss + penalty).backward() from the same
// start: fixtures/regularization_node_pytorch.json
// (generate_regularization_node_fixtures.py). The start is the Engine's own
// seeded initialisation, which the fixture records and the test checks first.
// Also: the compiled coefficients and the compiler's refusals.
#include "../src/core/arrow_dataset.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/layer_capture_store.h"
#include "../src/core/training_executor.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>
#include <cyxwiz/sequential.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
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
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "regularization_node_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_REGULARIZATION_NODE_FIXTURE;
}

std::shared_ptr<arrow::Array> Floats(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) Check(builder.Append(value).ok(), "append");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish");
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset(const json& fixture) {
    const auto x0 = fixture.at("x0").get<std::vector<float>>();
    const auto x1 = fixture.at("x1").get<std::vector<float>>();
    const auto label = fixture.at("label").get<std::vector<float>>();
    auto schema = arrow::schema({arrow::field("x0", arrow::float32()), arrow::field("x1", arrow::float32()),
                                 arrow::field("label", arrow::float32())});
    auto table = arrow::Table::Make(schema, {Floats(x0), Floats(x1), Floats(label)},
                                    static_cast<int64_t>(label.size()));
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "regularization_rows");
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

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

// The penalty node comes from the node factory: its pins and defaults are the
// contract. Node `id` has input pin id*100+1 and output pin id*100+11.
gui::MLNode Penalty(gui::NodeType type, int id, const std::map<std::string, std::string>& params) {
    int next_node_id = id, next_pin_id = 0;
    auto node = gui::CreateGraphNode(type, "Penalty " + std::to_string(id), next_node_id, next_pin_id);
    Check(node.inputs.size() == 1 && node.inputs[0].type == gui::PinType::Loss && node.outputs.size() == 1 &&
              node.outputs[0].type == gui::PinType::Loss,
          "penalty node: one Loss input, one Loss output");
    node.id = id;
    node.inputs[0].id = id * 100 + 1;
    node.outputs[0].id = id * 100 + 11;
    for (const auto& [key, value] : params) {
        Check(node.parameters.count(key) == 1, "penalty parameter exists: " + key);
        node.parameters[key] = value;
    }
    return node;
}

// Loss -> penalties[0] -> ... -> SGD; with no penalty, Loss -> SGD. With
// `bypass`, the loss also feeds the optimizer directly and the penalty output
// is left hanging.
Graph BuildGraph(const std::vector<gui::MLNode>& penalties, double learning_rate, bool bypass = false) {
    Graph g;
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "regularization_rows"}, {"shape", "[2]"}};
    gui::MLNode dense;
    dense.id = 3;
    dense.type = gui::NodeType::Dense;
    dense.name = "Dense";
    dense.inputs = {Pin(301, gui::PinType::Tensor, "Input", true)};
    dense.outputs = {Pin(311, gui::PinType::Tensor, "Output", false)};
    dense.parameters = {{"units", "1"}};
    gui::MLNode loss;
    loss.id = 9;
    loss.type = gui::NodeType::MSELoss;
    loss.name = "MSE";
    loss.inputs = {Pin(901, gui::PinType::Tensor, "Predictions", true), Pin(902, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(911, gui::PinType::Loss, "Loss", false)};
    gui::MLNode sgd;
    sgd.id = 10;
    sgd.type = gui::NodeType::SGD;
    sgd.name = "SGD";
    sgd.inputs = {Pin(1001, gui::PinType::Loss, "Loss", true)};
    sgd.outputs = {Pin(1011, gui::PinType::Optimizer, "State", false)};
    sgd.outputs[0].is_required = false;  // as the SGD node's metadata says
    sgd.parameters = {{"learning_rate", std::to_string(learning_rate)}, {"momentum", "0"}};
    g.nodes = {data, dense, loss, sgd};
    g.links = {Link(1, 1, 111, 3, 301), Link(2, 3, 311, 9, 901), Link(3, 1, 112, 9, 902)};
    int from_node = 9, from_pin = 911;
    for (const auto& penalty : penalties) {
        g.nodes.push_back(penalty);
        g.links.push_back(Link(100 + penalty.id, from_node, from_pin, penalty.id, penalty.inputs[0].id));
        from_node = penalty.id;
        from_pin = penalty.outputs[0].id;
    }
    if (bypass) {
        from_node = 9;
        from_pin = 911;
    }
    g.links.push_back(Link(4, from_node, from_pin, 10, 1001));
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

cyxwiz::TrainingConfiguration CompileValid(const Graph& graph, int batch_size, const std::string& what) {
    auto config = Compile(graph);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, what + " compiles");
    // Every row trains, in order, in batches of batch_size.
    config.dataset_name = "regularization_rows";
    config.batch_size = batch_size;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.model_seed = kModelSeed;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.checkpoint_dir =
        (std::filesystem::temp_directory_path() / "cyxwiz_regularization_node_checkpoints").string();
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
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
}

// Trains and returns the model's parameters, flattened in the Engine's layout.
// With `snapshots`, layer snapshots are taken on the schedule and handed back.
Parameters TrainAndRead(const cyxwiz::TrainingConfiguration& config, const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
                        int epochs, const std::string& what,
                        const std::optional<cyxwiz::TrainingExecutor::LayerSnapshotSchedule>& snapshots = std::nullopt,
                        std::vector<cyxwiz::TrainingExecutor::LayerSnapshot>* taken = nullptr) {
    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    if (snapshots) executor.EnableLayerSnapshots(*snapshots);
    cyxwiz::TrainingMetrics final_metrics;
    bool completed = false;
    executor.Train(epochs, config.batch_size, nullptr, nullptr, [&](const cyxwiz::TrainingMetrics& metrics) {
        final_metrics = metrics;
        completed = true;
    });
    if (!completed) final_metrics = executor.GetMetrics();
    Check(final_metrics.terminal_status == "completed",
          what + " completes (" + final_metrics.terminal_status + ": " + final_metrics.terminal_reason + ")");
    Check(executor.GetModel() != nullptr, what + ": the trained model");
    if (taken) *taken = executor.LayerSnapshots();
    Parameters result;
    for (const auto& [name, tensor] : executor.GetModel()->GetParameters()) {
        const float* values = tensor.ReadData<float>();
        result[name].assign(values, values + tensor.NumElements());
    }
    return result;
}

std::string Describe(const Parameters& parameters) {
    std::string text;
    for (const auto& [name, values] : parameters) {
        text += "  \"" + name + "\": [";
        for (size_t i = 0; i < values.size(); ++i) {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%s%.9g", i ? ", " : "", values[i]);
            text += buffer;
        }
        text += "]\n";
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

gui::NodeType NodeTypeNamed(const std::string& name) {
    if (name == "L1Regularization") return gui::NodeType::L1Regularization;
    if (name == "L2Regularization") return gui::NodeType::L2Regularization;
    Check(name == "ElasticNet", "fixture node " + name);
    return gui::NodeType::ElasticNet;
}

void CheckCompiledCoefficients() {
    auto config = Compile(BuildGraph({Penalty(gui::NodeType::L1Regularization, 11, {{"lambda", "0.05"}})}, 0.0625));
    Check(config.is_valid && config.regularization_l1 == 0.05f && config.regularization_l2 == 0.0f &&
              config.regularization_node_id == 11,
          "L1 compiles to l1 = lambda");
    config = Compile(BuildGraph({Penalty(gui::NodeType::L2Regularization, 11, {{"lambda", "0.05"}})}, 0.0625));
    Check(config.is_valid && config.regularization_l1 == 0.0f && config.regularization_l2 == 0.05f,
          "L2 compiles to l2 = lambda");
    config = Compile(BuildGraph({Penalty(gui::NodeType::ElasticNet, 11, {{"lambda", "0.1"}, {"l1_ratio", "0.25"}})},
                                0.0625));
    Check(config.is_valid && config.regularization_l1 == 0.025f && config.regularization_l2 == 0.075f,
          "Elastic Net compiles to l1 = lambda x l1_ratio, l2 = lambda x (1 - l1_ratio)");
    config = Compile(BuildGraph({}, 0.0625));
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid && config.regularization_l1 == 0.0f && config.regularization_l2 == 0.0f &&
              config.regularization_node_id == -1,
          "no penalty node: no penalty");
}

void CheckRefusals() {
    CheckRefused(BuildGraph({Penalty(gui::NodeType::L2Regularization, 11, {})}, 0.0625, true),
                 "must sit between the training loss and the optimizer", "a penalty off the loss-optimizer wire");
    CheckRefused(BuildGraph({Penalty(gui::NodeType::L1Regularization, 11, {}),
                             Penalty(gui::NodeType::L2Regularization, 12, {})},
                            0.0625),
                 "must sit between the training loss and the optimizer", "two chained penalties");
    {
        // The SGD node: Nesterov needs momentum (BuildGraph sets momentum 0).
        Graph nesterov = BuildGraph({}, 0.0625);
        for (auto& node : nesterov.nodes)
            if (node.type == gui::NodeType::SGD) node.parameters["nesterov"] = "true";
        CheckRefused(nesterov, "set momentum above 0", "Nesterov without momentum");
    }
    CheckRefused(BuildGraph({Penalty(gui::NodeType::L1Regularization, 11, {{"lambda", "-1"}})}, 0.0625),
                 "must be a number from 0 to 1000000", "a negative lambda");
    CheckRefused(BuildGraph({Penalty(gui::NodeType::ElasticNet, 11, {{"l1_ratio", "2"}})}, 0.0625),
                 "must be a number from 0 to 1", "l1_ratio 2");
    CheckRefused(BuildGraph({Penalty(gui::NodeType::L2Regularization, 11, {{"lambda", "strong"}})}, 0.0625),
                 "'strong' must be a number", "a lambda that is not a number");
}

}  // namespace

// TOFIX178 6.1b: layer snapshots during training leave the training as it
// was: the trained parameters are bit for bit the run's without them (and so
// still torch's), and the last snapshot holds the trained weights.
void CheckLayerSnapshots(const json& fixture, const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
                         double learning_rate, int batch_size, int epochs, double tolerance) {
    const json& plain_case = fixture.at("cases").at(0);
    Check(!plain_case.contains("node"), "the first fixture case is the plain run");
    const auto config = CompileValid(BuildGraph({}, learning_rate), batch_size, "snapshots");
    const auto without = TrainAndRead(config, dataset, epochs, "without snapshots");

    cyxwiz::TrainingExecutor::LayerSnapshotSchedule schedule;
    schedule.every_epochs = 2;
    const float row[2] = {fixture.at("x0").at(0).get<float>(), fixture.at("x1").at(0).get<float>()};
    const float target[1] = {fixture.at("label").at(0).get<float>()};
    schedule.features = cyxwiz::Tensor({1, 2}, row, cyxwiz::DataType::Float32);
    schedule.labels = cyxwiz::Tensor({1, 1}, target, cyxwiz::DataType::Float32);
    std::vector<cyxwiz::TrainingExecutor::LayerSnapshot> taken;
    const auto with = TrainAndRead(config, dataset, epochs, "with snapshots", schedule, &taken);
    Check(with == without, "snapshots leave the trained parameters bit for bit as without them");
    CheckParameters(with, plain_case.at("parameters_after"), tolerance, "with snapshots: trained parameters match torch");

    std::vector<int> expected;
    for (int e = 2; e <= epochs; e += 2) expected.push_back(e);
    if (expected.empty() || expected.back() != epochs) expected.push_back(epochs);
    Check(taken.size() == expected.size(), "one snapshot every 2 epochs and after the last: " +
                                               std::to_string(taken.size()) + " of " + std::to_string(expected.size()));
    // A saved snapshot keeps no tensors in memory; it reads back from the run store.
    const auto saved = [](const cyxwiz::TrainingExecutor::LayerSnapshot& snapshot) {
        std::string why;
        auto capture = cyxwiz::LoadLayerCapture(snapshot.run_id, snapshot.label, &why);
        Check(!snapshot.capture && capture != nullptr,
              "the snapshot after epoch " + std::to_string(snapshot.epoch) + " is on disk, not held in memory: " + why);
        if (!capture) throw std::runtime_error("a saved snapshot did not read back");
        return capture;
    };
    for (size_t i = 0; i < taken.size(); ++i) {
        Check(taken[i].epoch == expected[i], "snapshot " + std::to_string(i) + " after epoch " + std::to_string(expected[i]));
        const auto capture = saved(taken[i]);
        const auto& layers = capture->Layers();
        Check(layers.size() == 1 && layers[0].node_name == "Dense", "the snapshot holds the Dense layer");
        const auto* input = layers[0].Find(cyxwiz::CapturedTensorRole::Input);
        Check(input && input->tensor.Shape() == std::vector<size_t>({1, 2}), "the snapshot ran the given row");
        Check(layers[0].Find(cyxwiz::CapturedTensorRole::Gradient, "weight") != nullptr, "the snapshot holds the gradient");
    }
    // The last snapshot, after the last epoch, holds the trained weights.
    const auto last = saved(taken.back());
    for (const auto& [name, values] : with) {
        const auto field = name.substr(name.find('.') + 1);
        const auto* p = last->Layers()[0].Find(cyxwiz::CapturedTensorRole::Parameter, field);
        Check(p != nullptr, "the last snapshot holds " + name);
        const float* data = p->tensor.ReadData<float>();
        Check(std::vector<double>(data, data + p->tensor.NumElements()) == values,
              "the last snapshot's " + name + " is the trained one");
    }

    // Saved with the run as they are taken (6.1c): one capture per snapshot,
    // read back tensor for tensor.
    const std::string run_id = taken.back().run_id;
    Check(!run_id.empty(), "the snapshots are saved with the training run");
    std::vector<std::string> labels;
    for (const auto& snapshot : taken) {
        Check(snapshot.run_id == run_id && snapshot.label == cyxwiz::EpochCaptureLabel(snapshot.epoch),
              "snapshot after epoch " + std::to_string(snapshot.epoch) + " is saved under its epoch");
        labels.push_back(snapshot.label);
    }
    Check(cyxwiz::ListLayerCaptures(run_id) == labels, "the run lists its saved snapshots");
    // Each epoch's metrics are saved with the snapshots (6.3d), the last
    // snapshot's save holding every epoch.
    const auto record = cyxwiz::LoadTrainingRunRecord(run_id);
    Check(record.has_value(), "the run's epochs are saved with its snapshots");
    Check(record->total_epochs == epochs && static_cast<int>(record->epochs.size()) == epochs,
          "every epoch is in the saved record");
    for (int e = 0; e < epochs; ++e) {
        Check(record->epochs[static_cast<size_t>(e)].epoch == e + 1, "the record is in epoch order");
        Check(std::isfinite(record->epochs[static_cast<size_t>(e)].train_loss) &&
                  record->epochs[static_cast<size_t>(e)].train_loss > 0.0f,
              "each epoch's training loss is recorded");
    }
    Check(!record->label_values.empty() && !record->label_shape.empty(),
          "the snapshot items' labels are saved with the run");
    Check(cyxwiz::ListRunsWithLayerCaptures().front() == run_id, "the training run is listed first, newest");
    const auto infos = cyxwiz::ListLayerCaptureInfo(run_id);
    Check(infos.size() == labels.size() && infos.back().epoch == taken.back().epoch && infos.back().bytes > 0,
          "each snapshot is listed with its epoch and its size on disk");
    // On items by number from the run's own batchers: rows 0 and 1, as
    // validation reads them.
    cyxwiz::TrainingExecutor::LayerSnapshotSchedule by_number;
    by_number.every_epochs = epochs;
    by_number.source_index = 0;
    by_number.source_count = 2;
    by_number.save = false;
    std::vector<cyxwiz::TrainingExecutor::LayerSnapshot> numbered;
    const auto with_numbered = TrainAndRead(config, dataset, epochs, "with snapshots by number", by_number, &numbered);
    Check(with_numbered == without, "snapshots on items by number leave the training as it was");
    Check(numbered.size() == 1 && numbered[0].run_id.empty(), "one unsaved snapshot after the last epoch");
    const auto* rows = numbered[0].capture->Layers()[0].Find(cyxwiz::CapturedTensorRole::Input);
    Check(rows && rows->tensor.Shape() == std::vector<size_t>({2, 2}), "the snapshot ran rows 0 and 1");
    const float* r = rows->tensor.ReadData<float>();
    for (size_t i = 0; i < 2; ++i)
        Check(r[2 * i] == fixture.at("x0").at(i).get<float>() && r[2 * i + 1] == fixture.at("x1").at(i).get<float>(),
              "snapshot row " + std::to_string(i) + " is the dataset's row " + std::to_string(i));
    std::cout << "Layer snapshots: " << taken.size() << " taken and saved, training unchanged\n";
}

int main(int, char** argv) {
    namespace fs = std::filesystem;
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_regularization_node_training";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(work_dir / "debug_runs");

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "regularization_node_pytorch.json is readable");
    const json fixture = json::parse(in);
    const double tolerance = fixture.at("tolerance").get<double>();
    const double learning_rate = fixture.at("learning_rate").get<double>();
    const int batch_size = fixture.at("batch_size").get<int>();
    const int epochs = fixture.at("epochs").get<int>();

    CheckCompiledCoefficients();
    CheckRefusals();
    const auto dataset = MakeDataset(fixture);

    // The start: the Engine's seeded initialisation (learning rate 0 leaves
    // it untouched) is the one torch starts from.
    auto still = CompileValid(BuildGraph({}, learning_rate), batch_size, "the start");
    still.learning_rate = 0.0f;
    const auto start = TrainAndRead(still, dataset, 1, "the start");
    // 9 significant digits round-trip a float exactly.
    CheckParameters(start, fixture.at("initial_parameters"), 1.0e-8,
                    "the Engine's seed " + std::to_string(kModelSeed) +
                        " initialisation is the fixture's (if it changed, regenerate the fixture from these values)");

    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("name").get<std::string>();
        std::vector<gui::MLNode> penalties;
        if (c.contains("node")) {
            penalties.push_back(
                Penalty(NodeTypeNamed(c.at("node")), 11, c.at("parameters").get<std::map<std::string, std::string>>()));
        }
        Graph graph = BuildGraph(penalties, learning_rate);
        if (c.contains("optimizer"))  // the SGD node's own settings: momentum, Nesterov, weight decay
            for (auto& node : graph.nodes)
                if (node.type == gui::NodeType::SGD)
                    for (const auto& [key, value] : c.at("optimizer").get<std::map<std::string, std::string>>())
                        node.parameters[key] = value;
        const auto config = CompileValid(graph, batch_size, name);
        CheckParameters(TrainAndRead(config, dataset, epochs, name), c.at("parameters_after"), tolerance,
                        name + ": trained parameters match torch");
    }

    CheckLayerSnapshots(fixture, dataset, learning_rate, batch_size, epochs, tolerance);

    fs::remove_all(work_dir);
    std::cout << "Regularization nodes match PyTorch: " << fixture.at("cases").size() << " cases\n";
    return 0;
}
