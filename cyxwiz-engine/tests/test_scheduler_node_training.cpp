// Scheduler nodes through the Engine's own path (TOFIX140 A3), against
// PyTorch: Data Input -> Dense -> MSE -> SGD -> scheduler node. GraphCompiler
// turns the node into TrainingConfiguration::scheduler, TrainingExecutor
// attaches and steps it, and its learning-rate history matches
// fixtures/scheduler_node_pytorch.json (generate_scheduler_node_fixtures.py).
// Reduce LR steps on the run's own validation losses, so the test replays them
// through the backend ReduceLROnPlateau (held to torch by test_scheduler).
// Also: the compiler's refusals, and resume continuing the schedule.
#include "../src/core/arrow_dataset.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/training_executor.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>
#include <cyxwiz/optimizers/sgd.h>
#include <cyxwiz/scheduler.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void CheckNear(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual << " expected=" << expected << "\n";
        std::exit(1);
    }
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "scheduler_node_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_SCHEDULER_NODE_FIXTURE;
}

std::shared_ptr<arrow::Array> Floats(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) Check(builder.Append(value).ok(), "append");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish");
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    auto schema = arrow::schema({arrow::field("x0", arrow::float32()), arrow::field("x1", arrow::float32()),
                                 arrow::field("label", arrow::float32())});
    auto table = arrow::Table::Make(
        schema,
        {Floats({0.0f, 0.1f, 0.9f, 1.0f, 0.2f, 0.8f, 0.4f, 0.6f}),
         Floats({0.0f, 0.2f, 0.8f, 1.0f, 0.1f, 0.9f, 0.5f, 0.3f}),
         Floats({0.0f, 0.3f, 1.7f, 2.0f, 0.3f, 1.7f, 0.9f, 0.9f})},
        8);
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "scheduler_rows");
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

// The scheduler comes from the node factory: its pins and defaults are the
// contract. `id` 11 (input pin 1101) is linked from SGD's State output.
gui::MLNode Scheduler(gui::NodeType type, int id, const std::map<std::string, std::string>& params) {
    int next_node_id = id, next_pin_id = 0;
    auto node = gui::CreateGraphNode(type, "Scheduler " + std::to_string(id), next_node_id, next_pin_id);
    Check(node.inputs.size() == 1 && node.inputs[0].type == gui::PinType::Optimizer && node.outputs.empty(),
          "scheduler node: one Optimizer input, no output");
    node.id = id;
    node.inputs[0].id = id * 100 + 1;
    for (const auto& [key, value] : params) {
        Check(node.parameters.count(key) == 1, "scheduler parameter exists: " + key);
        node.parameters[key] = value;
    }
    return node;
}

Graph BuildGraph(const std::vector<gui::MLNode>& schedulers, bool link_schedulers = true) {
    Graph g;
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "scheduler_rows"}, {"shape", "[2]"}};
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
    sgd.parameters = {{"learning_rate", "0.0625"}, {"momentum", "0"}};
    g.nodes = {data, dense, loss, sgd};
    g.links = {Link(1, 1, 111, 3, 301), Link(2, 3, 311, 9, 901), Link(3, 1, 112, 9, 902), Link(4, 9, 911, 10, 1001)};
    for (const auto& scheduler : schedulers) {
        g.nodes.push_back(scheduler);
        if (link_schedulers) {
            g.links.push_back(Link(100 + scheduler.id, 10, 1011, scheduler.id, scheduler.inputs[0].id));
        }
    }
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

cyxwiz::TrainingConfiguration CompileValid(const Graph& graph, const std::string& what) {
    auto config = Compile(graph);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, what + " compiles");
    // The run itself: 4 training rows in batches of 2, 4 validation rows.
    config.dataset_name = "scheduler_rows";
    config.batch_size = 2;
    config.train_ratio = 0.5f;
    config.val_ratio = 0.5f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.checkpoint_dir = (std::filesystem::temp_directory_path() / "cyxwiz_scheduler_node_checkpoints").string();
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

cyxwiz::TrainingMetrics Train(cyxwiz::TrainingExecutor& executor, int epochs) {
    SelectArrayFireCpu();
    cyxwiz::TrainingMetrics final_metrics;
    bool completed = false;
    executor.Train(epochs, 2, nullptr, nullptr, [&](const cyxwiz::TrainingMetrics& metrics) {
        final_metrics = metrics;
        completed = true;
    });
    // A failed run reports through the executor's metrics, not the callback.
    return completed ? final_metrics : executor.GetMetrics();
}

void CheckCompleted(const cyxwiz::TrainingMetrics& metrics, const std::string& what) {
    Check(metrics.terminal_status == "completed" && metrics.terminal_reason == "completed_all_epochs",
          what + " completes (" + metrics.terminal_status + ": " + metrics.terminal_reason + ")");
}

gui::NodeType NodeTypeNamed(const std::string& name) {
    if (name == "StepLR") return gui::NodeType::StepLR;
    if (name == "CosineAnnealing") return gui::NodeType::CosineAnnealing;
    if (name == "ExponentialLR") return gui::NodeType::ExponentialLR;
    Check(name == "WarmupScheduler", "fixture node " + name);
    return gui::NodeType::WarmupScheduler;
}

// Each epoch scheduler against torch, from the graph node to the trained run.
void CheckPyTorchCase(const json& c, const std::shared_ptr<cyxwiz::ArrowDataset>& dataset, double tolerance) {
    const std::string name = c.at("name").get<std::string>();
    const auto params = c.at("parameters").get<std::map<std::string, std::string>>();
    const auto expected = c.at("learning_rate_history").get<std::vector<double>>();
    auto config = CompileValid(BuildGraph({Scheduler(NodeTypeNamed(c.at("node")), 11, params)}), name);
    Check(config.scheduler.has_value() && config.scheduler_node_id == 11, name + ": the node is the run's scheduler");
    Check(cyxwiz::GetTrainingSchedulerCadence(*config.scheduler) == cyxwiz::TrainingSchedulerCadence::CompletedEpoch,
          name + ": stepped after each completed epoch");

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    const auto metrics = Train(executor, static_cast<int>(expected.size()));
    CheckCompleted(metrics, name);
    Check(metrics.learning_rate_history.size() == expected.size(), name + ": one learning rate per epoch");
    for (size_t i = 0; i < expected.size(); ++i) {
        CheckNear(metrics.learning_rate_history[i], expected[i], tolerance,
                  name + ": learning rate after epoch " + std::to_string(i + 1) + " matches torch");
    }
}

void CheckCompiledSpecs() {
    auto config = Compile(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, {{"step_size", "3"}, {"gamma", "0.5"}})}));
    const auto* step = config.scheduler ? std::get_if<cyxwiz::StepLRSchedulerSpec>(&*config.scheduler) : nullptr;
    Check(config.is_valid && step && step->step_size == 3 && step->gamma == 0.5, "Step LR compiles to StepLR(3, 0.5)");

    config = Compile(BuildGraph({Scheduler(gui::NodeType::ReduceOnPlateau, 11,
                                           {{"factor", "0.5"}, {"patience", "2"}, {"threshold", "0.01"}, {"min_lr", "0.001"}})}));
    const auto* plateau =
        config.scheduler ? std::get_if<cyxwiz::ReduceLROnPlateauSchedulerSpec>(&*config.scheduler) : nullptr;
    Check(config.is_valid && plateau && plateau->mode == "min" && plateau->factor == 0.5 && plateau->patience == 2 &&
              plateau->threshold == 0.01 && plateau->min_lr == 0.001,
          "Reduce LR compiles to ReduceLROnPlateau(min, 0.5, 2, 0.01, 0.001)");

    config = Compile(BuildGraph({Scheduler(gui::NodeType::WarmupScheduler, 11, {{"warmup_epochs", "4"}, {"start_factor", "0.5"}})}));
    const auto* warmup = config.scheduler ? std::get_if<cyxwiz::LinearWarmupLRSchedulerSpec>(&*config.scheduler) : nullptr;
    Check(config.is_valid && warmup && warmup->warmup_epochs == 4 && warmup->base_lr == 0.0625f &&
              warmup->start_lr == 0.5 * 0.0625f,
          "Warmup LR compiles to LinearWarmupLR(4, learning_rate, 0.5 x learning_rate)");

    config = Compile(BuildGraph({}));
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid && !config.scheduler.has_value() && config.scheduler_node_id == -1,
          "no scheduler node: no scheduler");
}

void CheckRefusals() {
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, {})}, false), "not connected to the training optimizer",
                 "an unlinked scheduler");
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, {}), Scheduler(gui::NodeType::ExponentialLR, 12, {})}),
                 "one scheduler per optimizer", "two schedulers");
    auto conflict = BuildGraph({Scheduler(gui::NodeType::CosineAnnealing, 11, {})});
    for (auto& node : conflict.nodes) {
        if (node.type == gui::NodeType::SGD) node.parameters["lr_schedule"] = "warmup_cosine";
    }
    CheckRefused(conflict, "both set the learning rate", "a scheduler with the optimizer's lr_schedule");
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, {{"step_size", "0"}})}), "positive step_size",
                 "step_size 0");
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, {{"gamma", "fast"}})}), "is not a number",
                 "a gamma that is not a number");
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::WarmupScheduler, 11, {{"start_factor", "0"}})}),
                 "above 0 and at most 1", "start_factor 0");
    CheckRefused(BuildGraph({Scheduler(gui::NodeType::ReduceOnPlateau, 11, {{"factor", "1"}})}), "factor in [0,1)",
                 "factor 1");
}

// Reduce LR: stepped on each validated epoch's loss; the backend scheduler
// given the same losses must give the same rates.
void CheckPlateau(const std::shared_ptr<cyxwiz::ArrowDataset>& dataset) {
    auto config = CompileValid(
        BuildGraph({Scheduler(gui::NodeType::ReduceOnPlateau, 11,
                              {{"factor", "0.5"}, {"patience", "0"}, {"threshold", "0.05"}, {"min_lr", "0.01"}})}),
        "Reduce LR");
    Check(cyxwiz::GetTrainingSchedulerCadence(*config.scheduler) == cyxwiz::TrainingSchedulerCadence::ValidatedEpoch,
          "Reduce LR steps after validated epochs");
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    const auto metrics = Train(executor, 6);
    CheckCompleted(metrics, "Reduce LR");
    Check(metrics.val_loss_history.size() == 6 && metrics.learning_rate_history.size() == 6,
          "Reduce LR: one validation loss and one rate per epoch");

    cyxwiz::SGDOptimizer optimizer(0.0625);
    cyxwiz::ReduceLROnPlateau replay(&optimizer, "min", 0.5, 0, 0.05, 0.01);
    for (size_t i = 0; i < metrics.val_loss_history.size(); ++i) {
        replay.Step(static_cast<int>(i + 1), metrics.val_loss_history[i]);
        CheckNear(metrics.learning_rate_history[i], optimizer.GetLearningRate(), 1e-12,
                  "Reduce LR rate after epoch " + std::to_string(i + 1));
    }
}

// Resume continues the node's schedule from the checkpoint: epochs 4-6 of a
// resumed run give torch's rates for epochs 4-6.
void CheckResume(const json& step_case, const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
                 const std::filesystem::path& work_dir, double tolerance) {
    const auto params = step_case.at("parameters").get<std::map<std::string, std::string>>();
    const auto expected = step_case.at("learning_rate_history").get<std::vector<double>>();
    const auto config = CompileValid(BuildGraph({Scheduler(gui::NodeType::StepLR, 11, params)}), "resume");
    cyxwiz::TrainingResumeIdentity identity;
    identity.run_id = "scheduler-node-resume";

    std::map<int, std::filesystem::path> checkpoints;
    {
        cyxwiz::TrainingExecutor first(config, dataset, "label");
        first.EnableResumeCheckpoints(work_dir / "resume", identity, 10);
        first.SetResumeCheckpointCallback([&](const std::filesystem::path& path, int epoch, int) {
            checkpoints[epoch] = path;
        });
        CheckCompleted(Train(first, 3), "the first three epochs");
    }
    Check(checkpoints.count(3) == 1, "a resume checkpoint after epoch 3");

    cyxwiz::TrainingExecutor resumed(config, dataset, "label");
    resumed.SetResumeFrom(checkpoints.at(3));
    const auto metrics = Train(resumed, 6);
    CheckCompleted(metrics, "the resumed run");
    // The history may carry the checkpoint's first three rates; its last three
    // are the resumed epochs.
    const auto& history = metrics.learning_rate_history;
    Check(history.size() >= 3, "the resumed run records its three epochs");
    for (size_t i = 3; i < expected.size(); ++i) {
        CheckNear(history[history.size() - expected.size() + i], expected[i], tolerance,
                  "resumed rate after epoch " + std::to_string(i + 1) + " matches torch");
    }

    // A checkpoint without scheduler state cannot continue a run that has one.
    std::map<int, std::filesystem::path> plain_checkpoints;
    {
        auto plain = CompileValid(BuildGraph({}), "no scheduler");
        cyxwiz::TrainingExecutor first(plain, dataset, "label");
        first.EnableResumeCheckpoints(work_dir / "plain", identity, 10);
        first.SetResumeCheckpointCallback([&](const std::filesystem::path& path, int epoch, int) {
            plain_checkpoints[epoch] = path;
        });
        CheckCompleted(Train(first, 1), "a run without a scheduler");
    }
    cyxwiz::TrainingExecutor mismatched(config, dataset, "label");
    mismatched.SetResumeFrom(plain_checkpoints.at(1));
    const auto refused = Train(mismatched, 3);
    Check(refused.terminal_status != "completed" &&
              refused.terminal_reason.find("no scheduler state") != std::string::npos,
          "resuming a scheduler run from a checkpoint without one is refused: " + refused.terminal_status + ": " +
              refused.terminal_reason);
}

}  // namespace

int main(int, char** argv) {
    namespace fs = std::filesystem;
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_scheduler_node_training";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(work_dir / "debug_runs");

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "scheduler_node_pytorch.json is readable");
    const json fixture = json::parse(in);
    const double tolerance = fixture.at("tolerance").get<double>();

    CheckCompiledSpecs();
    CheckRefusals();
    const auto dataset = MakeDataset();
    for (const auto& c : fixture.at("cases")) CheckPyTorchCase(c, dataset, tolerance);
    CheckPlateau(dataset);
    CheckResume(fixture.at("cases").at(0), dataset, work_dir, tolerance);

    fs::remove_all(work_dir);
    std::cout << "Scheduler nodes match PyTorch: " << fixture.at("cases").size() << " cases, plateau, resume\n";
    return 0;
}
