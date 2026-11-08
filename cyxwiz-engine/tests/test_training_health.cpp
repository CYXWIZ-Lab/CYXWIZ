// Training health (TOFIX129 D11).
// 1. The recorder against numpy / torch (generate_training_health_fixtures.py):
//    ||W||, the 24-bin histogram, the mean of the steps' gradient norms, and
//    dead units of an image ReLU (channels of [H, W, C, N]) and of a dense
//    ReLU ([N, F]) over two validation batches.
// 2. Through TrainingExecutor: Dense 16 -> ReLU -> Dense 3 -> cross entropy
//    trains three epochs; health has one entry an epoch, ||W|| of each
//    weight at the end is the trained model's, gradients were recorded,
//    the ReLU's dead units are counted on the validation pass, and a range
//    test records none.
#include "../src/core/arrow_dataset.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_health.h"
#include "../src/core/training_health_presentation.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>
#include <cyxwiz/sequential.h>

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

bool Near(double a, double b, double relative = 1e-5) {
    return std::abs(a - b) <= relative * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

cyxwiz::Tensor ReadTensor(const json& value) {
    const auto shape = value.at("shape").get<std::vector<size_t>>();
    const auto values = value.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data());
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "training_health_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_TRAINING_HEALTH_FIXTURE;
}

void CheckRecorder(const json& f) {
    cyxwiz::TrainingHealthRecorder recorder;
    const std::vector<std::pair<std::string, std::string>> weights = {{"conv", "layer0.weights"}, {"dense", "layer3.weight"}};
    for (int step = 0; step < 3; ++step) {
        std::map<std::string, cyxwiz::Tensor> gradients;
        for (const auto& [key, name] : weights) gradients[name] = ReadTensor(f.at(key).at("gradients").at(step));
        gradients["layer3.bias"] = cyxwiz::Tensor(std::vector<size_t>{10});  // a bias is not a weight
        recorder.OnStep(gradients);
    }
    const cyxwiz::ReLUModule relu;
    const cyxwiz::LeakyReLUModule leaky(0.01f);
    for (const auto& batch : f.at("relu_image").at("batches")) recorder.OnValidationOutput(1, relu, ReadTensor(batch));
    for (const auto& batch : f.at("relu_dense").at("batches")) recorder.OnValidationOutput(4, relu, ReadTensor(batch));
    recorder.OnValidationOutput(6, leaky, ReadTensor(f.at("relu_dense").at("batches").at(0)));  // not a ReLU

    std::map<std::string, cyxwiz::Tensor> parameters;
    for (const auto& [key, name] : weights) parameters[name] = ReadTensor(f.at(key).at("weight"));
    parameters["layer3.bias"] = cyxwiz::Tensor(std::vector<size_t>{10});
    const auto health = recorder.EndEpoch(1, parameters);
    Check(health.epoch == 1 && health.weights.size() == 2, "two weight tensors, the bias left out");
    for (const auto& [key, name] : weights) {
        const auto found = std::find_if(health.weights.begin(), health.weights.end(),
                                        [&](const cyxwiz::WeightHealth& w) { return w.parameter == name; });
        Check(found != health.weights.end(), key + ": recorded");
        if (found == health.weights.end()) continue;
        const json& truth = f.at(key);
        Check(Near(found->norm, truth.at("norm")), key + ": ||W|| " + std::to_string(found->norm));
        Check(Near(found->mean_gradient_norm, truth.at("mean_gradient_norm")),
              key + ": mean gradient norm " + std::to_string(found->mean_gradient_norm));
        Check(Near(found->low, truth.at("low")) && Near(found->high, truth.at("high")), key + ": histogram range");
        const auto bins = truth.at("histogram").get<std::vector<long long>>();
        Check(found->histogram == bins, key + ": the 24 bins are numpy's");
    }
    Check(health.relus.size() == 2, "two ReLUs (the LeakyReLU is not one)");
    for (const auto& r : health.relus) {
        const json& truth = f.at(r.layer == 1 ? "relu_image" : "relu_dense");
        Check(r.dead == truth.at("dead").get<long long>() && r.units == truth.at("units").get<long long>(),
              "ReLU " + std::to_string(r.layer) + ": " + std::to_string(r.dead) + " of " + std::to_string(r.units) + " dead");
    }
    const auto next = recorder.EndEpoch(2, parameters);
    Check(next.relus.empty() && next.weights.size() == 2 && next.weights[0].mean_gradient_norm == 0.0,
          "the next epoch starts afresh");
    Check(cyxwiz::HealthLayerIndex("layer12.weights") == 12 && cyxwiz::HealthLayerIndex("head.weight") == std::string::npos,
          "layer index from the parameter name");
}

// ---- Through the executor ------------------------------------------------------

std::shared_ptr<arrow::Array> Floats(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Array> Ints(const std::vector<int32_t>& values) {
    arrow::Int32Builder builder;
    for (int32_t value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    std::vector<float> x0, x1, x2, x3;
    std::vector<int32_t> label;
    for (int i = 0; i < 90; ++i) {
        const int c = i % 3;
        x0.push_back(static_cast<float>(c) + 0.3f * std::sin(0.7f * static_cast<float>(i)));
        x1.push_back(static_cast<float>(2 - c) + 0.3f * std::cos(0.5f * static_cast<float>(i)));
        x2.push_back(0.1f * static_cast<float>(i % 7));
        x3.push_back(0.2f * std::sin(static_cast<float>(i)));
        label.push_back(c);
    }
    auto schema = arrow::schema({arrow::field("x0", arrow::float32()), arrow::field("x1", arrow::float32()),
                                 arrow::field("x2", arrow::float32()), arrow::field("x3", arrow::float32()),
                                 arrow::field("label", arrow::int32())});
    auto table = arrow::Table::Make(schema, {Floats(x0), Floats(x1), Floats(x2), Floats(x3), Ints(label)}, 90);
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "health_rows");
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

gui::MLNode Layer(int id, gui::NodeType type, const std::string& name, std::map<std::string, std::string> parameters) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 11, gui::PinType::Tensor, "Output", false)};
    node.parameters = std::move(parameters);
    return node;
}

cyxwiz::TrainingConfiguration CompileGraph() {
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "health_rows"}, {"shape", "[4]"}};
    const auto hidden = Layer(3, gui::NodeType::Dense, "Hidden", {{"units", "16"}});
    const auto relu = Layer(4, gui::NodeType::ReLU, "ReLU", {});
    const auto head = Layer(5, gui::NodeType::Dense, "Head", {{"units", "3"}});
    gui::MLNode loss;
    loss.id = 9;
    loss.type = gui::NodeType::CrossEntropyLoss;
    loss.name = "Loss";
    loss.inputs = {Pin(901, gui::PinType::Tensor, "Predictions", true), Pin(902, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(911, gui::PinType::Loss, "Loss", false)};
    gui::MLNode adam;
    adam.id = 10;
    adam.type = gui::NodeType::Adam;
    adam.name = "Adam";
    adam.inputs = {Pin(1001, gui::PinType::Loss, "Loss", true)};
    adam.parameters = {{"learning_rate", "0.01"}};
    const std::vector<gui::MLNode> nodes = {data, hidden, relu, head, loss, adam};
    const std::vector<gui::NodeLink> links = {Link(1, 1, 111, 3, 301), Link(2, 3, 311, 4, 401), Link(3, 4, 411, 5, 501),
                                              Link(4, 5, 511, 9, 901), Link(5, 1, 112, 9, 902), Link(6, 9, 911, 10, 1001)};
    auto config = cyxwiz::GraphCompiler{}.Compile(nodes, links, true);
    for (const auto& issue : config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
    Check(config.is_valid, "the health graph compiles");
    config.dataset_name = "health_rows";
    config.batch_size = 10;
    config.train_ratio = 0.7f;
    config.val_ratio = 0.3f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.log_interval = 0;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = (std::filesystem::temp_directory_path() / "cyxwiz_health_checkpoints").string();
    return config;
}

void SelectArrayFireCpu() {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = std::find_if(devices.begin(), devices.end(),
                                  [](const cyxwiz::DeviceInfo& device) { return device.type == cyxwiz::DeviceType::CPU; });
    Check(cpu != devices.end(), "an ArrayFire CPU route is required");
    cyxwiz::test::InstallQualifiedRouteSnapshot(devices);
    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(cyxwiz::DeviceType::CPU, cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(cyxwiz::ExecutionPolicy::Strict);
}

void CheckExecutor() {
    SelectArrayFireCpu();
    auto config = CompileGraph();
    cyxwiz::TrainingExecutor executor(config, MakeDataset(), "label");
    executor.Train(3, 10, nullptr, nullptr, nullptr);
    const auto metrics = executor.GetMetrics();
    const auto& health = metrics.health;
    Check(health.epochs.size() == 3, "health once an epoch: " + std::to_string(health.epochs.size()));
    if (health.epochs.size() != 3) return;
    auto* model = executor.GetModel();
    Check(model != nullptr, "a layer stack");
    if (!model) return;
    const auto parameters = model->GetParameters();
    const auto& last = health.epochs.back();
    Check(last.epoch == 3 && last.weights.size() == 2, "two weight tensors (Hidden, Head)");
    for (const auto& w : last.weights) {
        const auto found = parameters.find(w.parameter);
        Check(found != parameters.end(), "weight " + w.parameter + " is a parameter of the model");
        if (found == parameters.end()) continue;
        const float* v = found->second.Data<float>();
        double sum = 0.0;
        for (size_t i = 0; i < found->second.NumElements(); ++i) sum += static_cast<double>(v[i]) * v[i];
        Check(Near(w.norm, std::sqrt(sum)), w.parameter + ": ||W|| at the end is the model's");
        long long total = 0;
        for (long long c : w.histogram) total += c;
        Check(total == static_cast<long long>(w.count) && w.histogram.size() == 24, w.parameter + ": the histogram counts every weight");
        Check(w.mean_gradient_norm > 0.0, w.parameter + ": gradients were recorded");
    }
    for (const auto& epoch : health.epochs) {
        Check(epoch.relus.size() == 1 && epoch.relus[0].units == 16 && epoch.relus[0].dead >= 0 && epoch.relus[0].dead <= 16,
              "epoch " + std::to_string(epoch.epoch) + ": the ReLU's 16 units checked on the validation pass");
    }
    Check(health.epochs[0].weights[0].norm != health.epochs[2].weights[0].norm, "the weights change from epoch to epoch");

    // A range test records no health.
    auto range = CompileGraph();
    range.learning_rate_range.enabled = true;
    range.learning_rate_range.steps = 4;
    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor range_executor(range, MakeDataset(), "label");
    range_executor.Train(4, 10, nullptr, nullptr, nullptr);
    Check(range_executor.GetMetrics().health.epochs.empty(), "a range test records no health");
}

// The Health tab's words on a record shaped like the MNIST CNN's: Conv2D 32
// (layer 0, ReLU 1), Conv2D 64 (layer 3, ReLU 4), Dense 128 (layer 7, ReLU 8),
// Dense 10 (layer 10, no ReLU), three epochs.
void CheckView() {
    cyxwiz::TrainingHealth health;
    const long long dense_dead[3] = {48, 46, 46};
    for (int e = 0; e < 3; ++e) {
        cyxwiz::EpochHealth epoch;
        epoch.epoch = e + 1;
        const auto weight = [&](size_t layer, const char* name, double norm, double grad, size_t count) {
            cyxwiz::WeightHealth w;
            w.parameter = name;
            w.layer = layer;
            w.norm = norm + 0.1 * e;
            w.mean_gradient_norm = grad;
            w.histogram.assign(24, 0);
            w.histogram[12] = static_cast<long long>(count);
            w.low = -0.2;
            w.high = 0.25;
            w.count = count;
            epoch.weights.push_back(w);
        };
        weight(0, "layer0.weights", 3.74, 0.152, 288);
        weight(3, "layer3.weights", 6.52, 0.348, 18432);
        weight(7, "layer7.weight", 8.64, 1.43, 204800);
        weight(10, "layer10.weight", 2.10, 1.74, 1280);
        epoch.relus = {{1, 0, 32}, {4, 15, 64}, {8, dense_dead[e], 128}};
        health.epochs.push_back(epoch);
    }
    std::vector<std::string> names(11);
    names[0] = "Conv2D 32";
    names[3] = "Conv2D 64";
    names[7] = "Dense 128";
    names[10] = "Dense 10";
    const auto view = cyxwiz::BuildHealthView(health, names);
    Check(view.has_health && view.rows.size() == 4 && view.epochs == "3 epochs", "four weight layers over 3 epochs");
    Check(view.warning && view.headline == "46 of Dense 128's 128 units are dead: they give 0 for every validation row.",
          "headline: " + view.headline);
    Check(view.detail == "A unit that never fires gets no gradient, so it stops learning. 15 of Conv2D 64's 64 too. "
                         "Gradients stay in range (no exploding or vanishing).",
          "detail: " + view.detail);
    Check(view.advice == "LeakyReLU in place of ReLU after Dense 128, a lower learning rate, or a narrower layer (82 live "
                         "units do the work).",
          "advice: " + view.advice);
    if (view.rows.size() == 4) {
        Check(view.rows[0].dead == "0 of 32" && !view.rows[0].dead_warning, "Conv2D 32: none dead");
        Check(view.rows[1].dead == "15 of 64 (23 %)" && view.rows[1].dead_warning, "Conv2D 64: " + view.rows[1].dead);
        Check(view.rows[2].dead == "46 of 128 (36 %)" && view.rows[2].weight_norms == "8.64 → 8.74 → 8.84",
              "Dense 128: " + view.rows[2].dead + " / " + view.rows[2].weight_norms);
        Check(view.rows[2].words == "204,800 weights from -0.2 to 0.25. Dead after its ReLU by epoch: 48 → 46 "
                                    "→ 46 of 128.",
              "Dense 128 words: " + view.rows[2].words);
        Check(view.rows[3].dead == "no ReLU", "Dense 10: no ReLU after it");
    }
    cyxwiz::TrainingHealth none;
    Check(!cyxwiz::BuildHealthView(none, names).has_health, "no record: an empty state");
}

}  // namespace

int main(int, char** argv) {
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "the fixture opens");
    if (in.good()) CheckRecorder(json::parse(in));
    CheckView();
    CheckExecutor();
    if (failures > 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "training health: " << checks << " checks passed (numpy / torch statistics, executor)\n";
    return 0;
}
