// The learning rate range test (TOFIX129 D11).
// 1. Its rules against PyTorch: the raw losses of two torch runs
//    (generate_learning_rate_range_fixtures.py) replayed through
//    LearningRateRangeTest give torch's rates, smoothed losses and stop, and
//    AnalyseLearningRateRange gives numpy's slopes and torch's suggestions.
// 2. Through TrainingExecutor: a Dense -> MSE -> SGD graph trains with
//    config.learning_rate_range; every recorded step used the scheduled rate,
//    a rate far too high stops it early, a short range runs all its steps,
//    and no validation pass runs.
// 3. The Model window's words for a result.
#include "../src/core/arrow_dataset.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/learning_rate_range_test.h"
#include "../src/core/training_executor.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/device.h>

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

bool Near(double a, double b, double relative = 1e-12) {
    return std::abs(a - b) <= relative * std::max({1.0, std::abs(a), std::abs(b)});
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "learning_rate_range_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_LEARNING_RATE_RANGE_FIXTURE;
}

void CheckAgainstTorch(const std::string& name, const json& c) {
    cyxwiz::LearningRateRangeSettings settings;
    settings.enabled = true;
    settings.start = c.at("start");
    settings.end = c.at("end");
    settings.steps = c.at("steps");
    settings.smoothing = c.at("smoothing");
    settings.stop_factor = c.at("stop_factor");
    cyxwiz::LearningRateRangeTest test(settings);
    const auto losses = c.at("losses").get<std::vector<double>>();
    const auto rates = c.at("rates").get<std::vector<double>>();
    const auto smoothed = c.at("smoothed").get<std::vector<double>>();
    size_t recorded = 0;
    for (double loss : losses) {
        Check(Near(test.CurrentLearningRate(), rates[recorded]),
              name + ": step " + std::to_string(recorded + 1) + " rate " + std::to_string(test.CurrentLearningRate()));
        const bool done = test.Record(loss);
        ++recorded;
        if (done) break;
    }
    Check(recorded == losses.size(), name + ": stops after " + std::to_string(recorded) + " steps, torch after " +
                                         std::to_string(losses.size()));
    const std::string stop = c.at("stop");
    const std::string& reason = test.StopReason();
    Check((stop == "factor" && reason.find("Stopped early") == 0) ||
              (stop == "steps" && reason.find("All ") == 0) ||
              (stop == "not_finite" && reason.find("not a number") != std::string::npos),
          name + ": stop reason '" + reason + "' for torch's " + stop);
    for (size_t i = 0; i < test.Points().size() && i < smoothed.size(); ++i)
        Check(Near(test.Points()[i].smoothed, smoothed[i]), name + ": smoothed loss " + std::to_string(i + 1));

    const auto analysis = cyxwiz::AnalyseLearningRateRange(test.Points());
    Check(analysis.ok, name + ": the curve is read");
    const auto slopes = c.at("slopes").get<std::vector<double>>();
    Check(analysis.slopes.size() == slopes.size(), name + ": one slope per point");
    for (size_t i = 0; i < slopes.size() && i < analysis.slopes.size(); ++i)
        Check(Near(analysis.slopes[i], slopes[i], 1e-9), name + ": numpy slope " + std::to_string(i + 1) + ": " +
                                                             std::to_string(analysis.slopes[i]) + " vs " +
                                                             std::to_string(slopes[i]));
    Check(analysis.lowest == c.at("lowest").get<size_t>() && Near(analysis.lowest_rate, c.at("lowest_rate")),
          name + ": lowest at step " + std::to_string(analysis.lowest + 1));
    Check(analysis.steepest == c.at("steepest").get<size_t>() && Near(analysis.steepest_rate, c.at("steepest_rate")),
          name + ": steepest fall at step " + std::to_string(analysis.steepest + 1));
    Check(Near(analysis.lowest_over_ten, c.at("lowest_over_ten")), name + ": lowest / 10");
}

// ---- Through the executor ---------------------------------------------------

std::shared_ptr<arrow::Array> Floats(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeDataset() {
    std::vector<float> x0, x1, label;
    for (int i = 0; i < 64; ++i) {
        const float a = std::sin(0.37f * static_cast<float>(i)), b = std::cos(0.23f * static_cast<float>(i));
        x0.push_back(a);
        x1.push_back(b);
        label.push_back(2.0f * a - b + 0.5f);
    }
    auto schema = arrow::schema({arrow::field("x0", arrow::float32()), arrow::field("x1", arrow::float32()),
                                 arrow::field("label", arrow::float32())});
    auto table = arrow::Table::Make(schema, {Floats(x0), Floats(x1), Floats(label)}, 64);
    return std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "range_rows");
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

cyxwiz::TrainingConfiguration CompileGraph() {
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "range_rows"}, {"shape", "[2]"}};
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
    sgd.parameters = {{"learning_rate", "0.01"}, {"momentum", "0"}};
    const std::vector<gui::MLNode> nodes = {data, dense, loss, sgd};
    const std::vector<gui::NodeLink> links = {Link(1, 1, 111, 3, 301), Link(2, 3, 311, 9, 901), Link(3, 1, 112, 9, 902),
                                              Link(4, 9, 911, 10, 1001)};
    auto config = cyxwiz::GraphCompiler{}.Compile(nodes, links, true);
    for (const auto& issue : config.issues)
        if (issue.level == cyxwiz::IssueLevel::Error) std::cerr << "  issue: " << issue.message << "\n";
    Check(config.is_valid, "the range test graph compiles");
    config.dataset_name = "range_rows";
    config.batch_size = 8;
    config.train_ratio = 0.75f;
    config.val_ratio = 0.25f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.log_interval = 0;
    config.checkpoint_dir = (std::filesystem::temp_directory_path() / "cyxwiz_range_test_checkpoints").string();
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

cyxwiz::TrainingMetrics RunRange(const cyxwiz::LearningRateRangeSettings& settings, int* validations) {
    auto config = CompileGraph();
    config.learning_rate_range = settings;
    config.learning_rate_range.enabled = true;
    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor executor(config, MakeDataset(), "label");
    *validations = 0;
    executor.Train(settings.steps, 8, nullptr,
                   [validations](int, float, float, float val_loss, float, float) {
                       if (val_loss >= 0.0f) ++*validations;  // -1 when no validation ran
                   },
                   nullptr);
    return executor.GetMetrics();
}

void CheckExecutor() {
    cyxwiz::LearningRateRangeSettings wide;
    wide.start = 1e-4;
    wide.end = 1e4;
    wide.steps = 60;
    int validations = 0;
    const auto diverged = RunRange(wide, &validations);
    const cyxwiz::LearningRateRangeTest schedule(wide);
    Check(!diverged.range_points.empty() && diverged.range_points.size() < 60,
          "a rate far too high stops the test early: " + std::to_string(diverged.range_points.size()) + " steps (" +
              diverged.range_stop_reason + ")");
    for (size_t i = 0; i < diverged.range_points.size(); ++i)
        Check(Near(diverged.range_points[i].learning_rate, schedule.LearningRateAt(static_cast<int>(i))),
              "step " + std::to_string(i + 1) + " trained at the scheduled rate");
    Check(diverged.range_stop_reason.find("Stopped early") == 0 ||
              diverged.range_stop_reason.find("not a number") != std::string::npos,
          "it says why it stopped: " + diverged.range_stop_reason);
    Check(diverged.range_points.front().loss > 0.0 && std::isfinite(diverged.range_points.front().loss),
          "the first step's loss is the batch's MSE");
    Check(validations == 0, "no validation pass runs in a range test");

    cyxwiz::LearningRateRangeSettings narrow;
    narrow.start = 1e-4;
    narrow.end = 1e-2;
    narrow.steps = 12;
    const auto all = RunRange(narrow, &validations);
    Check(all.range_points.size() == 12 && all.range_stop_reason == "All 12 steps ran.",
          "a short safe range runs all its steps: " + std::to_string(all.range_points.size()) + " (" +
              all.range_stop_reason + ")");
    Check(Near(all.range_points.back().learning_rate, 1e-2), "the last step trains at To");
    Check(validations == 0, "still no validation pass");
}

void CheckWords() {
    std::vector<cyxwiz::LearningRateRangePoint> points;
    cyxwiz::LearningRateRangeSettings settings;
    settings.start = 1e-4;
    settings.end = 1.0;
    settings.steps = 5;
    cyxwiz::LearningRateRangeTest test(settings);
    for (double loss : {2.3, 2.0, 1.2, 1.0, 1.1}) test.Record(loss);
    const auto analysis = cyxwiz::AnalyseLearningRateRange(test.Points());
    const auto steep = cyxwiz::DescribeLearningRateRange(analysis, true, "Adam", 0.001);
    Check(steep.headline == "Suggested learning rate: " + steep.suggested_text && steep.use_label == "Use " + steep.suggested_text + " in Adam",
          "steepest fall words: " + steep.headline + " / " + steep.use_label);
    Check(steep.why.find("The graph's Adam uses 0.001 now.") != std::string::npos, "it names the rate in use: " + steep.why);
    const auto tenth = cyxwiz::DescribeLearningRateRange(analysis, false, "Adam", 0.001);
    Check(Near(tenth.suggested, analysis.lowest_rate / 10.0) && tenth.why.find("A tenth of the rate") == 0,
          "lowest / 10 words: " + tenth.why);
    Check(cyxwiz::FormatLearningRate(0.014849) == "0.0148" && cyxwiz::FormatLearningRate(1e-7) == "1e-07",
          "rates print with three digits");
    const std::string text = cyxwiz::LearningRateRangeText(test.Points(), ',');
    Check(text.rfind("step,learning_rate,loss,smoothed\n1,0.0001,2.3,2.3\n", 0) == 0, "curve text: " + text.substr(0, 60));
    cyxwiz::LearningRateRangeTest short_test(settings);
    short_test.Record(2.0);
    const auto too_few = cyxwiz::AnalyseLearningRateRange(short_test.Points());
    Check(!too_few.ok && !too_few.error.empty(), "too few steps: no suggestion, with a reason");
}

}  // namespace

int main(int, char** argv) {
    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "the PyTorch fixture opens");
    if (in.good()) {
        const json fixture = json::parse(in);
        for (const auto& [name, c] : fixture.at("cases").items()) CheckAgainstTorch(name, c);
    }
    CheckWords();
    CheckExecutor();
    if (failures > 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "learning rate range test: " << checks << " checks passed (PyTorch rules, executor, words)\n";
    return 0;
}
