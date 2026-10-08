// Conv1D graphs through the Engine's own path (TOFIX140), against PyTorch.
// GraphCompiler resolves the sequence section (sequence_conv_section.h),
// ModelBuilder builds it, and the built model - with torch's parameters set
// on it - gives torch's output and parameter gradients for both ways a
// sequence enters: the input rows (Conv1D first) and an Embedding's output
// (fixtures/conv1d_graph_pytorch.json, generate_conv1d_graph_fixtures.py).
// Also: compiled shapes, the time-series row layout, the compiler's refusals,
// and a Conv1D graph training with the TrainingExecutor.
#include "../src/core/arrow_dataset.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_compiler_dataset_hooks.h"
#include "../src/core/model_builder.h"
#include "../src/core/sequence_conv_section.h"
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
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text;
    for (size_t d : shape) text += (text.empty() ? "" : ", ") + std::to_string(d);
    return "[" + text + "]";
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "conv1d_graph_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return CYXWIZ_CONV1D_GRAPH_FIXTURE;
}

gui::NodePin Pin(int id, gui::PinType type, const std::string& name, bool is_input) {
    gui::NodePin pin;
    pin.id = id;
    pin.type = type;
    pin.name = name;
    pin.is_input = is_input;
    return pin;
}

// Model nodes: input pin id*100+1, output pin id*100+11.
gui::MLNode Layer(int id, gui::NodeType type, const std::string& name, std::map<std::string, std::string> params = {}) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = name;
    node.inputs = {Pin(id * 100 + 1, gui::PinType::Tensor, "Input", true)};
    node.outputs = {Pin(id * 100 + 11, gui::PinType::Tensor, "Output", false)};
    node.parameters = std::move(params);
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

struct Graph {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
};

// Data (features wide) -> model layers in order -> MSE -> SGD.
Graph Chain(size_t features, const std::vector<gui::MLNode>& model) {
    Graph g;
    gui::MLNode data;
    data.id = 1;
    data.type = gui::NodeType::DataInput;
    data.name = "Data";
    data.outputs = {Pin(111, gui::PinType::Tensor, "Data", false), Pin(112, gui::PinType::Labels, "Labels", false)};
    data.parameters = {{"dataset_name", "conv1d_rows"}, {"shape", "[" + std::to_string(features) + "]"}};
    gui::MLNode loss;
    loss.id = 90;
    loss.type = gui::NodeType::MSELoss;
    loss.name = "MSE";
    loss.inputs = {Pin(9001, gui::PinType::Tensor, "Predictions", true), Pin(9002, gui::PinType::Labels, "Targets", true)};
    loss.outputs = {Pin(9011, gui::PinType::Loss, "Loss", false)};
    gui::MLNode sgd;
    sgd.id = 91;
    sgd.type = gui::NodeType::SGD;
    sgd.name = "SGD";
    sgd.inputs = {Pin(9101, gui::PinType::Loss, "Loss", true)};
    sgd.outputs = {Pin(9111, gui::PinType::Optimizer, "State", false)};
    sgd.outputs[0].is_required = false;
    sgd.parameters = {{"learning_rate", "0.05"}, {"momentum", "0"}};
    g.nodes = {data};
    int from_node = 1, from_pin = 111, link_id = 1;
    for (const auto& layer : model) {
        g.nodes.push_back(layer);
        g.links.push_back(Link(link_id++, from_node, from_pin, layer.id, layer.inputs[0].id));
        from_node = layer.id;
        from_pin = layer.outputs[0].id;
    }
    g.nodes.push_back(loss);
    g.nodes.push_back(sgd);
    g.links.push_back(Link(link_id++, from_node, from_pin, 90, 9001));
    g.links.push_back(Link(link_id++, 1, 112, 90, 9002));
    g.links.push_back(Link(link_id++, 90, 9011, 91, 9101));
    return g;
}

cyxwiz::TrainingConfiguration Compile(const Graph& graph) {
    cyxwiz::GraphCompiler compiler;
    return compiler.Compile(graph.nodes, graph.links, true);
}

void PrintIssues(const cyxwiz::TrainingConfiguration& config) {
    for (const auto& issue : config.issues) {
        std::cerr << "  issue(" << (issue.level == cyxwiz::IssueLevel::Error ? "error" : "note") << "): "
                  << issue.node_name << ": " << issue.message << "\n";
    }
}

cyxwiz::TrainingConfiguration CompileValid(const Graph& graph, const std::string& what) {
    auto config = Compile(graph);
    if (!config.is_valid) PrintIssues(config);
    Check(config.is_valid, what + " compiles");
    return config;
}

const cyxwiz::CompiledLayer& LayerNamed(const cyxwiz::TrainingConfiguration& config, const std::string& name) {
    for (const auto& layer : config.layers) {
        if (layer.name == name) return layer;
    }
    Check(false, "compiled layer missing: " + name);
    return config.layers.front();
}

void CheckRefused(const Graph& graph, const std::string& node, const std::string& text, const std::string& what) {
    const auto config = Compile(graph);
    bool reported = false;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error && issue.node_name == node &&
            issue.message.find(text) != std::string::npos) {
            reported = true;
        }
    }
    if (!reported) PrintIssues(config);
    Check(!config.is_valid && reported, what + ": refused on '" + node + "' with '" + text + "'");
}

gui::MLNode Conv(int id, const std::string& name, int filters, int kernel, const std::string& padding) {
    return Layer(id, gui::NodeType::Conv1D, name,
                 {{"filters", std::to_string(filters)}, {"kernel_size", std::to_string(kernel)}, {"stride", "1"},
                  {"padding", padding}});
}

void CheckShapesAndRefusals() {
    // Table rows: one channel of 10 values.
    auto config = CompileValid(Chain(10, {Conv(3, "Conv", 4, 3, "same"), Layer(4, gui::NodeType::ReLU, "ReLU"),
                                          Layer(5, gui::NodeType::Flatten, "Flatten"),
                                          Layer(6, gui::NodeType::Dense, "Dense", {{"units", "3"}})}),
                               "rows -> Conv1D -> Flatten -> Dense");
    Check(LayerNamed(config, "Conv").input_shape == std::vector<size_t>{10, 1} &&
              LayerNamed(config, "Conv").output_shape == std::vector<size_t>{10, 4},
          "Conv1D reads the rows as [10, 1] and gives [10, 4], got " + ShapeText(LayerNamed(config, "Conv").output_shape));
    Check(LayerNamed(config, "Flatten").output_shape == std::vector<size_t>{40}, "Flatten -> 40");
    const auto section = cyxwiz::ResolveSequenceConvSection(config);
    Check(section && section->from_rows && section->features == 40 && !section->global_pool,
          "the sequence section opens on the rows and ends at Flatten with 40 features");

    // Time-series windows: blocks of input_width values per feature.
    config.sequence_input_shape = {8, 3};
    Check(cyxwiz::SequenceInputSample(config) == std::vector<size_t>{8, 3},
          "a time-series window of width 8 over 3 features is an [8, 3] sequence");

    // Stride and valid padding: L_out = floor((10 - 5) / 2) + 1 = 3.
    auto strided = Conv(3, "Conv", 2, 5, "valid");
    strided.parameters["stride"] = "2";
    config = CompileValid(Chain(10, {strided, Layer(5, gui::NodeType::GlobalAvgPool, "GAP"),
                                     Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                          "Conv1D stride 2 -> Global Avg Pool");
    Check(LayerNamed(config, "Conv").output_shape == std::vector<size_t>{3, 2}, "valid k5 s2 over 10 -> [3, 2]");
    Check(LayerNamed(config, "GAP").output_shape == std::vector<size_t>{2}, "Global Avg Pool over [3, 2] -> [2]");

    CheckRefused(Chain(10, {Layer(2, gui::NodeType::Dense, "Dense first", {{"units", "6"}}), Conv(3, "Conv", 4, 3, "same"),
                            Layer(5, gui::NodeType::Flatten, "Flatten"), Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                 "Conv", "make it the first model layer", "Conv1D after Dense");
    CheckRefused(Chain(10, {Conv(3, "Conv", 4, 3, "same"), Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                 "Dense", "needs Flatten or a global pool before", "Dense straight after Conv1D");
    CheckRefused(Chain(10, {Conv(3, "Conv", 4, 3, "same"), Layer(5, gui::NodeType::Flatten, "Flatten"),
                            Conv(4, "Conv 2", 4, 3, "same"), Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                 "Conv 2", "needs an [L, C] sequence and gets [40]", "Conv1D after Flatten");
    CheckRefused(Chain(4, {Conv(3, "Conv", 4, 7, "valid"), Layer(5, gui::NodeType::Flatten, "Flatten"),
                           Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                 "Conv", "does not fit", "a kernel longer than the sequence");
}

cyxwiz::Tensor ReadFloat(const json& fixture) {
    const auto shape = fixture.at("shape").get<std::vector<size_t>>();
    const auto values = fixture.at("values").get<std::vector<float>>();
    return cyxwiz::Tensor(shape, values.data(), cyxwiz::DataType::Float32);
}

void CheckClose(const cyxwiz::Tensor& actual, const json& expected, const json& tolerance, const std::string& what) {
    const auto shape = expected.at("shape").get<std::vector<size_t>>();
    const auto values = expected.at("values").get<std::vector<float>>();
    Check(actual.NumElements() == values.size(), what + ": " + ShapeText(actual.Shape()) + " vs torch " + ShapeText(shape));
    const float atol = tolerance.at("atol").get<float>();
    const float rtol = tolerance.at("rtol").get<float>();
    const float* data = actual.ReadData<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::fabs(data[i] - values[i]) > atol + rtol * std::fabs(values[i])) {
            Check(false, what + ": element " + std::to_string(i) + " " + std::to_string(data[i]) + " vs torch " +
                             std::to_string(values[i]));
        }
    }
}

// Sets torch's parameters on the built model's Embedding, Conv1D and Linear
// modules, runs forward and backward, compares with torch.
void CheckPyTorchCase(const json& c, const cyxwiz::TrainingConfiguration& config, const cyxwiz::Tensor& input,
                      const json& tolerance) {
    const std::string name = c.at("name").get<std::string>();
    auto built = cyxwiz::BuildExecutableFromConfig(config);
    Check(built.ok(), name + ": ModelBuilder builds it: " + built.error_message);
    Check(built.model->AsSequentialModel() != nullptr, name + ": a sequential model");
    auto& model = *built.model->AsSequentialModel();
    struct Owner {
        const char* prefix;   // module name prefix
        const char* fixture;  // fixture parameter prefix
        std::vector<std::pair<std::string, std::string>> keys;  // module key, fixture key
    };
    const std::vector<Owner> owners = {
        {"Embedding", "embedding.", {{"weight", "weight"}}},
        {"Conv1D", "conv.", {{"weights", "weights"}, {"bias", "bias"}}},
        {"Linear", "dense.", {{"weight", "weight"}, {"bias", "bias"}}},
    };
    const auto& parameters = c.at("parameters");
    std::map<std::string, cyxwiz::Module*> modules;
    for (size_t i = 0; i < model.Size(); ++i) {
        cyxwiz::Module* module = model.GetModule(i);
        for (const auto& owner : owners) {
            if (module->GetName().rfind(owner.prefix, 0) != 0) continue;
            if (!parameters.contains(std::string(owner.fixture) + owner.keys.front().second)) continue;
            Check(modules.emplace(owner.fixture, module).second, name + ": one " + owner.prefix + " module");
            std::map<std::string, cyxwiz::Tensor> values;
            for (const auto& [module_key, fixture_key] : owner.keys) {
                values[module_key] = ReadFloat(parameters.at(std::string(owner.fixture) + fixture_key));
            }
            module->SetParameters(values);
        }
    }
    for (const auto& owner : owners) {
        if (parameters.contains(std::string(owner.fixture) + owner.keys.front().second)) {
            Check(modules.count(owner.fixture) == 1, name + ": the model has its " + owner.prefix + " module");
        }
    }

    const cyxwiz::Tensor output = model.Forward(input);
    CheckClose(output, c.at("output"), tolerance, name + " output");
    model.Backward(ReadFloat(c.at("grad_output")));
    for (const auto& owner : owners) {
        const auto found = modules.find(owner.fixture);
        if (found == modules.end()) continue;
        const auto gradients = found->second->GetGradients();
        for (const auto& [module_key, fixture_key] : owner.keys) {
            const auto gradient = gradients.find(module_key);
            Check(gradient != gradients.end(), name + ": gradient " + owner.prefix + "." + module_key);
            CheckClose(gradient->second, c.at("parameter_gradients").at(std::string(owner.fixture) + fixture_key),
                       tolerance, name + " gradient " + owner.fixture + fixture_key);
        }
    }
    std::cout << "  ok " << name << "\n";
}

void CheckPyTorch(const json& fixture) {
    const auto& tolerance = fixture.at("tolerance");
    for (const auto& c : fixture.at("cases")) {
        const auto& g = c.at("graph");
        const std::string name = c.at("name").get<std::string>();
        if (name == "rows_flatten") {
            const size_t features = g.at("features").get<size_t>();
            const auto config = CompileValid(
                Chain(features, {Conv(3, "Conv", g.at("filters").get<int>(), g.at("kernel_size").get<int>(),
                                      g.at("padding").get<std::string>()),
                                 Layer(4, gui::NodeType::ReLU, "ReLU"), Layer(5, gui::NodeType::Flatten, "Flatten"),
                                 Layer(6, gui::NodeType::Dense, "Dense", {{"units", std::to_string(g.at("units").get<int>())}})}),
                name);
            CheckPyTorchCase(c, config, ReadFloat(c.at("input")), tolerance);
        } else {
            const size_t length = g.at("length").get<size_t>();
            const auto config = CompileValid(
                Chain(length,
                      {Layer(2, gui::NodeType::Embedding, "Embedding",
                             {{"num_embeddings", std::to_string(g.at("vocab").get<int>())},
                              {"embedding_dim", std::to_string(g.at("embedding_dim").get<int>())}}),
                       Conv(3, "Conv", g.at("filters").get<int>(), g.at("kernel_size").get<int>(),
                            g.at("padding").get<std::string>()),
                       Layer(4, gui::NodeType::ReLU, "ReLU"),
                       Layer(5, g.at("pool").get<std::string>() == "max" ? gui::NodeType::GlobalMaxPool
                                                                          : gui::NodeType::GlobalAvgPool,
                             "GAP"),
                       Layer(6, gui::NodeType::Dense, "Dense", {{"units", std::to_string(g.at("units").get<int>())}})}),
                name);
            Check(LayerNamed(config, "Conv").input_shape ==
                      std::vector<size_t>{length, g.at("embedding_dim").get<size_t>()},
                  "Conv1D after an Embedding reads [L, E], got " + ShapeText(LayerNamed(config, "Conv").input_shape));
            const auto section = cyxwiz::ResolveSequenceConvSection(config);
            Check(section && !section->from_rows && section->global_pool,
                  "the section opens after the Embedding and ends at the global pool");
            const auto shape = c.at("input").at("shape").get<std::vector<size_t>>();
            const auto ids = c.at("input").at("values").get<std::vector<int32_t>>();
            CheckPyTorchCase(c, config, cyxwiz::Tensor(shape, ids.data(), cyxwiz::DataType::Int32), tolerance);
        }
    }
}

std::shared_ptr<arrow::Array> Floats(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) Check(builder.Append(value).ok(), "append");
    std::shared_ptr<arrow::Array> array;
    Check(builder.Finish(&array).ok(), "finish");
    return array;
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

// A Conv1D graph trains with the TrainingExecutor: 16 rows of 10 values, the
// label is the mean of the first three; the loss falls.
void CheckTraining(const std::filesystem::path& work_dir) {
    constexpr size_t kRows = 16, kFeatures = 10;
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<float> label(kRows, 0.0f);
    for (size_t f = 0; f < kFeatures; ++f) {
        std::vector<float> column(kRows);
        for (size_t r = 0; r < kRows; ++r) {
            column[r] = static_cast<float>(std::sin(0.7 * static_cast<double>(r + 1) * static_cast<double>(f + 1)));
            if (f < 3) label[r] += column[r] / 3.0f;
        }
        fields.push_back(arrow::field("x" + std::to_string(f), arrow::float32()));
        columns.push_back(Floats(column));
    }
    fields.push_back(arrow::field("label", arrow::float32()));
    columns.push_back(Floats(label));
    auto table = arrow::Table::Make(arrow::schema(fields), columns, kRows);
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(std::move(table), "conv1d_rows");

    auto config = CompileValid(Chain(kFeatures, {Conv(3, "Conv", 4, 3, "same"), Layer(4, gui::NodeType::ReLU, "ReLU"),
                                                 Layer(5, gui::NodeType::Flatten, "Flatten"),
                                                 Layer(6, gui::NodeType::Dense, "Dense", {{"units", "1"}})}),
                               "the training graph");
    config.dataset_name = "conv1d_rows";
    config.batch_size = 4;
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.shuffle = false;
    config.num_workers = 0;
    config.model_seed = 52;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.log_interval = 0;
    config.checkpoint_dir = (work_dir / "checkpoints").string();

    SelectArrayFireCpu();
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    cyxwiz::TrainingMetrics metrics;
    bool completed = false;
    executor.Train(12, 4, nullptr, nullptr, [&](const cyxwiz::TrainingMetrics& m) {
        metrics = m;
        completed = true;
    });
    if (!completed) metrics = executor.GetMetrics();
    Check(metrics.terminal_status == "completed",
          "the Conv1D graph trains (" + metrics.terminal_status + ": " + metrics.terminal_reason + ")");
    Check(metrics.loss_history.size() == 12 && metrics.loss_history.back() < metrics.loss_history.front(),
          "the training loss falls: " + std::to_string(metrics.loss_history.front()) + " -> " +
              std::to_string(metrics.loss_history.back()));
}

}  // namespace

int main(int, char** argv) {
    namespace fs = std::filesystem;
    cyxwiz::GraphCompilerDatasetHooks hooks;
    hooks.is_dataset_registered = [](const std::string&) { return false; };
    cyxwiz::SetGraphCompilerDatasetHooks(hooks);

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_conv1d_graph_training";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(work_dir / "debug_runs");

    std::ifstream in(FixturePath(argv[0]));
    Check(in.good(), "conv1d_graph_pytorch.json is readable");
    const json fixture = json::parse(in);

    CheckShapesAndRefusals();
    SelectArrayFireCpu();
    CheckPyTorch(fixture);
    CheckTraining(work_dir);

    fs::remove_all(work_dir);
    std::cout << "Conv1D graphs match PyTorch and train\n";
    return 0;
}
