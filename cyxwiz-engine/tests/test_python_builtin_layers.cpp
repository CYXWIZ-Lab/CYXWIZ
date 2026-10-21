// The SDK's Python example layers in the Engine's training path (TOFIX125):
// examples/python_nodes/builtin_layers/nodes/builtin-layers.cyxnode is
// registered with the Engine's Python factory and trained from a Data Input,
// as a user's graph would be:
//   Data Input -> Split -> Loader -> Dense 16 (Python) -> ReLU (Python)
//   -> Layer Norm (Python) -> Dense 3 (Python) -> cross entropy -> Adam
// learns three separable classes; then each Python activation trains in
// place of the ReLU. The same graph with built-in nodes is the yardstick.
// Last, the example graph shipped beside the bundle (blobs_python_layers.
// cyxgraph, four classes from blobs.csv) trains as saved.
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/python_node_bundle.h"
#include "../src/extension/python/python_node_factory.h"
#include "route_qualification_test_fixture.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <pybind11/embed.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif
#ifndef CYXWIZ_TEST_PYTHONHOME
#error "CYXWIZ_TEST_PYTHONHOME must name the Python the build uses"
#endif

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

constexpr const char* kBundleId = "com.cyxwiz.examples.layers";

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

// Three separable classes over four features (as test_graph_training_job).
bool WriteTabularBlobs(const fs::path& path, int rows, int seed) {
    arrow::DoubleBuilder features[4];
    arrow::Int64Builder labels;
    uint32_t state = static_cast<uint32_t>(seed);
    const auto noise = [&state] {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>(state >> 8) / 16777216.0 - 0.5) * 0.6;
    };
    for (int row = 0; row < rows; ++row) {
        const int label = row % 3;
        for (int column = 0; column < 4; ++column) {
            if (!features[column].Append((column == label ? 2.0 : 0.0) + noise()).ok()) return false;
        }
        if (!labels.Append(label).ok()) return false;
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(5);
    for (int column = 0; column < 4; ++column) {
        if (!features[column].Finish(&arrays[column]).ok()) return false;
    }
    if (!labels.Finish(&arrays[4]).ok()) return false;
    auto table = arrow::Table::Make(arrow::schema({arrow::field("x0", arrow::float64()),
                                                   arrow::field("x1", arrow::float64()),
                                                   arrow::field("x2", arrow::float64()),
                                                   arrow::field("x3", arrow::float64()),
                                                   arrow::field("class", arrow::int64())}),
                                    arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 100).ok() &&
           (*out)->Close().ok();
}

// The example's blobs.csv (f0..f7, class) as Parquet: the headless job reads
// Parquet; in the Engine the Data Input dialog loads the CSV.
bool CsvToParquet(const fs::path& csv, const fs::path& path) {
    std::ifstream in(csv);
    std::string line;
    if (!std::getline(in, line)) return false;
    arrow::DoubleBuilder features[8];
    arrow::Int64Builder labels;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream cells(line);
        std::string cell;
        for (int column = 0; column < 8; ++column) {
            if (!std::getline(cells, cell, ',') || !features[column].Append(std::stod(cell)).ok()) return false;
        }
        if (!std::getline(cells, cell, ',') || !labels.Append(std::stoll(cell)).ok()) return false;
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(9);
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int column = 0; column < 8; ++column) {
        if (!features[column].Finish(&arrays[column]).ok()) return false;
        fields.push_back(arrow::field("f" + std::to_string(column), arrow::float64()));
    }
    if (!labels.Finish(&arrays[8]).ok()) return false;
    fields.push_back(arrow::field("class", arrow::int64()));
    auto table = arrow::Table::Make(arrow::schema(fields), arrays);
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 100).ok() &&
           (*out)->Close().ok();
}

struct Layer {
    std::string python;  // node name in the bundle; empty: a built-in node
    gui::NodeType builtin = gui::NodeType::Dense;
    std::map<std::string, std::string> parameters;
};

// Data Input -> Split -> Loader -> layers... -> cross entropy -> Adam -> Output.
nlohmann::json Graph(const std::vector<Layer>& layers, const std::string& content_hash) {
    using gui::NodeType;
    nlohmann::json nodes = nlohmann::json::array();
    const auto add = [&nodes](int id, NodeType type, const std::string& name, nlohmann::json parameters) {
        nodes.push_back({{"id", id}, {"type", static_cast<int>(type)}, {"name", name}, {"parameters", parameters}});
    };
    add(1, NodeType::DataInput, "Blobs",
        {{"dataset_name", "tabular_blobs"}, {"file_category", "tabular"}, {"file_type", "parquet"},
         {"label_column", "class"}});
    add(2, NodeType::DataSplit, "Split",
        {{"seed", "42"}, {"stratified", "false"}, {"train_ratio", "0.8"}, {"val_ratio", "0.1"}, {"test_ratio", "0.1"}});
    add(3, NodeType::DataLoader, "Loader",
        {{"batch_size", "32"}, {"epochs", "4"}, {"shuffle", "true"}, {"seed", "11"}, {"model_seed", "7"}});
    int id = 10;
    for (const auto& layer : layers) {
        nlohmann::json parameters = nlohmann::json::object();
        for (const auto& [key, value] : layer.parameters) parameters[key] = value;
        if (layer.python.empty()) {
            add(id, layer.builtin, "Built-in " + std::to_string(id), parameters);
        } else {
            nodes.push_back(
                {{"id", id}, {"type", static_cast<int>(NodeType::PluginCustom)},
                 {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", layer.python + " (Python)"},
                 {"parameters", parameters},
                 {"extension",
                  {{"contract", 1}, {"type_id", std::string(kBundleId) + ":" + layer.python}, {"version", "1.0.0"},
                   {"content_hash", content_hash},
                   {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
                   {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
        }
        ++id;
    }
    const int last_layer = id - 1;
    add(7, NodeType::CrossEntropyLoss, "Cross entropy", nlohmann::json::object());
    add(8, NodeType::Adam, "Adam", {{"learning_rate", "0.01"}});
    add(9, NodeType::Output, "Output", {{"num_classes", "3"}});
    nlohmann::json links = nlohmann::json::array();
    int link_id = 1;
    const auto link = [&](int from, int from_index, int to, int to_index) {
        links.push_back({{"id", link_id}, {"from_node", from}, {"from_pin", 100 + link_id}, {"from_pin_index", from_index},
                         {"to_node", to}, {"to_pin", 200 + link_id}, {"to_pin_index", to_index}, {"link_type", 0}});
        ++link_id;
    };
    link(1, 0, 2, 0);
    link(2, 0, 3, 0);
    link(3, 0, 10, 0);
    for (int layer = 10; layer < last_layer; ++layer) link(layer, 0, layer + 1, 0);
    link(last_layer, 0, 7, 0);
    link(3, 1, 7, 1);
    link(7, 0, 8, 0);
    link(8, 0, 9, 0);
    return {{"version", "1.0"}, {"data_boundary_version", 2}, {"nodes", nodes}, {"links", links}};
}

struct Outcome {
    bool ok = false;
    std::string error;
    std::vector<float> losses;
    float last_accuracy = 0.0f;
};

Outcome Train(const nlohmann::json& graph, const fs::path& parquet, const fs::path& checkpoints) {
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = graph.dump();
    request.dataset_files["tabular_blobs"] = parquet.string();
    request.checkpoint_dir_override = checkpoints.string();
    Outcome outcome;
    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_epoch = [&outcome](int, float train_loss, float train_accuracy, float, float, float) {
        outcome.losses.push_back(train_loss);
        outcome.last_accuracy = train_accuracy;
    };
    const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
    outcome.ok = result.ok;
    outcome.error = result.error;
    return outcome;
}

std::string Describe(const Outcome& outcome) {
    if (!outcome.ok) return "failed: " + outcome.error;
    if (outcome.losses.empty()) return "no epochs";
    return "loss " + std::to_string(outcome.losses.front()) + " -> " + std::to_string(outcome.losses.back()) +
           ", train accuracy " + std::to_string(outcome.last_accuracy);
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path bundle_dir =
        root / "examples" / "python_nodes" / "builtin_layers" / "nodes" / "builtin-layers.cyxnode";
    const fs::path work = fs::temp_directory_path() / "cyxwiz_python_builtin_layers_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "blobs.parquet";
    if (!WriteTabularBlobs(parquet, 600, 3)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    std::cout << "the example bundle registers like any Python node bundle\n";
    cyxwiz::PythonBundle bundle;
    std::string error;
    Check(cyxwiz::ReadPythonBundle(bundle_dir, bundle, error), "the bundle reads" + (error.empty() ? "" : ": " + error));
    Check(bundle.nodes.size() == 10, "it holds ten nodes (" + std::to_string(bundle.nodes.size()) + ")");
    for (const auto& node : bundle.nodes) {
        cyxwiz::ExtensionNodeRegistry::Registration registration;
        registration.descriptor = node.descriptor;
        registration.tensor_factory = cyxwiz::MakePythonTensorNodeFactory(bundle, node);
        if (!cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error))
            Check(false, "registers " + node.descriptor.type_id + ": " + error);
    }

    std::cout << "with Python running\n";
#ifdef _WIN32
    _putenv_s("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME);
#else
    setenv("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME, 1);
#endif
    py::scoped_interpreter interpreter;
    {
        py::module_ sys = py::module_::import("sys");
        py::module_ os = py::module_::import("os");
        const std::string exe_dir = fs::path(CYXWIZ_TEST_EXE_DIR).string();
        sys.attr("path").attr("insert")(0, exe_dir);
        sys.attr("path").attr("insert")(0, fs::path(CYXWIZ_TEST_PYCYXWIZ_DIR).string());
        for (const std::string& dir : {exe_dir, std::string(R"(C:\Program Files\ArrayFire\v3\lib)")}) {
            if (fs::is_directory(dir)) os.attr("add_dll_directory")(dir);
        }
        if (const char* cuda = std::getenv("CUDA_PATH")) {
            const std::string bin = (fs::path(cuda) / "bin").string();
            if (fs::is_directory(bin)) os.attr("add_dll_directory")(bin);
        }
        py::module_::import("pycyxwiz.nodes");
    }
    py::gil_scoped_release release;  // as the Engine: training threads take the lock per call

    {
        const auto dense = cyxwiz::ExtensionNodeRegistry::Instance().TensorFactory(std::string(kBundleId) + ":Dense");
        const auto shape = dense ? dense->InferOutputShape({4}, {{"units", "3"}})
                                 : cyxwiz::ExtensionShapeResult{};
        Check(shape.ok && shape.output_shape == std::vector<size_t>({3}),
              "Dense (Python): [4] -> [3] from its own output_shape" +
                  (shape.ok ? std::string() : ": " + shape.error));
    }

    std::cout << "the built-in graph, the yardstick\n";
    const auto builtin = Train(Graph({{"", gui::NodeType::Dense, {{"units", "16"}}},
                                      {"", gui::NodeType::ReLU, {}},
                                      {"", gui::NodeType::Dense, {{"units", "3"}}}},
                                     bundle.content_hash),
                               parquet, work / "ck_builtin");
    Check(builtin.ok && builtin.losses.size() == 4, "built-in Dense 16 -> ReLU -> Dense 3: " + Describe(builtin));

    std::cout << "the same graph from the Python example nodes, fed by the Data Input\n";
    const auto python = Train(Graph({{"Dense", {}, {{"units", "16"}}},
                                     {"ReLU", {}, {}},
                                     {"LayerNorm", {}, {}},
                                     {"Dense", {}, {{"units", "3"}}}},
                                    bundle.content_hash),
                              parquet, work / "ck_python");
    Check(python.ok && python.losses.size() == 4, "Dense 16 -> ReLU -> Layer Norm -> Dense 3, all Python, trains: " +
                                                      Describe(python));
    Check(python.ok && !python.losses.empty() && python.losses.back() < python.losses.front(), "the loss goes down");
    Check(python.ok && python.last_accuracy > 0.9f, "it learns the three classes (train accuracy above 90%)");

    std::cout << "each Python activation trains between two Python Dense layers\n";
    for (const char* activation : {"LeakyReLU", "ELU", "Sigmoid", "Tanh", "Softplus", "GELU", "SiLU"}) {
        const auto outcome = Train(Graph({{"Dense", {}, {{"units", "16"}}},
                                          {activation, {}, {}},
                                          {"Dense", {}, {{"units", "3"}}}},
                                         bundle.content_hash),
                                   parquet, work / (std::string("ck_") + activation));
        Check(outcome.ok && !outcome.losses.empty() && outcome.losses.back() < outcome.losses.front(),
              std::string(activation) + ": " + Describe(outcome));
    }

    std::cout << "the example graph shipped with the bundle trains as saved\n";
    {
        const fs::path example = root / "examples" / "python_nodes" / "builtin_layers";
        std::ifstream file(example / "blobs_python_layers.cyxgraph");
        const auto graph = nlohmann::json::parse(file, nullptr, false);
        Check(!graph.is_discarded(), "blobs_python_layers.cyxgraph parses");
        bool current = true;
        for (const auto& node : graph.value("nodes", nlohmann::json::array())) {
            if (node.contains("extension"))
                current = current && node["extension"].value("content_hash", "") == bundle.content_hash;
        }
        Check(current, "its nodes name the bundle's current content hash");
        const fs::path blobs = work / "blobs4.parquet";
        Check(CsvToParquet(example / "blobs.csv", blobs), "blobs.csv converts to Parquet");
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["python_layers_blobs"] = blobs.string();
        request.epochs_override = 10;
        request.checkpoint_dir_override = (work / "ck_example").string();
        Outcome outcome;
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_epoch = [&outcome](int, float train_loss, float train_accuracy, float, float, float) {
            outcome.losses.push_back(train_loss);
            outcome.last_accuracy = train_accuracy;
        };
        const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        outcome.ok = result.ok;
        outcome.error = result.error;
        Check(outcome.ok && !outcome.losses.empty() && outcome.losses.back() < outcome.losses.front() &&
                  outcome.last_accuracy > 0.9f,
              "four classes, Dense 16 -> ReLU -> Layer Norm -> Dense 4 in Python: " + Describe(outcome));
    }

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_python_builtin_layers: all checks passed\n";
    return 0;
}
