// Cost of a Python node (TOFIX125 P4 step 4.6, Part C9): Softsign written in
// Python (bundle fixture example-acts.cyxnode) against the same Softsign in
// the C++ example plugin, in the same graph (tiny causal-LM example), same
// data, same epochs. Prints the time per epoch of each and the ratio.
//
//   bench_python_node_cost [epochs] [repeats]
//
// Not a test (timings depend on the machine); the numbers go into track125.md.
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/python_node_bundle.h"
#include "../src/extension/python/python_node_factory.h"
#include "../src/plugin/plugin_manager.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace py = pybind11;

namespace {

constexpr int kExtensionNodeId = 50;

nlohmann::json GraphWith(const fs::path& root, const std::string& type_id) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& link : graph["links"]) {
        if (link["from_node"] == 8 && link["to_node"] == 9) {
            link["to_node"] = kExtensionNodeId;
            link["to_pin_index"] = 0;
        }
    }
    graph["nodes"].push_back(
        {{"id", kExtensionNodeId}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", "Softsign"}, {"description", ""},
         {"parameters", nlohmann::json::object()}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", type_id}, {"version", "1.0.0"}, {"content_hash", ""},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0}, {"from_pin_index", 0},
                              {"to_node", 9}, {"to_pin", 0}, {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
}

// Seconds per epoch after the first (the first includes building and warm-up).
double SecondsPerEpoch(const fs::path& root, const fs::path& parquet, const fs::path& work,
                       const std::string& type_id, int epochs) {
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = GraphWith(root, type_id).dump();
    request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
    request.epochs_override = epochs;
    request.checkpoint_dir_override = (work / ("ck_" + std::to_string(std::rand()))).string();
    std::vector<std::chrono::steady_clock::time_point> marks;
    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_epoch = [&marks](int, float, float, float, float, float) {
        marks.push_back(std::chrono::steady_clock::now());
    };
    const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
    if (!result.ok || marks.size() < 2) {
        std::cerr << type_id << ": the job failed: " << result.error << "\n";
        return -1.0;
    }
    return std::chrono::duration<double>(marks.back() - marks.front()).count() /
           static_cast<double>(marks.size() - 1);
}

}  // namespace

int main(int argc, char** argv) {
    const int epochs = argc > 1 ? std::max(2, std::atoi(argv[1])) : 6;
    const int repeats = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_bench_python_node_cost";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8)) return EXIT_FAILURE;
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    // The C++ plugin's Softsign.
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    const fs::path plugin_dir = root / "plugins" / "examples" / "activation_nodes";
    if (!manager.LoadPlugin(plugin_dir) || !manager.InitializePlugin("com.cyxwiz.examples.activation-nodes")) {
        std::cerr << "the example plugin did not load\n";
        return EXIT_FAILURE;
    }
    // The Python bundle's Softsign.
    cyxwiz::PythonBundle bundle;
    std::string error;
    if (!cyxwiz::ReadPythonBundle(root / "cyxwiz-engine" / "tests" / "fixtures" / "python_bundles" /
                                      "example-acts.cyxnode",
                                  bundle, error)) {
        std::cerr << "bundle: " << error << "\n";
        return EXIT_FAILURE;
    }
    for (const auto& node : bundle.nodes) {
        cyxwiz::ExtensionNodeRegistry::Registration registration;
        registration.descriptor = node.descriptor;
        registration.tensor_factory = cyxwiz::MakePythonTensorNodeFactory(bundle, node);
        cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error);
    }

#ifdef _WIN32
    _putenv_s("PYTHONHOME", CYXWIZ_TEST_PYTHONHOME);
#endif
    py::scoped_interpreter interpreter;
    {
        py::module_ sys = py::module_::import("sys");
        py::module_ os = py::module_::import("os");
        sys.attr("path").attr("insert")(0, std::string(CYXWIZ_TEST_EXE_DIR));
        sys.attr("path").attr("insert")(0, std::string(CYXWIZ_TEST_PYCYXWIZ_DIR));
        for (const std::string& dir : {std::string(CYXWIZ_TEST_EXE_DIR), std::string(R"(C:\Program Files\ArrayFire\v3\lib)")}) {
            if (fs::is_directory(dir)) os.attr("add_dll_directory")(dir);
        }
        if (const char* cuda = std::getenv("CUDA_PATH")) {
            const std::string bin = (fs::path(cuda) / "bin").string();
            if (fs::is_directory(bin)) os.attr("add_dll_directory")(bin);
        }
        py::module_::import("pycyxwiz.nodes");
    }
    py::gil_scoped_release release;

    std::vector<double> cpp;
    std::vector<double> python;
    for (int r = 0; r < repeats; ++r) {
        cpp.push_back(SecondsPerEpoch(root, parquet, work, "com.cyxwiz.examples.activation-nodes:Softsign", epochs));
        python.push_back(SecondsPerEpoch(root, parquet, work, "com.example.acts:Softsign", epochs));
    }
    std::sort(cpp.begin(), cpp.end());
    std::sort(python.begin(), python.end());
    const double cpp_median = cpp[cpp.size() / 2];
    const double python_median = python[python.size() / 2];
    std::printf("graph: tiny causal-LM example (token windows), %d epochs, %d repeats, median\n", epochs, repeats);
    std::printf("C++ plugin Softsign:  %.4f s per epoch\n", cpp_median);
    std::printf("Python node Softsign: %.4f s per epoch\n", python_median);
    std::printf("ratio Python / C++:   %.2f\n", cpp_median > 0 ? python_median / cpp_median : 0.0);

    // Per call: forward + backward of one [32, 32] batch, as the trainer
    // calls a node once per batch.
    for (const char* type_id : {"com.cyxwiz.examples.activation-nodes:Softsign", "com.example.acts:Softsign"}) {
        auto factory = cyxwiz::ExtensionNodeRegistry::Instance().TensorFactory(type_id);
        std::string create_error;
        auto module = factory->CreateModule({32}, {}, create_error);
        if (!module) {
            std::cerr << type_id << ": " << create_error << "\n";
            continue;
        }
        std::vector<float> values(32 * 32);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i % 17) / 8.0f - 1.0f;
        const cyxwiz::Tensor x(std::vector<size_t>{32, 32}, values.data());
        for (int i = 0; i < 20; ++i) module->Backward(module->Forward(x));  // warm-up
        constexpr int kCalls = 2000;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kCalls; ++i) module->Backward(module->Forward(x)).ReadData<float>();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() /
                          kCalls;
        std::printf("%-46s forward+backward [32, 32]: %.1f us per batch\n", type_id, us);
        {
            py::gil_scoped_acquire gil;  // a Python module is released under the lock (as in a model)
            module.reset();
        }
    }
    manager.UnloadPlugin("com.cyxwiz.examples.activation-nodes");
    fs::remove_all(work, ec);
    return 0;
}
