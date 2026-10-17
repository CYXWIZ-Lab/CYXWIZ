// Python nodes in the Engine's training path (TOFIX125 P4 step 4.3): the
// bundle pycyxwiz.nodes.bundle() wrote (fixture example-acts.cyxnode) is
// registered with its Python factory; Softsign and Snake pass the gradient
// check through the factory and train in the tiny causal-LM graph; Snake's
// `a` is learned; settings are refused without Python; a Python error stops
// training and names the node and the Python traceback.
//
// The test starts its own interpreter (as the Engine's scripting engine
// does), imports pycyxwiz from the folder of this exe, then releases the
// lock so training threads take it per call, as in the Engine.
#include <cyxwiz/gradient_check.h>
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/python_node_bundle.h"
#include "../src/extension/python/python_node_factory.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
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

constexpr const char* kBundleId = "com.example.acts";
constexpr int kExtensionNodeId = 50;
constexpr int kDecoderNodeId = 8;
constexpr int kDenseNodeId = 9;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

nlohmann::json GraphWith(const fs::path& root, const std::string& provider, const std::string& type_name,
                         const std::map<std::string, std::string>& parameters, const std::string& hash) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& link : graph["links"]) {
        if (link["from_node"] == kDecoderNodeId && link["to_node"] == kDenseNodeId) {
            link["to_node"] = kExtensionNodeId;
            link["to_pin_index"] = 0;
        }
    }
    nlohmann::json params = nlohmann::json::object();
    for (const auto& [key, value] : parameters) params[key] = value;
    graph["nodes"].push_back(
        {{"id", kExtensionNodeId}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", type_name},
         {"description", ""}, {"parameters", params}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", provider + ":" + type_name}, {"version", "1.0.0"},
           {"content_hash", hash},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0},
                              {"from_pin_index", 0}, {"to_node", kDenseNodeId}, {"to_pin", 0},
                              {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
}

cyxwiz::Module* FindModule(cyxwiz::SequentialModel& model, const std::string& name) {
    for (size_t i = 0; i < model.Size(); ++i) {
        if (model.GetModule(i) && model.GetModule(i)->GetName() == name) return model.GetModule(i);
    }
    return nullptr;
}

// Registers every node of a bundle with its Python factory.
bool RegisterBundle(const fs::path& folder, cyxwiz::PythonBundle& bundle) {
    std::string error;
    if (!cyxwiz::ReadPythonBundle(folder, bundle, error)) {
        std::cout << "  bundle not read: " << error << "\n";
        return false;
    }
    for (const auto& node : bundle.nodes) {
        cyxwiz::ExtensionNodeRegistry::Registration registration;
        registration.descriptor = node.descriptor;
        registration.tensor_factory = cyxwiz::MakePythonTensorNodeFactory(bundle, node);
        if (!cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error)) {
            std::cout << "  not registered: " << error << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path bundle_dir = root / "cyxwiz-engine" / "tests" / "fixtures" / "python_bundles" /
                                "example-acts.cyxnode";
    const fs::path work = fs::temp_directory_path() / "cyxwiz_python_node_training_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();

    std::cout << "without Python running, Python nodes say why\n";
    cyxwiz::PythonBundle bundle;
    Check(RegisterBundle(bundle_dir, bundle), "the fixture bundle registers (no Python needed)");
    const std::string snake_id = std::string(kBundleId) + ":Snake";
    {
        auto factory = registry.TensorFactory(snake_id);
        const auto shape = factory->InferOutputShape({8, 32}, {{"initial_a", "1.0"}});
        Check(shape.ok && shape.output_shape == std::vector<size_t>({8, 32}),
              "same_as_input: the shape is known without Python");
        const auto refused = factory->InferOutputShape({8, 32}, {{"initial_a", "-1"}});
        Check(!refused.ok && refused.error == "initial_a must be at least 0.01",
              "a refused setting is reported without Python: " + refused.error);
        const auto not_number = factory->InferOutputShape({8, 32}, {{"initial_a", "abc"}});
        Check(!not_number.ok && not_number.error == "initial_a must be a number, not 'abc'",
              "the same words as pycyxwiz.nodes: " + not_number.error);
        const auto unknown = factory->InferOutputShape({8, 32}, {{"speed", "1"}});
        Check(!unknown.ok && unknown.error == "Snake has no setting speed", "an unknown setting: " + unknown.error);
        cyxwiz::PythonNodeRuntime::Instance().ClearWanted();
        std::string error;
        Check(factory->CreateModule({8, 32}, {}, error) == nullptr && !error.empty() &&
                  cyxwiz::PythonNodeRuntime::Instance().Wanted(),
              "making the module needs Python: refused with a reason, and Python is wanted");
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
        // pycyxwiz as just built, and the runtime folder holding its DLLs.
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
    py::gil_scoped_release release;  // as the Engine: other threads take the lock per call

    std::cout << "both pass the gradient check through the Engine's factory\n";
    for (const char* name : {"Softsign", "Snake"}) {
        const auto factory = registry.TensorFactory(std::string(kBundleId) + ":" + name);
        const auto result = cyxwiz::CheckModuleGradients([&](const std::vector<size_t>& shape) {
            std::string error;
            const std::vector<size_t> sample(shape.begin() + 1, shape.end());
            const std::map<std::string, std::string> settings =
                std::string(name) == "Snake" ? std::map<std::string, std::string>{{"initial_a", "1.0"}}
                                             : std::map<std::string, std::string>{};
            auto module = factory->CreateModule(sample, settings, error);
            if (!module) std::cout << "  CreateModule: " << error << "\n";
            return module;
        });
        Check(result.passed, std::string(name) + ": " + cyxwiz::DescribeGradientCheck(result));
    }

    std::cout << "Snake trains in the example graph and learns its frequency\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = GraphWith(root, kBundleId, "Snake", {{"initial_a", "1.0"}}, bundle.content_hash).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 3;
        request.checkpoint_dir_override = (work / "ck_snake").string();
        std::vector<float> losses;
        cyxwiz::GraphTrainingJobCallbacks callbacks;
        callbacks.on_epoch = [&losses](int, float train_loss, float, float, float, float) {
            losses.push_back(train_loss);
        };
        auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        cyxwiz::Module* module = result.model ? FindModule(*result.model, "Snake") : nullptr;
        Check(module != nullptr, "the model holds the Python Snake module");
        if (module) {
            const auto params = module->GetParameters();
            const float a = params.count("a") ? params.at("a").ReadData<float>()[0] : 1.0f;
            Check(std::isfinite(a) && std::fabs(a - 1.0f) > 1e-6f,
                  "the optimizer changed a from 1.0 (now " + std::to_string(a) + ")");
        }
        Check(losses.size() == 3 && losses.back() < losses.front(),
              "the loss goes down (" + (losses.empty() ? std::string("none")
                                                       : std::to_string(losses.front()) + " -> " +
                                                             std::to_string(losses.back())) + ")");
    }

    std::cout << "Softsign (tier 1, f and df only) trains\n";
    {
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = GraphWith(root, kBundleId, "Softsign", {}, bundle.content_hash).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 2;
        request.checkpoint_dir_override = (work / "ck_softsign").string();
        auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        Check(result.model && FindModule(*result.model, "Softsign") != nullptr, "the model holds Softsign");
    }
    Check(!fs::exists(bundle_dir / "__pycache__"), "loading a bundle writes no __pycache__ into its folder");

    std::cout << "a Python error stops training and names the node\n";
    {
        // A bundle written now by pycyxwiz, whose forward raises.
        const fs::path code = work / "broken_nodes.py";
        {
            py::gil_scoped_acquire gil;
            py::dict scope;
            scope["code_path"] = code.string();
            scope["work"] = work.string();
            py::exec(R"PY(
import importlib.util, sys
with open(code_path, "w", newline="\n") as f:
    f.write('''import pycyxwiz as cyx
from pycyxwiz import nodes

@nodes.activation(id="com.example.broken", name="Broken", version="1.0.0")
class Broken:
    def f(self, x):
        raise ValueError("deliberate failure in f")
    def df(self, x):
        return x
''')
spec = importlib.util.spec_from_file_location("broken_nodes", code_path)
module = importlib.util.module_from_spec(spec)
sys.modules["broken_nodes"] = module
spec.loader.exec_module(module)
from pycyxwiz import nodes
bundle_path = nodes.bundle(module.Broken, folder=work, name="broken", check_gradients=False)
)PY",
                     scope);
        }
        cyxwiz::PythonBundle broken;
        Check(RegisterBundle(work / "broken.cyxnode", broken), "the broken bundle registers");
        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = GraphWith(root, "com.example.broken", "Broken", {}, broken.content_hash).dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 1;
        request.checkpoint_dir_override = (work / "ck_broken").string();
        auto result = cyxwiz::RunGraphTrainingJob(request);
        Check(!result.ok, "the job stops");
        Check(result.error.find("Broken") != std::string::npos &&
                  result.error.find("deliberate failure in f") != std::string::npos,
              "the error names the node and the Python message: " + result.error.substr(0, 300));
        Check(result.error.find("broken_nodes.py") != std::string::npos ||
                  result.error.find("node.py") != std::string::npos,
              "the error carries the Python traceback");
    }

    registry.RemoveByProvider(kBundleId);
    registry.RemoveByProvider("com.example.broken");
    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_python_node_training: all checks passed\n";
    return 0;
}
