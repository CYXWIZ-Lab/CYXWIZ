// Model files, code generation and export with extension nodes (TOFIX125 P6).
// 6.1: what cannot contain an extension node's computation refuses it by
// name instead of writing a different model: ONNX export, and generated code
// (PyTorch, TensorFlow, Keras, pycyxwiz) through ExtensionCodeExportRefusal.
// 6.2: a model trained with an extension node saves as a CyxWiz package and
// imports again (as a layer stack and as an executable model) with the same
// weights; a missing extension and an extension that cannot create its module
// are refused with the reason.
#include "../src/core/executable_model.h"
#include "../src/core/extension_node_metadata.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/guarded_extension_module.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/model_exporter.h"
#include "../src/core/model_importer.h"
#include "causal_lm_token_window_fixture.h"
#include "route_qualification_test_fixture.h"

#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>

#include <cmath>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifndef CYXWIZ_SOURCE_ROOT
#error "CYXWIZ_SOURCE_ROOT must name the repository root"
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kProvider = "test.files";
constexpr int kExtensionNodeId = 50;
constexpr int kDecoderNodeId = 8;
constexpr int kDenseNodeId = 9;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// y = a * x with one learnable scalar a.
class ScaleModule : public cyxwiz::Module {
public:
    explicit ScaleModule(float initial) : a_(std::vector<size_t>{1}, &initial) {}
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override { return x * a_.ReadData<float>()[0]; }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override { return grad * a_.ReadData<float>()[0]; }
    std::map<std::string, cyxwiz::Tensor> GetParameters() override { return {{"a", a_}}; }
    void SetParameters(const std::map<std::string, cyxwiz::Tensor>& params) override {
        const auto it = params.find("a");
        if (it != params.end()) a_ = it->second;
    }
    bool HasParameters() const override { return true; }
    std::string GetName() const override { return "Scale"; }

private:
    cyxwiz::Tensor a_;
};

// y = a * x with one learnable scalar a; Backward stores dLoss/da.
class TrainableScaleModule : public cyxwiz::Module {
public:
    explicit TrainableScaleModule(float initial) : a_(std::vector<size_t>{1}, &initial) {
        const float zero = 0.0f;
        grad_a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &zero);
    }
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        return x * a_.ReadData<float>()[0];
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        const float* g = grad.ReadData<float>();
        const float* x = input_cache_.ReadData<float>();
        double sum = 0.0;
        for (size_t i = 0; i < grad.NumElements(); ++i) sum += static_cast<double>(g[i]) * x[i];
        const float value = static_cast<float>(sum);
        grad_a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &value);
        return grad * a_.ReadData<float>()[0];
    }
    std::map<std::string, cyxwiz::Tensor> GetParameters() override { return {{"a", a_}}; }
    void SetParameters(const std::map<std::string, cyxwiz::Tensor>& params) override {
        const auto it = params.find("a");
        if (it != params.end()) a_ = it->second;
    }
    std::map<std::string, cyxwiz::Tensor> GetGradients() override { return {{"a", grad_a_}}; }
    bool HasParameters() const override { return true; }
    std::string GetName() const override { return "Scale"; }

private:
    cyxwiz::Tensor a_;
    cyxwiz::Tensor grad_a_;
};

class ScaleFactory : public cyxwiz::ITensorNodeFactory {
public:
    explicit ScaleFactory(std::string refuse = {}) : refuse_(std::move(refuse)) {}
    cyxwiz::ExtensionShapeResult InferOutputShape(const std::vector<size_t>& input_shape,
                                                  const std::map<std::string, std::string>&) const override {
        return {true, input_shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(const std::vector<size_t>&,
                                                 const std::map<std::string, std::string>&,
                                                 std::string& error) const override {
        if (!refuse_.empty()) {
            error = refuse_;
            return nullptr;
        }
        return std::make_unique<TrainableScaleModule>(0.5f);
    }

private:
    std::string refuse_;
};

// refuse: CreateModule fails with this reason (an extension that cannot run).
void RegisterScale(const std::string& refuse = {}) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = kProvider;
    d.type_name = "Scale";
    d.version = "1.0.0";
    d.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
    d.role = cyxwiz::ExtensionLayerRole::Activation;
    d.metadata.name = "Scale";
    d.metadata.inputs = {{"Input"}};
    d.metadata.outputs = {{"Output"}};
    registration.tensor_factory = std::make_shared<ScaleFactory>(refuse);
    std::string error;
    if (!cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error)) {
        std::cerr << "register failed: " << error << "\n";
        std::exit(EXIT_FAILURE);
    }
}

gui::MLNode ExtensionNode(const std::string& type_id, bool missing) {
    gui::MLNode node;
    node.id = 50;
    node.type = gui::NodeType::PluginCustom;
    node.name = "My Scale";
    node.extension_type_id = type_id;
    node.extension_missing = missing;
    return node;
}

// Linear 3 -> 3, then the extension's Scale inside the guard the model
// builder puts around it.
std::shared_ptr<cyxwiz::SequentialModel> ModelWithExtension() {
    auto model = std::make_shared<cyxwiz::SequentialModel>();
    model->Add<cyxwiz::LinearModule>(3, 3, true);
    model->AddModule(std::make_unique<cyxwiz::GuardedExtensionModule>(
        std::make_unique<ScaleModule>(0.5f), 50, "My Scale", std::string(kProvider) + ":Scale",
        std::vector<size_t>{3}, nullptr));
    return model;
}

// The tiny causal-LM example with the Scale node between decoder block 2 and
// the vocabulary Dense.
nlohmann::json GraphWithScale(const fs::path& root) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    for (auto& link : graph["links"]) {
        if (link["from_node"] == kDecoderNodeId && link["to_node"] == kDenseNodeId) {
            link["to_node"] = kExtensionNodeId;
            link["to_pin_index"] = 0;
        }
    }
    graph["nodes"].push_back(
        {{"id", kExtensionNodeId}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", "My Scale"},
         {"description", ""}, {"parameters", nlohmann::json::object()}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", std::string(kProvider) + ":Scale"}, {"version", "1.0.0"},
           {"content_hash", ""},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0},
                              {"from_pin_index", 0}, {"to_node", kDenseNodeId}, {"to_pin", 0},
                              {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
}

std::vector<float> Values(const cyxwiz::Tensor& tensor) {
    const float* data = tensor.ReadData<float>();
    return std::vector<float>(data, data + tensor.NumElements());
}

// Same keys, shapes and values.
bool SameParameters(const std::map<std::string, cyxwiz::Tensor>& a, const std::map<std::string, cyxwiz::Tensor>& b,
                    std::string& difference) {
    if (a.size() != b.size()) {
        difference = std::to_string(a.size()) + " vs " + std::to_string(b.size()) + " parameters";
        return false;
    }
    for (const auto& [name, tensor] : a) {
        const auto it = b.find(name);
        if (it == b.end()) {
            difference = name + " is missing";
            return false;
        }
        if (tensor.Shape() != it->second.Shape()) {
            difference = name + " has another shape";
            return false;
        }
        const auto x = Values(tensor);
        const auto y = Values(it->second);
        for (size_t i = 0; i < x.size(); ++i) {
            if (std::fabs(x[i] - y[i]) > 1e-6f) {
                difference = name + "[" + std::to_string(i) + "] differs";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    RegisterScale();
    const std::string scale_id = std::string(kProvider) + ":Scale";

    std::cout << "generated code refuses extension nodes it cannot contain\n";
    {
        const auto trainable = cyxwiz::ExtensionCodeExportRefusal(ExtensionNode(scale_id, false));
        Check(trainable && Contains(*trainable, "My Scale") && Contains(*trainable, scale_id) &&
                  Contains(*trainable, "cannot include an extension node's own code"),
              "a trainable extension node: " + trainable.value_or("(none)"));
        const auto missing = cyxwiz::ExtensionCodeExportRefusal(ExtensionNode("test.gone:Thing", true));
        Check(missing && Contains(*missing, "test.gone:Thing") && Contains(*missing, "not installed"),
              "a node whose extension is not installed: " + missing.value_or("(none)"));
        const auto unknown = cyxwiz::ExtensionCodeExportRefusal(ExtensionNode("test.never:Thing", false));
        Check(unknown && Contains(*unknown, "not installed"), "an extension id nobody registered");
        gui::MLNode dense;
        dense.type = gui::NodeType::Dense;
        dense.name = "Dense";
        Check(!cyxwiz::ExtensionCodeExportRefusal(dense), "a built-in node is not refused");
    }

    std::cout << "ONNX export refuses a model that holds an extension module\n";
    const fs::path work = fs::temp_directory_path() / "cyxwiz_test_extension_model_files";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    {
        auto model = ModelWithExtension();
        cyxwiz::ExportOptions options;
        options.format = cyxwiz::ModelFormat::ONNX;
        cyxwiz::ModelExporter exporter;
        const auto direct = exporter.ExportONNX(*model, (work / "direct.onnx").string(), options);
        Check(!direct.success && Contains(direct.error_message, "extension node My Scale") &&
                  Contains(direct.error_message, scale_id) && Contains(direct.error_message, ".cyxmodel"),
              "ExportONNX names the node: " + direct.error_message);
        Check(!fs::exists(work / "direct.onnx"), "no ONNX file is written");

        cyxwiz::SequentialExecutableModel executable(model);
        const auto through_export =
            exporter.Export(executable, nullptr, nullptr, "", (work / "export.onnx").string(), options);
        Check(!through_export.success && Contains(through_export.error_message, scale_id),
              "Export with format ONNX refuses it too");

        auto plain = std::make_shared<cyxwiz::SequentialModel>();
        plain->Add<cyxwiz::LinearModule>(3, 3, true);
        const auto built_in = exporter.ExportONNX(*plain, (work / "plain.onnx").string(), options);
        Check(!Contains(built_in.error_message, "extension node"),
              "a model of built-in layers is not refused for extension reasons" +
                  (built_in.success ? std::string() : " (" + built_in.error_message + ")"));
    }

    std::cout << "a trained model with an extension node saves and imports with the same weights\n";
    {
        const fs::path root = CYXWIZ_SOURCE_ROOT;
        const fs::path parquet = work / "tokens.parquet";
        Check(cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8), "the Parquet fixture is written");
        cyxwiz::test::InstallQualifiedRouteSnapshot();
        const nlohmann::json graph = GraphWithScale(root);

        cyxwiz::GraphTrainingJobRequest request;
        request.graph_json = graph.dump();
        request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
        request.epochs_override = 2;
        request.checkpoint_dir_override = (work / "checkpoints").string();
        auto trained = cyxwiz::RunGraphTrainingJob(request);
        Check(trained.ok && trained.model,
              "the graph trains" + (trained.error.empty() ? std::string() : " (" + trained.error + ")"));
        if (!trained.model) return EXIT_FAILURE;
        const auto trained_parameters = trained.model->GetParameters();
        std::string scale_key;
        for (const auto& [name, tensor] : trained_parameters)
            if (name.size() > 2 && name.compare(name.size() - 2, 2, ".a") == 0) scale_key = name;
        Check(!scale_key.empty(), "the trained model holds Scale's a (" + scale_key + ")");
        const float trained_a = scale_key.empty() ? 0.0f : Values(trained_parameters.at(scale_key))[0];
        Check(std::fabs(trained_a - 0.5f) > 1e-6f,
              "training changed a from 0.5 (now " + std::to_string(trained_a) + ")");

        // The trainer sized the vocabulary from the data (the example graph
        // says 64); export the graph as trained, so its import rebuilds the
        // same shapes. (Export from the editor's graph does not do this for
        // a text graph yet: a finding outside TOFIX125, see track125.)
        nlohmann::json exported_graph = graph;
        if (trained_parameters.count("layer0.weight")) {
            const std::string vocabulary = std::to_string(trained_parameters.at("layer0.weight").Shape()[0]);
            for (auto& node : exported_graph["nodes"]) {
                auto& params = node["parameters"];
                for (const char* key : {"sequence_vocab_size", "num_embeddings", "vocab_size", "num_classes"})
                    if (params.contains(key)) params[key] = vocabulary;
                if (node.value("id", 0) == kDenseNodeId) params["units"] = vocabulary;
            }
        }

        cyxwiz::ExportOptions options;
        options.format = cyxwiz::ModelFormat::CyxModel;
        options.include_graph = true;
        options.include_training_history = false;
        options.include_optimizer_state = false;
        const fs::path package = work / "scale.cyxmodel";
        cyxwiz::ModelExporter exporter;
        const auto exported =
            exporter.Export(*trained.model, nullptr, nullptr, exported_graph.dump(), package.string(), options);
        Check(exported.success,
              "the package is written" + (exported.success ? std::string() : ": " + exported.error_message));

        cyxwiz::ImportOptions import_options;
        import_options.strict_mode = true;
        cyxwiz::ModelImporter importer;
        cyxwiz::SequentialModel stack;
        const auto imported = importer.ImportCyxModel(package.string(), stack, import_options);
        Check(imported.success,
              "it imports as a layer stack" + (imported.success ? std::string() : ": " + imported.error_message));
        std::string difference;
        Check(imported.success && SameParameters(trained_parameters, stack.GetParameters(), difference),
              "every weight is back, Scale's a included" + (difference.empty() ? "" : " (" + difference + ")"));
        bool rebuilt = false;
        for (size_t i = 0; i < stack.Size(); ++i)
            if (const auto* guard = dynamic_cast<const cyxwiz::GuardedExtensionModule*>(stack.GetModule(i)))
                rebuilt = guard->TypeId() == std::string(kProvider) + ":Scale" && guard->NodeName() == "My Scale";
        Check(rebuilt, "the Scale node is rebuilt through the extension, inside the guard");

        std::shared_ptr<cyxwiz::IExecutableModel> executable;
        const auto as_executable = importer.ImportExecutableModel(package.string(), executable, import_options);
        difference.clear();
        Check(as_executable.success && executable &&
                  SameParameters(trained_parameters, executable->GetParameters(), difference),
              "it imports as an executable model with the same weights" +
                  (as_executable.success ? (difference.empty() ? std::string() : " (" + difference + ")")
                                         : ": " + as_executable.error_message));

        std::cout << "importing without the extension, or when it cannot create its module, says why\n";
        cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider(kProvider);
        cyxwiz::SequentialModel missing;
        const auto without = importer.ImportCyxModel(package.string(), missing, import_options);
        Check(!without.success && Contains(without.error_message, "not installed") &&
                  Contains(without.error_message, std::string(kProvider) + ":Scale"),
              "extension not installed: " + without.error_message);

        RegisterScale("Python is not running yet");
        cyxwiz::SequentialModel refused;
        const auto cannot = importer.ImportCyxModel(package.string(), refused, import_options);
        Check(!cannot.success && Contains(cannot.error_message, "Python is not running yet"),
              "the extension's own reason reaches the import: " + cannot.error_message);
        cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider(kProvider);
        RegisterScale();
    }
    fs::remove_all(work, ec);

    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_extension_model_files: all checks passed\n";
    return 0;
}
