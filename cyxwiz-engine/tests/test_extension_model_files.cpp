// Model files, code generation and export with extension nodes (TOFIX125 P6).
// 6.1: what cannot contain an extension node's computation refuses it by
// name instead of writing a different model: ONNX export, and generated code
// (PyTorch, TensorFlow, Keras, pycyxwiz) through ExtensionCodeExportRefusal.
#include "../src/core/executable_model.h"
#include "../src/core/extension_node_metadata.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/guarded_extension_module.h"
#include "../src/core/model_exporter.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kProvider = "test.files";

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

class ScaleFactory : public cyxwiz::ITensorNodeFactory {
public:
    cyxwiz::ExtensionShapeResult InferOutputShape(const std::vector<size_t>& input_shape,
                                                  const std::map<std::string, std::string>&) const override {
        return {true, input_shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(const std::vector<size_t>&,
                                                 const std::map<std::string, std::string>&,
                                                 std::string&) const override {
        return std::make_unique<ScaleModule>(0.5f);
    }
};

void RegisterScale() {
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
    registration.tensor_factory = std::make_shared<ScaleFactory>();
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
    fs::remove_all(work, ec);

    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_extension_model_files: all checks passed\n";
    return 0;
}
