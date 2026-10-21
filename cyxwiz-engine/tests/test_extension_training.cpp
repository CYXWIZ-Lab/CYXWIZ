// Trainable extension nodes (TOFIX125 P2 steps 2.1 and 2.2): a node supplied
// through ExtensionNodeRegistry compiles to a layer, builds a guarded module
// and trains inside a real graph, headless, on the core alone (what a Server
// Node links). The graph is the tiny causal-LM example with one extension node
// inserted between the second decoder block and the vocabulary Dense.
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/guarded_extension_module.h"
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

constexpr int kExtensionNodeId = 50;
constexpr int kDecoderNodeId = 8;
constexpr int kDenseNodeId = 9;
constexpr const char* kProvider = "test.trainable";

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

std::vector<float> HostValues(const cyxwiz::Tensor& tensor) {
    const float* data = tensor.ReadData<float>();
    return std::vector<float>(data, data + tensor.NumElements());
}

// y = x / (1 + |x|), no parameters.
class SoftsignModule : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        return x / (x.Abs() + 1.0f);
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        const auto d = input_cache_.Abs() + 1.0f;
        return grad / (d * d);
    }
    std::string GetName() const override { return "Softsign"; }
};

// y = a * x with one learnable scalar a.
class ScaleModule : public cyxwiz::Module {
public:
    explicit ScaleModule(float initial) : a_(std::vector<size_t>{1}, &initial) {
        const float zero = 0.0f;
        grad_a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &zero);
    }
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        input_cache_ = x;
        return x * HostValues(a_)[0];
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override {
        const auto g = HostValues(grad);
        const auto x = HostValues(input_cache_);
        double sum = 0.0;
        for (size_t i = 0; i < g.size(); ++i) sum += static_cast<double>(g[i]) * x[i];
        const float value = static_cast<float>(sum);
        grad_a_ = cyxwiz::Tensor(std::vector<size_t>{1}, &value);
        return grad * HostValues(a_)[0];
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

// Drops the last feature: the output does not match what its shape function said.
class WrongShapeModule : public cyxwiz::Module {
public:
    cyxwiz::Tensor Forward(const cyxwiz::Tensor& x) override {
        auto shape = x.Shape();
        shape.back() -= 1;
        return cyxwiz::Tensor(shape);
    }
    cyxwiz::Tensor Backward(const cyxwiz::Tensor& grad) override { return grad; }
    std::string GetName() const override { return "WrongShape"; }
};

enum class Kind { Softsign, Scale, WrongShape, RefusesShape };

class TestFactory : public cyxwiz::ITensorNodeFactory {
public:
    explicit TestFactory(Kind kind) : kind_(kind) {}
    cyxwiz::ExtensionShapeResult InferOutputShape(
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>&) const override {
        if (kind_ == Kind::RefusesShape) return {false, {}, "needs a rank-1 input"};
        return {true, input_shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(
        const std::vector<size_t>&, const std::map<std::string, std::string>& parameters,
        std::string&) const override {
        switch (kind_) {
            case Kind::Softsign: return std::make_unique<SoftsignModule>();
            case Kind::Scale: {
                const auto it = parameters.find("initial_a");
                return std::make_unique<ScaleModule>(it == parameters.end() ? 1.0f : std::stof(it->second));
            }
            case Kind::WrongShape: return std::make_unique<WrongShapeModule>();
            case Kind::RefusesShape: return nullptr;
        }
        return nullptr;
    }

private:
    Kind kind_;
};

void Register(const std::string& type_name, Kind kind) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = kProvider;
    d.type_name = type_name;
    d.version = "1.0.0";
    d.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
    d.role = cyxwiz::ExtensionLayerRole::Activation;
    d.metadata.name = type_name;
    d.metadata.inputs = {{"Input"}};
    d.metadata.outputs = {{"Output"}};
    d.metadata.parameters = {{"initial_a", "float", "0.5", "Starting value of a"}};
    registration.tensor_factory = std::make_shared<TestFactory>(kind);
    std::string error;
    if (!cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error)) {
        std::cerr << "register failed: " << error << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// The example graph with the extension node between decoder block 2 and the Dense.
nlohmann::json GraphWith(const fs::path& root, const std::string& type_name,
                         std::map<std::string, std::string> parameters = {}) {
    nlohmann::json graph = cyxwiz::test::LoadTokenWindowGraph(root);
    bool rewired = false;
    for (auto& link : graph["links"]) {
        if (link["from_node"] == kDecoderNodeId && link["to_node"] == kDenseNodeId) {
            link["to_node"] = kExtensionNodeId;
            link["to_pin_index"] = 0;
            rewired = true;
        }
    }
    if (!rewired) throw std::runtime_error("decoder block 2 is not linked to the Dense");
    nlohmann::json params = nlohmann::json::object();
    for (const auto& [key, value] : parameters) params[key] = value;
    graph["nodes"].push_back(
        {{"id", kExtensionNodeId}, {"type", static_cast<int>(gui::NodeType::PluginCustom)},
         {"category", static_cast<int>(gui::NodeCategory::Plugin)}, {"name", "My " + type_name},
         {"description", ""}, {"parameters", params}, {"pos_x", 0.0}, {"pos_y", 0.0},
         {"extension",
          {{"contract", 1}, {"type_id", std::string(kProvider) + ":" + type_name}, {"version", "1.0.0"},
           {"content_hash", ""},
           {"inputs", nlohmann::json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
           {"outputs", nlohmann::json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}});
    graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0},
                              {"from_pin_index", 0}, {"to_node", kDenseNodeId}, {"to_pin", 0},
                              {"to_pin_index", 0}, {"link_type", 0}});
    return graph;
}

cyxwiz::GraphTrainingJobResult Train(const nlohmann::json& graph, const fs::path& parquet,
                                     const fs::path& checkpoints, std::vector<float>* losses = nullptr) {
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = graph.dump();
    request.dataset_files["tiny_causal_lm_tokens"] = parquet.string();
    request.epochs_override = 3;
    request.checkpoint_dir_override = checkpoints.string();
    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_epoch = [losses](int, float train_loss, float, float, float, float) {
        if (losses) losses->push_back(train_loss);
    };
    return cyxwiz::RunGraphTrainingJob(request, callbacks);
}

// A module of the trained model's layer stack (the job returns any
// executable model; these graphs train as a layer stack).
cyxwiz::Module* FindModule(cyxwiz::IExecutableModel& executable, const std::string& name) {
    cyxwiz::SequentialModel* model = executable.AsSequentialModel();
    if (!model) return nullptr;
    for (size_t i = 0; i < model->Size(); ++i) {
        if (model->GetModule(i) && model->GetModule(i)->GetName() == name) return model->GetModule(i);
    }
    return nullptr;
}

}  // namespace

int main() {
    const fs::path root = CYXWIZ_SOURCE_ROOT;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_extension_training_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);
    const fs::path parquet = work / "tokens.parquet";
    if (!cyxwiz::test::WriteTokenWindows(root, parquet, 0, 8)) {
        std::cerr << "could not write the Parquet fixture\n";
        return EXIT_FAILURE;
    }
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    Register("Softsign", Kind::Softsign);
    Register("Scale", Kind::Scale);
    Register("WrongShape", Kind::WrongShape);
    Register("RefusesShape", Kind::RefusesShape);

    std::cout << "compiles a trainable extension node as a layer\n";
    {
        cyxwiz::GraphDocument document;
        std::string error;
        Check(cyxwiz::ParseGraphDocument(GraphWith(root, "Softsign").dump(), document, error),
              "the graph loads" + (error.empty() ? "" : " (" + error + ")"));
        const auto config = cyxwiz::GraphCompiler{}.Compile(document.nodes, document.links, true);
        bool found = false;
        for (const auto& layer : config.layers) {
            if (layer.node_id != kExtensionNodeId) continue;
            found = true;
            Check(layer.type == gui::NodeType::PluginCustom &&
                      layer.extension_type_id == "test.trainable:Softsign",
                  "the layer carries the extension's type id");
            Check(!layer.input_shape.empty() && layer.output_shape == layer.input_shape,
                  "its shapes come from the extension's shape function: " +
                      cyxwiz::FormatTensorShape(layer.output_shape));
        }
        Check(found, "the extension node became a compiled layer");
        bool node_error = false;
        for (const auto& issue : config.issues) {
            if (issue.node_id == kExtensionNodeId && issue.level == cyxwiz::IssueLevel::Error) {
                node_error = true;
                std::cout << "       issue: " << issue.message << "\n";
            }
        }
        Check(!node_error, "no error on the extension node");
        Check(config.is_valid, "the graph is valid");
    }

    std::cout << "trains with a Softsign extension node\n";
    {
        std::vector<float> losses;
        auto result = Train(GraphWith(root, "Softsign"), parquet, work / "ck_softsign", &losses);
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        Check(result.model && FindModule(*result.model, "Softsign") != nullptr,
              "the trained model contains the extension's module");
        Check(losses.size() == 3, "three epochs reported");
        bool finite = !losses.empty();
        for (float loss : losses) finite = finite && std::isfinite(loss);
        Check(finite, "every epoch loss is finite");
        Check(losses.size() == 3 && losses.back() < losses.front(),
              "the loss goes down (" + (losses.empty() ? std::string("none")
                                                       : std::to_string(losses.front()) + " -> " +
                                                             std::to_string(losses.back())) + ")");
    }

    std::cout << "trains a learnable extension parameter\n";
    {
        auto result = Train(GraphWith(root, "Scale", {{"initial_a", "0.5"}}), parquet, work / "ck_scale");
        Check(result.ok, "the job succeeded" + (result.error.empty() ? "" : " (" + result.error + ")"));
        cyxwiz::Module* scale = result.model ? FindModule(*result.model, "Scale") : nullptr;
        Check(scale != nullptr, "the model contains the Scale module");
        if (scale) {
            const auto params = scale->GetParameters();
            const float a = params.count("a") ? HostValues(params.at("a"))[0] : 0.5f;
            Check(std::fabs(a - 0.5f) > 1e-6f, "the optimizer changed a from 0.5 (now " + std::to_string(a) + ")");
            bool named = false;
            for (const auto& [key, value] : result.model->GetParameters()) {
                named = named || (key.find(".a") != std::string::npos && key.rfind("layer", 0) == 0);
            }
            Check(named, "the model names it layer<i>.a, as the compiled card counts parameters");
        }
    }

    std::cout << "a module that returns the wrong shape stops training at the node\n";
    {
        auto result = Train(GraphWith(root, "WrongShape"), parquet, work / "ck_wrong");
        Check(!result.ok, "the job fails");
        Check(result.error.find("My WrongShape") != std::string::npos &&
                  result.error.find("expected output shape") != std::string::npos,
              "the error names the node and the shapes: " + result.error);
    }

    std::cout << "a shape function that refuses its input is a compile error on the node\n";
    {
        cyxwiz::GraphDocument document;
        std::string error;
        Check(cyxwiz::ParseGraphDocument(GraphWith(root, "RefusesShape").dump(), document, error), "loads");
        const auto config = cyxwiz::GraphCompiler{}.Compile(document.nodes, document.links, true);
        bool reported = false;
        for (const auto& issue : config.issues) {
            reported = reported || (issue.node_id == kExtensionNodeId && issue.level == cyxwiz::IssueLevel::Error &&
                                    issue.message.find("needs a rank-1 input") != std::string::npos);
        }
        Check(reported, "the extension's reason is reported on the node");
        Check(!config.is_valid, "the graph is not valid");
    }

    std::cout << "a remote job refuses a graph with an extension node, before anything is sent\n";
    {
        cyxwiz::GraphJobDatasetPlan plan;
        std::string error;
        Check(cyxwiz::PlanGraphJobDatasets(cyxwiz::test::LoadTokenWindowGraph(root).dump(), plan, error),
              "the example graph without it can be planned" + (error.empty() ? "" : " (" + error + ")"));
        error.clear();
        Check(!cyxwiz::PlanGraphJobDatasets(GraphWith(root, "Softsign").dump(), plan, error),
              "with an extension node it is refused");
        Check(error.find("test.trainable:Softsign") != std::string::npos &&
                  error.find("Remote nodes do not run extension nodes") != std::string::npos,
              "the reason names the node: " + error);
    }

    cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider(kProvider);
    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "test_extension_training: all checks passed\n";
    return 0;
}
