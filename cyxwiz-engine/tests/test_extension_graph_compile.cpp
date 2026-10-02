// The compiler and extension nodes (TOFIX125 P1 step 1.4): an extension node
// on the training path is never passed over. It is reported on the node when
// it is not installed, when it is a simulation node, and (until the trainable
// contract lands) when it is a trainable one.
//
// The graph is the standard training benchmark graph with one extension node
// inserted between "Final normalization" and the per-token Dense.
#include "../src/core/extension_node_document.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_compiler.h"
#include "../src/core/graph_document.h"

#include <cyxwiz/error_codes.h>
#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

#include "../src/core/training_benchmark_graph.inc"

using json = nlohmann::json;

constexpr int kExtensionNodeId = 50;
constexpr int kNormalizationNodeId = 10;
constexpr int kDenseNodeId = 11;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class StubSignalProvider : public cyxwiz::plugin::INodeProvider {
public:
    std::vector<cyxwiz::plugin::PluginNodeTypeInfo> GetNodeTypes() override { return {}; }
    std::string GenerateCode(const std::string&, const std::map<std::string, std::string>&,
                             const std::string&) override {
        return {};
    }
};

class StubTensorFactory : public cyxwiz::ITensorNodeFactory {
public:
    cyxwiz::ExtensionShapeResult InferOutputShape(
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>&) const override {
        return {true, input_shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(
        const std::vector<size_t>&, const std::map<std::string, std::string>&,
        std::string& error) const override {
        error = "not used in this test";
        return nullptr;
    }
};

StubSignalProvider g_signal_provider;

void RegisterNode(cyxwiz::ExtensionNodeKind kind, const std::string& version = "1.0.0") {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = "test.nodes";
    d.type_name = "Scale";
    d.version = version;
    d.kind = kind;
    d.metadata.name = "Scale";
    d.metadata.inputs = {{"Input"}};
    d.metadata.outputs = {{"Output"}};
    if (kind == cyxwiz::ExtensionNodeKind::TensorLayer) {
        registration.tensor_factory = std::make_shared<StubTensorFactory>();
    } else {
        registration.signal_provider = &g_signal_provider;
    }
    std::string error;
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error),
          "register the test node: " + error);
}

void Unregister() { cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider("test.nodes"); }

json ExtensionEntry() {
    return {{"id", kExtensionNodeId},
            {"type", static_cast<int>(gui::NodeType::PluginCustom)},
            {"category", static_cast<int>(gui::NodeCategory::Plugin)},
            {"name", "My scale"},
            {"description", ""},
            {"parameters", json::object()},
            {"pos_x", 0.0},
            {"pos_y", 0.0},
            {"extension",
             {{"contract", 1},
              {"type_id", "test.nodes:Scale"},
              {"version", "1.0.0"},
              {"content_hash", ""},
              {"inputs", json::array({{{"name", "Input"}, {"type", "Tensor"}}})},
              {"outputs", json::array({{{"name", "Output"}, {"type", "Tensor"}}})}}}};
}

// on_path: Final normalization -> extension -> Dense. Otherwise the extension
// node is in the graph with no links.
json GraphWithExtension(bool on_path) {
    json graph = json::parse(kTrainingBenchmarkGraphJson);
    graph["nodes"].push_back(ExtensionEntry());
    if (on_path) {
        bool rewired = false;
        for (auto& link : graph["links"]) {
            if (link["from_node"] == kNormalizationNodeId && link["to_node"] == kDenseNodeId) {
                link["to_node"] = kExtensionNodeId;
                link["to_pin_index"] = 0;
                rewired = true;
            }
        }
        Check(rewired, "the benchmark graph links Final normalization to the Dense");
        graph["links"].push_back({{"id", 900}, {"from_node", kExtensionNodeId}, {"from_pin", 0},
                                  {"from_pin_index", 0}, {"to_node", kDenseNodeId}, {"to_pin", 0},
                                  {"to_pin_index", 0}, {"link_type", 0}});
    }
    return graph;
}

cyxwiz::TrainingConfiguration Compile(const json& graph_json) {
    cyxwiz::GraphDocument graph;
    std::string error;
    Check(cyxwiz::ParseGraphDocument(graph_json.dump(), graph, error), "the graph should load: " + error);
    return cyxwiz::GraphCompiler{}.Compile(graph.nodes, graph.links, true);
}

// Issues about the node itself. An unconnected node also gets the compiler's
// pin connectivity errors, as any node does; those are not the subject here.
std::vector<cyxwiz::ValidationIssue> IssuesOn(const cyxwiz::TrainingConfiguration& config, int node_id) {
    std::vector<cyxwiz::ValidationIssue> result;
    for (const auto& issue : config.issues) {
        if (issue.node_id != node_id) continue;
        if (issue.error_code == cyxwiz::errors::Compiler::InvalidConnectivity) continue;
        result.push_back(issue);
    }
    return result;
}

size_t ErrorCount(const cyxwiz::TrainingConfiguration& config) {
    size_t count = 0;
    for (const auto& issue : config.issues) {
        if (issue.level == cyxwiz::IssueLevel::Error) ++count;
    }
    return count;
}

std::string Describe(const std::vector<cyxwiz::ValidationIssue>& issues) {
    std::string text;
    for (const auto& issue : issues) {
        text += "[" + issue.error_code + "] " + issue.message + " | ";
    }
    return text.empty() ? "(no issues)" : text;
}

size_t g_baseline_errors = 0;

void TestBaseline() {
    const auto config = Compile(json::parse(kTrainingBenchmarkGraphJson));
    g_baseline_errors = ErrorCount(config);
    Check(!config.layers.empty(), "the benchmark graph compiles to layers");
    Check(IssuesOn(config, kExtensionNodeId).empty(), "the unchanged graph has no such node");
}

void TestMissingOnPath() {
    Unregister();
    const auto config = Compile(GraphWithExtension(true));
    const auto issues = IssuesOn(config, kExtensionNodeId);
    Check(issues.size() == 1, "one issue on the missing node: " + Describe(issues));
    Check(issues[0].level == cyxwiz::IssueLevel::Error, "it is an error");
    Check(issues[0].error_code == cyxwiz::errors::External::PluginDependencyFailure,
          "with the plugin dependency code: " + issues[0].error_code);
    Check(issues[0].node_name == "My scale", "it names the node");
    Check(issues[0].message.find("test.nodes:Scale") != std::string::npos &&
              issues[0].message.find("not installed") != std::string::npos,
          "it names the type id and says it is not installed: " + issues[0].message);
    Check(!config.is_valid, "a graph with a missing extension on the training path is not valid");
    Check(ErrorCount(config) == g_baseline_errors + 1, "it adds exactly one error");
}

void TestSignalOnPath() {
    Unregister();
    RegisterNode(cyxwiz::ExtensionNodeKind::Signal);
    const auto config = Compile(GraphWithExtension(true));
    const auto issues = IssuesOn(config, kExtensionNodeId);
    Check(issues.size() == 1, "one issue on the simulation node: " + Describe(issues));
    Check(issues[0].level == cyxwiz::IssueLevel::Error &&
              issues[0].error_code == cyxwiz::errors::Compiler::UnsupportedTrainingNode,
          "an unsupported training node error: " + Describe(issues));
    Check(issues[0].message.find("simulation") != std::string::npos,
          "it says the node is a simulation node: " + issues[0].message);
    Check(!config.is_valid, "a simulation node on the training path makes the graph invalid");
    Unregister();
}

void TestTrainableOnPathIsReported() {
    Unregister();
    RegisterNode(cyxwiz::ExtensionNodeKind::TensorLayer);
    const auto config = Compile(GraphWithExtension(true));
    const auto issues = IssuesOn(config, kExtensionNodeId);
    // Until the trainable contract is compiled (P2) the node must not be
    // passed over: the model would train without it.
    Check(issues.size() == 1 && issues[0].level == cyxwiz::IssueLevel::Error,
          "a trainable extension node is reported, not skipped: " + Describe(issues));
    Check(!config.is_valid, "the graph is not valid while the node cannot be compiled");
    for (const auto& layer : config.layers) {
        Check(layer.node_id != kExtensionNodeId, "no layer is invented for it");
    }
    Unregister();
}

void TestOffPath() {
    Unregister();
    auto config = Compile(GraphWithExtension(false));
    auto issues = IssuesOn(config, kExtensionNodeId);
    Check(issues.size() == 1 && issues[0].level == cyxwiz::IssueLevel::Warning,
          "a missing extension outside the training path is a warning: " + Describe(issues));
    Check(issues[0].error_code == cyxwiz::errors::External::PluginDependencyFailure,
          "with the plugin dependency code");

    RegisterNode(cyxwiz::ExtensionNodeKind::Signal);
    config = Compile(GraphWithExtension(false));
    Check(IssuesOn(config, kExtensionNodeId).empty(),
          "an installed simulation node outside the training path is not reported: " +
              Describe(IssuesOn(config, kExtensionNodeId)));
    Unregister();
}

void TestLoadNoteBecomesWarning() {
    Unregister();
    RegisterNode(cyxwiz::ExtensionNodeKind::Signal, "2.0.0");  // the graph was saved with 1.0.0
    const auto config = Compile(GraphWithExtension(false));
    const auto issues = IssuesOn(config, kExtensionNodeId);
    Check(issues.size() == 1 && issues[0].level == cyxwiz::IssueLevel::Warning,
          "a changed extension is a warning on the node: " + Describe(issues));
    Check(issues[0].message.find("1.0.0") != std::string::npos &&
              issues[0].message.find("2.0.0") != std::string::npos,
          "it names both versions: " + issues[0].message);
    Unregister();
}

}  // namespace

int main() {
    TestBaseline();
    TestMissingOnPath();
    TestSignalOnPath();
    TestTrainableOnPathIsReported();
    TestOffPath();
    TestLoadNoteBecomesWarning();
    std::cout << "test_extension_graph_compile: all checks passed\n";
    return 0;
}
