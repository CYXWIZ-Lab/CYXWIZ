// Extension node identity on the canvas (TOFIX125 P1 step 1.2): a node is
// created from its type id, carries that identity in its own fields, and its
// metadata resolves the same way for built-in, registered and missing nodes.
#include "../src/core/extension_node_metadata.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_node_factory.h"
#include "../src/core/node_metadata_registry.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

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

void RegisterGate() {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = "test.nodes";
    d.type_name = "Gate";
    d.version = "1.2.0";
    d.content_hash = "sha256:abc";
    d.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
    d.metadata.name = "Gate";
    d.metadata.brief_description = "Scales the signal by a control";
    d.metadata.inputs = {{"Signal", gui::PinType::Tensor, true, "The tensor to scale"},
                         {"Control", gui::PinType::Tensor, true, "Scale per sample"}};
    d.metadata.outputs = {{"Output", gui::PinType::Tensor, true, "Scaled tensor"}};
    d.metadata.parameters = {{"gain", "float", "2.5", "Extra gain"},
                             {"mode", "enum", "soft", "Gate mode", {"soft", "hard"}}};
    registration.tensor_factory = std::make_shared<StubTensorFactory>();
    std::string error;
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error),
          "register the test node: " + error);
}

void TestCreateRegisteredNode() {
    int next_node_id = 10;
    int next_pin_id = 100;
    const gui::MLNode node =
        gui::CreateExtensionGraphNode("test.nodes:Gate", next_node_id, next_pin_id);

    Check(node.type == gui::NodeType::PluginCustom, "an extension node is a PluginCustom node");
    Check(node.category == gui::NodeCategory::Plugin, "its category is Plugin");
    Check(node.id == 10 && next_node_id == 11, "the node id comes from the cursor and advances it");
    Check(node.name == "Gate", "the node shows the display name");

    Check(node.extension_type_id == "test.nodes:Gate", "the node carries its type id");
    Check(node.extension_version == "1.2.0", "the node carries the contract version");
    Check(node.extension_content_hash == "sha256:abc", "the node carries the content hash");
    Check(!node.extension_missing, "a registered extension is not missing");

    Check(node.inputs.size() == 2 && node.inputs[0].name == "Signal" &&
              node.inputs[1].name == "Control",
          "inputs follow the descriptor, in order");
    Check(node.outputs.size() == 1 && node.outputs[0].name == "Output", "outputs follow the descriptor");
    Check(node.inputs[0].is_input && !node.outputs[0].is_input, "pin direction is set");
    Check(node.inputs[0].description == "The tensor to scale", "pin descriptions are kept");
    Check(node.inputs[0].id == 100 && node.inputs[1].id == 101 && node.outputs[0].id == 102 &&
              next_pin_id == 103,
          "pin ids come from the cursor and advance it");

    Check(node.parameters.count("gain") && node.parameters.at("gain") == "2.5",
          "parameters start at their defaults");
    Check(node.parameters.count("mode") && node.parameters.at("mode") == "soft",
          "enum parameters start at their default");
}

void TestCreateThroughGeneralFactory() {
    int next_node_id = 1;
    int next_pin_id = 1;
    const gui::MLNode node = gui::CreateGraphNode(gui::NodeType::PluginCustom, "test.nodes:Gate",
                                                  next_node_id, next_pin_id);
    Check(node.extension_type_id == "test.nodes:Gate" && node.inputs.size() == 2 &&
              !node.extension_missing,
          "the general factory treats a PluginCustom name as the type id");
}

void TestCreateMissingNode() {
    int next_node_id = 1;
    int next_pin_id = 1;
    const gui::MLNode node =
        gui::CreateExtensionGraphNode("nobody.home:Ghost", next_node_id, next_pin_id);
    Check(node.type == gui::NodeType::PluginCustom, "a missing extension is still a PluginCustom node");
    Check(node.extension_type_id == "nobody.home:Ghost", "a missing extension keeps its type id");
    Check(node.extension_missing, "a missing extension is marked missing");
    Check(node.inputs.size() == 1 && node.outputs.size() == 1,
          "with nothing saved, a missing extension gets one input and one output");
    Check(node.name.find("Ghost") != std::string::npos,
          "a missing extension is named after its type: " + node.name);
}

void TestIdentityIsNotAParameter() {
    int next_node_id = 1;
    int next_pin_id = 1;
    const gui::MLNode node =
        gui::CreateExtensionGraphNode("test.nodes:Gate", next_node_id, next_pin_id);
    // The saved-file format still reads this key until step 1.3 replaces it
    // with the extension block; it must agree with the node's own field.
    const auto it = node.parameters.find("plugin_qualified_name");
    Check(it != node.parameters.end() && it->second == node.extension_type_id,
          "the transitional parameter agrees with the identity field");
}

void TestResolveMetadata() {
    auto& catalog = cyxwiz::NodeMetadataRegistry::Instance();
    catalog.Initialize();

    int next_node_id = 1;
    int next_pin_id = 1;

    const gui::MLNode dense =
        gui::CreateGraphNode(gui::NodeType::Dense, "Dense", next_node_id, next_pin_id);
    const auto dense_metadata = cyxwiz::ResolveNodeMetadata(dense);
    const auto* catalog_dense = catalog.GetMetadata(gui::NodeType::Dense);
    Check(dense_metadata.has_value() && catalog_dense != nullptr &&
              dense_metadata->name == catalog_dense->name &&
              dense_metadata->type == gui::NodeType::Dense,
          "a built-in node resolves to its catalog entry");

    const gui::MLNode gate =
        gui::CreateExtensionGraphNode("test.nodes:Gate", next_node_id, next_pin_id);
    const auto gate_metadata = cyxwiz::ResolveNodeMetadata(gate);
    Check(gate_metadata.has_value() && gate_metadata->name == "Gate",
          "an extension node resolves to its own name, not the generic plugin entry");
    Check(gate_metadata->parameters.size() == 2 && gate_metadata->parameters[1].type == "enum" &&
              gate_metadata->parameters[1].enum_values.size() == 2,
          "an extension node resolves to its typed parameters");
    Check(gate_metadata->IsImplemented(), "a registered extension node is implemented");
    Check(cyxwiz::CanAddNodeToGraph(*gate_metadata),
          "a registered extension node passes the add gate");

    const gui::MLNode ghost =
        gui::CreateExtensionGraphNode("nobody.home:Ghost", next_node_id, next_pin_id);
    const auto ghost_metadata = cyxwiz::ResolveNodeMetadata(ghost);
    Check(ghost_metadata.has_value(), "a missing extension still resolves to metadata");
    Check(ghost_metadata->type == gui::NodeType::PluginCustom, "its type is PluginCustom");
    Check(ghost_metadata->badge == "Not installed", "a missing extension is badged: " + ghost_metadata->badge);
    Check(!cyxwiz::CanAddNodeToGraph(*ghost_metadata), "a missing extension cannot be added");
    Check(ghost_metadata->inputs.size() == ghost.inputs.size() &&
              ghost_metadata->outputs.size() == ghost.outputs.size(),
          "a missing extension's ports are the node's own pins");
    Check(ghost_metadata->brief_description.find("nobody.home:Ghost") != std::string::npos,
          "the description names the missing type id");

    // A node that was created while the extension was installed, looked at
    // after the extension is gone.
    cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider("test.nodes");
    const auto orphan_metadata = cyxwiz::ResolveNodeMetadata(gate);
    Check(orphan_metadata.has_value() && orphan_metadata->badge == "Not installed",
          "a node whose extension was removed resolves as missing");
    Check(orphan_metadata->inputs.size() == 2 && orphan_metadata->inputs[1].name == "Control",
          "and keeps its own pins");
    RegisterGate();
}

}  // namespace

int main() {
    RegisterGate();
    TestCreateRegisteredNode();
    TestCreateThroughGeneralFactory();
    TestCreateMissingNode();
    TestIdentityIsNotAParameter();
    TestResolveMetadata();
    cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider("test.nodes");
    std::cout << "test_extension_node_factory: all checks passed\n";
    return 0;
}
