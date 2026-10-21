// Saved graphs with extension nodes (TOFIX125 P1 step 1.3): the "extension"
// block carries the type id and the node's pins, so a graph loads without
// loss whether or not the extension is installed. Headless: the same loader
// the Engine and the Server Node use.
#include "../src/core/extension_node_document.h"
#include "../src/core/extension_node_registry.h"
#include "../src/core/graph_document.h"
#include "../src/core/graph_node_factory.h"

#include <cyxwiz/sequential.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

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

// Two inputs and one output, so a link to the second input proves that pins
// keep their positions.
void RegisterGate(const std::string& version = "1.0.0", const std::string& hash = "sha256:one",
                  bool with_third_input = false) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    auto& d = registration.descriptor;
    d.provider_id = "test.nodes";
    d.type_name = "Gate";
    d.version = version;
    d.content_hash = hash;
    d.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
    d.metadata.name = "Gate";
    d.metadata.inputs = {{"Signal"}, {"Control"}};
    if (with_third_input) d.metadata.inputs.push_back({"Bias"});
    d.metadata.outputs = {{"Output"}};
    d.metadata.parameters = {{"gain", "float", "2.5", "Extra gain"}};
    registration.tensor_factory = std::make_shared<StubTensorFactory>();
    std::string error;
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error),
          "register the test node: " + error);
}

void UnregisterGate() { cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider("test.nodes"); }

json GateBlock() {
    return {{"contract", 1},
            {"type_id", "test.nodes:Gate"},
            {"version", "1.0.0"},
            {"content_hash", "sha256:one"},
            {"inputs", json::array({{{"name", "Signal"}, {"type", "Tensor"}},
                                    {{"name", "Control"}, {"type", "Tensor"}}})},
            {"outputs", json::array({{{"name", "Output"}, {"type", "Tensor"}}})}};
}

json NodeEntry(int id, int type, const std::string& name) {
    return {{"id", id}, {"type", type}, {"category", 0}, {"name", name}, {"description", ""},
            {"parameters", json::object()}, {"pos_x", 0.0}, {"pos_y", 0.0}};
}

json LinkEntry(int id, int from_node, int from_index, int to_node, int to_index) {
    return {{"id", id}, {"from_node", from_node}, {"from_pin", 0}, {"from_pin_index", from_index},
            {"to_node", to_node}, {"to_pin", 0}, {"to_pin_index", to_index}, {"link_type", 0}};
}

// Dense -> Gate.Control, Gate.Output -> Dense.
json GraphWithGate(const json& extension_block, bool with_block = true) {
    const int dense = static_cast<int>(gui::NodeType::Dense);
    const int plugin = static_cast<int>(gui::NodeType::PluginCustom);
    json gate = NodeEntry(2, plugin, "My gate");
    gate["parameters"] = {{"gain", "7"}};
    if (with_block) gate["extension"] = extension_block;
    json document;
    document["version"] = "2.1";
    document["data_boundary_version"] = 2;
    document["data_validator_contract_version"] = 2;
    document["evaluation_table_contract_version"] = 2;
    document["classical_tree_table_contract_version"] = 2;
    document["nodes"] = json::array({NodeEntry(1, dense, "Dense A"), gate, NodeEntry(3, dense, "Dense B")});
    document["links"] = json::array({LinkEntry(1, 1, 0, 2, 1), LinkEntry(2, 2, 0, 3, 0)});
    return document;
}

const gui::MLNode& NodeById(const cyxwiz::GraphDocument& graph, int id) {
    for (const auto& node : graph.nodes) {
        if (node.id == id) return node;
    }
    Check(false, "node " + std::to_string(id) + " should exist");
    return graph.nodes.front();
}

cyxwiz::GraphDocument Load(const json& document, const std::string& what) {
    cyxwiz::GraphDocument graph;
    std::string error;
    Check(cyxwiz::ParseGraphDocument(document.dump(), graph, error), what + " should load: " + error);
    return graph;
}

std::string LoadError(const json& document, const std::string& what) {
    cyxwiz::GraphDocument graph;
    std::string error;
    Check(!cyxwiz::ParseGraphDocument(document.dump(), graph, error), what + " should not load");
    Check(!error.empty(), what + " should give a reason");
    return error;
}

void CheckGateLinks(const cyxwiz::GraphDocument& graph, const std::string& what) {
    Check(graph.links.size() == 2, what + ": both links load");
    const auto& gate = NodeById(graph, 2);
    const auto& dense_a = NodeById(graph, 1);
    const auto& dense_b = NodeById(graph, 3);
    bool into_control = false;
    bool out_of_gate = false;
    for (const auto& link : graph.links) {
        if (link.from_node == 1 && link.to_node == 2) {
            into_control = link.to_pin == gate.inputs[1].id && link.from_pin == dense_a.outputs[0].id;
        }
        if (link.from_node == 2 && link.to_node == 3) {
            out_of_gate = link.from_pin == gate.outputs[0].id && link.to_pin == dense_b.inputs[0].id;
        }
    }
    Check(into_control, what + ": the link still ends at the second input, Control");
    Check(out_of_gate, what + ": the link still starts at Output");
}

void TestPinTypeText() {
    const gui::PinType all[] = {gui::PinType::Tensor, gui::PinType::Labels, gui::PinType::Parameters,
                                gui::PinType::Loss, gui::PinType::Optimizer, gui::PinType::Dataset};
    for (const auto type : all) {
        gui::PinType back = gui::PinType::Tensor;
        Check(cyxwiz::PinTypeFromText(cyxwiz::PinTypeToText(type), back) && back == type,
              std::string("pin type text round trip: ") + cyxwiz::PinTypeToText(type));
    }
    gui::PinType ignored = gui::PinType::Tensor;
    Check(!cyxwiz::PinTypeFromText("Image", ignored), "an unknown pin type is refused");
    Check(!cyxwiz::PinTypeFromText("", ignored), "an empty pin type is refused");
}

void TestWriteBlock() {
    RegisterGate();
    int next_node_id = 1;
    int next_pin_id = 1;
    const auto node = gui::CreateExtensionGraphNode("test.nodes:Gate", next_node_id, next_pin_id);
    const json block = cyxwiz::WriteExtensionBlock(node);
    Check(block == GateBlock(), "the written block holds type id, version, hash and pins: " + block.dump());
    Check(node.parameters.count("plugin_qualified_name") == 0,
          "the identity is no longer a node parameter");
    UnregisterGate();
}

void TestLoadInstalled() {
    RegisterGate();
    const auto graph = Load(GraphWithGate(GateBlock()), "a graph with an installed extension");
    const auto& gate = NodeById(graph, 2);
    Check(gate.type == gui::NodeType::PluginCustom && gate.extension_type_id == "test.nodes:Gate",
          "the node is identified by the saved type id, not by its name");
    Check(!gate.extension_missing, "an installed extension is not missing");
    Check(gate.name == "My gate", "the user's node name is kept");
    Check(gate.inputs.size() == 2 && gate.inputs[1].name == "Control" && gate.outputs.size() == 1,
          "pins come from the registry");
    Check(gate.parameters.count("gain") && gate.parameters.at("gain") == "7",
          "saved parameter values are kept");
    Check(gate.extension_version == "1.0.0" && gate.extension_content_hash == "sha256:one",
          "version and hash are kept");
    Check(gate.extension_load_note.empty(), "nothing changed, so there is no note: " + gate.extension_load_note);
    CheckGateLinks(graph, "installed");
    UnregisterGate();
}

void TestLoadMissingKeepsEverything() {
    UnregisterGate();
    const auto graph = Load(GraphWithGate(GateBlock()), "a graph whose extension is not installed");
    const auto& gate = NodeById(graph, 2);
    Check(gate.type == gui::NodeType::PluginCustom, "the node is kept as an extension node");
    Check(gate.extension_missing, "the node is marked missing");
    Check(gate.extension_type_id == "test.nodes:Gate", "the type id is kept");
    Check(gate.extension_version == "1.0.0" && gate.extension_content_hash == "sha256:one",
          "version and hash are kept");
    Check(gate.name == "My gate", "the user's node name is kept");
    Check(gate.inputs.size() == 2 && gate.inputs[0].name == "Signal" &&
              gate.inputs[1].name == "Control" && gate.outputs.size() == 1 &&
              gate.outputs[0].name == "Output",
          "the saved pins are rebuilt, in order");
    Check(gate.parameters.count("gain") && gate.parameters.at("gain") == "7",
          "saved parameter values are kept");
    CheckGateLinks(graph, "missing");

    // Saving it again loses nothing.
    Check(cyxwiz::WriteExtensionBlock(gate) == GateBlock(),
          "a missing node writes back the block it was loaded from");

    // Installed later: the same document loads as a real node.
    RegisterGate();
    json resaved = GraphWithGate(cyxwiz::WriteExtensionBlock(gate));
    const auto later = Load(resaved, "the re-saved graph after installing the extension");
    Check(!NodeById(later, 2).extension_missing, "after installing, the node is no longer missing");
    CheckGateLinks(later, "installed later");
    UnregisterGate();
}

void TestRefusals() {
    RegisterGate();

    const std::string no_block = LoadError(GraphWithGate(json(), false), "an extension node without a block");
    Check(no_block.find("My gate") != std::string::npos && no_block.find("extension") != std::string::npos,
          "the reason names the node and the missing block: " + no_block);

    json no_type_id = GateBlock();
    no_type_id.erase("type_id");
    Check(LoadError(GraphWithGate(no_type_id), "a block without a type id").find("type_id") !=
              std::string::npos,
          "the reason names the missing field");

    json bad_type_id = GateBlock();
    bad_type_id["type_id"] = "Gate";
    LoadError(GraphWithGate(bad_type_id), "a type id without a provider");

    json bad_pin = GateBlock();
    bad_pin["inputs"][0]["type"] = "Image";
    Check(LoadError(GraphWithGate(bad_pin), "an unknown pin type").find("Image") != std::string::npos,
          "the reason names the pin type");

    json newer = GateBlock();
    newer["contract"] = 2;
    Check(LoadError(GraphWithGate(newer), "a block from a newer contract").find("newer") !=
              std::string::npos,
          "the reason says the file is newer");

    json no_pins = GateBlock();
    no_pins.erase("inputs");
    LoadError(GraphWithGate(no_pins), "a block without its pin list");

    UnregisterGate();
}

void TestChangedExtensionIsNoted() {
    // Same pins, newer code.
    RegisterGate("1.1.0", "sha256:two");
    auto graph = Load(GraphWithGate(GateBlock()), "a graph saved with an older version");
    const auto& updated = NodeById(graph, 2);
    Check(!updated.extension_missing && updated.extension_version == "1.1.0" &&
              updated.extension_content_hash == "sha256:two",
          "the node follows the installed version");
    Check(updated.extension_load_note.find("1.0.0") != std::string::npos &&
              updated.extension_load_note.find("1.1.0") != std::string::npos,
          "the note names both versions: " + updated.extension_load_note);
    CheckGateLinks(graph, "newer version");
    UnregisterGate();

    // Same version, different code.
    RegisterGate("1.0.0", "sha256:edited");
    graph = Load(GraphWithGate(GateBlock()), "a graph saved before the code was edited");
    Check(NodeById(graph, 2).extension_load_note.find("code") != std::string::npos,
          "the note says the code changed: " + NodeById(graph, 2).extension_load_note);
    UnregisterGate();

    // A pin was added.
    RegisterGate("2.0.0", "sha256:three", true);
    graph = Load(GraphWithGate(GateBlock()), "a graph saved before a pin was added");
    const auto& wider = NodeById(graph, 2);
    Check(wider.inputs.size() == 3, "the node has the installed pins");
    Check(wider.extension_load_note.find("pins") != std::string::npos,
          "the note says the pins changed: " + wider.extension_load_note);
    CheckGateLinks(graph, "pin added at the end");
    UnregisterGate();
}

void TestBuiltInNodesHaveNoBlock() {
    int next_node_id = 1;
    int next_pin_id = 1;
    const auto dense = gui::CreateGraphNode(gui::NodeType::Dense, "Dense", next_node_id, next_pin_id);
    Check(cyxwiz::WriteExtensionBlock(dense).is_null(), "a built-in node writes no extension block");
}

}  // namespace

int main() {
    UnregisterGate();
    TestPinTypeText();
    TestWriteBlock();
    TestLoadInstalled();
    TestLoadMissingKeepsEverything();
    TestRefusals();
    TestChangedExtensionIsNoted();
    TestBuiltInNodesHaveNoBlock();
    std::cout << "test_extension_graph_document: all checks passed\n";
    return 0;
}
