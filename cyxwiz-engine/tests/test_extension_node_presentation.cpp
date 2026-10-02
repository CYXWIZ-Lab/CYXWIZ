// The editor's presentation of extension nodes (TOFIX125 P1 step 1.5), on
// the real MuJoCo plugin's node list (plugins/simulation/mujoco/src/
// mujoco_plugin.cpp, GetNodeTypes), described the way the plugin manager
// describes it.
#include "../src/core/extension_node_presentation.h"
#include "../src/core/extension_node_registry.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

constexpr const char* kMuJoCo = "com.cyxwiz.simulation.mujoco";

cyxwiz::plugin::PluginNodeTypeInfo Info(const char* type_name, const char* display_name,
                                        const char* category, uint32_t color,
                                        std::vector<cyxwiz::plugin::PluginNodeTypeInfo::PinInfo> pins,
                                        std::map<std::string, std::string> parameters) {
    cyxwiz::plugin::PluginNodeTypeInfo info;
    info.type_name = type_name;
    info.display_name = display_name;
    info.category = category;
    info.color = color;
    info.pins = std::move(pins);
    info.default_parameters = std::move(parameters);
    return info;
}

// The five MuJoCo nodes, in the plugin's own order.
std::vector<cyxwiz::ExtensionNodeDescriptor> MuJoCoNodes() {
    std::vector<cyxwiz::plugin::PluginNodeTypeInfo> infos = {
        Info("MuJoCoEnv", "MuJoCo Environment", "RL / Simulation", 0xFF44AA88,
             {{"env", "Environment", false}},
             {{"mjcf_path", "inverted_pendulum.xml"}, {"max_steps", "1000"}}),
        Info("RewardFunction", "Reward Function", "RL / Simulation", 0xFF44AADD,
             {{"qpos", "Tensor", true}, {"qvel", "Tensor", true}, {"ctrl", "Tensor", true},
              {"sensor", "Tensor", true}, {"reward", "Float", false}},
             {{"alive_bonus", "1.0"}}),
        Info("ObservationFilter", "Observation Filter", "RL / Simulation", 0xFFAAAA44,
             {{"qpos", "Tensor", true}, {"qvel", "Tensor", true}, {"sensor", "Tensor", true},
              {"obs", "Tensor", false}},
             {{"normalize", "true"}}),
        Info("RLAgent", "RL Agent", "RL / Simulation", 0xFF4488DD,
             {{"env", "Environment", true}, {"policy", "Model", false}},
             {{"algorithm", "PPO"}}),
        Info("MuJoCoPlant", "MuJoCo Plant", "Simulation / Control", 0xFF22BB66,
             {{"u", "Tensor", true}, {"sensor", "Tensor", false}, {"qpos", "Tensor", false},
              {"qvel", "Tensor", false}, {"rgb", "Image", false}, {"depth", "Image", false}},
             {{"mjcf_path", ""}, {"timestep", "0.002"}, {"frame_skip", "1"}, {"interface", "bus"},
              {"camera", "0"}}),
    };
    infos.back().supports_dynamic_pins = true;
    infos.back().dynamic_pin_trigger = "mjcf_path";
    std::vector<cyxwiz::ExtensionNodeDescriptor> result;
    for (const auto& info : infos) result.push_back(cyxwiz::DescribeSignalNode(kMuJoCo, info));
    return result;
}

void TestPalette() {
    const auto entries = cyxwiz::BuildExtensionPalette(MuJoCoNodes());
    Check(entries.size() == 5, "five palette entries");
    Check(entries[0].group == "RL / Simulation" && entries[4].group == "Simulation / Control",
          "groups keep the plugin's order: RL first, then control");
    Check(entries[0].name == "MuJoCo Environment" && entries[1].name == "Observation Filter" &&
              entries[2].name == "Reward Function" && entries[3].name == "RL Agent",
          "names are sorted inside a group");
    Check(entries[4].type_id == "com.cyxwiz.simulation.mujoco:MuJoCoPlant",
          "an entry carries the type id, not the name");
    Check(entries[4].category_label == "Plugin/Simulation / Control",
          "the category label is what the search popup has always shown");
    Check(entries[4].color == 0xFF22BB66, "the declared colour is kept");
    Check(entries[4].pins_summary == "Pins follow the mjcf_path setting",
          "a dynamic-pin node says its pins follow a setting: " + entries[4].pins_summary);
    Check(entries[2].pins_summary == "In: qpos, qvel, ctrl, sensor. Out: reward",
          "a fixed node lists its pins: " + entries[2].pins_summary);
    Check(entries[0].hint == "Double-click or drag to add", "every entry says how to add it");

    Check(cyxwiz::ExtensionPaletteEntryMatches(entries[4], "mujoco"), "search by product name");
    Check(cyxwiz::ExtensionPaletteEntryMatches(entries[4], "PLANT"), "search ignores case");
    Check(cyxwiz::ExtensionPaletteEntryMatches(entries[2], "simulation"), "search by group");
    Check(cyxwiz::ExtensionPaletteEntryMatches(entries[2], "plugin"), "search by the word plugin");
    Check(!cyxwiz::ExtensionPaletteEntryMatches(entries[2], "dense"), "no false match");
    Check(cyxwiz::ExtensionPaletteEntryMatches(entries[2], ""), "an empty query matches all");

    Check(cyxwiz::ExtensionPaletteSummary(entries, {{kMuJoCo, "MuJoCo Simulation"}}) ==
              "5 nodes from MuJoCo Simulation",
          "the section summary names the plugin");
    Check(cyxwiz::ExtensionPaletteSummary(entries, {}) == "5 nodes from com.cyxwiz.simulation.mujoco",
          "an unknown provider name falls back to its id");
    Check(cyxwiz::ExtensionPaletteSummary({}, {}) == "No nodes", "the empty summary");
    std::vector<cyxwiz::ExtensionPaletteEntry> two = {entries[0]};
    two.push_back(entries[1]);
    two.back().type_id = "com.other.plugin:Thing";
    Check(cyxwiz::ExtensionPaletteSummary(two, {}) == "2 nodes from 2 plugins", "two providers");
}

gui::MLNode Node(const std::string& type_id, bool missing) {
    gui::MLNode node{};
    node.type = gui::NodeType::PluginCustom;
    node.name = "MuJoCo Plant";
    node.extension_type_id = type_id;
    node.extension_missing = missing;
    return node;
}

void TestCanvasStyle() {
    const auto nodes = MuJoCoNodes();
    const uint32_t steel_blue = 0xFFAA8844;  // IM_COL32(68, 136, 170)
    const auto installed = cyxwiz::BuildExtensionCanvasStyle(
        Node(nodes[4].type_id, false), &nodes[4], steel_blue);
    Check(!installed.missing && installed.box_color == 0xFF22BB66 && installed.status_line.empty(),
          "an installed MuJoCo Plant uses its declared green");

    auto undeclared = nodes[4];
    undeclared.color = 0;
    Check(cyxwiz::BuildExtensionCanvasStyle(Node(nodes[4].type_id, false), &undeclared, steel_blue)
                  .box_color == steel_blue,
          "a node without a declared colour keeps the old steel blue");

    const auto missing = cyxwiz::BuildExtensionCanvasStyle(Node(nodes[4].type_id, true), nullptr, steel_blue);
    Check(missing.missing && missing.box_color == cyxwiz::kExtensionMissingBoxColor &&
              missing.outline_color == cyxwiz::kExtensionMissingOutlineColor &&
              missing.status_line == "Not installed",
          "a missing node is grey with an orange outline and says Not installed");
}

void TestInfoFacts() {
    const auto nodes = MuJoCoNodes();
    auto plant = nodes[4];
    plant.source_path = "D:/dev/CyxWiz_Engine/plugins/mujoco_simulation";
    const auto facts = cyxwiz::BuildExtensionInfoFacts(
        plant, {"MuJoCo Simulation", "1.0.0", "CyxWiz Lab"});
    Check(facts.category_line == "Plugins / Simulation / Control", "category line");
    Check(facts.chips.size() == 4 && facts.chips[0] == "Plugin" && facts.chips[1] == "Simulation node" &&
              facts.chips[2] == "Cannot train" && facts.chips[3] == "Pins follow its settings",
          "chips say plugin, simulation, cannot train, dynamic pins");
    std::map<std::string, std::string> rows(facts.provided_by.begin(), facts.provided_by.end());
    Check(rows["Plugin"] == "MuJoCo Simulation" && rows["Version"] == "1.0.0" &&
              rows["Author"] == "CyxWiz Lab" &&
              rows["Type id"] == "com.cyxwiz.simulation.mujoco:MuJoCoPlant" &&
              rows["Source"] == "D:/dev/CyxWiz_Engine/plugins/mujoco_simulation",
          "provided-by rows");
    Check(rows.count("Node version") == 0, "an empty value is left out");

    const auto unknown = cyxwiz::BuildExtensionInfoFacts(nodes[0], {});
    std::map<std::string, std::string> unknown_rows(unknown.provided_by.begin(), unknown.provided_by.end());
    Check(unknown_rows["Plugin"] == kMuJoCo && unknown_rows.count("Version") == 0,
          "without plugin details the provider id is shown");
    Check(unknown.chips.size() == 3, "a node with fixed pins has no dynamic-pin chip");
}

void TestMissingCard() {
    gui::MLNode node = Node("com.cyxwiz.simulation.mujoco:MuJoCoPlant", true);
    node.parameters = {{"mjcf_path", "cartpole.xml"}, {"timestep", "0.002"}, {"_meta_nu", "2"}};
    gui::NodePin slide{};
    slide.name = "slide";
    slide.type = gui::PinType::Tensor;
    gui::NodePin hinge = slide;
    hinge.name = "hinge";
    gui::NodePin qpos = slide;
    qpos.name = "qpos";
    node.inputs = {slide, hinge};
    node.outputs = {qpos};

    const auto card = cyxwiz::BuildExtensionMissingCard(node);
    Check(card.title == "Not installed", "card title");
    Check(card.lines.size() == 2 &&
              card.lines[1].find("com.cyxwiz.simulation.mujoco") != std::string::npos,
          "the card names the plugin to load: " + card.lines[1]);
    std::map<std::string, std::string> details(card.details.begin(), card.details.end());
    Check(details["Type id"] == "com.cyxwiz.simulation.mujoco:MuJoCoPlant" &&
              details["Saved version"] == "(none)" && details["Saved pins"] == "2 inputs, 1 output" &&
              details["Code"] == "CW-X-0601",
          "details rows");
    Check(card.settings.size() == 2 && card.settings[0].first == "mjcf_path" &&
              card.settings[0].second == "cartpole.xml",
          "saved settings are listed; model info kept by the plugin is not");
    Check(card.pins.size() == 3 && card.pins[0].second == "(Tensor, in)" &&
              card.pins[2].first == "qpos" && card.pins[2].second == "(Tensor, out)",
          "saved pins are listed with direction");
}

void TestExactTypeName() {
    Check(cyxwiz::IsExtensionTypeName(Node("com.cyxwiz.simulation.mujoco:MuJoCoPlant", false), "MuJoCoPlant"),
          "the MuJoCo Plant matches");
    Check(!cyxwiz::IsExtensionTypeName(Node("com.other:MyMuJoCoPlantWrapper", false), "MuJoCoPlant"),
          "a name that only contains MuJoCoPlant does not match");
    Check(!cyxwiz::IsExtensionTypeName(Node("com.cyxwiz.simulation.mujoco:MuJoCoEnv", false), "MuJoCoPlant"),
          "another MuJoCo node does not match");
    gui::MLNode dense{};
    dense.type = gui::NodeType::Dense;
    dense.extension_type_id = "x:MuJoCoPlant";
    Check(!cyxwiz::IsExtensionTypeName(dense, "MuJoCoPlant"), "a built-in node never matches");
}

}  // namespace

int main() {
    TestPalette();
    TestCanvasStyle();
    TestInfoFacts();
    TestMissingCard();
    TestExactTypeName();
    std::cout << "test_extension_node_presentation: all checks passed\n";
    return 0;
}
