// Unloading a plugin removes everything it registered (TOFIX125, found in the
// step 1.5 check). UnloadPlugin freed the plugin library but left its nodes and
// panels registered, pointing into the freed library; only Shutdown and Disable
// removed them. Uses the image example plugin, built with the engine:
// plugins/examples/image_nodes (three nodes, one panel, no dangerous
// permissions, so it initialises without the approval dialog).
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_panel_registry.h"
#include "../src/plugin/registries/plugin_command_registry.h"
#include "../src/plugin/registries/plugin_simulation_registry.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char* kPluginId = "com.cyxwiz.examples.image-nodes";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::filesystem::path PluginDir() {
    auto dir = std::filesystem::current_path();
    while (!dir.empty()) {
        const auto candidate = dir / "plugins" / "examples" / "image_nodes";
        if (std::filesystem::exists(candidate / "plugin.json")) return candidate;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

size_t NodesOfPlugin() {
    size_t count = 0;
    for (const auto& descriptor : cyxwiz::ExtensionNodeRegistry::Instance().All()) {
        if (descriptor.provider_id == kPluginId) ++count;
    }
    return count;
}

void LoadAndInitialize(cyxwiz::plugin::PluginManager& manager, const std::filesystem::path& dir) {
    Check(manager.LoadPlugin(dir), "the image example plugin loads from " + dir.string());
    Check(manager.InitializePlugin(kPluginId), "the image example plugin initialises");
    Check(NodesOfPlugin() == 3, "its three nodes are registered, got " + std::to_string(NodesOfPlugin()));
    Check(cyxwiz::plugin::PluginPanelRegistry::Instance().HasPanel(
              "com.cyxwiz.examples.image-nodes.settings"),
          "its settings panel is registered");
    auto& panels = cyxwiz::plugin::PluginPanelRegistry::Instance();
    Check(!panels.ShowPanel("com.example.other", "com.cyxwiz.examples.image-nodes.settings"),
          "a plugin cannot open another plugin's panel");
    Check(!panels.ShowPanel(kPluginId, "com.cyxwiz.examples.image-nodes.missing"),
          "showing a panel the plugin does not have fails");
    panels.SetPanelVisible("com.cyxwiz.examples.image-nodes.settings", false);
    Check(panels.ShowPanel(kPluginId, "com.cyxwiz.examples.image-nodes.settings") &&
              panels.IsPanelVisible("com.cyxwiz.examples.image-nodes.settings"),
          "a plugin opens its own panel");
}

// A plugin's menu commands (plugin API 1.4): one enabled, one disabled with a reason
class FakeCommands final : public cyxwiz::plugin::ICommandProvider {
public:
    int runs = 0;
    std::vector<cyxwiz::plugin::PluginCommandInfo> GetCommands() override {
        return {{"go", "Simulation", "run", "Go", "Goes.", false}, {"halt", "Simulation", "run", "Halt", "Halts.", false}};
    }
    cyxwiz::plugin::PluginCommandState GetCommandState(const std::string& id) override {
        if (id == "halt") return {false, "Nothing is running.", false};
        return {};
    }
    void RunCommand(const std::string& id) override {
        if (id == "go") ++runs;
    }
};

void CheckCommandRegistry() {
    auto& commands = cyxwiz::plugin::PluginCommandRegistry::Instance();
    FakeCommands fake;
    for (const auto& info : fake.GetCommands()) commands.Register("com.example.sim", info, &fake);
    commands.Register("com.example.sim", fake.GetCommands()[0], &fake);  // a duplicate is skipped
    const auto snapshot = commands.Snapshot();
    Check(snapshot.size() == 2 && snapshot[0].key == "com.example.sim:go" && snapshot[0].state.enabled,
          "commands are listed in order with their state");
    Check(!snapshot[1].state.enabled && snapshot[1].state.disabled_reason == "Nothing is running.",
          "a disabled command keeps its reason");
    Check(commands.Run("com.example.sim:go") && fake.runs == 1, "a command runs");
    Check(!commands.Run("com.example.sim:missing"), "an unknown command does not run");
    commands.RemoveByPlugin("com.example.sim");
    Check(commands.GetCount() == 0 && !commands.Run("com.example.sim:go"), "removing the plugin removes its commands");
}

// A plugin whose nodes run in graph simulations (plugin API 1.4)
class FakeSimulation final : public cyxwiz::plugin::ISimulationProvider {
public:
    bool refuse = false;
    int begins = 0;
    int ends = 0;
    std::vector<cyxwiz::plugin::PluginSimulationGraphNode> nodes;
    std::vector<cyxwiz::plugin::PluginSimulationNodeInfo> GetSimulationNodes() override {
        return {{"Plant", "sim.view"}, {"Probe", ""}};
    }
    cyxwiz::plugin::PluginSimulationBeginResult BeginGraphRun(
        const std::vector<cyxwiz::plugin::PluginSimulationGraphNode>& graph_nodes) override {
        ++begins;
        nodes = graph_nodes;
        if (refuse) return {false, "The viewport is running the model."};
        return {};
    }
    void EndGraphRun() override { ++ends; }
};

gui::MLNode PluginNode(int id, const char* type_id, const char* name) {
    gui::MLNode node{};
    node.id = id;
    node.type = gui::NodeType::PluginCustom;
    node.name = name;
    node.extension_type_id = type_id;
    node.parameters["mjcf_path"] = "robot.xml";
    return node;
}

void CheckSimulationRegistry() {
    auto& simulation = cyxwiz::plugin::PluginSimulationRegistry::Instance();
    FakeSimulation fake;
    for (const auto& info : fake.GetSimulationNodes()) simulation.Register("com.example.sim", info, &fake);
    simulation.Register("com.example.sim", fake.GetSimulationNodes()[0], &fake);  // a duplicate is skipped
    Check(simulation.GetCount() == 2, "both simulation nodes are registered once");
    Check(simulation.IsSimulationNode("com.example.sim:Plant") && !simulation.IsSimulationNode("com.example.sim:Other"),
          "only the named node types run in a simulation");

    std::string reason;
    auto lease = simulation.BeginGraphRun(
        {PluginNode(1, "com.example.sim:Plant", "Plant"), PluginNode(2, "com.example.sim:Probe", "Probe")}, reason);
    Check(lease != nullptr, "a graph of simulation nodes begins: " + reason);
    Check(fake.begins == 1 && fake.nodes.size() == 2 && fake.nodes[0].node_type == "Plant" &&
              fake.nodes[0].node_id == 1 && fake.nodes[0].parameters.at("mjcf_path") == "robot.xml",
          "the plugin begins once, with its nodes and their settings");
    Check(lease->Panels().size() == 1 && lease->Panels()[0].second == "sim.view" &&
              lease->PluginIds().count("com.example.sim") == 1,
          "the run names the plugin and its one panel");
    Check(fake.ends == 0, "the plugin is not ended while the lease is held");
    lease.reset();
    Check(fake.ends == 1, "dropping the lease ends the plugin's run");

    lease = simulation.BeginGraphRun({PluginNode(3, "com.example.other:Thing", "Thing")}, reason);
    Check(lease == nullptr && reason.find("'Thing'") != std::string::npos && fake.begins == 1,
          "a plugin node no plugin runs in a simulation refuses the run, naming it");

    fake.refuse = true;
    lease = simulation.BeginGraphRun({PluginNode(1, "com.example.sim:Plant", "Plant")}, reason);
    Check(lease == nullptr && reason == "The viewport is running the model." && fake.ends == 1,
          "a plugin's refusal is the reason, and a refused plugin is not ended");

    lease = simulation.BeginGraphRun({}, reason);
    Check(lease != nullptr && lease->PluginIds().empty(), "a graph without plugin nodes needs no plugin");

    simulation.RemoveByPlugin("com.example.sim");
    Check(simulation.GetCount() == 0, "removing the plugin removes its simulation nodes");
}

}  // namespace

int main() {
    CheckCommandRegistry();
    CheckSimulationRegistry();
    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/image_nodes not found; run from the repository root");
    auto& manager = cyxwiz::plugin::PluginManager::Instance();

    // Unload from the running state: the Plugin Manager's Unload button.
    // Whatever runs the plugin's code (a graph run) hears of it first.
    LoadAndInitialize(manager, dir);
    const uint64_t generation = cyxwiz::ExtensionNodeRegistry::Instance().Generation();
    std::vector<std::string> stopping;
    manager.SetPluginStoppingHandler([&stopping](const std::string& id) { stopping.push_back(id); });
    manager.UnloadPlugin(kPluginId);
    Check(stopping.size() == 1 && stopping[0] == kPluginId, "unloading tells the stopping handler first");
    Check(manager.GetLoadedPlugin(kPluginId) == nullptr, "the plugin is unloaded");
    Check(NodesOfPlugin() == 0, "unloading removes its nodes, " + std::to_string(NodesOfPlugin()) + " left");
    Check(!cyxwiz::plugin::PluginPanelRegistry::Instance().HasPanel(
              "com.cyxwiz.examples.image-nodes.settings"),
          "unloading removes its panel");
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Generation() > generation,
          "the registry generation changes, so the editor refreshes");

    // Load again: nothing left over blocks the second registration.
    LoadAndInitialize(manager, dir);

    // Disable still cleans up as before, and tells the stopping handler too.
    manager.DisablePlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "disabling removes its nodes");
    Check(stopping.size() == 2, "disabling tells the stopping handler");
    manager.SetPluginStoppingHandler(nullptr);
    manager.UnloadPlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "unloading a disabled plugin leaves nothing");

    std::cout << "test_plugin_unload_registrations: all checks passed\n";
    return 0;
}
