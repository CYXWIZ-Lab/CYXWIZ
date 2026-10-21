// Plugins across sessions and what each adds (TOFIX125 P2 steps 2.9 and
// 2.10), on the two real plugins: the example activation nodes (2 trainable
// nodes, no window) and MuJoCo (2 windows, 5 simulation nodes).
#include "../src/core/plugin_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void Equal(const std::string& actual, const std::string& expected, const std::string& message) {
    if (actual != expected) {
        std::cerr << "FAIL: " << message << "\n  expected: " << expected << "\n  actual:   " << actual << '\n';
        std::exit(1);
    }
}

constexpr const char* kActivation = "com.cyxwiz.examples.activation-nodes";
constexpr const char* kMuJoCo = "com.cyxwiz.simulation.mujoco";
constexpr const char* kActivationDir = "D:/dev/CyxWiz_t125/plugins/examples/activation_nodes/bin";
constexpr const char* kMuJoCoDir = "D:/dev/CyxWiz_t125/plugins/simulation/mujoco/bin";

}  // namespace

int main() {
    // Startup list: loading records a plugin with startup on.
    cyxwiz::PluginStartupList list;
    cyxwiz::RecordPluginLoaded(list, kActivation, kActivationDir);
    cyxwiz::RecordPluginLoaded(list, kMuJoCo, kMuJoCoDir);
    Check(list.entries.size() == 2, "two plugins recorded");
    Check(cyxwiz::PluginLoadsAtStartup(list, kActivation), "on by default");
    Check(!cyxwiz::PluginLoadsAtStartup(list, "com.example.unknown"), "unknown plugin does not load");

    // The user's choice survives a reload from another folder.
    cyxwiz::SetPluginLoadAtStartup(list, kMuJoCo, false);
    cyxwiz::RecordPluginLoaded(list, kMuJoCo, "D:/other/mujoco");
    Check(!cyxwiz::PluginLoadsAtStartup(list, kMuJoCo), "choice kept on reload");
    Equal(list.entries[1].folder, "D:/other/mujoco", "folder updated");
    Check(list.entries.size() == 2, "no duplicate on reload");

    const auto to_load = cyxwiz::PluginsToLoadAtStartup(list);
    Check(to_load.size() == 1 && to_load[0].id == kActivation, "only plugins switched on load");

    // Saved and read back.
    {
        const std::string text = cyxwiz::SerializePluginStartupList(list);
        std::string error = "unset";
        const auto back = cyxwiz::ParsePluginStartupList(text, &error);
        Check(error.empty(), "round trip has no error");
        Check(back.entries.size() == 2, "round trip keeps both");
        Equal(back.entries[0].id, kActivation, "order kept");
        Equal(back.entries[0].folder, kActivationDir, "folder kept");
        Check(back.entries[0].load_at_startup && !back.entries[1].load_at_startup, "choices kept");
    }
    // Damaged or foreign files load nothing and say why.
    {
        std::string error;
        Check(cyxwiz::ParsePluginStartupList("not json", &error).entries.empty() && !error.empty(), "damaged");
        Check(cyxwiz::ParsePluginStartupList(R"({"version":9,"plugins":[]})", &error).entries.empty() &&
                  error == "unknown version",
              "unknown version");
        const auto dup = cyxwiz::ParsePluginStartupList(
            R"({"version":1,"plugins":[{"id":"a","folder":"x"},{"id":"a","folder":"y"},{"folder":"z"}]})", &error);
        Check(dup.entries.size() == 1 && dup.entries[0].folder == "x", "duplicates and blank ids dropped");
    }

    // Unloading removes the plugin from the list.
    cyxwiz::RecordPluginRemoved(list, kActivation);
    Check(list.entries.size() == 1 && list.entries[0].id == kMuJoCo, "unload removes");
    Check(cyxwiz::PluginsToLoadAtStartup(list).empty(), "nothing left to load");

    Equal(cyxwiz::PluginStartupNote(true), "Graphs that use its nodes open ready after a restart.", "note on");
    Equal(cyxwiz::PluginStartupNote(false),
          "After a restart its nodes show as not installed until you load it here.", "note off");

    // What each plugin adds.
    cyxwiz::PluginContribution activation;
    activation.trainable_nodes = 2;
    activation.node_categories = {"Activations"};
    Equal(cyxwiz::PluginAddsLine(activation), "Adds: 2 trainable nodes (Activations)", "activation adds");
    Equal(cyxwiz::PluginMenuDetail(activation), "2 nodes", "activation menu detail");

    cyxwiz::PluginContribution mujoco;
    mujoco.windows = 2;
    mujoco.simulation_nodes = 5;
    mujoco.node_categories = {"RL / Simulation"};
    Equal(cyxwiz::PluginAddsLine(mujoco), "Adds: 2 windows, 5 simulation nodes (RL / Simulation)", "mujoco adds");
    Equal(cyxwiz::PluginMenuDetail(mujoco), "2 windows, 5 nodes", "mujoco menu detail");

    cyxwiz::PluginContribution one;
    one.windows = 1;
    Equal(cyxwiz::PluginAddsLine(one), "Adds: 1 window", "singular");
    Equal(cyxwiz::PluginMenuDetail(one), "1 window", "singular detail");
    Equal(cyxwiz::PluginAddsLine({}), "", "adds nothing");
    Equal(cyxwiz::PluginMenuDetail({}), "", "no detail");

    std::cout << "test_plugin_presentation: all checks passed\n";
    return 0;
}
