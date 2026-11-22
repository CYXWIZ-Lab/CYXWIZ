// Plugins across sessions and what each adds (TOFIX125 P2 steps 2.9 and
// 2.10), on the two real plugins: the example activation nodes (2 trainable
// nodes, no window) and MuJoCo (2 windows, 5 simulation nodes).
#include "../src/core/plugin_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

    // What each plugin adds.
    cyxwiz::PluginContribution activation;
    activation.trainable_nodes = 2;
    activation.node_categories = {"Activations"};
    Equal(cyxwiz::PluginStartupNote(true, activation), "Graphs that use its nodes open ready after a restart.",
          "note on");
    Equal(cyxwiz::PluginStartupNote(false, activation),
          "After a restart its nodes show as not installed until you load it here.", "note off");
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
    Equal(cyxwiz::PluginStartupNote(true, {}), "It loads again after a restart.", "note, adds nothing");

    // A data plugin (the SDK's data-formats example): no nodes to speak of.
    cyxwiz::PluginContribution data;
    data.dataset_formats = 1;
    data.analytics = 1;
    Equal(cyxwiz::PluginAddsLine(data), "Adds: 1 dataset format, 1 analytic", "data plugin adds");
    Equal(cyxwiz::PluginMenuDetail(data), "1 format, 1 analytic", "data plugin menu detail");
    Equal(cyxwiz::PluginStartupNote(true, data), "Its dataset formats and analytics are there after a restart.",
          "data plugin note on");
    Equal(cyxwiz::PluginStartupNote(false, data),
          "After a restart its dataset formats and analytics are missing until you load it here.",
          "data plugin note off");
    cyxwiz::PluginContribution mixed = activation;
    mixed.dataset_formats = 2;
    Equal(cyxwiz::PluginAddsLine(mixed), "Adds: 2 trainable nodes (Activations), 2 dataset formats",
          "categories stay with the nodes");

    // Plugin packs from the installed runtime load from their current folder.
    {
        cyxwiz::PluginStartupList packs_list;
        cyxwiz::RecordPluginLoaded(packs_list, "com.cyxwiz.simulation.mujoco",
                                   "C:/CyxWiz/runtime/plugins/com.cyxwiz.simulation.mujoco/mujoco-v1");
        cyxwiz::RecordPluginLoaded(packs_list, "com.example.off", "C:/plugins/off");
        cyxwiz::SetPluginLoadAtStartup(packs_list, "com.example.off", false);
        const std::vector<cyxwiz::PluginStartupEntry> packs = {
            {"com.cyxwiz.simulation.mujoco",
             "C:/CyxWiz/runtime/plugins/com.cyxwiz.simulation.mujoco/mujoco-v2", true},
            {"com.example.off", "C:/CyxWiz/runtime/plugins/com.example.off/off-v1", true},
            {"com.example.new", "C:/CyxWiz/runtime/plugins/com.example.new/new-v1", true}};
        const auto pack_load = cyxwiz::PluginsToLoadAtStartup(packs_list, packs);
        Check(pack_load.size() == 2, "a pack loads unless the user set it off");
        Equal(pack_load[0].folder, "C:/CyxWiz/runtime/plugins/com.cyxwiz.simulation.mujoco/mujoco-v2",
              "a listed pack loads from its active folder, not the recorded one");
        Equal(pack_load[1].id, "com.example.new", "a newly installed pack loads at startup");
        Check(cyxwiz::PluginsToLoadAtStartup(packs_list).size() == 1, "without packs the list is unchanged");
    }

    // The Plugin Manager card (TOFIX129 C6): state words, reasons, actions, order, footer
    {
        using cyxwiz::PluginRunState;
        cyxwiz::PluginCardFacts mujoco_card{kMuJoCo, "MuJoCo Simulation", PluginRunState::Running, "", false};
        const auto active = cyxwiz::DescribePluginCard(mujoco_card);
        Check(active.state_word == "Active" && active.reason.empty() && active.action_label == "Disable",
              "an active plugin: no reason, Disable");

        cyxwiz::PluginCardFacts waits{kActivation, "Activation nodes", PluginRunState::Loaded,
                                      "waits for com.cyxwiz.simulation.mujoco_card, which has not started yet", false};
        const auto waiting = cyxwiz::DescribePluginCard(waits);
        Equal(waiting.state_word, "Waiting", "a plugin waiting for another says Waiting");
        Equal(waiting.reason,
              "Waits for com.cyxwiz.simulation.mujoco_card, which has not started yet. It starts by itself when that one does.",
              "and why, in a sentence (today the card hid it)");
        Check(waiting.action_label == "Start", "Start is still offered");

        cyxwiz::PluginCardFacts asked{kActivation, "Activation nodes", PluginRunState::Loaded, "", true};
        const auto answer = cyxwiz::DescribePluginCard(asked);
        Check(answer.state_word == "Waiting" && answer.action == cyxwiz::PluginAction::None,
              "waiting for a permission answer: no button that would ask again");

        cyxwiz::PluginCardFacts fresh{kActivation, "Activation nodes", PluginRunState::Loaded, "", false};
        Equal(cyxwiz::DescribePluginCard(fresh).state_word, "Not started", "loaded, not started");

        cyxwiz::PluginCardFacts failed{kMuJoCo, "MuJoCo Simulation", PluginRunState::Failed, "mujoco.dll was not found", false};
        const auto bad = cyxwiz::DescribePluginCard(failed);
        Equal(bad.reason, "Failed to start: Mujoco.dll was not found.", "a failure says why");
        Check(bad.action_label == "Try again" && bad.tone == cyxwiz::PluginTone::Problem, "Try again, problem tone");
        failed.error.clear();
        Equal(cyxwiz::DescribePluginCard(failed).reason, "Failed to start; the Console has the details.",
              "a failure without a message points to the Console");

        cyxwiz::PluginCardFacts off{kMuJoCo, "MuJoCo Simulation", PluginRunState::Disabled, "", false};
        Check(cyxwiz::DescribePluginCard(off).action_label == "Enable", "disabled: Enable");

        const std::vector<cyxwiz::PluginCardFacts> cards = {
            {"com.z", "mujoco_card Simulation", PluginRunState::Running, "", false},
            {"com.a", "Activation nodes", PluginRunState::Loaded, "", false},
            {"com.b", "MuJoCo Simulation", PluginRunState::Failed, "x", false}};
        const auto order = cyxwiz::PluginCardOrder(cards);
        Check(order.size() == 3 && order[0] == 1 && order[1] == 2 && order[2] == 0,
              "cards by name ignoring case, then by id");
        Equal(cyxwiz::PluginFooter(cards), "3 plugins: 1 active, 1 not started, 1 failed",
              "the footer counts every state a plugin is in (today it left out loaded ones)");
        Equal(cyxwiz::PluginFooter({mujoco_card}), "1 plugin: 1 active", "one active plugin");
        Equal(cyxwiz::PluginFooter({}), "No plugins", "none");
    }
    // Permissions on the card. MuJoCo asks for Training Hooks and UI Modification, both low
    // risk: allowed without a question, nothing to change.
    {
        using cyxwiz::PermissionAnswer;
        const std::vector<cyxwiz::PluginPermissionFacts> mujoco_perms = {
            {"Training Hooks", "Hook into the training pipeline", false, PermissionAnswer::NotAsked},
            {"UI Modification", "Add custom panels and nodes to the UI", false, PermissionAnswer::NotAsked}};
        Equal(cyxwiz::PluginPermissionsLink(mujoco_perms), "Permissions (2 allowed)", "MuJoCo's two low-risk permissions");
        Check(!cyxwiz::PluginPermissionRows(mujoco_perms)[0].can_change, "a low-risk permission has nothing to answer");
        // A plugin asking for File System and Network (they need the user's answer)
        std::vector<cyxwiz::PluginPermissionFacts> perms = {
            {"File System", "Read and write files on your computer", true, PermissionAnswer::Allowed},
            {"Network Access", "Make network connections", true, PermissionAnswer::Allowed}};
        Equal(cyxwiz::PluginPermissionsLink(perms), "Permissions (2 allowed)", "both allowed");
        perms[1].answer = PermissionAnswer::Denied;
        Equal(cyxwiz::PluginPermissionsLink(perms), "Permissions (1 allowed, 1 denied)", "one denied");
        perms[1].answer = PermissionAnswer::NotAsked;
        Equal(cyxwiz::PluginPermissionsLink(perms), "Permissions (1 allowed, 1 not asked yet)", "one not asked yet");
        perms.push_back({"GPU Compute", "Access GPU compute resources", false, PermissionAnswer::NotAsked});
        const auto rows = cyxwiz::PluginPermissionRows(perms);
        Check(rows.size() == 3 && rows[0].answer == "Allowed" && rows[0].can_change, "an answered one can change");
        Check(rows[1].answer == "Not asked yet", "an unanswered one says so");
        Check(rows[2].answer == "Allowed (low risk, not asked)" && !rows[2].can_change,
              "a low-risk one is allowed without a question and has nothing to change");
        Check(cyxwiz::PluginPermissionsLink({}).empty(), "no permissions: no link");
    }
    // The Unload question names what goes
    {
        cyxwiz::PluginContribution mujoco_adds;
        mujoco_adds.windows = 2;
        mujoco_adds.simulation_nodes = 5;
        Equal(cyxwiz::PluginUnloadQuestion("MuJoCo Simulation", mujoco_adds),
              "Unload MuJoCo Simulation? Its 2 windows close. Graphs that use its nodes cannot run until it is loaded again.",
              "MuJoCo's windows and nodes");
        cyxwiz::PluginContribution one_window;
        one_window.windows = 1;
        Equal(cyxwiz::PluginUnloadQuestion("X", one_window), "Unload X? Its 1 window closes.", "one window");
    }

    std::cout << "test_plugin_presentation: all checks passed\n";
    return 0;
}
