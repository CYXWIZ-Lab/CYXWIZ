// Plugins across sessions and what each adds (TOFIX125 P2 steps 2.9 and
// 2.10), on the two real plugins: the example activation nodes (2 trainable
// nodes, no window) and MuJoCo (2 windows, 5 simulation nodes).
#include "../src/core/plugin_presentation.h"

#include <cstdlib>
#include <filesystem>
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

    // Where a plugin comes from (the Plugin Manager's trust line, mockup 2026-11-25).
    {
        using cyxwiz::PluginTrustKind;
        using cyxwiz::PluginTrustTone;
        const std::string mujoco_dll = std::filesystem::path("bin/mujoco.dll").make_preferred().string();
        const auto detail = [](const cyxwiz::PluginTrustView& view, const std::string& key) {
            for (const auto& [k, v] : view.details)
                if (k == key) return v;
            return std::string("(none)");
        };

        cyxwiz::PluginTrustFacts signed_pack;
        signed_pack.kind = PluginTrustKind::SignedAddOn;
        signed_pack.package_version = "2";
        signed_pack.pack_id = "cyxwiz-plugin-mujoco-1.0.0-2-win64-x86_64";
        signed_pack.companion_base_id = "cyxwiz-base-0.2.0-1-win64-x86_64";
        signed_pack.files_checked = 15;
        signed_pack.bytes_checked = 8656679;
        signed_pack.check_ms = 41.6;
        const auto verified = cyxwiz::PluginTrustLine(signed_pack);
        Equal(verified.line, "Signed add-on, verified at start", "a verified add-on's line");
        Equal(verified.note, "package version 2", "its package version");
        Check(verified.tone == PluginTrustTone::Verified, "a verified add-on is green");
        Equal(detail(verified, "Checked"), "15 files, 8.7 MB, in 42 ms", "what the check read and its time");
        Equal(detail(verified, "Built for"), "CPU Engine cyxwiz-base-0.2.0-1-win64-x86_64", "its CPU Engine");
        Equal(detail(verified, "Pack"), signed_pack.pack_id, "its pack under Details");
        Check(verified.body.empty() && verified.hint.empty(), "a verified add-on needs no explanation");

        cyxwiz::PluginTrustFacts developer;
        developer.folder = std::filesystem::path("plugins/examples/activation_nodes").make_preferred().string();
        const auto dev = cyxwiz::PluginTrustLine(developer);
        Equal(dev.line, "Developer plugin, not signed", "a developer plugin's line");
        Equal(dev.note, "loaded from " + developer.folder, "where it was loaded from");
        Check(dev.tone == PluginTrustTone::Muted, "a developer plugin is muted, not a warning");

        cyxwiz::PluginTrustFacts refused = signed_pack;
        refused.kind = PluginTrustKind::RefusedAddOn;
        refused.reason = mujoco_dll + " does not match its signed package";
        refused.files_listed = 15;
        refused.issues = {{"bin/mujoco.dll", cyxwiz::plugin::PackFileProblem::ContentDiffers}};
        refused.checked_at = "14:02";
        const auto failed = cyxwiz::PluginTrustLine(refused);
        Equal(failed.line, "Signed add-on, check failed at start", "a refused add-on's line");
        Check(failed.tone == PluginTrustTone::Failed, "a refused add-on is red");
        Equal(failed.body,
              mujoco_dll + " does not match its signed package, so the Engine did not load this add-on. "
                           "Graphs that use its nodes show them as not installed.",
              "what happened, in words");
        Equal(failed.hint,
              "Repair it to put the signed files back. It also opens from Preferences > Devices > Installed packs.",
              "what to do");
        Equal(detail(failed, "Check"), "1 of 15 files differs from the signed list", "how many files differ");
        Equal(detail(failed, "File"), mujoco_dll + ": same size, different contents", "which file and how");
        Equal(detail(failed, "Checked"), "at this start, 14:02", "when it was checked");

        refused.issues.push_back({"bin/extra.dll", cyxwiz::plugin::PackFileProblem::NotListed});
        Equal(detail(cyxwiz::PluginTrustLine(refused), "Check"),
              "1 of 15 files differs from the signed list; 1 file is not in the signed list",
              "an unlisted file is counted apart");
        Equal(cyxwiz::plugin::PackRefusalReason(refused.issues),
              mujoco_dll + " does not match its signed package (and 1 other file)", "the log's reason");

        cyxwiz::PluginTrustFacts metadata;
        metadata.kind = PluginTrustKind::RefusedAddOn;
        metadata.reason = "its signed package record could not be verified";
        metadata.detail = "No trusted signature";
        const auto record = cyxwiz::PluginTrustLine(metadata);
        Equal(detail(record, "Check"), "its signed package record could not be verified", "a metadata refusal");
        Equal(detail(record, "Why"), "No trusted signature", "with the check's own words");
        Equal(detail(record, "Signed with"), "(none)", "no key to name when the record did not verify");

        const auto summary = cyxwiz::PluginListSummary({2, 1, 0, 0, 1, 0});
        Check(summary.size() == 1, "no verified add-on line when none verified");
        Equal(summary[0], "2 plugins: 1 active, 1 not loaded (check failed), 0 disabled", "a refused add-on counted");
        const auto clean = cyxwiz::PluginListSummary({2, 2, 0, 0, 0, 1});
        Equal(clean[0], "2 plugins: 2 active, 0 failed, 0 disabled", "the usual counts");
        Equal(clean[1], "1 signed add-on verified at start", "the verified add-ons");
        Equal(cyxwiz::FormatPackBytes(512000), "512 KB", "small sizes in KB");
    }

    std::cout << "test_plugin_presentation: all checks passed\n";
    return 0;
}
