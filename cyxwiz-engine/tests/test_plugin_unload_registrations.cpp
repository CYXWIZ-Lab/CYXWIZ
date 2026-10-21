// Unloading a plugin removes everything it registered (TOFIX125, found in the
// step 1.5 check). UnloadPlugin freed the plugin library but left its nodes and
// panels registered, pointing into the freed library; only Shutdown and Disable
// removed them. Uses the image example plugin, built with the engine:
// plugins/examples/image_nodes (three nodes, one panel, no dangerous
// permissions, so it initialises without the approval dialog).
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_panel_registry.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

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
}

}  // namespace

int main() {
    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/image_nodes not found; run from the repository root");
    auto& manager = cyxwiz::plugin::PluginManager::Instance();

    // Unload from the running state: the Plugin Manager's Unload button.
    LoadAndInitialize(manager, dir);
    const uint64_t generation = cyxwiz::ExtensionNodeRegistry::Instance().Generation();
    manager.UnloadPlugin(kPluginId);
    Check(manager.GetLoadedPlugin(kPluginId) == nullptr, "the plugin is unloaded");
    Check(NodesOfPlugin() == 0, "unloading removes its nodes, " + std::to_string(NodesOfPlugin()) + " left");
    Check(!cyxwiz::plugin::PluginPanelRegistry::Instance().HasPanel(
              "com.cyxwiz.examples.image-nodes.settings"),
          "unloading removes its panel");
    Check(cyxwiz::ExtensionNodeRegistry::Instance().Generation() > generation,
          "the registry generation changes, so the editor refreshes");

    // Load again: nothing left over blocks the second registration.
    LoadAndInitialize(manager, dir);

    // Disable still cleans up as before.
    manager.DisablePlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "disabling removes its nodes");
    manager.UnloadPlugin(kPluginId);
    Check(NodesOfPlugin() == 0, "unloading a disabled plugin leaves nothing");

    std::cout << "test_plugin_unload_registrations: all checks passed\n";
    return 0;
}
