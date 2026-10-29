// cyxwiz-plugin-check: load C++ plugin folders the way the Engine does and
// check their trainable nodes (TOFIX125 P5 step 5.3, Part D8).
//
//   cyxwiz-plugin-check <plugin folder> [<plugin folder> ...]
//
// For each folder: read plugin.json, load the library (the build fingerprint
// is checked before the plugin is created; the library's static
// initialisers have already run), grant the permissions plugin.json declares
// for this run, initialise the plugin, list the
// nodes it registers, and for each trainable node check its output shape and
// its gradients with the node's default settings, as the Plugin Manager's
// "Check gradient" does; list its dataset formats and analytics (TOFIX181).
// Exit code 0 when every folder loads, registers something, and every node
// passes.
#include <cyxwiz/gradient_check.h>
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_loader.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_analytics_registry.h"
#include "../src/plugin/registries/plugin_data_loader_registry.h"
#include "../src/plugin/security/permission_store.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string ShapeText(const std::vector<size_t>& shape) {
    std::string text;
    for (const size_t d : shape) text += (text.empty() ? "" : ", ") + std::to_string(d);
    return "[" + text + "]";
}

bool CheckFolder(const fs::path& folder) {
    std::cout << folder.string() << "\n";
    cyxwiz::plugin::PluginManifest manifest;
    std::string error;
    if (!cyxwiz::plugin::PluginLoader::ParseManifest(folder, manifest, error)) {
        std::cout << "  FAIL plugin.json: " << error << "\n";
        return false;
    }
    std::cout << "  plugin " << manifest.id << " " << manifest.version.ToString() << " (API "
              << manifest.api_version.ToString() << ")\n";

    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    std::string load_error;
    if (!manager.LoadPlugin(folder, &load_error)) {
        std::cout << "  FAIL the plugin did not load: " << load_error << "\n";
        return false;
    }
    // The Engine asks the user before a plugin with a dangerous permission
    // starts; a command line cannot, so the check grants what plugin.json
    // declares for this run only (the store is never loaded or saved here).
    auto& permissions = cyxwiz::plugin::security::PermissionStore::Instance();
    const std::string version = manifest.version.ToString();
    for (const auto permission : permissions.GetUndecidedDangerous(manifest.id, version, manifest.permissions)) {
        permissions.SetDecision(manifest.id, version, permission,
                                cyxwiz::plugin::security::PermissionDecision::Allowed);
        for (const auto& name : cyxwiz::plugin::PermissionsToStrings(static_cast<uint32_t>(permission)))
            std::cout << "  permission " << name << ": granted for this check only (the Engine asks the user)\n";
    }
    if (!manager.InitializePlugin(manifest.id)) {
        const auto* loaded = manager.GetLoadedPlugin(manifest.id);
        std::cout << "  FAIL the plugin did not initialise"
                  << (loaded && !loaded->error_message.empty() ? ": " + loaded->error_message : std::string())
                  << "\n";
        manager.UnloadPlugin(manifest.id);
        return false;
    }

    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    bool ok = true;
    size_t nodes = 0;
    for (const auto& node : registry.All()) {
        if (node.provider_id != manifest.id) continue;
        ++nodes;
        const auto factory = registry.TensorFactory(node.type_id);
        if (!factory) {
            std::cout << "  node " << node.type_id << ": not trainable, nothing to check\n";
            continue;
        }
        std::map<std::string, std::string> settings;
        for (const auto& parameter : node.metadata.parameters) settings[parameter.name] = parameter.default_value;

        // The gradient check's batches (GradientCheckOptions::input_shapes);
        // a node that refuses one of their sample shapes is checked with the
        // others, and says which it refused and why.
        cyxwiz::GradientCheckOptions options;
        std::vector<std::vector<size_t>> accepted;
        for (const auto& batch : options.input_shapes) {
            const std::vector<size_t> sample(batch.begin() + 1, batch.end());
            const auto shape = factory->InferOutputShape(sample, settings);
            if (shape.ok) {
                accepted.push_back(batch);
            } else {
                std::cout << "  node " << node.type_id << ": refuses a " << ShapeText(sample)
                          << " input with its default settings: " << shape.error << "\n";
            }
        }
        if (accepted.empty()) {
            std::cout << "  FAIL node " << node.type_id << ": refuses every input shape the check uses\n";
            ok = false;
            continue;
        }
        options.input_shapes = accepted;
        const auto result = cyxwiz::CheckModuleGradients(
            [&](const std::vector<size_t>& input_shape) {
                std::string create_error;
                const std::vector<size_t> sample(input_shape.begin() + 1, input_shape.end());
                auto module = factory->CreateModule(sample, settings, create_error);
                if (!module && !create_error.empty()) std::cout << "  CreateModule: " << create_error << "\n";
                return module;
            },
            options);
        std::cout << (result.passed ? "  ok   node " : "  FAIL node ") << node.type_id << ": "
                  << cyxwiz::DescribeGradientCheck(result) << "\n";
        ok = ok && result.passed;
    }
    size_t formats = 0;
    for (const auto& loader : cyxwiz::plugin::PluginDataLoaderRegistry::Instance().Loaders()) {
        if (loader.plugin_id != manifest.id) continue;
        ++formats;
        std::string extensions;
        for (const auto& extension : loader.info.extensions) extensions += (extensions.empty() ? "." : ", .") + extension;
        std::cout << "  ok   dataset format " << loader.info.format_name << " (" << extensions << ")\n";
    }
    size_t analytics = 0;
    for (const auto& analytic : cyxwiz::plugin::PluginAnalyticsRegistry::Instance().Analytics()) {
        if (analytic.plugin_id != manifest.id) continue;
        ++analytics;
        std::cout << "  ok   analytic " << analytic.info.name
                  << (analytic.info.category.empty() ? std::string() : " (" + analytic.info.category + ")") << "\n";
    }
    if (const auto* loaded = manager.GetLoadedPlugin(manifest.id)) {
        for (const auto& reason : loaded->skipped_nodes) {
            std::cout << "  FAIL not added: " << reason << "\n";
            ok = false;
        }
    }
    if (nodes + formats + analytics == 0) {
        std::cout << "  FAIL the plugin registered no nodes, dataset formats or analytics\n";
        ok = false;
    }
    manager.UnloadPlugin(manifest.id);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: cyxwiz-plugin-check <plugin folder> [<plugin folder> ...]\n";
        return 2;
    }
    bool ok = true;
    for (int i = 1; i < argc; ++i) ok = CheckFolder(fs::absolute(argv[i])) && ok;
    std::cout << (ok ? "cyxwiz-plugin-check: all plugins pass\n" : "cyxwiz-plugin-check: FAILED\n");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
