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
// "Check gradient" does. Exit code 0 when every folder loads and every node
// passes.
#include <cyxwiz/gradient_check.h>
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/plugin_loader.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/security/permission_store.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// One sample of the shapes the gradient check uses, without the batch.
constexpr size_t kCheckWidth = 3;

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
    if (!manager.LoadPlugin(folder)) {
        std::cout << "  FAIL the plugin did not load (the reason is in the log above)\n";
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
        std::cout << "  FAIL the plugin did not initialise\n";
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

        const auto shape = factory->InferOutputShape({kCheckWidth}, settings);
        if (!shape.ok) {
            std::cout << "  FAIL node " << node.type_id << ": refuses a [" << kCheckWidth
                      << "] input with its default settings: " << shape.error << "\n";
            ok = false;
            continue;
        }
        const auto result = cyxwiz::CheckModuleGradients([&](const std::vector<size_t>& input_shape) {
            std::string create_error;
            const std::vector<size_t> sample(input_shape.begin() + 1, input_shape.end());
            auto module = factory->CreateModule(sample, settings, create_error);
            if (!module && !create_error.empty()) std::cout << "  CreateModule: " << create_error << "\n";
            return module;
        });
        std::cout << (result.passed ? "  ok   node " : "  FAIL node ") << node.type_id << ": "
                  << cyxwiz::DescribeGradientCheck(result) << "\n";
        ok = ok && result.passed;
    }
    if (nodes == 0) {
        std::cout << "  FAIL the plugin registered no nodes\n";
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
