#pragma once

// Installed add-ons (plugin packs) are checked against their signed
// inventory before they load. This file is the bridge from the pack service
// (redist: signed metadata, file inventory, SHA-256) to the PluginManager,
// which knows only plain data and callbacks (plugin_pack.h).

#include "plugin_pack.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz::runtime {
class BackendPackMetadataVerifier;
}

namespace cyxwiz::plugin {

class PluginManager;

// The active add-ons of the runtime at `runtime_root` (active-runtime.json,
// active-plugins.json), each one's signed installed metadata verified with
// `verifier`: a PackPlugin, or a PackRefusal saying why not. `skipped` gets
// the entries the layout itself drops (a missing folder). False with `error`
// when the active state cannot be read at all.
bool ReadInstalledPluginPacks(const std::filesystem::path& runtime_root,
                              const runtime::BackendPackMetadataVerifier& verifier,
                              InstalledPluginPacks& output, std::vector<std::string>& skipped,
                              std::string& error);

// The pack service's file checks (exact file set, size, SHA-256), as the
// loader's callbacks.
PackFileChecks InstalledPackFileChecks();

// Start-up of an installed Engine (and of the Python environment host): reads
// runtime/trust/trusted-keys.json once, verifies every active add-on and hands
// the result to `manager`. Logs one line per add-on that is not loaded.
// False, logged, when the trust file or the installation's state cannot be
// read: the manager is still told the runtime, so no add-on loads unchecked
// (the active ones are refused, and any other folder under its plugins/).
bool ConfigureInstalledPluginPacks(PluginManager& manager, const std::filesystem::path& runtime_root);

}  // namespace cyxwiz::plugin
