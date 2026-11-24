#include "backend_pack_removal_target.h"

#include "backend_pack_path.h"
#include "runtime_layout.h"

#include <algorithm>
#include <system_error>

namespace cyxwiz::runtime {

bool ResolveBackendPackRemovalTarget(
    const std::filesystem::path& runtime_root,
    const std::string& pack_id,
    BackendPackRemovalTarget& target,
    std::string& error) {
    target = {};
    error.clear();
    ActiveRuntimeState active;
    if (!LoadActiveRuntimeState(runtime_root / "active-runtime.json", active, error)) {
        error = "Cannot load the packaged runtime for component removal: " + error;
        return false;
    }
    const auto selected = std::find_if(
        active.packs.begin(), active.packs.end(),
        [&](const ActivePackState& pack) { return pack.pack_id == pack_id; });
    if (selected != active.packs.end()) {
        // A backend pack needs active-runtime.json only.
        target.backend = selected->backend;
        return true;
    }
    const auto plugins_path = runtime_root / kActivePluginsFileName;
    ActivePluginsState plugins;
    if (!LoadActivePluginsState(plugins_path, plugins, error)) {
        error = "Cannot read the active plugin packs in " + plugins_path.string() + ": " + error +
            ". Delete that file to turn every plugin pack off (installed packs stay), then remove "
            "the pack again";
        return false;
    }
    for (const auto& plugin : plugins.plugins) {
        if (plugin.pack_id == pack_id) {
            target.plugin_id = plugin.plugin_id;
            return true;
        }
    }
    // An installed plugin pack that is not active (another base became
    // active since, or it was deactivated): found by its folder,
    // runtime/plugins/<plugin-id>/<pack-id>/plugin.json
    if (IsCanonicalBackendPackRelativePath(pack_id) && pack_id.find('/') == std::string::npos) {
        std::error_code scan_error;
        for (const auto& entry :
             std::filesystem::directory_iterator(runtime_root / "plugins", scan_error)) {
            std::error_code entry_error;
            if (entry.is_directory(entry_error) &&
                std::filesystem::is_regular_file(entry.path() / pack_id / "plugin.json", entry_error)) {
                target.plugin_id = entry.path().filename().string();
                return true;
            }
        }
    }
    error = "The selected optional component is not installed and active";
    return false;
}

}  // namespace cyxwiz::runtime
