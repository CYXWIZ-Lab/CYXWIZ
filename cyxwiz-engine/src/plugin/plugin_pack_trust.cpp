#include "plugin_pack_trust.h"

#include "plugin_manager.h"
#include "plugin_pack_checks.h"
#include "../core/addon_identity.h"

#include "backend_pack_installed_files.h"
#include "backend_pack_metadata_verifier.h"
#include "runtime_layout.h"

#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <utility>

namespace cyxwiz::plugin {
namespace {

runtime::VerifiedPackComponent ServiceComponent(const PackComponent& component) {
    return {component.path, component.size, component.sha256};
}

std::vector<runtime::VerifiedPackComponent> ServiceComponents(const std::vector<PackComponent>& components) {
    std::vector<runtime::VerifiedPackComponent> result;
    result.reserve(components.size());
    for (const auto& component : components) result.push_back(ServiceComponent(component));
    return result;
}

// The add-ons active for the runtime's base; false when the runtime's state
// cannot be read.
bool ActivePacks(const std::filesystem::path& runtime_root, std::vector<runtime::ActivePluginPack>& packs,
                 std::vector<std::string>& skipped, std::string& error) {
    runtime::ActiveRuntimeState active;
    return runtime::LoadActiveRuntimeState(runtime_root / "active-runtime.json", active, error) &&
           runtime::ResolveActivePluginPacks(runtime_root, active.base_pack_id, packs, skipped, error);
}

PackRefusal RefusalOf(const runtime::ActivePluginPack& active_pack, std::string reason, std::string detail) {
    PackRefusal refusal;
    refusal.plugin_id = active_pack.plugin_id;
    refusal.pack_id = active_pack.pack_id;
    refusal.title = AddOnTitle(active_pack.plugin_id);
    refusal.directory = active_pack.directory;
    refusal.reason = std::move(reason);
    refusal.detail = std::move(detail);
    refusal.checked_at = std::chrono::system_clock::now();
    return refusal;
}

}  // namespace

PackFileChecks InstalledPackFileChecks() {
    PackFileChecks checks;
    checks.check_folder = [](const std::filesystem::path& folder, const std::vector<PackComponent>& components) {
        return PackFolderCheckOf(runtime::CheckInstalledPackFiles(folder, ServiceComponents(components)));
    };
    checks.check_file = [](const std::filesystem::path& file, const PackComponent& component,
                           PackFileProblem& problem, std::string& error) {
        runtime::PackFileFault fault = runtime::PackFileFault::None;
        if (runtime::VerifyPackComponentFile(file, ServiceComponent(component), error, {}, &fault)) return true;
        problem = PackFileProblemOf(fault);
        return false;
    };
    return checks;
}

bool ReadInstalledPluginPacks(const std::filesystem::path& runtime_root,
                              const runtime::BackendPackMetadataVerifier& verifier,
                              InstalledPluginPacks& output, std::vector<std::string>& skipped,
                              std::string& error) {
    output = {};
    output.runtime_root = runtime_root;
    output.checks = InstalledPackFileChecks();
    std::vector<runtime::ActivePluginPack> active_packs;
    if (!ActivePacks(runtime_root, active_packs, skipped, error)) return false;
    for (const auto& active_pack : active_packs) {
        runtime::VerifiedBackendPackManifest manifest;
        std::string reason;
        if (!runtime::VerifyInstalledPluginPack(runtime_root, active_pack.plugin_id, active_pack.pack_id, verifier,
                                                manifest, reason)) {
            output.refusals.push_back(
                RefusalOf(active_pack, "its signed package record could not be verified", reason));
            continue;
        }
        PackPlugin pack;
        pack.plugin_id = active_pack.plugin_id;
        pack.pack_id = active_pack.pack_id;
        pack.title = AddOnTitle(active_pack.plugin_id);
        pack.package_version = manifest.package_version;
        pack.companion_base_id = manifest.companion_base_id;
        pack.directory = active_pack.directory;
        for (const auto& component : manifest.components)
            pack.components.push_back({component.relative_path, component.size, component.sha256});
        output.packs.push_back(std::move(pack));
    }
    return true;
}

bool ConfigureInstalledPluginPacks(PluginManager& manager, const std::filesystem::path& runtime_root) {
    // Whatever cannot be read, the manager knows the runtime: no add-on loads
    // unchecked, and every folder under its plugins/ is refused (fail closed).
    InstalledPluginPacks packs;
    packs.runtime_root = runtime_root;
    packs.checks = InstalledPackFileChecks();
    std::vector<std::string> skipped;
    bool read = false;
    std::string error;
    const auto verifier = runtime::LoadBackendPackMetadataVerifier(
        runtime_root / "trust" / "trusted-keys.json", GetVersionString(), error);
    if (!verifier) {
        spdlog::warn("Installed add-ons are not loaded: the install's trust file could not be read ({})", error);
        std::vector<runtime::ActivePluginPack> active_packs;
        std::string active_error;
        if (ActivePacks(runtime_root, active_packs, skipped, active_error)) {
            for (const auto& active_pack : active_packs) {
                packs.refusals.push_back(
                    RefusalOf(active_pack, "the install's trust file could not be read", error));
            }
        }
    } else if (!ReadInstalledPluginPacks(runtime_root, *verifier, packs, skipped, error)) {
        spdlog::warn("Installed add-ons are not loaded: the installation's state could not be read ({})", error);
    } else {
        read = true;
    }
    for (const auto& reason : skipped) spdlog::warn("Plugin pack not loaded: {}", reason);
    for (const auto& refusal : packs.refusals) {
        spdlog::warn("Add-on {} was not loaded: {}. Repair it in CyxWiz Installer > Add-ons.", refusal.title,
                     refusal.reason);
        spdlog::debug("Add-on {} ({}): {}", refusal.title, refusal.pack_id, refusal.detail);
    }
    manager.SetPackPlugins(std::move(packs));
    return read;
}

}  // namespace cyxwiz::plugin
