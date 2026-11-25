#include "plugin_pack_trust.h"

#include "plugin_manager.h"
#include "../core/addon_identity.h"

#include "backend_pack_installed_files.h"
#include "backend_pack_metadata_verifier.h"
#include "backend_pack_path.h"
#include "backend_pack_platform.h"
#include "runtime_layout.h"

#include <cyxwiz/cyxwiz.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <utility>

namespace cyxwiz::plugin {
namespace {

PackFileProblem ProblemOf(runtime::PackFileFault fault) {
    switch (fault) {
        case runtime::PackFileFault::Missing: return PackFileProblem::Missing;
        case runtime::PackFileFault::SizeDiffers: return PackFileProblem::SizeDiffers;
        case runtime::PackFileFault::Unreadable: return PackFileProblem::Unreadable;
        case runtime::PackFileFault::NotListed: return PackFileProblem::NotListed;
        case runtime::PackFileFault::ContentDiffers:
        case runtime::PackFileFault::None: break;
    }
    return PackFileProblem::ContentDiffers;
}

runtime::VerifiedPackComponent ServiceComponent(const PackComponent& component) {
    return {component.path, component.size, component.sha256};
}

std::vector<runtime::VerifiedPackComponent> ServiceComponents(const std::vector<PackComponent>& components) {
    std::vector<runtime::VerifiedPackComponent> result;
    result.reserve(components.size());
    for (const auto& component : components) result.push_back(ServiceComponent(component));
    return result;
}

}  // namespace

PackFileChecks InstalledPackFileChecks() {
    PackFileChecks checks;
    checks.check_folder = [](const std::filesystem::path& folder, const std::vector<PackComponent>& components) {
        const auto check = runtime::CheckInstalledPackFiles(folder, ServiceComponents(components));
        PackFolderCheck result;
        result.files_checked = check.files_checked;
        result.bytes_checked = check.bytes_checked;
        for (const auto& finding : check.findings)
            result.issues.push_back({finding.relative_path, ProblemOf(finding.fault)});
        // A check that stopped on a link or special file has no file to name.
        if (!check.error.empty() && check.findings.empty()) result.error = check.error;
        return result;
    };
    checks.check_file = [](const std::filesystem::path& file, const PackComponent& component,
                           PackFileProblem& problem, std::string& error) {
        runtime::PackFileFault fault = runtime::PackFileFault::None;
        if (runtime::VerifyPackComponentFile(file, ServiceComponent(component), error, {}, &fault)) return true;
        problem = ProblemOf(fault);
        return false;
    };
    return checks;
}

bool ReadInstalledPluginPacks(const std::filesystem::path& runtime_root,
                              const runtime::BackendPackMetadataVerifier& verifier,
                              InstalledPluginPacks& output, std::vector<std::string>& skipped,
                              std::string& error) {
    runtime::ActiveRuntimeState active;
    std::vector<runtime::ActivePluginPack> active_packs;
    if (!runtime::LoadActiveRuntimeState(runtime_root / "active-runtime.json", active, error) ||
        !runtime::ResolveActivePluginPacks(runtime_root, active.base_pack_id, active_packs, skipped, error)) {
        return false;
    }
    output = {};
    output.runtime_root = runtime_root;
    output.checks = InstalledPackFileChecks();
    for (const auto& active_pack : active_packs) {
        runtime::VerifiedBackendPackManifest manifest;
        std::string reason;
        if (!runtime::VerifyInstalledPluginPack(runtime_root, active_pack.plugin_id, active_pack.pack_id, verifier,
                                                manifest, reason)) {
            PackRefusal refusal;
            refusal.plugin_id = active_pack.plugin_id;
            refusal.pack_id = active_pack.pack_id;
            refusal.title = AddOnTitle(active_pack.plugin_id);
            refusal.directory = active_pack.directory;
            refusal.reason = "its signed package record could not be verified";
            refusal.detail = reason;
            refusal.checked_at = std::chrono::system_clock::now();
            output.refusals.push_back(std::move(refusal));
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
    std::string error;
    auto trust = runtime::BackendPackTrustStore::Load(runtime_root / "trust" / "trusted-keys.json", error);
    if (!trust) {
        spdlog::warn("Installed add-ons were not loaded: the install's trust file could not be read ({})", error);
        return false;
    }
    const runtime::BackendPackMetadataVerifier verifier(
        std::move(*trust), GetVersionString(), std::string(runtime::CurrentBackendPackPlatformId()),
        std::string(runtime::CurrentBackendPackArchitectureId()));
    InstalledPluginPacks packs;
    std::vector<std::string> skipped;
    if (!ReadInstalledPluginPacks(runtime_root, verifier, packs, skipped, error)) {
        spdlog::warn("Installed plugin packs were not read: {}", error);
        return false;
    }
    for (const auto& reason : skipped) spdlog::warn("Plugin pack not loaded: {}", reason);
    for (const auto& refusal : packs.refusals) {
        spdlog::warn("Add-on {} was not loaded: {}. Repair it in CyxWiz Installer > Add-ons.", refusal.title,
                     refusal.reason);
        spdlog::debug("Add-on {} ({}): {}", refusal.title, refusal.pack_id, refusal.detail);
    }
    manager.SetPackPlugins(std::move(packs));
    return true;
}

}  // namespace cyxwiz::plugin
