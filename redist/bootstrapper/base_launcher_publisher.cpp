#include "base_launcher_publisher.h"

#include "atomic_file_publisher.h"
#include "backend_pack_hash.h"
#include "backend_pack_platform.h"

#include <algorithm>
#include <system_error>
#include <utility>

namespace cyxwiz::runtime {

BaseLauncherPublishResult PublishVerifiedBaseLauncher(
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_base_directory,
    const std::filesystem::path& runtime_root) {
    BaseLauncherPublishResult result;
    if (manifest.kind != BackendPackManifestKind::Base ||
        manifest.backend != "cpu" || !runtime_root.is_absolute() ||
        !installed_base_directory.is_absolute() ||
        installed_base_directory.lexically_normal() !=
            (runtime_root / "base" / manifest.pack_id).lexically_normal()) {
        result.message =
            "Verified CPU-base identity and absolute installed paths are required";
        return result;
    }
    const std::string launcher_name(
        CurrentRuntimeBootstrapperExecutableName());
    const auto component = std::find_if(
        manifest.components.begin(), manifest.components.end(),
        [&](const VerifiedPackComponent& candidate) {
            return candidate.relative_path == launcher_name;
        });
    if (component == manifest.components.end() || component->size == 0) {
        result.message =
            "The verified CPU base does not contain its app-level launcher";
        return result;
    }
    const auto source = installed_base_directory / launcher_name;
    std::error_code filesystem_error;
    const auto source_size = std::filesystem::file_size(source, filesystem_error);
    if (filesystem_error || source_size != component->size) {
        result.message =
            "The installed app-level launcher size differs from the verified manifest";
        return result;
    }
    std::string error;
    result.installed_path = runtime_root.parent_path() / launcher_name;
    if (!PublishRegularFileAtomic(
            source, result.installed_path, component->size, error,
            [&](const std::filesystem::path& temporary,
                std::string& validation_error) {
                std::string temporary_hash;
                if (!Sha256File(
                        temporary, temporary_hash, validation_error)) {
                    return false;
                }
                if (temporary_hash != component->sha256) {
                    validation_error =
                        "The copied app-level launcher hash differs from the verified manifest";
                    return false;
                }
                return true;
            })) {
        result.installed_path.clear();
        result.message = std::move(error);
        return result;
    }
    result.published = true;
    result.message = "Verified app-level launcher published atomically";
    return result;
}

}  // namespace cyxwiz::runtime
