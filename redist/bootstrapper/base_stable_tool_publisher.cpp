#include "base_stable_tool_publisher.h"

#include "atomic_file_publisher.h"
#include "backend_pack_hash.h"
#include "backend_pack_platform.h"

#include <algorithm>
#include <string_view>
#include <system_error>

namespace cyxwiz::runtime {
namespace {

bool PublishVerifiedStableTool(
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_base_directory,
    const std::filesystem::path& install_root,
    std::string_view tool_name,
    std::filesystem::path& published_path,
    std::string& error) {
    const auto component = std::find_if(
        manifest.components.begin(), manifest.components.end(),
        [&](const VerifiedPackComponent& candidate) {
            return candidate.relative_path == tool_name;
        });
    if (component == manifest.components.end() || component->size == 0) {
        error = "The verified CPU base does not contain required stable tool " +
            std::string(tool_name);
        return false;
    }

    const auto source = installed_base_directory / std::string(tool_name);
    std::error_code filesystem_error;
    const auto source_status = std::filesystem::symlink_status(
        source, filesystem_error);
    const auto source_size = std::filesystem::file_size(
        source, filesystem_error);
    if (filesystem_error ||
        source_status.type() != std::filesystem::file_type::regular ||
        source_size != component->size) {
        error =
            "The installed stable tool differs from its verified manifest: " +
            std::string(tool_name);
        return false;
    }

    published_path = install_root / std::string(tool_name);
    if (!PublishRegularFileAtomic(
            source, published_path, component->size, error,
            [&](const std::filesystem::path& candidate,
                std::string& validation_error) {
                std::string candidate_hash;
                if (!Sha256File(
                        candidate, candidate_hash, validation_error)) {
                    return false;
                }
                if (candidate_hash != component->sha256) {
                    validation_error =
                        "The copied stable tool hash differs from the verified manifest";
                    return false;
                }
                return true;
            })) {
        published_path.clear();
        return false;
    }
    return true;
}

}  // namespace

BaseStableToolsPublishResult PublishVerifiedBaseStableTools(
    const VerifiedBackendPackManifest& manifest,
    const std::filesystem::path& installed_base_directory,
    const std::filesystem::path& runtime_root) {
    BaseStableToolsPublishResult result;
    if (manifest.kind != BackendPackManifestKind::Base ||
        manifest.backend != "cpu" || !runtime_root.is_absolute() ||
        !installed_base_directory.is_absolute() ||
        installed_base_directory.lexically_normal() !=
            (runtime_root / "base" / manifest.pack_id).lexically_normal()) {
        result.message =
            "Verified CPU-base identity and absolute installed paths are required";
        return result;
    }

    const auto install_root = runtime_root.parent_path();
    if (!PublishVerifiedStableTool(
            manifest, installed_base_directory, install_root,
            CurrentProductRemovalFinalizerExecutableName(),
            result.finalizer_path, result.message) ||
        !PublishVerifiedStableTool(
            manifest, installed_base_directory, install_root,
            CurrentRuntimeBootstrapperExecutableName(),
            result.launcher_path, result.message)) {
        return result;
    }
    result.published = true;
    result.message =
        "Verified stable launcher and removal finalizer published atomically";
    return result;
}

}  // namespace cyxwiz::runtime
