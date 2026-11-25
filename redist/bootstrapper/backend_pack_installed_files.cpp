#include "backend_pack_installed_files.h"

#include "backend_pack_installed_metadata.h"
#include "backend_pack_path.h"

#include <algorithm>
#include <iterator>

namespace cyxwiz::runtime {

bool VerifyPackComponentFile(
    const std::filesystem::path& path,
    const VerifiedPackComponent& component,
    std::string& error,
    const Sha256FileProgress& hash_progress,
    PackFileFault* fault) {
    const auto fail = [fault](PackFileFault kind) {
        if (fault) *fault = kind;
        return false;
    };
    if (fault) *fault = PackFileFault::None;
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
        error = "Component is missing or is not a regular file: " +
                component.relative_path;
        return fail(PackFileFault::Missing);
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size != component.size) {
        error = "Component size differs from signed metadata: " +
                component.relative_path;
        return fail(PackFileFault::SizeDiffers);
    }
    std::string digest;
    if (!Sha256File(path, digest, error, hash_progress)) {
        return fail(PackFileFault::Unreadable);
    }
    if (digest != component.sha256) {
        error = "Component SHA-256 differs from signed metadata: " +
                component.relative_path;
        return fail(PackFileFault::ContentDiffers);
    }
    return true;
}

std::set<std::string> FoldedPackInventory(
    const std::vector<VerifiedPackComponent>& components) {
    std::set<std::string> folded;
    for (const auto& component : components) {
        folded.insert(FoldBackendPackPath(component.relative_path));
    }
    return folded;
}

bool VerifyExactPackInventory(
    const std::filesystem::path& root,
    const std::set<std::string>& expected,
    std::string& error,
    std::vector<std::string>* unlisted,
    std::vector<std::string>* missing) {
    std::error_code filesystem_error;
    const auto root_status =
        std::filesystem::symlink_status(root, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root_status) ||
        std::filesystem::is_symlink(root_status)) {
        error = "Pack payload root is missing or unsafe";
        return false;
    }
    std::set<std::string> observed;
    for (std::filesystem::recursive_directory_iterator iterator(root), end;
         iterator != end; ++iterator) {
        const auto status = iterator->symlink_status(filesystem_error);
        if (filesystem_error || std::filesystem::is_symlink(status)) {
            error = "Pack payload contains a link or unreadable entry";
            return false;
        }
        if (std::filesystem::is_directory(status)) continue;
        if (!std::filesystem::is_regular_file(status)) {
            error = "Pack payload contains an unsupported filesystem entry";
            return false;
        }
        const auto relative =
            std::filesystem::relative(iterator->path(), root, filesystem_error);
        if (filesystem_error) {
            error = "Cannot resolve a pack payload path";
            return false;
        }
        observed.insert(FoldBackendPackPath(relative.generic_string()));
    }
    if (observed != expected) {
        if (unlisted) {
            std::set_difference(observed.begin(), observed.end(),
                                expected.begin(), expected.end(),
                                std::back_inserter(*unlisted));
        }
        if (missing) {
            std::set_difference(expected.begin(), expected.end(),
                                observed.begin(), observed.end(),
                                std::back_inserter(*missing));
        }
        error = "Pack payload files differ from signed component inventory";
        return false;
    }
    return true;
}

bool VerifyInstalledPluginPack(
    const std::filesystem::path& runtime_root,
    const std::string& plugin_id,
    const std::string& pack_id,
    const BackendPackMetadataVerifier& verifier,
    VerifiedBackendPackManifest& output,
    std::string& error) {
    VerifiedBackendPackManifest manifest;
    if (!ReadInstalledBackendPackMetadata(
            runtime_root, pack_id, BackendPackManifestKind::PluginPack,
            verifier, manifest, error)) {
        return false;
    }
    if (manifest.plugin_id != plugin_id || manifest.pack_id != pack_id) {
        error = "The signed metadata of " + pack_id + " is for " +
                manifest.plugin_id + " " + manifest.pack_id + ", not " +
                plugin_id;
        return false;
    }
    output = std::move(manifest);
    return true;
}

PackFilesCheck CheckInstalledPackFiles(
    const std::filesystem::path& folder,
    const std::vector<VerifiedPackComponent>& components) {
    PackFilesCheck check;
    std::vector<std::string> unlisted;
    std::vector<std::string> missing;
    std::string error;
    if (!VerifyExactPackInventory(folder, FoldedPackInventory(components),
                                  error, &unlisted, &missing)) {
        check.error = error;
        for (const auto& path : unlisted) {
            check.findings.push_back({path, PackFileFault::NotListed});
        }
    }
    // A link or special file stops the walk: nothing more can be trusted.
    if (!check.error.empty() && unlisted.empty() && missing.empty()) {
        return check;
    }
    for (const auto& component : components) {
        PackFileFault fault = PackFileFault::None;
        std::string file_error;
        if (!VerifyPackComponentFile(
                folder / BackendPackNativeRelativePath(component.relative_path),
                component, file_error, {}, &fault)) {
            check.findings.push_back({component.relative_path, fault});
            if (check.error.empty()) check.error = file_error;
            continue;
        }
        ++check.files_checked;
        check.bytes_checked += component.size;
    }
    return check;
}

}  // namespace cyxwiz::runtime
