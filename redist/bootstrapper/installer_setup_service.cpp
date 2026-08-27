#include "installer_setup_service.h"

#include "backend_pack_archive_extractor.h"
#include "backend_pack_hash.h"
#include "backend_pack_path.h"
#include "installer_bundle_verifier.h"

#include <map>
#include <set>

namespace cyxwiz::runtime {
namespace {

InstallerSetupResult Finish(
    InstallerSetupStatus status, std::string message,
    std::string bundle_id = {}, std::filesystem::path installer_path = {}) {
    return {status, std::move(message), std::move(bundle_id),
            std::move(installer_path)};
}

bool VerifyCachedBundle(
    const std::filesystem::path& root,
    const VerifiedInstallerBundle& bundle,
    std::string& error) {
    std::error_code filesystem_error;
    const auto root_status = std::filesystem::symlink_status(root, filesystem_error);
    if (filesystem_error || !std::filesystem::is_directory(root_status) ||
        std::filesystem::is_symlink(root_status)) {
        error = "Cached installer bundle is not a regular directory";
        return false;
    }
    std::map<std::string, const VerifiedInstallerBundleComponent*> expected;
    for (const auto& component : bundle.components) {
        expected.emplace(FoldBackendPackPath(component.relative_path), &component);
    }
    std::set<std::string> observed;
    std::filesystem::recursive_directory_iterator iterator(
        root, std::filesystem::directory_options::none, filesystem_error);
    const std::filesystem::recursive_directory_iterator end;
    while (!filesystem_error && iterator != end) {
        const auto status = iterator->symlink_status(filesystem_error);
        if (filesystem_error || std::filesystem::is_symlink(status)) break;
        if (std::filesystem::is_regular_file(status)) {
            const auto relative = iterator->path().lexically_relative(root).generic_string();
            const auto found = expected.find(FoldBackendPackPath(relative));
            if (found == expected.end() ||
                relative != found->second->relative_path ||
                !observed.insert(found->first).second ||
                std::filesystem::file_size(iterator->path(), filesystem_error) !=
                    found->second->size || filesystem_error) {
                error = "Cached installer files differ from signed inventory";
                return false;
            }
#ifndef _WIN32
            if (found->second->executable) {
                const auto permissions = status.permissions();
                const auto executable_bits =
                    std::filesystem::perms::owner_exec |
                    std::filesystem::perms::group_exec |
                    std::filesystem::perms::others_exec;
                if ((permissions & executable_bits) == std::filesystem::perms::none) {
                    error = "Cached installer entry point is not executable";
                    return false;
                }
            }
#endif
            std::string digest;
            if (!Sha256File(iterator->path(), digest, error) ||
                digest != found->second->sha256) {
                error = "Cached installer file hash differs from signed inventory";
                return false;
            }
        } else if (!std::filesystem::is_directory(status)) {
            error = "Cached installer contains an unsupported filesystem entry";
            return false;
        }
        iterator.increment(filesystem_error);
    }
    if (filesystem_error || observed.size() != expected.size()) {
        error = filesystem_error
            ? "Cannot inspect cached installer: " + filesystem_error.message()
            : "Cached installer is missing signed files";
        return false;
    }
    return true;
}

}  // namespace

InstallerSetupResult PrepareInstallerBundle(
    const InstallerSetupRequest& request,
    BackendPackArtifactSource& archive_source) {
    if (!request.descriptor_path.is_absolute() ||
        !request.trust_store_path.is_absolute() ||
        !request.cache_root.is_absolute() || request.current_utc.empty() ||
        request.setup_version.empty() || request.platform.empty() ||
        request.architecture.empty()) {
        return Finish(
            InstallerSetupStatus::InvalidRequest,
            "Absolute descriptor, trust, cache paths and setup identity are required");
    }
    std::string error;
    auto trust_store = BackendPackTrustStore::Load(request.trust_store_path, error);
    if (!trust_store) {
        return Finish(
            InstallerSetupStatus::TrustFailure,
            "Cannot load installer trust store: " + error);
    }
    InstallerBundleVerifier verifier(
        std::move(*trust_store), request.setup_version,
        request.platform, request.architecture);
    VerifiedInstallerBundle bundle;
    if (!verifier.Verify(
            request.descriptor_path, request.current_utc, bundle, error)) {
        return Finish(
            InstallerSetupStatus::TrustFailure,
            "Installer descriptor was rejected: " + error);
    }
    const auto archive_path = request.cache_root / "archives" /
        BackendPackNativeRelativePath(bundle.archive.file_name);
    BackendPackArtifactAcquirer acquirer;
    const auto acquisition = acquirer.Acquire(
        archive_source, archive_path, bundle.archive.size,
        bundle.archive.sha256, bundle.archive.size);
    if (acquisition.status != BackendPackAcquisitionStatus::Downloaded &&
        acquisition.status != BackendPackAcquisitionStatus::AlreadyPresent) {
        return Finish(
            InstallerSetupStatus::AcquisitionFailure,
            "Cannot acquire installer archive: " + acquisition.message,
            bundle.bundle_id);
    }
    const auto bundle_root = request.cache_root / "bundles" /
        BackendPackNativeRelativePath(bundle.bundle_id);
    std::error_code filesystem_error;
    if (std::filesystem::exists(bundle_root, filesystem_error)) {
        if (filesystem_error || !VerifyCachedBundle(bundle_root, bundle, error)) {
            return Finish(
                InstallerSetupStatus::CacheFailure,
                error.empty() ? "Cannot inspect cached installer bundle" : error,
                bundle.bundle_id);
        }
    } else {
        BackendPackArchiveExtractor extractor;
        const auto extraction = extractor.ExtractInstallerBundle(
            acquisition.artifact_path, bundle, bundle_root,
            512U * 1024U * 1024U);
        if (extraction.status != BackendPackExtractionStatus::Extracted) {
            return Finish(
                InstallerSetupStatus::ExtractionFailure,
                "Cannot publish installer bundle: " + extraction.message,
                bundle.bundle_id);
        }
    }
    const auto installer_path = bundle_root /
        BackendPackNativeRelativePath(bundle.InstallerEntryPoint());
    return Finish(
        InstallerSetupStatus::Ready,
        "Signed installer bundle is verified and ready",
        bundle.bundle_id, installer_path);
}

const char* InstallerSetupStatusName(InstallerSetupStatus status) {
    switch (status) {
        case InstallerSetupStatus::Ready: return "ready";
        case InstallerSetupStatus::InvalidRequest: return "invalid_request";
        case InstallerSetupStatus::TrustFailure: return "trust_failure";
        case InstallerSetupStatus::AcquisitionFailure: return "acquisition_failure";
        case InstallerSetupStatus::ExtractionFailure: return "extraction_failure";
        case InstallerSetupStatus::CacheFailure: return "cache_failure";
    }
    return "unknown";
}

}  // namespace cyxwiz::runtime
