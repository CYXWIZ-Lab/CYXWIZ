#include "backend_pack_metadata_cache.h"

#include "atomic_file_publisher.h"

namespace cyxwiz::runtime {
namespace {

constexpr std::uintmax_t kMaximumMetadataBytes = 16U * 1024U * 1024U;

}  // namespace

bool PublishVerifiedBackendPackMetadata(
    const std::filesystem::path& trusted_keys_path,
    const VerifiedBackendPackCatalogSnapshot& snapshot,
    const std::filesystem::path& runtime_root,
    std::string& error) {
    if (!runtime_root.is_absolute() || !trusted_keys_path.is_absolute() ||
        !snapshot.catalog_path.is_absolute()) {
        error = "Absolute bootstrap metadata and runtime paths are required";
        return false;
    }
    if (!PublishRegularFileAtomic(
            trusted_keys_path,
            runtime_root / "trust" / "trusted-keys.json",
            kMaximumMetadataBytes, error)) {
        return false;
    }
    for (const auto& record : snapshot.records) {
        if (!record.manifest) continue;
        if (record.manifest->pack_id != record.catalog_entry.pack_id ||
            !record.manifest_path.is_absolute()) {
            error = "Verified bootstrap manifest identity is inconsistent";
            return false;
        }
        if (!PublishRegularFileAtomic(
                record.manifest_path,
                BackendPackCachedManifestPath(
                    runtime_root, record.catalog_entry.pack_id),
                kMaximumMetadataBytes, error)) {
            return false;
        }
    }
    if (!PublishRegularFileAtomic(
            snapshot.catalog_path,
            BackendPackCurrentCatalogPath(runtime_root),
            kMaximumMetadataBytes, error)) {
        return false;
    }
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
