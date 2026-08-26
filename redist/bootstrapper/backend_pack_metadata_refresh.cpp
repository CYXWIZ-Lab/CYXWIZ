#include "backend_pack_metadata_refresh.h"

#include "atomic_file_publisher.h"
#include "backend_pack_acquisition.h"
#include "backend_pack_lifecycle_service.h"
#include "backend_pack_metadata_cache.h"

#include <chrono>
#include <fstream>
#include <system_error>
#include <utility>

namespace cyxwiz::runtime {
namespace {

constexpr std::uint64_t kMaximumMetadataBytes = 16U * 1024U * 1024U;

class RemoveRefreshStaging {
public:
    explicit RemoveRefreshStaging(std::filesystem::path path)
        : path_(std::move(path)) {}

    ~RemoveRefreshStaging() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

private:
    std::filesystem::path path_;
};

BackendPackMetadataRefreshResult Finish(
    BackendPackMetadataRefreshStatus status,
    std::string message,
    std::string catalog_id = {},
    std::size_t verified_pack_count = 0) {
    return {status, std::move(message), std::move(catalog_id),
            verified_pack_count};
}

bool CreateRefreshStaging(
    const std::filesystem::path& destination_root,
    std::filesystem::path& output,
    std::string& error) {
    const auto parent = destination_root / "cache" / "metadata-refresh";
    std::error_code filesystem_error;
    std::filesystem::create_directories(parent, filesystem_error);
    if (filesystem_error) {
        error = "Cannot create the metadata refresh cache: " +
            filesystem_error.message();
        return false;
    }
    const auto seed = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    for (unsigned int attempt = 0; attempt < 16; ++attempt) {
        const auto candidate = parent /
            ("refresh-" + std::to_string(seed) + "-" +
             std::to_string(attempt));
        if (std::filesystem::create_directory(candidate, filesystem_error)) {
            output = candidate;
            return true;
        }
        if (filesystem_error) {
            error = "Cannot create private metadata refresh staging: " +
                filesystem_error.message();
            return false;
        }
    }
    error = "Cannot allocate private metadata refresh staging";
    return false;
}

}  // namespace

bool HttpsBackendPackMetadataSource::Fetch(
    const std::string& url,
    const std::filesystem::path& destination,
    std::uint64_t maximum_bytes,
    std::string& error) {
    if (!destination.is_absolute() || maximum_bytes == 0 ||
        maximum_bytes > kMaximumMetadataBytes) {
        error = "Absolute metadata destination and a valid byte bound are required";
        return false;
    }
    std::error_code filesystem_error;
    std::filesystem::create_directories(
        destination.parent_path(), filesystem_error);
    if (filesystem_error || std::filesystem::exists(
            destination, filesystem_error) || filesystem_error) {
        error = "Cannot prepare a new metadata document destination";
        return false;
    }

    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "Cannot create the metadata document staging file";
        return false;
    }
    std::uint64_t written = 0;
    HttpsBackendPackArtifactSource transfer(url);
    std::uint64_t received = 0;
    const bool transferred = transfer.TransferBounded(
        maximum_bytes,
        [&](const char* data, std::size_t size, std::string& sink_error) {
            if (size > maximum_bytes - written) {
                sink_error = "Metadata document exceeds its byte bound";
                return false;
            }
            output.write(data, static_cast<std::streamsize>(size));
            if (!output) {
                sink_error = "Cannot write the metadata document staging file";
                return false;
            }
            written += size;
            return true;
        },
        [] { return false; }, received, error);
    output.flush();
    const bool complete = transferred && output && written == received;
    output.close();
    if (!complete) {
        if (error.empty()) error = "Metadata document transfer is incomplete";
        std::filesystem::remove(destination, filesystem_error);
        return false;
    }
    const auto size = std::filesystem::file_size(destination, filesystem_error);
    if (filesystem_error || size != written || size == 0) {
        error = "Metadata document staging size is inconsistent";
        std::filesystem::remove(destination, filesystem_error);
        return false;
    }
    error.clear();
    return true;
}

BackendPackMetadataRefreshResult RefreshBackendPackMetadata(
    const BackendPackMetadataRefreshRequest& request,
    const BackendPackMetadataVerifier& verifier,
    BackendPackMetadataSource& source) {
    if (!request.destination_root.is_absolute() ||
        !request.trusted_keys_path.is_absolute() ||
        request.catalog_url.rfind("https://", 0) != 0 ||
        request.current_utc.empty()) {
        return Finish(
            BackendPackMetadataRefreshStatus::InvalidRequest,
            "HTTPS catalog URL, absolute metadata paths, and current UTC are required");
    }

    std::string error;
    std::filesystem::path staging_root;
    if (!CreateRefreshStaging(
            request.destination_root, staging_root, error)) {
        return Finish(
            BackendPackMetadataRefreshStatus::PublicationFailure,
            std::move(error));
    }
    RemoveRefreshStaging cleanup(staging_root);

    const auto staged_trusted_keys =
        staging_root / "trust" / "trusted-keys.json";
    if (!PublishRegularFileAtomic(
            request.trusted_keys_path, staged_trusted_keys,
            kMaximumMetadataBytes, error)) {
        return Finish(
            BackendPackMetadataRefreshStatus::PublicationFailure,
            "Cannot stage the trusted signing keys: " + error);
    }

    VerifiedBackendPackCatalogSnapshot snapshot;
    snapshot.catalog_path = staging_root / "catalogs" / "current.json";
    if (!source.Fetch(
            request.catalog_url, snapshot.catalog_path,
            kMaximumMetadataBytes, error)) {
        return Finish(
            BackendPackMetadataRefreshStatus::SourceFailure,
            "Cannot download the signed catalog: " + error);
    }
    if (!verifier.VerifyCatalog(
            snapshot.catalog_path, request.current_utc,
            snapshot.catalog, error)) {
        return Finish(
            BackendPackMetadataRefreshStatus::VerificationFailure,
            "Downloaded catalog was not trusted: " + error);
    }

    snapshot.records.reserve(snapshot.catalog.packs.size());
    std::size_t verified_pack_count = 0;
    for (const auto& entry : snapshot.catalog.packs) {
        VerifiedBackendPackCatalogRecord record;
        record.catalog_entry = entry;
        record.manifest_path =
            BackendPackCachedManifestPath(staging_root, entry.pack_id);
        if (entry.support_status == BackendPackSupportStatus::Blocked ||
            entry.support_status == BackendPackSupportStatus::Revoked) {
            record.manifest_error =
                "Catalog policy blocks this backend pack";
            snapshot.records.push_back(std::move(record));
            continue;
        }
        if (!source.Fetch(
                entry.manifest_url, record.manifest_path,
                kMaximumMetadataBytes, error)) {
            return Finish(
                BackendPackMetadataRefreshStatus::SourceFailure,
                "Cannot download signed manifest for " + entry.pack_id +
                    ": " + error,
                snapshot.catalog.catalog_id, verified_pack_count);
        }
        VerifiedBackendPackManifest manifest;
        if (!verifier.VerifyManifest(
                record.manifest_path, entry, manifest, error)) {
            std::string base_error;
            if (!verifier.VerifyManifest(
                    record.manifest_path, entry, manifest, base_error,
                    BackendPackManifestKind::Base)) {
                return Finish(
                    BackendPackMetadataRefreshStatus::VerificationFailure,
                    "Downloaded manifest for " + entry.pack_id +
                        " was not trusted: " + error,
                    snapshot.catalog.catalog_id, verified_pack_count);
            }
            error.clear();
        }
        record.manifest = std::move(manifest);
        ++verified_pack_count;
        snapshot.records.push_back(std::move(record));
    }

    if (!PublishVerifiedBackendPackMetadata(
            staged_trusted_keys, snapshot,
            request.destination_root, error)) {
        return Finish(
            BackendPackMetadataRefreshStatus::PublicationFailure,
            "Cannot publish the verified catalog: " + error,
            snapshot.catalog.catalog_id, verified_pack_count);
    }
    return Finish(
        BackendPackMetadataRefreshStatus::Refreshed,
        "Verified signed catalog " + snapshot.catalog.catalog_id +
            " is current",
        snapshot.catalog.catalog_id, verified_pack_count);
}

const char* BackendPackMetadataRefreshStatusName(
    BackendPackMetadataRefreshStatus status) {
    switch (status) {
        case BackendPackMetadataRefreshStatus::Refreshed:
            return "refreshed";
        case BackendPackMetadataRefreshStatus::InvalidRequest:
            return "invalid_request";
        case BackendPackMetadataRefreshStatus::SourceFailure:
            return "source_failure";
        case BackendPackMetadataRefreshStatus::VerificationFailure:
            return "verification_failure";
        case BackendPackMetadataRefreshStatus::PublicationFailure:
            return "publication_failure";
    }
    return "unknown";
}

}  // namespace cyxwiz::runtime
